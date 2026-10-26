"""Checkpoint 2's numbers, from rows and judgments on disk: nothing is re-generated or re-judged.

Reads `rows/`, `judgments/<judge>/`, `provenance/generation_points_*.json` and writes
`analysis/pilot_validation.json` plus the tables that `analysis/pilot_report.md` quotes.
What it computes (PREREG 9; every one of these is a plumbing number, not a result):

* per point: rows, generation statuses, truncation, judge statuses, coherence exclusion share;
* the generation controls: k = 0 byte-identity against the unhooked run, the unhooked run's
  reproducibility, the hook-fire count against its exact expectation, and that k != 0 changes
  ids - the identity check is void unless all three positive controls hold;
* judge throughput as achieved (seconds per call from the run provenance, and from the
  timestamps of the calls that were actually sent);
* the determinism re-send: exact match and the score deltas;
* the bridge: parse coverage, P(other-regarding) per point, and each judge's AUC of the trait
  score against the parsed A/B choice, with a question-clustered bootstrap interval;
* judge-judge agreement on the rows both judges scored (Spearman, quadratic-weighted kappa and
  Krippendorff's alpha on binned scores, bootstrap over items);
* pairwise: P(steered chosen), order consistency, position bias, judge-judge agreement.

Human agreement is not computed here: it needs the two labellers' exports
(`scripts/measurement/agreement.py`), which do not exist yet. It is reported as not run.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
for sub in ("judging", "measurement"):
    sys.path.insert(0, str(PACKAGE / "scripts" / sub))

import agreement  # noqa: E402
import bridge_parse  # noqa: E402
import gqt_judging as gj  # noqa: E402
import score_stats as ss  # noqa: E402

N_BOOT = 10000
SEED = 20261028


def point_key(row):
    return "%s k%+g" % (row["arm"], row["k"])


# --- generation controls -----------------------------------------------------------

def _by_id(rows):
    return {(r["question_id"], r["sample_index"]): r for r in rows}


def identity_checks(points, rows):
    """k = 0 vs unhooked byte-identity and its three positive controls (PREREG 3).

    `points` maps arm/k/set -> rows. The check is void unless (a) the unhooked run
    reproduces itself, (b) the hook fired exactly as often as the batches' lengths say, and
    (c) a k != 0 point changes the ids. Anything that could not be checked is `None`.
    """
    def ids(name):
        got = points.get(name)
        return None if got is None else {k: r["output_ids_sha256"] for k, r in _by_id(got).items()}

    base, unhooked, repeat = ids("baseline k+0|heldout"), ids("unhooked k+0|heldout"), ids("unhooked_repeat k+0|heldout")

    def same(a, b):
        if a is None or b is None or set(a) != set(b):
            return None
        return sum(a[k] == b[k] for k in a), len(a)

    steered = {}
    for name in points:
        head, qset = name.split("|")
        if qset != "heldout" or head.startswith(("baseline", "unhooked")):
            continue
        steered[head] = same(base, ids(name))
    return {
        "k0_hooked_vs_unhooked_identical_rows": same(base, unhooked),
        "unhooked_vs_unhooked_repeat_identical_rows": same(unhooked, repeat),
        "k_nonzero_vs_k0_identical_rows": steered,
    }


def control_verdict(identity, hook_points):
    """pass / fail / not_run for the byte-identity check, with its positive controls."""
    parts = {}
    k0 = identity["k0_hooked_vs_unhooked_identical_rows"]
    parts["k0_byte_identical"] = None if k0 is None else k0[0] == k0[1]
    rep = identity["unhooked_vs_unhooked_repeat_identical_rows"]
    parts["control_a_unhooked_reproducible"] = None if rep is None else rep[0] == rep[1]
    fires = [p for p in hook_points.values() if p.get("hooked")]
    parts["control_b_hook_fired_exactly_as_expected"] = (
        None if not fires else all(p["hook_fires"] == p["hook_fires_expected"] and p["hook_fires"] > 0
                                   for p in fires))
    changed = [v for v in identity["k_nonzero_vs_k0_identical_rows"].values() if v is not None]
    parts["control_c_k_nonzero_changes_ids"] = (None if not changed else all(v[0] < v[1] for v in changed))
    parts["check_is_valid"] = all(parts[k] is True for k in parts if k.startswith("control_"))
    parts["verdict"] = ("not_run" if any(v is None for v in parts.values())
                        else "pass" if parts["k0_byte_identical"] and parts["check_is_valid"] else "fail")
    return parts


# --- judge joins --------------------------------------------------------------------

def load_judgments(judgments_dir):
    """judge dir name -> kind -> key -> latest row."""
    out = {}
    for judge_dir in sorted(p for p in Path(judgments_dir).iterdir() if p.is_dir()):
        out[judge_dir.name] = {kind: gj.latest_by_key(gj.read_jsonl(judge_dir / ("%s.jsonl" % kind)))
                               for kind in gj.KINDS}
    return out


def joined(rows, judgments, judge):
    """Generation rows with that judge's trait / coherence columns, as `score_stats` reads them."""
    trait = {r["row_id"]: r for r in judgments.get(judge, {}).get("trait", {}).values()}
    coherence = {r["row_id"]: r for r in judgments.get(judge, {}).get("coherence", {}).values()}
    out = []
    for row in rows:
        merged = dict(row)
        for name, table in (("trait", trait), ("coherence", coherence)):
            got = table.get(row["row_id"])
            merged[name] = None if got is None else got["score"]
            merged[name + "_status"] = "not_judged" if got is None else got["status"]
        out.append(merged)
    return out


def status_table(judgments, judge, kind):
    """kind status counts overall."""
    counts = {}
    for r in judgments.get(judge, {}).get(kind, {}).values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


# --- throughput and determinism -------------------------------------------------------

def throughput(judge_dir):
    """Seconds per call as achieved: from the run provenance, and from the call timestamps."""
    import datetime as dt
    runs = []
    for path in sorted(Path(judge_dir).glob("run_*.json")):
        run = json.loads(path.read_text())
        if run.get("phase") == "judge" and run.get("n_sent"):
            runs.append({"run_id": run["run_id"], "kinds": run["kinds"], "n_sent": run["n_sent"],
                         "seconds": run["seconds"], "seconds_per_call": run["seconds"] / run["n_sent"]})
    stamps = []
    for kind in gj.KINDS:
        for r in gj.read_jsonl(Path(judge_dir) / ("%s.jsonl" % kind)):
            if not r["from_cache"]:
                stamps.append((r["timestamp"], kind))
    stamps.sort()
    gaps = {}
    times = [(dt.datetime.fromisoformat(t), k) for t, k in stamps]
    for (t0, _), (t1, k1) in zip(times, times[1:]):
        gap = (t1 - t0).total_seconds()
        if gap < 120:        # consecutive calls of one run, not the hours between runs
            gaps.setdefault(k1, []).append(gap)
    per_kind = {k: {"n": len(v), "mean_s": statistics.mean(v), "median_s": statistics.median(v)}
                for k, v in gaps.items() if v}
    total = sum(r["n_sent"] for r in runs)
    return {"runs": runs, "n_sent": total,
            "seconds_per_call_overall": (sum(r["seconds"] for r in runs) / total) if total else None,
            "inter_call_gaps_by_kind": per_kind}


def determinism(judge_dir):
    rows = gj.read_jsonl(Path(judge_dir) / "determinism_resend.jsonl")
    if not rows:
        return {"n": 0}
    deltas = [abs(r["score_delta"]) for r in rows if r["score_delta"] is not None]
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    return {
        "n": len(rows), "exact_match": sum(r["exact_match"] for r in rows),
        "same_status": sum(r["same_status"] for r in rows),
        "n_with_score_both": len(deltas),
        "mean_abs_delta": statistics.mean(deltas) if deltas else None,
        "max_abs_delta": max(deltas) if deltas else None,
        "n_delta_nonzero": sum(d != 0 for d in deltas),
        "by_kind": {k: {"n": len(v), "exact_match": sum(r["exact_match"] for r in v)}
                    for k, v in by_kind.items()},
        "mismatches": [{"key": r["key"], "score_original": r["score_original"],
                        "score_resend": r["score_resend"]} for r in rows if not r["exact_match"]][:20],
    }


# --- bridge ------------------------------------------------------------------------

def auc_vectorised(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def clustered_auc(items, n_boot=N_BOOT, seed=SEED, level=0.95):
    """AUC of score against choice, `items` = [(question_id, score, other_regarding 0/1)].

    Questions are resampled with replacement; every row of a drawn question comes along.
    """
    by_q = {}
    for q, score, label in items:
        by_q.setdefault(q, []).append((score, label))
    keys = sorted(by_q)
    point = auc_vectorised([s for _, s, l in items if l == 1], [s for _, s, l in items if l == 0])
    if point is None:
        return {"auc": None, "lo": None, "hi": None, "n": len(items)}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(keys), size=len(keys))
        sample = [pair for i in picks for pair in by_q[keys[i]]]
        value = auc_vectorised([s for s, l in sample if l == 1], [s for s, l in sample if l == 0])
        if value is not None:
            draws.append(value)
    alpha = (1 - level) / 2
    return {"auc": point, "lo": float(np.quantile(draws, alpha)), "hi": float(np.quantile(draws, 1 - alpha)),
            "n": len(items), "n_other_regarding": sum(l for _, _, l in items),
            "n_questions": len(keys), "n_boot": len(draws), "seed": seed}


def bridge_labels(bridge_json):
    items = json.loads(Path(bridge_json).read_text())["items"]
    return {it["id"]: it["other_regarding_option"] for it in items}


def bridge_summary(rows, labels):
    """Parsed choices per point; rows gain `parsed_status` and `other_regarding`."""
    per_point, parsed = {}, {}
    for r in rows:
        if r["question_set"] != "bridge":
            continue
        choice = bridge_parse.parse_choice(r["answer"])
        value = bridge_parse.other_regarding(choice, labels[r["question_id"]])
        parsed[r["row_id"]] = (choice.status, value)
        per_point.setdefault(point_key(r), []).append(value)
    return {k: ss.bridge_point(v) for k, v in per_point.items()}, parsed


def bridge_auc(rows, parsed, judgments, judge):
    trait = {r["row_id"]: r for r in judgments.get(judge, {}).get("trait", {}).values()}
    items = []
    n_candidates = 0
    for r in rows:
        if not gj.bridge_auc_row(r):
            continue
        n_candidates += 1
        got, (_, value) = trait.get(r["row_id"]), parsed[r["row_id"]]
        if got is None or got["status"] != "ok" or value is None:
            continue
        items.append((r["question_id"], got["score"], value))
    result = clustered_auc(items)
    result["n_candidate_rows"] = n_candidates
    result["by_k"] = {}
    for k in (-3.0, 0.0, 3.0):
        keep = {r["row_id"] for r in rows if gj.bridge_auc_row(r) and r["k"] == k}
        sub = [(q, s, l) for (q, s, l), r in zip(items, [x for x in rows if gj.bridge_auc_row(x)
                                                         and trait.get(x["row_id"]) and trait[x["row_id"]]["status"] == "ok"
                                                         and parsed[x["row_id"]][1] is not None])
               if r["row_id"] in keep]
        result["by_k"]["%+g" % k] = {"n": len(sub), "auc": auc_vectorised([s for _, s, l in sub if l == 1],
                                                                          [s for _, s, l in sub if l == 0])}
    return result


# --- agreement and pairwise -----------------------------------------------------------

def judge_pair_agreement(judgments, kind, a="llama-3.3-70b-instruct-q8_0", b="gpt-oss-20b-mxfp4",
                         row_filter=None):
    ta = {r["row_id"]: r for r in judgments.get(a, {}).get(kind, {}).values()}
    tb = {r["row_id"]: r for r in judgments.get(b, {}).get(kind, {}).values()}
    ids = sorted(set(ta) & set(tb))
    if row_filter is not None:
        ids = [i for i in ids if row_filter(i)]
    xs = [ta[i]["score"] if ta[i]["status"] == "ok" else None for i in ids]
    ys = [tb[i]["score"] if tb[i]["status"] == "ok" else None for i in ids]
    if not ids:
        return {"n_items": 0}
    result = agreement.bootstrap_pair(xs, ys, n_boot=2000, seed=SEED)
    result["status_counts_a"] = _counts([ta[i]["status"] for i in ids])
    result["status_counts_b"] = _counts([tb[i]["status"] for i in ids])
    return result


def _counts(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def pairwise_summary(judgments, judge, rows):
    """P(steered answer judged more other-regarding), order consistency, position bias."""
    got = judgments.get(judge, {}).get("pairwise", {})
    by_row = {}
    positions = []
    for r in got.values():
        if r["status"] != "ok":
            continue
        positions.append(r["score"])
        by_row.setdefault(r["row_id"], {})[r["order"]] = r["score"] == r["steered_position"]
    points = {}
    row_point = {r["row_id"]: point_key(r) for r in rows}
    consistent = both = 0
    for rid, orders in by_row.items():
        points.setdefault(row_point[rid], []).extend(orders.values())
        if len(orders) == 2:
            both += 1
            consistent += orders["baseline_first"] == orders["steered_first"]
    return {
        "n_calls": len(got), "status_counts": _counts([r["status"] for r in got.values()]),
        "p_steered_chosen_by_point": {k: sum(v) / len(v) for k, v in sorted(points.items())},
        "order_consistency": (consistent / both) if both else None, "n_rows_both_orders": both,
        "position_bias_p_first_minus_half": (sum(p == 1 for p in positions) / len(positions) - 0.5)
        if positions else None,
    }


def pairwise_agreement(judgments, a="llama-3.3-70b-instruct-q8_0", b="gpt-oss-20b-mxfp4"):
    ga = judgments.get(a, {}).get("pairwise", {})
    gb = judgments.get(b, {}).get("pairwise", {})
    keys = [k for k in ga if k in gb and ga[k]["status"] == "ok" and gb[k]["status"] == "ok"]
    agree = sum(ga[k]["score"] == gb[k]["score"] for k in keys)
    return {"n_both_ok": len(keys), "n_agree": agree, "share": (agree / len(keys)) if keys else None}


# --- per point tables ---------------------------------------------------------------------

def point_tables(rows, judgments, judge):
    heldout = [r for r in joined(rows, judgments, judge) if r["question_set"] == "heldout"]
    by_point = {}
    for r in heldout:
        by_point.setdefault(point_key(r), []).append(r)
    out = {}
    for name, group in sorted(by_point.items()):
        s = ss.point_summary(group, n_boot=N_BOOT)
        out[name] = {k: s[k] for k in ("n_rows", "status_counts", "share_truncated", "share_excluded",
                                       "share_coherence_unscored", "coverage_filtered", "trait_filtered",
                                       "trait_unfiltered", "coherence", "degeneracy")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--judgments-dir", required=True)
    ap.add_argument("--provenance-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    all_rows = []
    for path in sorted(Path(args.rows_dir).rglob("*.jsonl")):
        all_rows.extend(gj.read_jsonl(path))
    judged = [r for r in all_rows if r["arm"] not in gj.CONTROL_ARMS]
    judgments = load_judgments(args.judgments_dir)
    primary, secondary = gj.JUDGE_DIRS["primary"], gj.JUDGE_DIRS["secondary"]

    points = {}
    for r in all_rows:
        points.setdefault("%s|%s" % (point_key(r), r["question_set"]), []).append(r)
    hook_points = {}
    for path in sorted(Path(args.provenance_dir).glob("generation_points_*.json")):
        state = json.loads(path.read_text())
        for p in state["points"].values():
            hook_points["%s k%+g|%s" % (p["arm"], p["k"], p["question_set"])] = p
    identity = identity_checks(points, all_rows)
    generation = {
        "points": {name: {"n_rows": len(rows),
                          "statuses": _counts([r["generation_status"] for r in rows]),
                          "share_truncated": sum(r["generation_status"] == "truncated" for r in rows) / len(rows),
                          "mean_generated_tokens": statistics.mean(r["n_generated_tokens"] for r in rows)}
                   for name, rows in sorted(points.items())},
        "hook_points": {k: {f: v[f] for f in ("hooked", "hook_fires", "hook_fires_expected", "n_rows",
                                              "minutes") if f in v} for k, v in sorted(hook_points.items())},
        "identity": identity,
        "controls": control_verdict(identity, hook_points),
    }

    labels = bridge_labels(PACKAGE / "questions" / "bridge.json")
    bridge_points, parsed = bridge_summary(judged, labels)
    heldout_ids = {r["row_id"] for r in judged if r["question_set"] == "heldout"}
    report = {
        "generation": generation,
        "judge_status": {j: {k: status_table(judgments, j, k) for k in gj.KINDS} for j in judgments},
        "throughput": {j: throughput(Path(args.judgments_dir) / j) for j in judgments},
        "determinism": {j: determinism(Path(args.judgments_dir) / j) for j in judgments},
        "points_primary": point_tables(judged, judgments, primary) if primary in judgments else None,
        "bridge": {"points": bridge_points,
                   "auc": {j: bridge_auc(judged, parsed, judgments, j) for j in judgments}},
        "judge_judge": {
            "trait_heldout": judge_pair_agreement(judgments, "trait", row_filter=lambda i: i in heldout_ids),
            "coherence_heldout": judge_pair_agreement(judgments, "coherence", row_filter=lambda i: i in heldout_ids),
            "trait_bridge_auc_subset": judge_pair_agreement(judgments, "trait", row_filter=lambda i: i not in heldout_ids),
            "pairwise_choice_agreement": pairwise_agreement(judgments),
        } if primary in judgments and secondary in judgments else None,
        "pairwise": {j: pairwise_summary(judgments, j, judged) for j in judgments},
        "human_agreement": "not run: the labellers' exports do not exist yet",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
