"""EXPLORATORY analysis: fairness vs altruism on the pilot's held-out answers (2x2 interaction).

Implements docs/fairness_exploratory_prereg.md sections 5-8. Reads only the generation rows
(`rows/`) and the judgments (`judgments/`); never the human sample key or `human/`.

    python analyze_fairness.py        # writes results.json, scores_by_row.csv, tables.md

Measures per row: F = fairness (the new rubric), A = altruism (the existing trait score), both
from the same judge. Primary analysis: a row enters only if its primary-judge coherence is ok and
>= 50 (score_stats.kept) and BOTH F and A are ok. Sensitivity: the same rows with no coherence
filter. The secondary judge (gpt-oss) is descriptive: its 30-row-per-point subset, both scores ok.

Bootstrap: question-clustered, 10,000 draws, seed 20261028; a drawn question is resampled
within (its samples, per point), exactly as score_stats._boot_means does for one arm. All points
of one contrast share the same question draws, so the contrasts are paired by question.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE / "scripts" / "judging"))
sys.path.insert(0, str(PACKAGE / "scripts" / "measurement"))

import gqt_judging as gj  # noqa: E402
import score_stats as ss  # noqa: E402

N_BOOT = ss.N_BOOT
BOOT_SEED = ss.BOOT_SEED
COHERENCE_THRESHOLD = ss.COHERENCE_THRESHOLD
ROLES = {"baseline": "base", "decision": "D", "trait": "T", "shuffled_null": "S", "orthogonal_null": "O"}
JUDGES = {"primary": gj.JUDGE_DIRS["primary"], "secondary": gj.JUDGE_DIRS["secondary"]}
POINTS = [("base", 0.0), ("D", -3.0), ("D", 3.0), ("T", -3.0), ("T", 3.0),
          ("S", -3.0), ("S", 3.0), ("O", -3.0), ("O", 3.0)]


def label(point):
    role, k = point
    return "base" if role == "base" else "%s%+d" % (role, int(k))


def latest(judge, kind):
    return gj.latest_by_key(gj.read_jsonl(PACKAGE / "judgments" / JUDGES[judge] / ("%s.jsonl" % kind)))


def value(record):
    """A score if the call is ok and numeric, else None."""
    if record is None or record["status"] != "ok" or record["score"] is None:
        return None
    return float(record["score"])


def load_rows():
    rows = [r for r in gj.load_generation_rows(PACKAGE / "rows") if r["question_set"] == "heldout"]
    judged = {}
    for judge in JUDGES:
        table = {kind: latest(judge, kind) for kind in ("trait", "coherence", "fairness")}
        out = []
        for r in rows:
            key = r["row_id"] + "|%s"
            point = (ROLES[r["arm_role"]], float(r["k"]))
            if judge == "primary":
                coh = value(table["coherence"].get(key % "coherence"))
            else:
                coh = None
            f = table["fairness"].get(key % "fairness")
            t = table["trait"].get(key % "trait")
            out.append({"row_id": r["row_id"], "point": point, "arm": r["arm"], "k": float(r["k"]),
                        "question_id": r["question_id"], "sample_index": r["sample_index"],
                        "coherence": coh, "F": value(f), "A": value(t),
                        "F_status": None if f is None else f["status"],
                        "A_status": None if t is None else t["status"],
                        "coherence_status": None if table["coherence"].get(key % "coherence") is None
                        else table["coherence"][key % "coherence"]["status"]})
        judged[judge] = [x for x in out if x["F_status"] is not None or x["A_status"] is not None]
    return judged


def kept_row(x, judge):
    if judge == "secondary":
        return True
    return x["coherence"] is not None and x["coherence"] >= COHERENCE_THRESHOLD


def cells_for(judged, judge, filtered=True):
    """point -> question_id -> list of (F, A), for rows with both scores ok (and kept, if filtered)."""
    cells = {}
    for x in judged[judge]:
        if x["F"] is None or x["A"] is None:
            continue
        if filtered and not kept_row(x, judge):
            continue
        cells.setdefault(x["point"], {}).setdefault(x["question_id"], []).append((x["F"], x["A"]))
    return cells


def _resample_means(rng, arrays, picks, Q):
    """For each point: the per-draw, per-pick mean of (F, A) over a fresh within-question resample."""
    n_boot = picks.shape[0]
    out = np.full((n_boot, picks.shape[1], 2), np.nan)
    for j in range(Q):
        where = np.nonzero(picks == j)
        if where[0].size == 0:
            continue
        arr = arrays[j]
        m = arr.shape[0]
        idx = rng.integers(0, m, size=(where[0].size, m))
        out[where[0], where[1], :] = arr[idx].mean(axis=1)
    return out


def contrast(cells, terms, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """terms: list of (weight, point, measure) with measure 0 = F, 1 = A.

    Questions are those with at least one row in every point of the contrast. The estimate is the
    mean over questions of the question-level combination; the interval is the question-clustered
    percentile interval of the same statistic."""
    points = sorted({p for _, p, _ in terms}, key=str)
    qids = sorted(set.intersection(*[set(cells.get(p, {})) for p in points]))
    if not qids:
        return {"estimate": None, "lo": None, "hi": None, "n_questions": 0, "n_rows": 0}
    Q = len(qids)
    per_q = {}
    for p in points:
        per_q[p] = [np.asarray(cells[p][q], dtype=float) for q in qids]
    estimate_by_q = np.zeros(Q)
    for w, p, m in terms:
        estimate_by_q += w * np.array([a[:, m].mean() for a in per_q[p]])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, Q, size=(n_boot, Q))
    total = np.zeros((n_boot, Q))
    for w, p, m in terms:
        means = _resample_means(rng, per_q[p], picks, Q)
        total += w * means[:, :, m]
    boots = total.mean(axis=1)
    alpha = (1 - level) / 2
    n_rows = sum(len(cells[p][q]) for p in points for q in qids)
    return {"estimate": float(estimate_by_q.mean()), "lo": float(np.quantile(boots, alpha)),
            "hi": float(np.quantile(boots, 1 - alpha)), "n_questions": Q, "n_rows": int(n_rows),
            "n_boot": int(n_boot), "seed": seed, "estimator": "question_clustered_bootstrap"}


def verdict(res, direction):
    """supported: CI excludes zero in the predicted direction; contrary: excludes it the other way;
    not resolved: the CI includes zero. direction is +1 (expect > 0) or -1 (expect < 0)."""
    if res["estimate"] is None:
        return "undetermined"
    if direction > 0:
        return "supported" if res["lo"] > 0 else ("contrary" if res["hi"] < 0 else "not resolved")
    return "supported" if res["hi"] < 0 else ("contrary" if res["lo"] > 0 else "not resolved")


def point_rows(judged, judge, filtered=True):
    """The scores table: one line per point, fairness and altruism side by side."""
    out = []
    for p in POINTS:
        rows = [x for x in judged[judge] if x["point"] == p]
        ok_both = [x for x in rows if x["F"] is not None and x["A"] is not None]
        kept_both = [x for x in ok_both if kept_row(x, judge)] if filtered else ok_both
        f_vals = [x["F"] for x in kept_both]
        a_vals = [x["A"] for x in kept_both]
        out.append({
            "point": label(p), "rows": len(rows), "F_ok": sum(x["F"] is not None for x in rows),
            "A_ok": sum(x["A"] is not None for x in rows), "analysed": len(kept_both),
            "F_mean": None if not f_vals else round(float(np.mean(f_vals)), 2),
            "A_mean": None if not a_vals else round(float(np.mean(a_vals)), 2),
            "F_refusals": sum(x["F_status"] == "refusal" for x in rows),
        })
    return out


def contrasts_for(cells, judge):
    """Every contrast of prereg section 6, with the predicted direction where there is one."""
    def F(p, w=1.0):
        return (w, p, 0)

    def A(p, w=1.0):
        return (w, p, 1)

    base = ("base", 0.0)
    D3, D_3, T3, T_3 = ("D", 3.0), ("D", -3.0), ("T", 3.0), ("T", -3.0)
    S3, S_3, O3, O_3 = ("S", 3.0), ("S", -3.0), ("O", 3.0), ("O", -3.0)
    spec = {
        "a1: F(D+3)-F(base)": ([F(D3), F(base, -1)], None),
        "a2: F(D+3)-F(T+3)  [P1]": ([F(D3), F(T3, -1)], +1),
        "a2 at -3: F(D-3)-F(T-3)": ([F(D_3), F(T_3, -1)], None),
        "F(D+3)-F(S+3)": ([F(D3), F(S3, -1)], None),
        "F(D+3)-F(O+3)": ([F(D3), F(O3, -1)], None),
        "F(D-3)-F(S-3)": ([F(D_3), F(S_3, -1)], None),
        "F(D-3)-F(O-3)": ([F(D_3), F(O_3, -1)], None),
        "A(D+3)-A(base)": ([A(D3), A(base, -1)], None),
        "A(D+3)-A(T+3)  [P2]": ([A(D3), A(T3, -1)], -1),
        "A(D-3)-A(T-3)": ([A(D_3), A(T_3, -1)], None),
        "b1 interaction +3 vs T  [P3]": ([F(D3), F(T3, -1), A(D3, -1), A(T3, 1)], +1),
        "c1 interaction -3 vs T": ([F(D_3), F(T_3, -1), A(D_3, -1), A(T_3, 1)], None),
        "c2 interaction +3 vs S": ([F(D3), F(S3, -1), A(D3, -1), A(S3, 1)], None),
        "c3 interaction +3 vs O": ([F(D3), F(O3, -1), A(D3, -1), A(O3, 1)], None),
        "c4 interaction -3 vs S": ([F(D_3), F(S_3, -1), A(D_3, -1), A(S_3, 1)], None),
        "c5 interaction -3 vs O": ([F(D_3), F(O_3, -1), A(D_3, -1), A(O_3, 1)], None),
    }
    out = {}
    for name, (terms, direction) in spec.items():
        res = contrast(cells, terms)
        # the secondary judge is descriptive only (prereg section 2): it enters no verdict
        res["verdict"] = verdict(res, direction) if direction and judge == "primary" else "descriptive"
        out[name] = res
    return out


def discriminant(judged, filtered=True, n_boot=N_BOOT, seed=BOOT_SEED):
    """Pearson r between F and A over the baseline and null rows, question-clustered interval."""
    keep_points = {("base", 0.0), ("S", -3.0), ("S", 3.0), ("O", -3.0), ("O", 3.0)}
    rows = [x for x in judged["primary"] if x["point"] in keep_points and x["F"] is not None
            and x["A"] is not None and (kept_row(x, "primary") if filtered else True)]
    qids = sorted({x["question_id"] for x in rows})
    by_q = {q: np.array([[x["F"], x["A"]] for x in rows if x["question_id"] == q]) for q in qids}

    def r_of(arr):
        if arr.shape[0] < 3 or np.std(arr[:, 0]) == 0 or np.std(arr[:, 1]) == 0:
            return None
        return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])

    everything = np.vstack([by_q[q] for q in qids])
    point_r = r_of(everything)
    spearman = float(scipy_stats.spearmanr(everything[:, 0], everything[:, 1]).statistic)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    boots = []
    for b in range(n_boot):
        arr = np.vstack([by_q[qids[j]] for j in picks[b]])
        r = r_of(arr)
        if r is not None:
            boots.append(r)
    boots = np.asarray(boots)
    return {"pearson_r": point_r, "spearman_r": spearman,
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "n_rows": int(everything.shape[0]), "n_questions": len(qids), "n_boot": int(boots.size),
            "seed": seed, "exceeds_0_8": bool(point_r is not None and point_r > 0.8)}


def fmt(x):
    return "n/a" if x is None else ("%.2f" % x)


def tables_md(results):
    lines = ["# Tables (EXPLORATORY; aggregates only)", ""]
    for judge in ("primary", "secondary"):
        lines += ["## Scores by point: %s judge (%s)" % (judge, JUDGES[judge]), "",
                  "| point | rows | F ok | A ok | analysed | fairness mean | altruism mean | F refusals |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in results["points"][judge]:
            lines.append("| %s | %d | %d | %d | %d | %s | %s | %d |" % (
                r["point"], r["rows"], r["F_ok"], r["A_ok"], r["analysed"], fmt(r["F_mean"]),
                fmt(r["A_mean"]), r["F_refusals"]))
        lines.append("")
    for judge in ("primary", "secondary"):
        lines += ["## Contrasts: %s judge (95%% question-clustered bootstrap)" % judge, "",
                  "| contrast | estimate | 95% CI | questions | rows | verdict |", "|---|---|---|---|---|---|"]
        for name, r in results["contrasts"][judge].items():
            lines.append("| %s | %s | [%s, %s] | %d | %d | %s |" % (
                name, fmt(r["estimate"]), fmt(r["lo"]), fmt(r["hi"]), r["n_questions"], r["n_rows"],
                r["verdict"]))
        lines.append("")
    d = results["discriminant"]
    lines += ["## Discriminant validity (baseline and null rows, primary judge)", "",
              "Pearson r = %s, 95%% CI [%s, %s]; Spearman = %s; rows = %d, questions = %d; r > 0.8: %s"
              % (fmt(d["pearson_r"]), fmt(d["lo"]), fmt(d["hi"]), fmt(d["spearman_r"]), d["n_rows"],
                 d["n_questions"], d["exceeds_0_8"]), ""]
    return "\n".join(lines) + "\n"


def write_scores_csv(judged):
    fields = ["row_id", "judge", "arm", "k", "question_id", "sample_index", "coherence_primary",
              "fairness", "fairness_status", "altruism", "altruism_status"]
    with (HERE / "scores_by_row.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for judge in ("primary", "secondary"):
            for x in judged[judge]:
                writer.writerow({"row_id": x["row_id"], "judge": JUDGES[judge], "arm": x["arm"],
                                 "k": x["k"], "question_id": x["question_id"],
                                 "sample_index": x["sample_index"],
                                 "coherence_primary": x["coherence"] if judge == "primary" else "",
                                 "fairness": "" if x["F"] is None else x["F"],
                                 "fairness_status": x["F_status"] or "",
                                 "altruism": "" if x["A"] is None else x["A"],
                                 "altruism_status": x["A_status"] or ""})


def main():
    judged = load_rows()
    results = {"status": "EXPLORATORY (docs/fairness_exploratory_prereg.md)", "points": {},
               "contrasts": {}, "contrasts_unfiltered": {}, "judges": JUDGES}
    for judge in ("primary", "secondary"):
        results["points"][judge] = point_rows(judged, judge, filtered=(judge == "primary"))
        results["contrasts"][judge] = contrasts_for(cells_for(judged, judge, filtered=(judge == "primary")),
                                                    judge)
    results["contrasts_unfiltered"]["primary"] = contrasts_for(cells_for(judged, "primary", False),
                                                               "primary")
    results["discriminant"] = discriminant(judged)
    results["discriminant_unfiltered"] = discriminant(judged, filtered=False)
    (HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    write_scores_csv(judged)
    (HERE / "tables.md").write_text(tables_md(results), encoding="utf-8")
    print(tables_md(results))


if __name__ == "__main__":
    main()
