"""Draw the human calibration sample, export the blind sheet, the grader page and the judge-reveal files.

PREREG 9.1: about 200 held-out answers, stratified over (arm role x k) of the rows available
at checkpoint 2, plus the k = 0 baseline as its own stratum: strata = {decision, relaxed +
Dictator-only, trait, shuffled null, orthogonal null} x k in {-3, +3} and the baseline; 18 rows
per steered stratum, 20 for the baseline, uniform random within a stratum, seed 20261025.
A stratum with no rows (the pilot has no relaxed or Dictator-only arm) contributes none, and
the shortfall is printed and recorded - the sample is never topped up from another stratum.

Outputs (the operator opens `human/grader.html`; the key and the reveal files stay out of the
labellers' sight until they have finished):

    <human>/calibration_sheet.csv      item_id, question, answer - nothing else, shuffled, opaque ids
    <human>/grader.html                the page (scripts/grader_page.py), items embedded
    <human>/judge_reveal_<judge>.json  audit-mode file: a judge's scores per item (written when judgments exist)
    <provenance>/human_sample_key.json item_id -> row_id and stratum, the seed, the shortfalls

    python export_calibration.py --rows-dir rows --human-dir human --key provenance/human_sample_key.json \\
        [--judgments-dir judgments]
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE / "scripts"))
sys.path.insert(0, str(PACKAGE / "scripts" / "judging"))

import grader_page  # noqa: E402
import gqt_judging as gj  # noqa: E402

SEED = 20261025
PER_STEERED_STRATUM = 18
BASELINE_N = 20
ROLES = ("decision", "relaxed_dictator_only", "trait", "shuffled_null", "orthogonal_null")
KS = (-3.0, 3.0)


def role_of(arm: str):
    if arm == "baseline":
        return "baseline"
    head = arm.split("/")[0]
    return {"decision": "decision", "relaxed": "relaxed_dictator_only",
            "dictator_only": "relaxed_dictator_only", "trait": "trait",
            "shuffled_null": "shuffled_null", "orthogonal_null": "orthogonal_null"}.get(head)


def strata_order():
    return [(role, k) for role in ROLES for k in KS] + [("baseline", 0.0)]


def draw(rows, seed=SEED):
    """(items, key): the shuffled blind sheet rows and the id-to-row key."""
    heldout = [r for r in rows if r["question_set"] == "heldout" and role_of(r["arm"])]
    pools = {}
    for r in heldout:
        pools.setdefault((role_of(r["arm"]), r["k"]), []).append(r["row_id"])
    by_id = {r["row_id"]: r for r in heldout}
    rng = random.Random(seed)
    picked, strata = [], []
    for role, k in strata_order():
        pool = sorted(pools.get((role, k), []))
        want = BASELINE_N if role == "baseline" else PER_STEERED_STRATUM
        take = min(want, len(pool))
        chosen = rng.sample(pool, take) if take else []
        strata.append({"role": role, "k": k, "available": len(pool), "wanted": want, "drawn": take})
        picked.extend((rid, role, k) for rid in chosen)
    rng.shuffle(picked)
    items, key, used = [], {}, set()
    for rid, role, k in picked:
        while True:
            item_id = "i%08x" % rng.getrandbits(32)
            if item_id not in used:
                used.add(item_id)
                break
        row = by_id[rid]
        items.append({"item_id": item_id, "question": row["question"], "answer": row["answer"]})
        key[item_id] = {"row_id": rid, "stratum": "%s k%+g" % (role, k)}
    return items, key, strata


def reveal_file(judge_dir, items, key):
    """Audit-mode JSON for one judge: its trait and coherence scores per item (a missing call is absent)."""
    rows = {}
    for kind in ("trait", "coherence"):
        for k, r in gj.latest_by_key(gj.read_jsonl(Path(judge_dir) / ("%s.jsonl" % kind))).items():
            rows[(r["row_id"], kind)] = r
    out = {}
    for item in items:
        rid = key[item["item_id"]]["row_id"]
        entry = {}
        for kind in ("trait", "coherence"):
            r = rows.get((rid, kind))
            if r is None:
                continue
            entry[kind] = r["score"]
            entry[kind + "_status"] = r["status"]
            detail = r.get("detail") or {}
            top = sorted(detail.get("top_logprobs") or [], key=lambda t: -t[1])[:3]
            entry[kind + "_detail"] = (
                "number_mass=%.3f; top: %s" % (detail.get("number_mass", float("nan")),
                                              ", ".join("%s (%.2f)" % (t, p) for t, p in top))
                if top else str(detail.get("final_text") or detail.get("reason") or ""))
        if entry:
            out[item["item_id"]] = entry
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--human-dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--judgments-dir")
    args = ap.parse_args(argv)
    rows = gj.load_generation_rows(args.rows_dir)
    items, key, strata = draw(rows)
    human = Path(args.human_dir)
    human.mkdir(parents=True, exist_ok=True)
    sheet = human / "calibration_sheet.csv"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grader_page.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(items)
    grader_page.write_page(sheet, human / "grader.html",
                           title="Calibration sheet: score each answer (trait, coherence)")
    Path(args.key).parent.mkdir(parents=True, exist_ok=True)
    Path(args.key).write_text(json.dumps(
        {"seed": SEED, "per_steered_stratum": PER_STEERED_STRATUM, "baseline_n": BASELINE_N,
         "n_items": len(items), "strata": strata, "items": key}, indent=1, sort_keys=True) + "\n")
    short = [s for s in strata if s["drawn"] < s["wanted"]]
    print("%d items drawn; strata short of their quota: %s"
          % (len(items), [("%s k%+g" % (s["role"], s["k"]), s["drawn"], s["wanted"]) for s in short]))
    if args.judgments_dir:
        for judge_dir in sorted(p for p in Path(args.judgments_dir).iterdir() if p.is_dir()):
            reveal = reveal_file(judge_dir, items, key)
            if not reveal:
                continue
            path = human / ("judge_reveal_%s.json" % judge_dir.name)
            path.write_text(json.dumps({"judge": judge_dir.name, "items": reveal}, indent=1) + "\n",
                            encoding="utf-8")
            print("%s: %d of %d items scored -> %s" % (judge_dir.name, len(reveal), len(items), path))


if __name__ == "__main__":
    main()
