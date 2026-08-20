"""The master comparison table across all four arms, plus the plot."""
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, "scratch")
from analyze_sweep import describe, welch, summarize_point, ols

UNIT = 10.508308410644531
OUT = Path("/home/marco/dockmaster/data/steer-decision")
DIRS = {
    "decision": OUT / "rows/decision",
    "shuffled-null": OUT / "rows/shuffled-null",
    "orthogonal-null": OUT / "rows/orthogonal-null",
}
NORMS = {"decision": 6.872108459472656, "shuffled-null": 0.5799206060523804,
         "orthogonal-null": 0.5626369046258461}


def path_for(arm, k):
    if arm == "theirs":
        return Path("/home/marco/dockmaster/data/audit-steer/rows") / (
            "altruism_v3-dictator_free_layer20_all_coef%d.csv" % k)
    return DIRS[arm] / ("altruism_v3-dictator_free_layer20_all_unit_coef%s.csv"
                        % ("%g" % (k * UNIT) if k else "0"))


def point(arm, k):
    p = path_for(arm, k)
    return summarize_point(p) if p.is_file() else None


ARMS = ["decision", "theirs", "shuffled-null", "orthogonal-null"]
KS = list(range(-5, 6))
table = {}
for arm in ARMS:
    for k in KS:
        e = point(arm, k)
        if e is not None:
            table[(arm, k)] = e

rows_out = []
for arm in ARMS:
    for k in KS:
        e = table.get((arm, k))
        if e is None:
            continue
        zero = table[(arm, 0)] if (arm, 0) in table else table[("decision", 0)]
        move = welch(e, zero) if k != 0 else {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0}
        rows_out.append({
            "arm": arm, "their_raw_beta": k, "unit_beta": k * UNIT,
            "own_raw_beta": (k * UNIT / NORMS[arm]) if arm != "theirs" else k,
            "n": e["n_rows"], "parsed": e["n_parsed"],
            "parse_coverage": round(e["parse_coverage"], 4),
            "mean": round(e["mean"], 3), "sd": round(e["sd"], 3), "se": round(e["se"], 3),
            "ci_low": round(e["ci_low"], 3), "ci_high": round(e["ci_high"], 3),
            "median": e["median"],
            "share_zero": round(e["modes"]["share_zero"], 4),
            "share_half": round(e["modes"]["share_half"], 4),
            "share_full": round(e["modes"]["share_full"], 4),
            "move_from_zero": round(move["diff"], 3) if move["diff"] is not None else None,
            "move_ci_low": round(move["ci_low"], 3) if move.get("ci_low") is not None else None,
            "move_ci_high": round(move["ci_high"], 3) if move.get("ci_high") is not None else None,
        })

with (OUT / "comparison_table.csv").open("w", encoding="utf-8", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print("arm              k  unit_beta  own_raw     n  parsed   mean    sd     se    95%% CI            move from 0")
for r in rows_out:
    print("%-16s %+2d %9.3f %8.3f %4d %6d %7.2f %6.2f %5.2f [%6.2f,%6.2f]  %+7.2f [%+6.2f,%+6.2f]" % (
        r["arm"], r["their_raw_beta"], r["unit_beta"], r["own_raw_beta"], r["n"], r["parsed"],
        r["mean"], r["sd"], r["se"], r["ci_low"], r["ci_high"],
        r["move_from_zero"], r["move_ci_low"], r["move_ci_high"]))

# pairwise vs decision at matched size
print()
print("difference vs the decision arm at matched intervention size")
diffs = []
for arm in ("theirs", "shuffled-null", "orthogonal-null"):
    for k in KS:
        if (arm, k) not in table or ("decision", k) not in table or k == 0:
            continue
        w = welch(table[(arm, k)], table[("decision", k)])
        diffs.append({"arm": arm, "their_raw_beta": k, "unit_beta": k * UNIT,
                      "minus_decision": round(w["diff"], 3),
                      "ci_low": round(w["ci_low"], 3), "ci_high": round(w["ci_high"], 3),
                      "p": w["p"]})
        print("  %-16s k=%+d  %+7.2f [%+7.2f,%+7.2f] p=%.2e" % (
            arm, k, w["diff"], w["ci_low"], w["ci_high"], w["p"]))
with (OUT / "differences_vs_decision.csv").open("w", encoding="utf-8", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(diffs[0].keys()))
    w.writeheader(); w.writerows(diffs)

json.dump({"table": rows_out, "differences_vs_decision": diffs},
          (OUT / "comparison.json").open("w"), indent=2)
