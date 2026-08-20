"""The answer distribution behind every mean, for all three arms.

A mean plus an SD cannot tell a behavioural shift from a collapse onto one value.
This prints, per coefficient: how many distinct amounts were given, the modal
amount and its share, the top four amounts, the normalised Shannon entropy of the
amount distribution, and the surface-degeneracy statistics.
"""
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, "scratch")
from analyze_sweep import summarize_point, read_rows

ARMS = {
    "decision": "/home/marco/dockmaster/data/steer-decision/rows/decision/altruism_v3-dictator_free_layer20_all_unit_coef%s.csv",
    "null":     "/home/marco/dockmaster/data/steer-decision/rows/shuffled-null/altruism_v3-dictator_free_layer20_all_unit_coef%s.csv",
    "orth":     "/home/marco/dockmaster/data/steer-decision/rows/orthogonal-null/altruism_v3-dictator_free_layer20_all_unit_coef%s.csv",
    "theirs":   "/home/marco/dockmaster/data/audit-steer/rows/altruism_v3-dictator_free_layer20_all_coef%s.csv",
}
ARM_KS = {"null": (-5, -3, -1, 0, 1, 3, 5), "orth": (-5, -1, 1, 5)}
UNIT = 10.508308410644531


def label(arm, k):
    if arm == "theirs":
        return "%d" % k
    return "%g" % (k * UNIT) if k else "0"


def entropy(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = len(values)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h, (h / math.log(len(counts)) if len(counts) > 1 else 0.0), counts


out = {}
for arm, template in ARMS.items():
    ks = ARM_KS.get(arm, range(-5, 6))
    for k in ks:
        path = Path(template % label(arm, k))
        if not path.is_file():
            raise SystemExit("missing %s" % path)
        rows = read_rows(path)
        values = [float(r["value"]) for r in rows if r["value"] != ""]
        h, hn, counts = entropy(values)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        s = summarize_point(path)
        out.setdefault(arm, {})[k] = {
            "n_rows": len(rows), "n_parsed": len(values),
            "parse_coverage": s["parse_coverage"], "tags": s["tags"],
            "mean": s["mean"], "sd": s["sd"], "se": s["se"],
            "ci": [s["ci_low"], s["ci_high"]], "median": s["median"],
            "n_distinct_amounts": len(counts),
            "modal_amount": top[0][0], "modal_share": top[0][1] / len(values),
            "top4": [[v, c, c / len(values)] for v, c in top],
            "entropy_nats": h, "entropy_normalised": hn,
            "share_zero": s["modes"]["share_zero"],
            "share_half": s["modes"]["share_half"],
            "share_full": s["modes"]["share_full"],
            "degeneracy": s["degeneracy"],
        }

Path("/home/marco/dockmaster/data/steer-decision/distributions.json").write_text(json.dumps(out, indent=2))

for arm in ("decision", "theirs", "null", "orth"):
    print("=== %s ===" % arm)
    print("  k  parsed  mean    sd   median  distinct  mode(share)   top4                                  H_norm  %0    %50   %100")
    for k in sorted(out[arm]):
        e = out[arm][k]
        top = " ".join("%g:%.0f%%" % (v, 100 * f) for v, _c, f in e["top4"])
        print("%+3d %5d %7.2f %6.2f %6.1f %8d  %5g(%.0f%%)  %-36s %.3f  %.2f %.2f %.2f" % (
            k, e["n_parsed"], e["mean"], e["sd"], e["median"], e["n_distinct_amounts"],
            e["modal_amount"], 100 * e["modal_share"], top, e["entropy_normalised"],
            e["share_zero"], e["share_half"], e["share_full"]))
    print()
