"""Two-proportion tests on P(gives exactly $0), the contrast the vector was built on.

Newcombe's hybrid-score interval for a difference of independent proportions: it is
built from the two Wilson intervals and stays valid at shares near 0 and 1, where a
Wald interval on a difference does not.
"""
import csv, json, math
from pathlib import Path
UNIT = 10.508308410644531
OUT = Path("/home/marco/dockmaster/data/steer-decision")
D = {"decision": "decision", "shuffled-null": "shuffled-null",
     "orthogonal-null": "orthogonal-null"}


def counts(arm, k):
    p = OUT / "rows" / D[arm] / ("altruism_v3-dictator_free_layer20_all_unit_coef%s.csv"
                                 % ("%g" % (k * UNIT) if k else "0"))
    v = [float(r["value"]) for r in csv.DictReader(open(p, encoding="utf-8"))
         if r["value"] != ""]
    return sum(1 for x in v if x == 0.0), len(v)


def wilson(k, n, z=1.959963985):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, lo, hi


base = counts("decision", 0)
print("P(gives exactly $0)  -  baseline %d/%d = %.3f\n" % (base[0], base[1], base[0] / base[1]))
out = []
for k in (-5, -1, 1, 5):
    print("unit beta %+.2f" % (k * UNIT))
    row = {"their_raw_beta": k, "unit_beta": k * UNIT}
    for arm in ("decision", "orthogonal-null"):
        c = counts(arm, k)
        d, lo, hi = newcombe(c[0], c[1], base[0], base[1])
        print("   %-16s %3d/%3d = %.3f   vs baseline: %+.3f [%+.3f, %+.3f]%s"
              % (arm, c[0], c[1], c[0] / c[1], d, lo, hi,
                 "   (contains 0 - no detectable move)" if lo <= 0 <= hi else ""))
        row[arm] = {"k0": c[0], "n": c[1], "p": c[0] / c[1],
                    "vs_baseline": d, "lo": lo, "hi": hi}
    cd = counts("decision", k)
    co = counts("orthogonal-null", k)
    d, lo, hi = newcombe(cd[0], cd[1], co[0], co[1])
    print("   decision - orthogonal-null:  %+.3f [%+.3f, %+.3f]" % (d, lo, hi))
    row["decision_minus_orthogonal"] = {"diff": d, "lo": lo, "hi": hi}
    out.append(row)
    print()
json.dump(out, (OUT / "pole_tests.json").open("w"), indent=2)
