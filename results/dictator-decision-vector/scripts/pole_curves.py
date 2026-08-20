"""The two pole probabilities the vector was actually constructed from.

The decision vector's poles were "gave exactly 0" and "gave at least half". The
mean in dollars is a derived quantity; these two shares are the contrast itself,
so they are the metric the steering test should be read on. Wilson 95% intervals,
which behave at shares near 0 and 1 where a normal interval does not.
"""
import csv, json, math
from pathlib import Path

UNIT = 10.508308410644531
OUT = Path("/home/marco/dockmaster/data/steer-decision")
DIRS = {"decision": "decision", "shuffled-null": "shuffled-null",
        "orthogonal-null": "orthogonal-null"}


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, centre - half, centre + half


def path_for(arm, k):
    if arm == "theirs":
        return Path("/home/marco/dockmaster/data/audit-steer/rows") / (
            "altruism_v3-dictator_free_layer20_all_coef%d.csv" % k)
    return OUT / "rows" / DIRS[arm] / (
        "altruism_v3-dictator_free_layer20_all_unit_coef%s.csv"
        % ("%g" % (k * UNIT) if k else "0"))


rows = []
print("arm               k  unit_beta   n   P(give 0)            P(give >= half)")
for arm in ("decision", "theirs", "shuffled-null", "orthogonal-null"):
    for k in range(-5, 6):
        p = path_for(arm, k)
        if not p.is_file():
            continue
        v = [float(r["value"]) for r in csv.DictReader(open(p, encoding="utf-8"))
             if r["value"] != ""]
        n = len(v)
        z0, z0lo, z0hi = wilson(sum(1 for x in v if x == 0.0), n)
        h, hlo, hhi = wilson(sum(1 for x in v if x >= 50.0), n)
        rows.append({"arm": arm, "their_raw_beta": k, "unit_beta": k * UNIT, "n_parsed": n,
                     "p_give_zero": z0, "p_give_zero_lo": z0lo, "p_give_zero_hi": z0hi,
                     "p_give_half_or_more": h, "p_half_lo": hlo, "p_half_hi": hhi})
        print("%-16s %+2d %9.3f %4d   %.3f [%.3f,%.3f]    %.3f [%.3f,%.3f]"
              % (arm, k, k * UNIT, n, z0, z0lo, z0hi, h, hlo, hhi))
    print()

with (OUT / "pole_curves.csv").open("w", encoding="utf-8", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(rows, (OUT / "pole_curves.json").open("w"), indent=2)
