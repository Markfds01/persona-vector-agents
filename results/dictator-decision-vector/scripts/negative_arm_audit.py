"""Hand audit of every non-zero answer on the decision vector's negative arm.

A mean of $1.03 at the extreme could be a real decision to give nothing or a
collapse into unusable text. The parsed values say 96-98% exact zeros, so the
question reduces to what the 26 non-zero rows actually are. Every one of them was
read in full and classified into exactly three kinds:

  genuine    the model reasons coherently and commits to a non-zero gift
  inverted   the model states a payoff-MAXIMISING goal and then concludes it
             should hand over the whole endowment - a broken payoff logic, not
             generosity
  misparse   the model committed to $0 (or to no amount at all) and the scorer
             read a number out of the surrounding payoff arithmetic

The classification is recorded by position so it can be re-checked against the
rows; it is a human read of 26 responses, not an automated rule.
"""
import csv, json
from pathlib import Path

UNIT = 10.508308410644531
BASE = ("/home/marco/dockmaster/data/steer-decision/rows/decision/"
        "altruism_v3-dictator_free_layer20_all_unit_coef%g.csv")

# position within that coefficient's non-zero, parsed rows -> classification
AUDIT = {
    -1: ["genuine", "genuine", "inverted", "misparse", "genuine"],
    -2: ["inverted", "misparse", "misparse", "misparse", "inverted", "inverted"],
    -3: ["inverted", "inverted", "inverted", "inverted", "inverted"],
    -4: ["inverted", "inverted", "misparse", "inverted", "misparse", "misparse",
         "inverted"],
    -5: ["misparse", "inverted", "inverted"],
}
# the one row that committed to NO amount at all; scored $100 off the payoff line
NO_COMMITMENT = {(-4, 5)}

out = {}
totals = {"genuine": 0, "inverted": 0, "misparse": 0}
for k, classes in sorted(AUDIT.items()):
    rows = list(csv.DictReader(open(BASE % (k * UNIT), encoding="utf-8")))
    parsed = [r for r in rows if r["value"] != ""]
    nonzero = [r for r in parsed if float(r["value"]) != 0.0]
    if len(nonzero) != len(classes):
        raise SystemExit("k=%d: audited %d rows but found %d non-zero"
                         % (k, len(classes), len(nonzero)))
    values = [float(r["value"]) for r in parsed]
    mean = sum(values) / len(values)
    # corrected: a misparse is a row where the model committed to $0 or to nothing
    corrected = []
    for r in parsed:
        v = float(r["value"])
        if v == 0.0:
            corrected.append(0.0)
    for r, cls in zip(nonzero, classes):
        corrected.append(0.0 if cls == "misparse" else float(r["value"]))
    counts = {c: classes.count(c) for c in ("genuine", "inverted", "misparse")}
    for c, n in counts.items():
        totals[c] += n
    out[k] = {
        "n_parsed": len(parsed), "n_nonzero": len(nonzero), "counts": counts,
        "mean_as_scored": mean,
        "mean_after_audit": sum(corrected) / len(corrected),
        "share_gave_nothing_as_scored": sum(1 for v in values if v == 0.0) / len(values),
        "share_gave_nothing_after_audit": sum(1 for v in corrected if v == 0.0) / len(corrected),
        "nonzero_values": [float(r["value"]) for r in nonzero],
        "classes": classes,
    }
out["totals"] = totals
out["n_audited"] = sum(totals.values())
out["note_no_commitment_rows"] = sorted("k=%d[%d]" % kv for kv in NO_COMMITMENT)
Path("/home/marco/dockmaster/data/steer-decision/negative_arm_audit.json").write_text(
    json.dumps(out, indent=2))

print("  k  parsed nonzero  genuine inverted misparse | mean scored -> audited | %%zero scored -> audited")
for k in sorted(AUDIT):
    e = out[k]; c = e["counts"]
    print("%+3d %6d %7d %8d %8d %8d | %10.2f -> %7.2f | %11.3f -> %7.3f" % (
        k, e["n_parsed"], e["n_nonzero"], c["genuine"], c["inverted"], c["misparse"],
        e["mean_as_scored"], e["mean_after_audit"],
        e["share_gave_nothing_as_scored"], e["share_gave_nothing_after_audit"]))
print()
print("totals over the 26 audited non-zero rows:", totals)
