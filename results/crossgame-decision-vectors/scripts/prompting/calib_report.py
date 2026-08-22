"""Pole yield per family and stake, from generated rows alone. No GPU, no activations.

The point of the calibration pass: a pole with no rows in it produces no vector.
Two of the six games are known to sit at a corner on the published parameters -
the Prisoner's Dilemma defected 50/50 and Overfishing took the maximum 5/50 - so
the stake ladder has to be checked to actually populate both poles BEFORE the
full grid is paid for in GPU hours.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# this directory holds the grid and pole definitions; the repo root is four up
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

import crossgame_grid  # noqa: E402
import poles  # noqa: E402

crossgame_grid.register()
POLICY = sys.argv[2] if len(sys.argv) > 2 else "strict"
print("pole policy: %s" % POLICY)
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
print("generated rows: %d" % len(rows))

by_family = {}
for row in rows:
    game = crossgame_grid.GRID_BY_ID[row["game_id"]]
    fam = by_family.setdefault(game.family, {})
    stake = fam.setdefault(game.stake, {"self": 0, "alt": 0, "mid": 0,
                                        "derived": 0, "unresolved": 0, "unknown": 0,
                                        "values": []})
    cls = poles.tag_class(row["tag"], game.scorer)
    if cls == "unresolved":
        stake["unresolved"] += 1
        continue
    if cls == "unknown":
        stake["unknown"] += 1
        continue
    value = float(row["value"])
    stake["values"].append(value)
    if cls == "derived":
        stake["derived"] += 1
        continue
    pole = poles.classify(value, game, POLICY)
    stake[{poles.SELF: "self", poles.ALT: "alt", poles.MIDDLE: "mid"}[pole]] += 1

for family in crossgame_grid.FAMILIES:
    if family not in by_family:
        continue
    print("\n== %s ==" % family)
    tot = {"self": 0, "alt": 0, "mid": 0}
    for stake, s in by_family[family].items():
        vals = sorted(s["values"])
        span = ("%g..%g" % (vals[0], vals[-1])) if vals else "-"
        print("  %-10s self=%-3d alt=%-3d mid=%-3d derived=%-2d unresolved=%-2d "
              "unknown=%-2d values %s"
              % (stake, s["self"], s["alt"], s["mid"], s["derived"],
                 s["unresolved"], s["unknown"], span))
        for k in tot:
            tot[k] += s[k]
    print("  TOTAL      self=%d alt=%d mid=%d   %s"
          % (tot["self"], tot["alt"], tot["mid"],
             "BOTH POLES POPULATED" if tot["self"] and tot["alt"] else "*** A POLE IS EMPTY ***"))
