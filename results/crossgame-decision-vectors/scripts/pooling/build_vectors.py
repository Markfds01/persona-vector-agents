"""Build the game-balanced pooled vectors under every weighting and both policies.

Two different checks are reported on the `cell_balanced` rebuild, and they answer
different questions:

* against the SIX-GAME stage's own pooled vector, which `analyze_crossgame.py`
  built from the same activations by a different code path. This is the pipeline
  check: it is what licenses "any difference between the schemes is the weighting
  and not the pipeline". A comparison against another run of THIS file could not
  say that.
* against what round 1 committed, which is the rebuild check: did any published
  number move.
"""

import json
import os
from pathlib import Path

import torch

import common

VEC_DIR = common.OUT / "vectors"
#: what round 1 published; the comparison this rebuild has to explain
COMMITTED = common.RESULTS / "vectors"
#: the six-game stage's output directory, written before this stage runs.
#: run_analysis.sh puts it beside this stage's; env-overridable so the coupling to
#: that layout is stated rather than assumed.
CROSSGAME = Path(os.environ.get("DM_CROSSGAME_OUT", common.OUT.parent / "crossgame"))


def build(acts, labels, policy):
    usable, seen = common.cell_index(labels, policy)
    families = [f for f in common.FAMILIES if f in usable]
    cell_vecs = {f: common.cell_vectors(acts, usable[f]) for f in families}
    game_vecs = {f: torch.stack([v for _c, v in sorted(cell_vecs[f].items())]).mean(dim=0)
                 for f in families}
    pooled = {s: common.combine(game_vecs, usable, s, cell_vecs)
              for s in common.WEIGHTINGS}

    census = {}
    for f in families:
        cells = usable[f]
        alt_all, self_all = common.pole_rows(labels, policy, f)
        census[f] = {
            "usable_cells": len(cells),
            "cells_seen_with_a_pole_row": len(seen[f]),
            "alt_rows_total": len(alt_all),
            "self_rows_total": len(self_all),
            "alt_rows_in_usable_cells": sum(len(a) for a, _s in cells.values()),
            "self_rows_in_usable_cells": sum(len(s) for _a, s in cells.values()),
            "effective_n": common.effective_n(cells),
            "vector_norm_layer20": game_vecs[f].norm(dim=1)[20].item(),
            "vector_norm_layer0": game_vecs[f].norm(dim=1)[0].item(),
        }
    total_eff = sum(c["effective_n"] for c in census.values())
    for f in census:
        census[f]["precision_weight"] = census[f]["effective_n"] / total_eff
        census[f]["cell_weight"] = census[f]["usable_cells"] / sum(
            c["usable_cells"] for c in census.values())
        census[f]["equal_weight"] = 1.0 / len(census)
    return usable, cell_vecs, game_vecs, pooled, census


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    acts, labels = common.load_response_avg()
    report = {"policies": {}}
    for policy in ("strict", "relaxed"):
        usable, cell_vecs, game_vecs, pooled, census = build(acts, labels, policy)
        report["policies"][policy] = {"census": census}
        for scheme, vec in pooled.items():
            torch.save(vec.float(),
                       VEC_DIR / ("decision_pooled_%s_response_avg_diff_%s.pt"
                                  % (scheme, policy)))
        for f, vec in game_vecs.items():
            torch.save(vec.float(),
                       VEC_DIR / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                                  % (f, policy)))

        # the PIPELINE check: this stage's cell_balanced pool against the one the
        # six-game stage built from the same activations by another code path
        other_path = CROSSGAME / ("decision_pooled_response_avg_diff_cellbalanced_%s.pt"
                                  % policy)
        if not other_path.is_file():
            raise SystemExit("%s is missing; run the six-game stage before this one, "
                             "or the pipeline check would be vacuous" % other_path)
        other = torch.load(other_path, map_location="cpu")
        pcos = common.cosines(pooled["cell_balanced"], other)
        report["policies"][policy]["pipeline_check_cos_vs_six_game_pool"] = {
            "layer0": pcos[0].item(), "layer20": pcos[20].item(),
            "min_over_layers": pcos.min().item(),
            "against": str(other_path),
        }

        # the REBUILD check: this rebuild against what round 1 committed
        archived = torch.load(
            COMMITTED / ("decision_pooled_cell_balanced_response_avg_diff_%s.pt" % policy),
            map_location="cpu")
        cos = common.cosines(pooled["cell_balanced"], archived)
        report["policies"][policy]["rebuild_check_cos_vs_archived_pool"] = {
            "layer0": cos[0].item(), "layer20": cos[20].item(),
            "min_over_layers": cos.min().item(),
        }
        for f in game_vecs:
            arch_g = torch.load(
                COMMITTED / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                             % (f, policy)), map_location="cpu")
            c = common.cosines(game_vecs[f], arch_g).min().item()
            report["policies"][policy].setdefault(
                "rebuild_check_cos_vs_archived_per_game", {})[f] = c
        print("policy %s: pipeline check vs six-game pool min cos %.12f; "
              "rebuild check vs round 1 min cos %.6f"
              % (policy, pcos.min().item(), cos.min().item()), flush=True)

    (common.OUT / "build.json").write_text(json.dumps(report, indent=2))
    print("wrote %s" % (common.OUT / "build.json"))


if __name__ == "__main__":
    main()
