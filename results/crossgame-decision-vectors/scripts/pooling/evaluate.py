"""Measure the game-balanced pooled vector against everything that matters.

Every separation number here is out of sample: either the game it scores was
held out of the fit entirely (leave-one-game-out), or the rows it scores were
held out (split-half within cell). In-sample numbers are computed too, and are
labelled in-sample wherever they appear, because they are a decomposition of the
fit and not evidence of transfer.

Two label nulls are computed, not one, because they answer different questions
and the predecessor used only the wider of the two:

  within_cell  permute the pole labels inside each cell, keeping every cell's
               pole counts. This holds the design fixed and randomises only the
               model's choice, so it is the apt null for a cell-balanced vector -
               and it is the NARROWER of the two, i.e. the easier bar.
  game_wide    permute the pole labels across a whole game and rebuild the
               unbalanced difference, which is what `data/crossgame-vector` did.
               Wider, more conservative, and reported so its bar is comparable.
"""

import json

import torch

import common

DICTATOR = common.DICTATOR_VECTOR
THEIRS = common.ALTRUISM_VECTOR
LAYERS = (0, 20)
SEED = 20260820


# --- single-layer combination (the nulls run one layer at a time) --------------

def combine_1d(game_vecs, weights, unit_first):
    families = sorted(game_vecs)
    stack = torch.stack([game_vecs[f] for f in families])
    if unit_first:
        stack = stack / stack.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w = torch.tensor([weights[f] for f in families], dtype=stack.dtype)
    w = w / w.sum()
    return (stack * w.view(-1, 1)).sum(dim=0)


#: re-exported so the scheme table has exactly one definition, in common.py
SCHEME_SPEC = common.SCHEME_SPEC


def scheme_weights(usable, kind):
    return common.weights_for(kind, usable)


# --- nulls ---------------------------------------------------------------------

class Null:
    """Draws null per-game vectors at ONE layer by permuting pole labels.

    `mode` is "within_cell" (permute inside each cell, rebuild the cell-balanced
    game vector) or "game_wide" (permute across the game, rebuild the unbalanced
    difference). Blocks and column sums are materialised once; a draw is then a
    permutation plus one index_select per group, which is what makes hundreds of
    draws affordable on a CPU.
    """

    def __init__(self, acts, usable, labels, policy, layer, mode):
        self.mode = mode
        self.groups = {}
        for family, cells in usable.items():
            groups = []
            if mode == "within_cell":
                items = sorted(cells.items())
                spans = [(a, s) for _c, (a, s) in items]
            else:
                alt, self_ = common.pole_rows(labels, policy, family)
                spans = [(alt, self_)]
            for alt, self_ in spans:
                idx = torch.tensor(alt + self_, dtype=torch.long)
                # float64, because common.mean_of reduces the OBSERVED statistic
                # in float64: a null must be computed at the statistic's precision
                block = acts.index_select(0, idx)[:, layer, :].double()
                groups.append({"x": block, "total": block.sum(dim=0),
                               "n": block.shape[0], "n_alt": len(alt)})
            self.groups[family] = groups

    def draw(self, generator):
        out = {}
        for family, groups in self.groups.items():
            diffs = []
            for g in groups:
                perm = torch.randperm(g["n"], generator=generator)[:g["n_alt"]]
                s_alt = g["x"].index_select(0, perm).sum(dim=0)
                diffs.append(s_alt / g["n_alt"]
                             + (s_alt - g["total"]) / (g["n"] - g["n_alt"]))
            out[family] = torch.stack(diffs).mean(dim=0)
        return out


def cos1(a, b):
    return float(torch.dot(a.double(), b.double())
                 / (a.double().norm() * b.double().norm()).clamp_min(1e-30))


def run_nulls(acts, usable, labels, policy, game_vecs, targets, draws, mode):
    """{layer: {scheme: {target: summary}}} of null cosines, targets rebuilt too.

    A target that is one of the six per-game vectors is rebuilt from the same
    permuted labels, because both sides of that cosine are estimated from the
    same rows; an external target (the archived Dictator vector, the shipped
    altruism vector) is fixed and only the pooled side is redrawn.
    """
    out = {}
    for layer in LAYERS:
        null = Null(acts, usable, labels, policy, layer, mode)
        fixed = {name: v[layer].double() for name, v in targets.items()}
        g = torch.Generator().manual_seed(SEED + layer)
        collected = {s: {} for s in SCHEME_SPEC}
        for _ in range(draws):
            fake_games = null.draw(g)
            for scheme, (kind, unit_first) in SCHEME_SPEC.items():
                pooled = combine_1d(fake_games, scheme_weights(usable, kind), unit_first)
                bucket = collected[scheme]
                for name, target in fixed.items():
                    bucket.setdefault(name, []).append(cos1(pooled, target))
                for family, fv in fake_games.items():
                    bucket.setdefault("game:" + family, []).append(cos1(pooled, fv))
        out[layer] = {s: {k: common.summarize(torch.tensor(v, dtype=torch.double))
                          for k, v in b.items()} for s, b in collected.items()}
        print("  null %s layer %d done" % (mode, layer), flush=True)
    return out


# --- leave-one-game-out ---------------------------------------------------------

def logo(acts, labels, usable, cell_vecs, game_vecs, policy):
    """Fit on five games, score the sixth game's own poles and its own vector."""
    out = {}
    for held in sorted(usable):
        rest = {f: v for f, v in game_vecs.items() if f != held}
        rest_cells = {f: v for f, v in usable.items() if f != held}
        rest_cellvecs = {f: v for f, v in cell_vecs.items() if f != held}
        alt, self_ = common.pole_rows(labels, policy, held)
        entry = {"n_held_altruistic": len(alt), "n_held_self": len(self_),
                 "fit_families": sorted(rest), "schemes": {}}
        for scheme in common.WEIGHTINGS:
            v = common.combine(rest, rest_cells, scheme, rest_cellvecs)
            layers = common.auc_by_layer(acts, v, alt, self_)
            best = max(layers, key=lambda r: r["auc"])
            entry["schemes"][scheme] = {
                "auc_layer0": layers[0]["auc"], "auc_layer20": layers[20]["auc"],
                "best_layer": best["layer"], "best_auc": best["auc"],
                "cos_vs_held_game_vector_layer0":
                    common.cosines(v, game_vecs[held])[0].item(),
                "cos_vs_held_game_vector_layer20":
                    common.cosines(v, game_vecs[held])[20].item(),
                "auc_by_layer": [r["auc"] for r in layers],
            }
        out[held] = entry
    return out


# --- split-half within cell -----------------------------------------------------

def split_half(acts, labels, usable, policy, seed):
    """Fit every scheme on half of each cell's rows, score the other half.

    Splitting INSIDE the cell keeps every cell usable on both sides, so the fit
    half and the score half have the identical design and the only thing that
    differs is which rows landed where.
    """
    g = torch.Generator().manual_seed(seed)
    fit, score = {}, {}
    for family, cells in usable.items():
        for cell, (alt, self_) in cells.items():
            halves = []
            for group in (alt, self_):
                order = torch.randperm(len(group), generator=g).tolist()
                cut = max(1, len(group) // 2)
                halves.append(([group[i] for i in order[:cut]],
                               [group[i] for i in order[cut:]]))
            (alt_a, alt_b), (self_a, self_b) = halves
            if alt_a and self_a:
                fit.setdefault(family, {})[cell] = (alt_a, self_a)
            if alt_b and self_b:
                score.setdefault(family, {})[cell] = (alt_b, self_b)
    fit_cellvecs = {f: common.cell_vectors(acts, c) for f, c in fit.items()}
    fit_games = {f: torch.stack([v for _c, v in sorted(fit_cellvecs[f].items())]).mean(dim=0)
                 for f in fit}
    out = {"n_fit_cells": {f: len(c) for f, c in fit.items()},
           "n_score_cells": {f: len(c) for f, c in score.items()}, "schemes": {}}
    for scheme in common.WEIGHTINGS:
        v = common.combine(fit_games, fit, scheme, fit_cellvecs)
        entry = {"per_game": {}}
        all_alt, all_self = [], []
        for family, cells in sorted(score.items()):
            alt = [p for a, _s in cells.values() for p in a]
            self_ = [p for _a, s in cells.values() for p in s]
            all_alt.extend(alt)
            all_self.extend(self_)
            layers = common.auc_by_layer(acts, v, alt, self_)
            entry["per_game"][family] = {
                "n_alt": len(alt), "n_self": len(self_),
                "auc_layer0": layers[0]["auc"], "auc_layer20": layers[20]["auc"]}
        layers = common.auc_by_layer(acts, v, all_alt, all_self)
        best = max(layers, key=lambda r: r["auc"])
        entry["pooled"] = {
            "n_alt": len(all_alt), "n_self": len(all_self),
            "auc_layer0": layers[0]["auc"], "auc_layer20": layers[20]["auc"],
            "best_layer": best["layer"], "best_auc": best["auc"],
            "auc_by_layer": [r["auc"] for r in layers]}
        # macro-average over games: the pooled figure is dominated by the games
        # with the most rows, which is the exact bias this task is about
        entry["pooled"]["macro_auc_layer0"] = sum(
            e["auc_layer0"] for e in entry["per_game"].values()) / len(entry["per_game"])
        entry["pooled"]["macro_auc_layer20"] = sum(
            e["auc_layer20"] for e in entry["per_game"].values()) / len(entry["per_game"])
        out["schemes"][scheme] = entry
    return out


def main():
    acts, labels = common.load_response_avg()
    dictator = torch.load(DICTATOR, map_location="cpu").double()
    theirs = torch.load(THEIRS, map_location="cpu").double()
    report = {"seed": SEED, "layers_reported": list(LAYERS),
              "dictator_vector": str(DICTATOR), "their_altruism_vector": str(THEIRS),
              "cos_dictator_vs_their_altruism_layer20":
                  common.cosines(dictator, theirs)[20].item(),
              "policies": {}}

    for policy in ("strict", "relaxed"):
        print("=== policy %s" % policy, flush=True)
        usable, _seen = common.cell_index(labels, policy)
        cell_vecs = {f: common.cell_vectors(acts, usable[f]) for f in usable}
        game_vecs = {f: torch.stack([v for _c, v in sorted(cell_vecs[f].items())]).mean(dim=0)
                     for f in usable}
        pooled = {s: common.combine(game_vecs, usable, s, cell_vecs)
                  for s in common.WEIGHTINGS}
        entry = {"weights": {kind: scheme_weights(usable, kind)
                             for kind in ("equal", "cells", "precision",
                                          "family", "non_dollar")},
                 "cosines": {}, "in_sample_auc": {}}

        for scheme, v in pooled.items():
            c_dict = common.cosines(v, dictator)
            c_theirs = common.cosines(v, theirs)
            c_pool = common.cosines(v, pooled["cell_balanced"])
            entry["cosines"][scheme] = {
                "vs_dictator": {"layer0": c_dict[0].item(), "layer20": c_dict[20].item(),
                                "by_layer": c_dict.tolist()},
                "vs_their_altruism": {"layer0": c_theirs[0].item(),
                                      "layer20": c_theirs[20].item()},
                "vs_cell_balanced_pool": {"layer0": c_pool[0].item(),
                                          "layer20": c_pool[20].item()},
                "norm_layer20": v.norm(dim=1)[20].item(),
                "vs_each_game_in_sample": {
                    f: {"layer0": common.cosines(v, game_vecs[f])[0].item(),
                        "layer20": common.cosines(v, game_vecs[f])[20].item()}
                    for f in sorted(game_vecs)},
            }
            alt_all, self_all = [], []
            for f in sorted(usable):
                alt, self_ = common.pole_rows(labels, policy, f)
                alt_all.extend(alt)
                self_all.extend(self_)
            layers = common.auc_by_layer(acts, v, alt_all, self_all)
            entry["in_sample_auc"][scheme] = {
                "note": "scored over every direct-tag pole row of every game, "
                        "which is a superset of the fit: rows in cells that filled "
                        "only one pole are scored here but entered no vector",
                "auc_layer0": layers[0]["auc"], "auc_layer20": layers[20]["auc"]}
            print("  %s cos_vs_dictator L20=%.4f" % (scheme, c_dict[20].item()), flush=True)

        entry["leave_one_game_out"] = logo(acts, labels, usable, cell_vecs,
                                           game_vecs, policy)
        print("  logo done", flush=True)
        entry["split_half_within_cell"] = split_half(acts, labels, usable, policy, SEED)
        print("  split-half done", flush=True)

        targets = {"dictator_vector": dictator, "their_altruism": theirs}
        entry["null_within_cell"] = run_nulls(acts, usable, labels, policy, game_vecs,
                                              targets, 300, "within_cell")
        entry["null_game_wide"] = run_nulls(acts, usable, labels, policy, game_vecs,
                                            targets, 300, "game_wide")
        report["policies"][policy] = entry
        (common.OUT / "analysis.json").write_text(json.dumps(report, indent=2))
        print("  checkpointed analysis.json after %s" % policy, flush=True)

    print("wrote %s" % (common.OUT / "analysis.json"))


if __name__ == "__main__":
    main()
