"""Three follow-ups the main battery leaves open. CPU only.

1. A null for the LEAVE-ONE-GAME-OUT cosine. The null in `evaluate.py` redraws a
   pool that CONTAINS the game it is compared against, which is the right bar for
   the in-pool cosine and the wrong one for the leave-one-out cosine. This
   rebuilds the pool from the other five games under the same permuted labels.

2. Is there anything in the pooled vector that the Dictator vector does not
   already have? Project the archived Dictator direction out of the pooled
   direction, per layer, and re-run the leave-one-game-out separation. If the
   Prisoner's Dilemma still separates at depth with the Dictator direction gone,
   there is a shared component that is not the Dictator vector. If it collapses,
   the pooled vector is the Dictator vector plus noise.

3. How much of the pooled direction is the four dollar games? Cosine the pool
   against the dollar-only sub-pool and the non-dollar-only sub-pool, built by
   the same weighting.
"""

import json

import torch

import common
from evaluate import (DICTATOR, LAYERS, SEED, Null, combine_1d, cos1,
                      scheme_weights, SCHEME_SPEC)

DOLLAR = common.DOLLAR_FAMILIES
NON_DOLLAR = common.NON_DOLLAR_FAMILIES
DRAWS = 300

#: the schemes the expensive per-game checks are run for
FOCUS = ("game_equal_unit", "cell_balanced", "family_balanced_unit",
         "non_dollar_unit")


def logo_null(acts, usable, labels, policy):
    """p97.5 of cos(pool fit without game g, game g's own vector) under the null."""
    # The weights are recomputed over the SURVIVING games, which is what
    # common.combine does for the observed statistic. Renormalising the six-game
    # weights instead would spread a dropped game's share proportionally, and
    # under family balancing its share belongs to its own format group: dropping
    # dictator gives the other dollar games 1/9 each, not 1/11.
    kinds = {kind for kind, _unit_first in SCHEME_SPEC.values()}
    held_weights = {(kind, held): scheme_weights(
                        {f: c for f, c in usable.items() if f != held}, kind)
                    for kind in kinds for held in usable}
    out = {}
    for layer in LAYERS:
        null = Null(acts, usable, labels, policy, layer, "within_cell")
        g = torch.Generator().manual_seed(SEED + 7 + layer)
        collected = {s: {} for s in SCHEME_SPEC}
        for _ in range(DRAWS):
            fake = null.draw(g)
            for scheme, (kind, unit_first) in SCHEME_SPEC.items():
                for held in fake:
                    rest = {f: v for f, v in fake.items() if f != held}
                    pooled = combine_1d(rest, held_weights[(kind, held)], unit_first)
                    collected[scheme].setdefault(held, []).append(
                        cos1(pooled, fake[held]))
        out[layer] = {s: {k: common.summarize(torch.tensor(v, dtype=torch.double))
                          for k, v in b.items()} for s, b in collected.items()}
        print("  logo null layer %d done" % layer, flush=True)
    return out


def orthogonalise(v, other):
    """v with `other` projected out, per layer."""
    u = other / other.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return v - (v * u).sum(dim=1, keepdim=True) * u


def logo_orthogonal(acts, labels, usable, cell_vecs, game_vecs, policy, dictator):
    """Leave-one-game-out separation before and after removing the Dictator axis."""
    out = {}
    for held in sorted(usable):
        rest = {f: v for f, v in game_vecs.items() if f != held}
        rest_cells = {f: v for f, v in usable.items() if f != held}
        rest_cellvecs = {f: v for f, v in cell_vecs.items() if f != held}
        alt, self_ = common.pole_rows(labels, policy, held)
        entry = {"n_held_altruistic": len(alt), "n_held_self": len(self_), "schemes": {}}
        for scheme in FOCUS:
            v = common.combine(rest, rest_cells, scheme, rest_cellvecs)
            perp = orthogonalise(v, dictator)
            layers = common.auc_by_layer(acts, perp, alt, self_)
            best = max(layers, key=lambda r: r["auc"])
            entry["schemes"][scheme] = {
                "residual_norm_fraction_layer20":
                    (perp.norm(dim=1) / v.norm(dim=1))[20].item(),
                "residual_norm_fraction_layer0":
                    (perp.norm(dim=1) / v.norm(dim=1))[0].item(),
                "auc_layer0": layers[0]["auc"], "auc_layer20": layers[20]["auc"],
                "best_layer": best["layer"], "best_auc": best["auc"],
            }
        out[held] = entry
    return out


def dollar_decomposition(game_vecs, usable, cell_vecs):
    out = {}
    for scheme in common.WEIGHTINGS:
        if scheme.startswith("non_dollar"):
            continue          # a dollar-only sub-pool cannot carry this weighting
        full = common.combine(game_vecs, usable, scheme, cell_vecs)
        sub = {}
        for name, families in (("dollar_four", DOLLAR), ("non_dollar_two", NON_DOLLAR)):
            keep = {f: v for f, v in game_vecs.items() if f in families}
            v = common.combine(keep, {f: usable[f] for f in keep}, scheme,
                               {f: cell_vecs[f] for f in keep})
            c = common.cosines(full, v)
            sub[name] = {"layer0": c[0].item(), "layer20": c[20].item()}
        c = common.cosines(
            common.combine({f: game_vecs[f] for f in DOLLAR},
                           {f: usable[f] for f in DOLLAR}, scheme,
                           {f: cell_vecs[f] for f in DOLLAR}),
            common.combine({f: game_vecs[f] for f in NON_DOLLAR},
                           {f: usable[f] for f in NON_DOLLAR}, scheme,
                           {f: cell_vecs[f] for f in NON_DOLLAR}))
        sub["dollar_vs_non_dollar"] = {"layer0": c[0].item(), "layer20": c[20].item()}
        out[scheme] = sub
    return out


def main():
    acts, labels = common.load_response_avg()
    dictator = torch.load(DICTATOR, map_location="cpu").double()
    report = {"draws": DRAWS, "policies": {}}
    for policy in ("strict", "relaxed"):
        print("=== %s" % policy, flush=True)
        usable, _seen = common.cell_index(labels, policy)
        cell_vecs = {f: common.cell_vectors(acts, usable[f]) for f in usable}
        game_vecs = {f: torch.stack([v for _c, v in sorted(cell_vecs[f].items())]).mean(dim=0)
                     for f in usable}
        report["policies"][policy] = {
            "dollar_decomposition": dollar_decomposition(game_vecs, usable, cell_vecs),
            "logo_orthogonal_to_dictator": logo_orthogonal(
                acts, labels, usable, cell_vecs, game_vecs, policy, dictator),
            "logo_null_within_cell": logo_null(acts, usable, labels, policy),
        }
        (common.OUT / "addendum.json").write_text(json.dumps(report, indent=2))
        print("  checkpointed addendum.json after %s" % policy, flush=True)


if __name__ == "__main__":
    main()
