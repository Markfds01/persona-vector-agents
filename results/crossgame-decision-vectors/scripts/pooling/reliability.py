"""Is the non-alignment of overfishing and PD real, or just measurement noise?

Splits every cell's rows into two disjoint halves, rebuilds each game's vector and
each leave-one-out pool from each half, and reports the split-half reliability -
the ceiling any cosine between two independently estimated vectors could reach.
Dividing an observed cosine by sqrt(rel_a * rel_b) corrects it for attenuation.
If a corrected cosine is still near zero, the two directions genuinely differ.

Also counts the focal-point concentration of each pole, because it bounds what
any vector built from this contrast can mean.
"""

import csv
import json

import torch

import common

SEED = 20260820
DOLLAR = ("dictator", "trust", "ultimatum", "apology")
NON_DOLLAR = ("overfishing", "prisoners_dilemma")


def halves(usable, seed):
    g = torch.Generator().manual_seed(seed)
    a, b = {}, {}
    for family, cells in usable.items():
        for cell, (alt, self_) in cells.items():
            split = []
            for group in (alt, self_):
                order = torch.randperm(len(group), generator=g).tolist()
                cut = max(1, len(group) // 2)
                split.append(([group[i] for i in order[:cut]],
                              [group[i] for i in order[cut:]]))
            (alt_a, alt_b), (self_a, self_b) = split
            if alt_a and self_a:
                a.setdefault(family, {})[cell] = (alt_a, self_a)
            if alt_b and self_b:
                b.setdefault(family, {})[cell] = (alt_b, self_b)
    return a, b


def build(acts, cells_by_family):
    cell_vecs = {f: common.cell_vectors(acts, c) for f, c in cells_by_family.items()}
    game_vecs = {f: torch.stack([v for _c, v in sorted(cell_vecs[f].items())]).mean(dim=0)
                 for f in cell_vecs}
    return cell_vecs, game_vecs


def focal_points(policy):
    """Share of each pole sitting on one single answer value, per game."""
    out = {}
    for family in common.FAMILIES:
        path = common.ACTS / family / "rows.csv"
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts = {}
        for row in rows:
            game = common.crossgame_grid.GRID_BY_ID[row["game_id"]]
            if common.poles.tag_class(row["tag"], game.scorer) != "direct":
                continue
            value = float(row["value"])
            pole = common.poles.classify(value, game, policy)
            if pole == common.poles.MIDDLE:
                continue
            # normalise to the game's own scale so one focal value is comparable
            key = value / float(game.pole_scale)
            bucket = counts.setdefault(pole, {})
            bucket[key] = bucket.get(key, 0) + 1
        out[family] = {}
        for pole, bucket in counts.items():
            total = sum(bucket.values())
            top, n = max(bucket.items(), key=lambda kv: kv[1])
            out[family][pole] = {"n": total, "modal_value_as_fraction_of_scale": top,
                                 "modal_share": n / total}
    return out


def main():
    acts, labels = common.load_response_avg()
    report = {"seed": SEED, "policies": {}}
    for policy in ("strict", "relaxed"):
        usable, _seen = common.cell_index(labels, policy)
        cells_a, cells_b = halves(usable, SEED)
        _cva, games_a = build(acts, cells_a)
        _cvb, games_b = build(acts, cells_b)
        cell_a = {f: common.cell_vectors(acts, c) for f, c in cells_a.items()}
        cell_b = {f: common.cell_vectors(acts, c) for f, c in cells_b.items()}

        cv = {f: common.cell_vectors(acts, usable[f]) for f in usable}
        gv = {f: torch.stack([v for _c, v in sorted(cv[f].items())]).mean(dim=0)
              for f in cv}
        entry = {"game_reliability": {}, "logo_corrected": {}, "focal_points":
                 focal_points(policy)}
        for f in sorted(usable):
            c = common.cosines(games_a[f], games_b[f])
            entry["game_reliability"][f] = {"layer0": c[0].item(), "layer20": c[20].item()}

        for scheme in ("game_equal_unit", "cell_balanced",
                       "family_balanced_unit", "non_dollar_unit"):
            bucket = {}
            for held in sorted(usable):
                rest_a = {f: v for f, v in games_a.items() if f != held}
                rest_b = {f: v for f, v in games_b.items() if f != held}
                pool_a = common.combine(rest_a, {f: cells_a[f] for f in rest_a}, scheme,
                                        {f: cell_a[f] for f in rest_a})
                pool_b = common.combine(rest_b, {f: cells_b[f] for f in rest_b}, scheme,
                                        {f: cell_b[f] for f in rest_b})
                rel_pool = common.cosines(pool_a, pool_b)
                rel_game = common.cosines(games_a[held], games_b[held])
                # the observed cosine, from the full-data vectors
                rest = {f: v for f, v in gv.items() if f != held}
                pool = common.combine(rest, {f: usable[f] for f in rest}, scheme,
                                      {f: cv[f] for f in rest})
                obs = common.cosines(pool, gv[held])
                row = {}
                for layer in (0, 20):
                    # A split-half reliability is a cosine and can come back
                    # negative for a noisy game. Clamping the product would turn
                    # that into a denominator near zero and publish a corrected
                    # "cosine" in the thousands, so it is reported as undefined.
                    product = (rel_pool[layer] * rel_game[layer]).item()
                    corrected = (obs[layer].item() / product ** 0.5
                                 if product > 0 else None)
                    row["layer%d" % layer] = {
                        "observed": obs[layer].item(),
                        "reliability_pool": rel_pool[layer].item(),
                        "reliability_game": rel_game[layer].item(),
                        "attenuation_corrected": corrected}
                bucket[held] = row
            entry["logo_corrected"][scheme] = bucket
        report["policies"][policy] = entry
        print("policy %s done" % policy, flush=True)
    (common.OUT / "reliability.json").write_text(json.dumps(report, indent=2))
    print("wrote %s" % (common.OUT / "reliability.json"))


if __name__ == "__main__":
    main()
