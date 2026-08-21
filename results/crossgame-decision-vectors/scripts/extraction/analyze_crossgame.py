"""Build the per-game and pooled decision vectors and measure them. CPU only.

The question this answers is whether there is ONE decision direction across the
six Table-1 games, or six surface directions that each encode their own answer
tokens. Three measurements settle it, and they are computed here:

  1. layer-0 vs layer-20 AUC. Layer 0 is the raw token embeddings averaged over
     the response, so a direction that separates the poles there is separating
     ANSWER WORDING. The Dictator-only vector reaches 0.903 at layer 0 because
     both its poles are dollar amounts. Pooling six games whose poles share no
     tokens ($0 / 0 fish / Defect against $50 / 8 fish / Cooperate) removes that
     shared surface, so a pooled layer-0 AUC near chance means the confound is
     gone and a pooled layer-20 AUC that holds means something else is there.

  2. the 6x6 agreement matrix - each game's vector cosined against every other's.

  3. leave-one-game-out transfer: fit the direction on five games, score the
     sixth game's own poles with it. This is the strongest of the three. A cosine
     says two directions point the same way; a transferred AUC says a direction
     built without ever seeing a game still tells that game's poles apart.

Every separation number is OUT OF SAMPLE. In-sample separation is circular - the
direction is by construction the thing that maximises it - so every AUC here is
scored on rows that were not used to fit the direction it is scored against.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# this directory holds the grid and pole definitions; the repo root is four up
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

import crossgame_grid  # noqa: E402
import poles  # noqa: E402

VARIANTS = ("prompt_avg", "prompt_last", "response_avg")


# --- loading -----------------------------------------------------------------

def load_family(rows_csv, act_dir):
    with open(rows_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    shards = sorted(Path(act_dir).glob("shard_*.pt"))
    if not shards:
        raise SystemExit("no activation shards in %s" % act_dir)
    index, acts = [], {v: [] for v in VARIANTS}
    for path in shards:
        payload = torch.load(path, map_location="cpu")
        index.extend(payload["row_index"].tolist())
        for v in VARIANTS:
            acts[v].append(payload[v])
    acts = {v: torch.cat(t, dim=0) for v, t in acts.items()}
    if len(index) != acts["response_avg"].shape[0]:
        raise SystemExit("%s: shard row count disagrees with activation count" % act_dir)
    meta = json.loads((Path(act_dir) / "meta.json").read_text())
    return rows, index, acts, meta


def load_all(manifest):
    """Concatenate every family's activations into one block, keeping labels aligned.

    `offset` is what makes a per-family activation index usable in the pooled
    block; it is applied once here so nothing downstream has to remember it.
    """
    labels, blocks, metas, tag_counts = [], {v: [] for v in VARIANTS}, {}, {}
    offset = 0
    for entry in manifest["families"]:
        family = entry["family"]
        rows, index, acts, meta = load_family(entry["rows_csv"], entry["acts_dir"])
        metas[family] = meta
        counts = tag_counts.setdefault(family, {})
        for row in rows:
            counts[row["tag"]] = counts.get(row["tag"], 0) + 1
        for position, row_index in enumerate(index):
            row = rows[row_index]
            game = crossgame_grid.GRID_BY_ID[row["game_id"]]
            if game.family != family:
                raise SystemExit("%s: row %s is family %s"
                                 % (family, row["game_id"], game.family))
            value = float(row["value"])
            labels.append({
                "position": offset + position,
                "family": family,
                "stake": game.stake,
                "wording": game.wording,
                "cell": game.cell,
                "scorer": game.scorer,
                "value": value,
                "tag": row["tag"],
                "tag_class": poles.tag_class(row["tag"], game.scorer),
                "pole_strict": poles.classify(value, game, "strict"),
                "pole_relaxed": poles.classify(value, game, "relaxed"),
            })
        for v in VARIANTS:
            blocks[v].append(acts[v])
        offset += len(index)
    acts = {v: torch.cat(t, dim=0) for v, t in blocks.items()}
    if acts["response_avg"].shape[0] != offset:
        raise SystemExit("pooled activation count disagrees with label count")
    return labels, acts, metas, tag_counts


# --- vectors -----------------------------------------------------------------

def split_poles(labels, allow_derived, policy):
    ok = ("direct", "derived") if allow_derived else ("direct",)
    key = "pole_" + policy
    alt, self_, middle, excluded = [], [], [], []
    for item in labels:
        if item["tag_class"] not in ok:
            excluded.append(item)
        elif item[key] == poles.ALT:
            alt.append(item)
        elif item[key] == poles.SELF:
            self_.append(item)
        else:
            middle.append(item)
    return alt, self_, middle, excluded


def mean_of(acts, items):
    idx = torch.tensor([i["position"] for i in items], dtype=torch.long)
    return acts.index_select(0, idx).double().mean(dim=0)


def naive_vector(acts, alt, self_):
    return mean_of(acts, alt) - mean_of(acts, self_)


def balanced_vector(acts, alt, self_):
    """Equal weight per prompt cell, so pole composition cannot fake a direction.

    A cell is one (family, stake, wording). Only a cell holding rows in BOTH poles
    can contribute a within-cell difference; the rest are reported, not used.
    Pooled, this also equalises the games, because every game contributes the same
    30 cells - so a game whose poles happened to fill up cannot dominate the sum.
    """
    by_cell = {}
    for item in alt:
        by_cell.setdefault(item["cell"], ([], []))[0].append(item)
    for item in self_:
        by_cell.setdefault(item["cell"], ([], []))[1].append(item)
    usable = [(c, a, s) for c, (a, s) in sorted(by_cell.items()) if a and s]
    if not usable:
        return None, [], len(by_cell)
    stack = torch.stack([mean_of(acts, a) - mean_of(acts, s) for _c, a, s in usable])
    return stack.mean(dim=0), [c for c, _a, _s in usable], len(by_cell)


def cosines(a, b):
    """Per-layer cosine between two (L, H) tensors, NaN on a zero row."""
    a, b = a.double(), b.double()
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(den > 0, num / den, torch.full_like(num, float("nan")))


# --- separation --------------------------------------------------------------

def auc(pos, neg):
    """Mann-Whitney AUC: P(a random altruistic projection > a random self one)."""
    values = torch.cat([pos, neg])
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=values.dtype)
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    n1, n2 = len(pos), len(neg)
    return ((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)).item()


def project(acts, items, direction):
    idx = torch.tensor([i["position"] for i in items], dtype=torch.long)
    x = acts.index_select(0, idx).double()                       # (n, L, H)
    unit = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (x * unit.unsqueeze(0)).sum(dim=2)                    # (n, L)


def separation(acts, direction, alt, self_):
    """Per-layer AUC and Cohen's d of `alt` against `self_` along `direction`."""
    pa = project(acts, alt, direction)
    ps = project(acts, self_, direction)
    out = []
    for l in range(pa.shape[1]):
        a, s = pa[:, l], ps[:, l]
        pooled = math.sqrt((a.var(unbiased=True).item()
                            + s.var(unbiased=True).item()) / 2)
        out.append({
            "layer": l,
            "auc": auc(a, s),
            "cohens_d": (a.mean().item() - s.mean().item()) / pooled
                        if pooled > 0 else float("nan"),
        })
    return out


def halve(items, generator):
    order = torch.randperm(len(items), generator=generator).tolist()
    cut = len(items) // 2
    return [items[i] for i in order[:cut]], [items[i] for i in order[cut:]]


def split_half(acts, alt, self_, seed):
    """Fit on half the rows, score the other half. Per layer, out of sample."""
    g = torch.Generator().manual_seed(seed)
    alt_a, alt_b = halve(alt, g)
    self_a, self_b = halve(self_, g)
    if min(len(alt_a), len(alt_b), len(self_a), len(self_b)) < 2:
        return None
    direction = naive_vector(acts, alt_a, self_a)
    reliability = cosines(direction, naive_vector(acts, alt_b, self_b))
    return {
        "fit_altruistic": len(alt_a), "fit_self": len(self_a),
        "score_altruistic": len(alt_b), "score_self": len(self_b),
        "by_layer": separation(acts, direction, alt_b, self_b),
        # how much of the direction survives being rebuilt from a disjoint half:
        # the ceiling any cosine against another vector could reach
        "reliability_cos_by_layer": reliability.tolist(),
    }


# --- nulls -------------------------------------------------------------------

class LabelNull:
    """Rebuilds the difference vector from the SAME rows with the pole labels permuted.

    The activation block and its column sum are materialised once, because a null
    is drawn hundreds of times per game and re-slicing the block on every draw is
    the whole cost of the analysis.

    What this is a null OF matters and is stated wherever it is reported: it
    permutes the LABELS, so it answers "could a random split of these same
    activations produce this". It is NOT a null of two unrelated directions -
    shuffled vectors keep whatever structure the activations share, so it is not
    centred on zero, and its spread is the scale a real cosine has to beat.
    """

    def __init__(self, acts, alt, self_):
        positions = torch.tensor([i["position"] for i in alt + self_], dtype=torch.long)
        self.x = acts.index_select(0, positions).double()
        self.total = self.x.sum(dim=0)
        self.n = self.x.shape[0]
        self.n_alt = len(alt)
        if not 0 < self.n_alt < self.n:
            raise ValueError("a label null needs both poles: n_alt=%d of n=%d"
                             % (self.n_alt, self.n))

    def draw(self, generator):
        perm = torch.randperm(self.n, generator=generator)[:self.n_alt]
        sum_alt = self.x.index_select(0, perm).sum(dim=0)
        return (sum_alt / self.n_alt
                - (self.total - sum_alt) / (self.n - self.n_alt))


def summarize(values):
    v = values[torch.isfinite(values)]
    if v.numel() == 0:
        return {"n": 0}
    return {"n": int(v.numel()), "mean": v.mean().item(),
            "sd": v.std(unbiased=True).item(),
            "min": v.min().item(), "max": v.max().item(),
            "abs_max": v.abs().max().item(),
            "p2.5": v.quantile(0.025).item(), "p97.5": v.quantile(0.975).item()}


# --- the run -----------------------------------------------------------------

def pole_census(labels, policy):
    """Counts by pole, by tag, by tag class, per cell and per stake."""
    key = "pole_" + policy
    census = {"n": len(labels), "policy": policy, "by_pole": {}, "by_tag": {},
              "by_tag_class": {}, "by_stake": {}, "by_wording": {}}
    for item in labels:
        census["by_pole"][item[key]] = census["by_pole"].get(item[key], 0) + 1
        census["by_tag"][item["tag"]] = census["by_tag"].get(item["tag"], 0) + 1
        census["by_tag_class"][item["tag_class"]] = \
            census["by_tag_class"].get(item["tag_class"], 0) + 1
    for field in ("stake", "wording"):
        bucket = census["by_" + field]
        for item in labels:
            if item["tag_class"] != "direct":
                continue
            bucket.setdefault(item[field], {}).setdefault(item[key], 0)
            bucket[item[field]][item[key]] += 1
    return census


def build_family(acts, labels, family, out_dir, seed, save, policy):
    """Everything about one game: poles, both vectors, its own out-of-sample AUC."""
    mine = [i for i in labels if i["family"] == family]
    alt, self_, middle, excluded = split_poles(mine, False, policy)
    alt_d, self_d, _m, _e = split_poles(mine, True, policy)
    entry = {
        "n_captured": len(mine),
        "n_altruistic": len(alt), "n_self_interested": len(self_),
        "n_middle_discarded": len(middle), "n_tag_excluded": len(excluded),
        "n_altruistic_with_derived": len(alt_d),
        "n_self_interested_with_derived": len(self_d),
        "census": pole_census(mine, policy),
    }
    if not alt or not self_:
        entry["usable"] = False
        entry["reason"] = ("no altruistic rows" if not alt else "no self-interested rows")
        return entry, None, None
    entry["usable"] = True

    naive = naive_vector(acts, alt, self_)
    bal, usable_cells, n_cells = balanced_vector(acts, alt, self_)
    entry["usable_cells"] = usable_cells
    entry["n_cells_seen"] = n_cells
    entry["norms_unbalanced"] = naive.norm(dim=1).tolist()
    entry["norms_balanced"] = bal.norm(dim=1).tolist() if bal is not None else None
    entry["cos_balanced_vs_unbalanced"] = \
        cosines(naive, bal).tolist() if bal is not None else None
    if len(alt_d) > len(alt) or len(self_d) > len(self_):
        sens = naive_vector(acts, alt_d, self_d)
        entry["cos_sensitivity_vs_primary"] = cosines(sens, naive).tolist()
    entry["split_half"] = split_half(acts, alt, self_, seed)
    if save is not None:
        torch.save(naive.float(),
                   save / ("decision_%s_response_avg_diff_%s.pt" % (family, policy)))
        if bal is not None:
            torch.save(bal.float(),
                       save / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                               % (family, policy)))
    return entry, naive, bal


def agreement(acts, labels, families, layer, draws, seed, use_balanced, policy):
    """The 6x6 cosine matrix at one layer, with a shuffled-LABEL null per pair.

    The null permutes the pole labels within each game and rebuilds both vectors,
    so it answers "could this much agreement arise from a random split of these
    same activations". It is a null with respect to the LABELS. It is NOT a null
    of two orthogonal directions: shuffled vectors keep whatever structure the
    activations share, so the null cosine is not centred on zero and its spread is
    the scale a real cosine has to beat.
    """
    per_family = {}
    for family in families:
        mine = [i for i in labels if i["family"] == family]
        alt, self_, _m, _e = split_poles(mine, False, policy)
        if not alt or not self_:
            continue
        naive = naive_vector(acts, alt, self_)
        bal, _c, _n = balanced_vector(acts, alt, self_)
        vec = bal if (use_balanced and bal is not None) else naive
        per_family[family] = {"null": LabelNull(acts, alt, self_), "vec": vec}

    names = [f for f in families if f in per_family]
    real = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            real[a + "|" + b] = cosines(per_family[a]["vec"],
                                        per_family[b]["vec"])[layer].item()

    g = torch.Generator().manual_seed(seed)
    null_draws = {k: [] for k in real}
    for _ in range(draws):
        shuffled = {f: per_family[f]["null"].draw(g) for f in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                null_draws[a + "|" + b].append(
                    cosines(shuffled[a], shuffled[b])[layer].item())

    matrix = {"layer": layer, "policy": policy,
              "vector": "cellbalanced" if use_balanced else "unbalanced",
              "families": names, "draws": draws, "pairs": {}}
    for key, value in real.items():
        null = torch.tensor(null_draws[key], dtype=torch.double)
        matrix["pairs"][key] = {"cosine": value, "label_null": summarize(null)}
    return matrix


def leave_one_out(acts, labels, families, seed, policy):
    """Fit the direction on five games; score the sixth game's own poles with it.

    The strongest form of the question. A cosine says two directions point the
    same way; this says a direction that never saw a game still separates that
    game's poles. Nothing about the held-out game enters the fit, so the AUC is
    out of sample in the strict sense.
    """
    out = {}
    for held in families:
        rest = [i for i in labels if i["family"] != held]
        mine = [i for i in labels if i["family"] == held]
        alt_r, self_r, _m, _e = split_poles(rest, False, policy)
        alt_h, self_h, _m2, _e2 = split_poles(mine, False, policy)
        if not (alt_r and self_r and alt_h and self_h):
            out[held] = {"usable": False}
            continue
        direction, _c, _n = balanced_vector(acts, alt_r, self_r)
        if direction is None:
            out[held] = {"usable": False}
            continue
        out[held] = {
            "usable": True,
            "fit_families": sorted({i["family"] for i in rest}),
            "n_held_altruistic": len(alt_h), "n_held_self": len(self_h),
            "by_layer": separation(acts, direction, alt_h, self_h),
        }
    return out


def run_policy(acts_all, labels, families, out_dir, args, policy, dictator, theirs):
    """The whole battery under one pole policy. Every separation number is out of sample."""
    a = acts_all["response_avg"]
    result = {"policy": policy}

    # ---- per game -----------------------------------------------------------
    result["per_game"] = {}
    vectors = {}
    for family in families:
        entry, naive, bal = build_family(a, labels, family, out_dir, args.seed,
                                         out_dir, policy)
        result["per_game"][family] = entry
        if entry["usable"]:
            vectors[family] = {"unbalanced": naive, "balanced": bal}
    result["games_with_a_vector"] = sorted(vectors)
    result["games_without_a_vector"] = {
        f: result["per_game"][f].get("reason")
        for f in families if not result["per_game"][f]["usable"]}

    # A game that filled only one pole is dropped from every POOLED structure, not
    # just from cell balancing. Its rows would enter the pooled difference on one
    # side only, which makes part of that difference "which game is this" rather
    # than "what did the model decide" - the exact confound the pooling is for.
    # Cell balancing already refuses such a game's cells; the unbalanced vector,
    # the split-half AUC and the leave-one-out fits would not, so the filter is
    # applied to the label set itself.
    pooled_families = [f for f in families if f in vectors]
    labels = [i for i in labels if i["family"] in pooled_families]
    result["pooled_families"] = pooled_families
    result["families_excluded_from_pool"] = [f for f in families
                                             if f not in pooled_families]

    # ---- pooled -------------------------------------------------------------
    alt, self_, middle, excluded = split_poles(labels, False, policy)
    if not alt or not self_:
        result["pooled"] = {"usable": False}
        return result, None
    pooled_naive = naive_vector(a, alt, self_)
    pooled_bal, usable_cells, n_cells = balanced_vector(a, alt, self_)
    torch.save(pooled_naive.float(),
               out_dir / ("decision_pooled_response_avg_diff_%s.pt" % policy))
    if pooled_bal is not None:
        torch.save(pooled_bal.float(),
                   out_dir / ("decision_pooled_response_avg_diff_cellbalanced_%s.pt"
                              % policy))
    result["pooled"] = {
        "usable": True,
        "n_altruistic": len(alt), "n_self_interested": len(self_),
        "n_middle_discarded": len(middle), "n_tag_excluded": len(excluded),
        "usable_cells": len(usable_cells), "n_cells_seen": n_cells,
        "usable_cells_by_family": {
            f: len([c for c in usable_cells if c.startswith(f + "/")])
            for f in pooled_families},
        "pole_mix_by_family": {
            f: {"altruistic": len([i for i in alt if i["family"] == f]),
                "self_interested": len([i for i in self_ if i["family"] == f])}
            for f in pooled_families},
        "norms_unbalanced": pooled_naive.norm(dim=1).tolist(),
        "norms_balanced": pooled_bal.norm(dim=1).tolist() if pooled_bal is not None else None,
        "cos_balanced_vs_unbalanced": (cosines(pooled_naive, pooled_bal).tolist()
                                       if pooled_bal is not None else None),
        "split_half": split_half(a, alt, self_, args.seed),
    }
    reference = pooled_bal if pooled_bal is not None else pooled_naive

    # The pooled direction scored inside each single game. In sample - the game's
    # own rows are part of the pooled fit - so it is a decomposition of the pooled
    # number, not evidence of transfer. `leave_one_game_out` is the honest version.
    result["pooled_direction_within_game"] = {}
    for family in pooled_families:
        mine = [i for i in labels if i["family"] == family]
        alt_f, self_f, _m, _e = split_poles(mine, False, policy)
        if not alt_f or not self_f:
            result["pooled_direction_within_game"][family] = {"usable": False}
            continue
        result["pooled_direction_within_game"][family] = {
            "usable": True, "n_altruistic": len(alt_f), "n_self_interested": len(self_f),
            "in_sample": True,
            "by_layer": separation(a, reference, alt_f, self_f),
        }

    # ---- the three headline structures --------------------------------------
    result["agreement_matrix_cellbalanced"] = agreement(
        a, labels, pooled_families, args.layer, args.pair_shuffles, args.seed, True, policy)
    result["agreement_matrix_unbalanced"] = agreement(
        a, labels, pooled_families, args.layer, args.pair_shuffles, args.seed + 1,
        False, policy)
    result["agreement_matrix_layer0_cellbalanced"] = agreement(
        a, labels, pooled_families, 0, args.pair_shuffles, args.seed + 2, True, policy)
    result["leave_one_game_out"] = leave_one_out(a, labels, pooled_families,
                                                 args.seed, policy)

    # ---- against the vectors that already exist ------------------------------
    g = torch.Generator().manual_seed(args.seed)
    pooled_null = LabelNull(a, alt, self_)
    null_dict, null_theirs, null_self = [], [], []
    for _ in range(args.shuffles):
        fake = pooled_null.draw(g)
        null_dict.append(cosines(fake, dictator))
        null_theirs.append(cosines(fake, theirs))
        null_self.append(cosines(fake, reference))
    result["vs_existing"] = {
        "cos_pooled_balanced_vs_dictator": cosines(reference, dictator).tolist(),
        "cos_pooled_unbalanced_vs_dictator": cosines(pooled_naive, dictator).tolist(),
        "cos_pooled_balanced_vs_their_altruism": cosines(reference, theirs).tolist(),
        "cos_pooled_unbalanced_vs_their_altruism": cosines(pooled_naive, theirs).tolist(),
        "label_null_vs_dictator_layer_%d" % args.layer:
            summarize(torch.stack(null_dict)[:, args.layer]),
        "label_null_vs_their_altruism_layer_%d" % args.layer:
            summarize(torch.stack(null_theirs)[:, args.layer]),
        "label_null_vs_pooled_itself_layer_%d" % args.layer:
            summarize(torch.stack(null_self)[:, args.layer]),
        "label_null_vs_dictator_p97.5_by_layer":
            torch.stack(null_dict).quantile(0.975, dim=0).tolist(),
        "label_null_vs_their_altruism_p97.5_by_layer":
            torch.stack(null_theirs).quantile(0.975, dim=0).tolist(),
        "per_game": {
            f: {"cos_vs_dictator_vector": cosines(
                    v["balanced"] if v["balanced"] is not None else v["unbalanced"],
                    dictator).tolist(),
                "cos_vs_their_altruism": cosines(
                    v["balanced"] if v["balanced"] is not None else v["unbalanced"],
                    theirs).tolist(),
                "cos_vs_pooled": cosines(
                    v["balanced"] if v["balanced"] is not None else v["unbalanced"],
                    reference).tolist()}
            for f, v in vectors.items()},
    }

    # ---- the prompt-side variants, kept as a check, not a result -------------
    result["prompt_side_check"] = {}
    for variant in ("prompt_avg", "prompt_last"):
        v = naive_vector(acts_all[variant], alt, self_)
        result["prompt_side_check"][variant] = {
            "norms": v.norm(dim=1).tolist(),
            "split_half": split_half(acts_all[variant], alt, self_, args.seed),
            "note": "causal masking makes prompt-side activations identical within a "
                    "prompt cell, so any separation here is between-cell pole "
                    "composition, not a decision signal",
        }
    return result, reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="directory for vectors + analysis.json")
    ap.add_argument("--dictator-vector", default=str(
        HERE.parents[3] / "results/dictator-decision-vector/vectors"
        / "decision_response_avg_diff_cellbalanced.pt"))
    ap.add_argument("--their-vectors", default=str(
        HERE.parents[3] / "persona_vectors/Qwen2.5-7B-Instruct"))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--pair-shuffles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, acts, metas, tag_counts = load_all(manifest)
    families = [f for f in crossgame_grid.FAMILIES
                if any(i["family"] == f for i in labels)]

    dictator = torch.load(args.dictator_vector, map_location="cpu").double()
    theirs = torch.load(Path(args.their_vectors) / "altruism_response_avg_diff.pt",
                        map_location="cpu").double()

    report = {
        "seed": args.seed, "headline_layer": args.layer,
        "shuffles": args.shuffles, "pair_shuffles": args.pair_shuffles,
        "hidden_size": acts["response_avg"].shape[2],
        "n_layers_plus_embedding": acts["response_avg"].shape[1],
        "theoretical_random_cosine_sd": 1.0 / math.sqrt(acts["response_avg"].shape[2]),
        "theoretical_null_note": "the theoretical figure is quoted only to be "
                                 "dismissed: the measured label-null below is very "
                                 "much wider, and it is the one every claim uses",
        "families": families,
        "n_captured_total": len(labels),
        "dictator_vector_path": str(args.dictator_vector),
        "activation_meta": metas,
        "tag_counts_all_generated_rows": tag_counts,
        "cos_dictator_vs_their_altruism_by_layer": cosines(dictator, theirs).tolist(),
    }

    report["by_policy"] = {}
    for policy in poles.POLICIES:
        result, _ref = run_policy(acts, labels, families, out_dir, args, policy,
                                  dictator, theirs)
        report["by_policy"][policy] = result
        print("policy %s done" % policy, flush=True)

    with (out_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote %s" % (out_dir / "analysis.json"), flush=True)


if __name__ == "__main__":
    main()
