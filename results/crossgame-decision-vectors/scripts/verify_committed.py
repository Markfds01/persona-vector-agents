"""Independent recomputation of every load-bearing number in README.md.

Written during packaging, not part of either run. It exists so the README's
tables can be checked by `regenerate and diff` rather than taken on trust, and so
that the check does not go through either run's analysis code: it imports only
`crossgame_grid` and `poles`, which are the grid and pole DEFINITIONS, and does
its own linear algebra.

Sections A and A2 are the only ones with a committed artifact to disagree with,
and they are a GATE: every committed vector must come back at cosine >= the
`--min-cosine` floor AND at the same norm ON EVERY LAYER, or this exits non-zero.
A cosine is scale-invariant, so the norm is checked separately — without it a
vector twice as long still reads 1.000000 — and it is checked per layer, because
gating layer 20 alone left a doubled layer 5 passing at 1.000000 on both numbers
while sections 1, 3 and 5 read layer 0. The two tolerances sit on different
floors: a cosine is blind to how the committed vector is STORED, a norm is not,
and the committed `.pt` files are float32, whose rounding alone moves the norm by
~1e-8. `--max-norm-drift` is set above that floor and still a thousandfold below
any real scale error. The report is written either way, so a failure leaves its
evidence on disk.

What it recomputes, all from the activation shards:

  A  the six per-game cell-balanced vectors, against the committed .pt files
  A2 the nine pooled vectors, against the committed .pt files
  B  the 6x6 agreement matrix at layer 20, and at layer 0
  C  leave-one-game-out AUC under the cell-balanced pool
  D  leave-one-game-out AUC under every one of the nine weightings
  E  pooled cosines against the Dictator-only vector and the shipped altruism
     vector, per scheme
  F  the Dictator-projected-out collapse
  G  the layer-0 digit-span share, its control bands, the per-digit decomposition
     and the token decode
  H  the pole census, tag counts, parse coverage, focal shares, effective n
  I  the PD wording breakdown and the payoff-matrix probe
  J  split-half within cell at an INDEPENDENT seed - a robustness check on the
     run's own split, not a reproduction of it: the number depends on the draw
  K  the leave-one-game-out cosine between a pool and the game it never saw

The nine weightings and `effective_n` are restated in this file rather than
imported from `scripts/pooling/common.py`. That duplication is the point: sharing
the code would make this a re-execution of the analysis instead of a check on it.

CPU only. No GPU, no model loaded, no forward pass: G reads the embedding matrix
off the safetensors shard because layer 0 of `response_avg` IS an average of rows
of that matrix.

Usage, from a checkout of this repository:

    python results/crossgame-decision-vectors/scripts/verify_committed.py \
        --acts   <dir holding acts/<game>/shard_*.pt and rows.csv> \
        --grid   results/crossgame-decision-vectors/scripts/prompting \
        --dictator <path to the archived Dictator-only cell-balanced vector> \
        --snapshot <local Qwen2.5-7B-Instruct snapshot dir> \
        --out    results/crossgame-decision-vectors/analysis/verification.json

`--acts` is the one input that is not committed (5.3 GB); README.md says how to
regenerate it. Without `--snapshot` the digit-share and token-decode section is
skipped and everything else still runs.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DIR = HERE.parent

FAMILY_ORDER = ("dictator", "trust", "ultimatum", "apology", "overfishing",
                "prisoners_dilemma")
DIGIT_CONTROL_SEED = 20260820

#: the ANSWER-FORMAT partition: four games answer in dollars, one in fish counts,
#: one in the words Cooperate/Defect. This is the axis the confound lives on.
FORMAT_GROUPS = (("dollar", ("dictator", "trust", "ultimatum", "apology")),
                 ("fish", ("overfishing",)),
                 ("binary", ("prisoners_dilemma",)))
NON_DOLLAR = ("overfishing", "prisoners_dilemma")

#: scheme -> (across-game weight kind, whether each game is unit-normalised first).
#: `cell_balanced` is not here: it is the flat mean over every usable cell of every
#: game, which is not a weighted sum of the per-game vectors.
SCHEMES = {
    "game_equal_raw": ("equal", False),
    "game_equal_unit": ("equal", True),
    "game_precision_raw": ("precision", False),
    "game_precision_unit": ("precision", True),
    "family_balanced_raw": ("family", False),
    "family_balanced_unit": ("family", True),
    "non_dollar_raw": ("non_dollar", False),
    "non_dollar_unit": ("non_dollar", True),
}
ALL_SCHEMES = ("cell_balanced",) + tuple(SCHEMES)


# --- provenance ---------------------------------------------------------------
#
# This report is committed to a public repository, so nothing recorded here may
# carry the absolute path of the checkout that produced it: an input is named by
# where it sits in the repository, or by what identifies it.

def repo_path(path, root):
    """A path as `root`-relative posix, or a bare name when it is outside `root`."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return "(outside the repository) %s" % resolved.name


def acts_identity(acts_root):
    """What the activation directory IS, from each game's extraction meta.

    The directory is 5.3 GB and not committed, so its own name plus the model and
    the digest of the rows each game was extracted from is what tells a reader
    which corpus the gate ran against.
    """
    games = {}
    for family in FAMILY_ORDER:
        meta_path = Path(acts_root) / family / "meta.json"
        if not meta_path.is_file():
            games[family] = {"error": "no meta.json beside the shards"}
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        games[family] = {key: meta.get(key) for key in
                         ("model_id", "model_revision", "rows_csv_sha256",
                          "extractor", "n_captured")}
    return {"what": "the uncommitted activation directory, outside the repository",
            "dir_name": Path(acts_root).resolve().name, "games": games}


def snapshot_identity(snapshot):
    """The model behind a local weights directory, not the cache path to it.

    HuggingFace lays its cache out as `models--<org>--<name>/snapshots/<sha>`,
    which is exactly the model id and revision; anything else keeps its name.
    """
    path = Path(snapshot).resolve()
    revisions = path.parent
    repo = revisions.parent
    if revisions.name == "snapshots" and repo.name.startswith("models--"):
        return {"model_id": repo.name[len("models--"):].replace("--", "/"),
                "revision": path.name}
    return {"what": "not a HuggingFace cache snapshot; only its name is recorded",
            "dir_name": path.name}


# --- loading ------------------------------------------------------------------

def load(acts_root, grid, poles, policy):
    """(N, 29, 3584) response_avg for all six games, plus aligned labels.

    Read straight off the shards, in the manifest's family order, so a position
    index means the same thing here as it does in the runs' own code.
    """
    labels, blocks, offset = [], [], 0
    for family in FAMILY_ORDER:
        game_dir = Path(acts_root) / family
        with open(game_dir / "rows.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        index, chunks = [], []
        for path in sorted(game_dir.glob("shard_*.pt")):
            payload = torch.load(path, map_location="cpu")
            index.extend(payload["row_index"].tolist())
            chunks.append(payload["response_avg"].clone())
            del payload
        if not chunks:
            raise SystemExit("no activation shards in %s" % game_dir)
        block = torch.cat(chunks, dim=0)
        del chunks
        if len(index) != block.shape[0]:
            raise SystemExit("%s: shard row count disagrees with activations" % family)
        if len(set(index)) != len(index):
            raise SystemExit("%s: a row index appears in more than one shard; it "
                             "would be double-weighted in every pole mean" % game_dir)
        for position, row_index in enumerate(index):
            row = rows[row_index]
            game = grid.GRID_BY_ID[row["game_id"]]
            if game.family != family:
                raise SystemExit("%s: row %s is family %s"
                                 % (family, row["game_id"], game.family))
            value = float(row["value"])
            labels.append({
                "position": offset + position,
                "family": family,
                "cell": game.cell,
                "wording": game.wording,
                "value": value,
                "scale": float(game.pole_scale),
                "scorer": game.scorer,
                "tag": row["tag"],
                "tag_class": poles.tag_class(row["tag"], game.scorer),
                "pole": poles.classify(value, game, policy),
            })
        blocks.append(block)
        offset += block.shape[0]
        print("loaded %s: %d rows" % (family, block.shape[0]), flush=True)
    acts = torch.cat(blocks, dim=0)
    del blocks
    if acts.shape[0] != len(labels):
        raise SystemExit("activation count disagrees with label count")
    return acts, labels


# --- the pieces of the construction ------------------------------------------

def usable_cells(labels, poles, family):
    """{cell: (alt_positions, self_positions)} for cells holding BOTH poles."""
    seen = {}
    for item in labels:
        if item["family"] != family or item["tag_class"] != "direct":
            continue
        if item["pole"] == poles.MIDDLE:
            continue
        bucket = seen.setdefault(item["cell"], ([], []))
        bucket[0 if item["pole"] == poles.ALT else 1].append(item["position"])
    return {c: v for c, v in sorted(seen.items()) if v[0] and v[1]}


def pole_positions(labels, poles, family):
    alt, self_ = [], []
    for item in labels:
        if item["family"] != family or item["tag_class"] != "direct":
            continue
        if item["pole"] == poles.ALT:
            alt.append(item["position"])
        elif item["pole"] == poles.SELF:
            self_.append(item["position"])
    return alt, self_


def mean_of(acts, positions):
    idx = torch.tensor(positions, dtype=torch.long)
    return acts.index_select(0, idx).double().mean(dim=0)


def cell_balanced(acts, cells):
    """Unweighted mean over cells of (alt mean - self mean). float64."""
    return torch.stack([mean_of(acts, a) - mean_of(acts, s)
                        for _c, (a, s) in sorted(cells.items())]).mean(dim=0)


def unit(v):
    n = v.norm(dim=1, keepdim=True)
    return torch.where(n > 0, v / n.clamp_min(1e-12), torch.zeros_like(v))


def cosines(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1))


def effective_n(cells):
    c = len(cells)
    inv = sum(1.0 / len(a) + 1.0 / len(s) for a, s in cells.values())
    return (c * c) / inv if inv > 0 else 0.0


def across_game_weights(cells_by_family, scheme):
    """{family: weight}, normalised to sum 1.

    `equal` still leaves 4/6 of the weight on dollar-format answers, because four
    of the six games are dollar games: balancing the game axis is not balancing
    the format axis. `family` equalises the format axis instead, and was chosen
    AFTER the agreement matrix showed the dollar / non-dollar split.
    """
    kind, _unit_first = SCHEMES[scheme]
    families = sorted(cells_by_family)
    if kind == "equal":
        raw = {f: 1.0 for f in families}
    elif kind == "precision":
        raw = {f: effective_n(cells_by_family[f]) for f in families}
    elif kind == "family":
        present = [(name, [f for f in group if f in cells_by_family])
                   for name, group in FORMAT_GROUPS]
        present = [(name, group) for name, group in present if group]
        share = 1.0 / len(present)
        raw = {f: 0.0 for f in families}
        for _name, group in present:
            for f in group:
                raw[f] = share / len(group)
    elif kind == "non_dollar":
        keep = [f for f in NON_DOLLAR if f in cells_by_family]
        raw = {f: (1.0 / len(keep) if f in keep else 0.0) for f in families}
    else:
        raise ValueError("unknown weight kind %r" % (kind,))
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("%s: no game carries any weight here" % scheme)
    return {f: w / total for f, w in raw.items()}


def weighted_pool(game_vecs, cells_by_family, scheme):
    """One pooled direction: sum_g w_g * (v_g, unit-normalised per layer or not)."""
    _kind, unit_first = SCHEMES[scheme]
    weights = across_game_weights(cells_by_family, scheme)
    families = sorted(game_vecs)
    stack = torch.stack([unit(game_vecs[f]) if unit_first else game_vecs[f]
                         for f in families])
    w = torch.tensor([weights[f] for f in families], dtype=torch.double)
    return (stack * w.view(-1, 1, 1)).sum(dim=0)


def auc(pos, neg):
    """Mann-Whitney AUC with tie correction."""
    values = torch.cat([pos, neg])
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=values.dtype)
    ordered = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or ordered[i] != ordered[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    n1, n2 = len(pos), len(neg)
    return ((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)).item()


def auc_by_layer(acts, direction, alt, self_):
    u = unit(direction)

    def project(positions):
        idx = torch.tensor(positions, dtype=torch.long)
        return (acts.index_select(0, idx).double() * u.unsqueeze(0)).sum(dim=2)

    pa, ps = project(alt), project(self_)
    return [auc(pa[:, l], ps[:, l]) for l in range(pa.shape[1])]


def summarise_auc(values):
    best = max(range(len(values)), key=lambda l: values[l])
    return {"layer0": values[0], "layer20": values[20],
            "best": values[best], "best_layer": best}


# --- G: the layer-0 digit share and token decode ------------------------------

def digit_section(snapshot, acts_root, vectors, control_draws):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    snapshot = Path(snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    emb = load_file(str(snapshot / index["weight_map"][key]))[key].float()
    tok = AutoTokenizer.from_pretrained(str(snapshot))

    digit_ids = [tok.encode(str(d))[0] for d in range(10)]
    basis, _ = torch.linalg.qr(emb[digit_ids].double().T)

    # The control is empirical, not spherical: token embeddings are not isotropic,
    # so ANY direction overlaps ANY ten of them more than a sphere would predict.
    # The right pool is the tokens the model actually emitted in these responses.
    ids = set()
    for family in FAMILY_ORDER:
        with open(Path(acts_root) / family / "rows.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        step = max(1, len(rows) // 200)
        for row in rows[::step]:
            ids.update(tok.encode(row["continuation"]))
    pool = torch.tensor(sorted(ids))

    # The run drew 32 vocab-wide control subspaces BEFORE its response-token ones,
    # off one generator. Reproducing its numbers means reproducing that order.
    g = torch.Generator().manual_seed(DIGIT_CONTROL_SEED)
    vocab32 = [torch.linalg.qr(
        emb[torch.randperm(emb.shape[0], generator=g)[:10].tolist()].double().T)[0]
        for _ in range(32)]
    response32 = [torch.linalg.qr(
        emb[pool[torch.randperm(len(pool), generator=g)[:10]].tolist()].double().T)[0]
        for _ in range(32)]
    # 32 draws is few for a p97.5 and one draw containing a digit token inflates
    # it, so the same band is re-estimated with many more.
    g2 = torch.Generator().manual_seed(DIGIT_CONTROL_SEED)
    response_many = [torch.linalg.qr(
        emb[pool[torch.randperm(len(pool), generator=g2)[:10]].tolist()].double().T)[0]
        for _ in range(control_draws)]

    big = emb[[tok.encode(d)[0] for d in "4567"]].double().mean(dim=0)
    zero = emb[tok.encode("0")[0]].double()
    axis = big - zero
    axis = axis / axis.norm()
    # the aggregate share is unsigned and cannot tell "'5' marks generous" from
    # "'0' marks the extreme", which is exactly what changes across weightings
    zero_basis = (zero / zero.norm()).view(-1, 1)

    normed = emb.double()
    normed = normed / normed.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def band(controls, v):
        s = torch.tensor([float((c.T @ v).norm() / v.norm()) for c in controls])
        return {"n_draws": len(controls), "mean": float(s.mean()),
                "p97.5": float(s.quantile(0.975)), "max": float(s.max())}

    out = {"n_distinct_response_tokens": len(ids),
           "digit_token_ids": digit_ids,
           "theoretical_spherical_baseline": math.sqrt(10 / emb.shape[1]),
           "per_vector": {}}
    for name, vec in vectors.items():
        v = vec.double()[0]
        sims = normed @ (v / v.norm())
        top = torch.topk(sims, 12)
        bottom = torch.topk(-sims, 12)
        out["per_vector"][name] = {
            "digit_span_share": float((basis.T @ v).norm() / v.norm()),
            "cos_vs_big_digit_minus_zero": float(torch.dot(axis, v) / v.norm()),
            "zero_only_span_share": float((zero_basis.T @ v).norm() / v.norm()),
            "cos_per_digit": {str(d): float(
                torch.dot(emb[digit_ids[d]].double(), v)
                / (emb[digit_ids[d]].double().norm() * v.norm())) for d in range(10)},
            "response_control_32": band(response32, v),
            "response_control_many": band(response_many, v),
            "vocab_control_32": band(vocab32, v),
            "toward_altruistic": [[tok.convert_ids_to_tokens(int(i)), round(float(s), 3)]
                                  for s, i in zip(top.values, top.indices)],
            "toward_self_interested": [[tok.convert_ids_to_tokens(int(i)),
                                        round(-float(s), 3)]
                                       for s, i in zip(bottom.values, bottom.indices)],
        }
    return out


# --- J: split-half within cell, at a seed the run did not use -----------------

def independent_split(cells_by_family, seed):
    """Two disjoint halves of every cell, from a generator this file owns.

    Splitting INSIDE each cell keeps the design identical on both sides, so the
    only thing that differs is which rows landed where. The seed is deliberately
    NOT the run's: reproducing a randomised statistic would mean replaying its
    exact draw, which tests the RNG rather than the conclusion. This asks the
    weaker and more useful question - does the ordering survive a different split.
    """
    generator = torch.Generator().manual_seed(seed)
    fit, score = {}, {}
    for family in FAMILY_ORDER:
        for cell, (alt, self_) in sorted(cells_by_family[family].items()):
            halves = []
            for group in (alt, self_):
                order = torch.randperm(len(group), generator=generator).tolist()
                cut = max(1, len(group) // 2)
                halves.append(([group[i] for i in order[:cut]],
                               [group[i] for i in order[cut:]]))
            (alt_a, alt_b), (self_a, self_b) = halves
            if alt_a and self_a:
                fit.setdefault(family, {})[cell] = (alt_a, self_a)
            if alt_b and self_b:
                score.setdefault(family, {})[cell] = (alt_b, self_b)
    return fit, score


def split_half_section(acts, cells_by_family, seed):
    fit, score = independent_split(cells_by_family, seed)
    fit_games = {f: cell_balanced(acts, c) for f, c in fit.items()}
    flat_fit = {}
    for f, c in fit.items():
        flat_fit.update({"%s/%s" % (f, cell): v for cell, v in c.items()})
    directions = {"cell_balanced": cell_balanced(acts, flat_fit)}
    for scheme in SCHEMES:
        directions[scheme] = weighted_pool(fit_games, fit, scheme)

    out = {"seed": seed, "n_fit_cells": {f: len(c) for f, c in sorted(fit.items())},
           "n_score_cells": {f: len(c) for f, c in sorted(score.items())},
           "schemes": {}}
    all_alt = [p for c in score.values() for a, _s in c.values() for p in a]
    all_self = [p for c in score.values() for _a, s in c.values() for p in s]
    for scheme, direction in directions.items():
        per_game = {}
        for family, cells in sorted(score.items()):
            alt = [p for a, _s in cells.values() for p in a]
            self_ = [p for _a, s in cells.values() for p in s]
            layers = auc_by_layer(acts, direction, alt, self_)
            per_game[family] = {"n_alt": len(alt), "n_self": len(self_),
                                "layer0": layers[0], "layer20": layers[20]}
        layers = auc_by_layer(acts, direction, all_alt, all_self)
        out["schemes"][scheme] = {
            "pooled": summarise_auc(layers),
            "macro_layer0": sum(e["layer0"] for e in per_game.values()) / len(per_game),
            "macro_layer20": sum(e["layer20"] for e in per_game.values()) / len(per_game),
            "per_game": per_game,
        }
    return out


# --- H, I: census straight from the row CSVs ----------------------------------

def census_section(acts_root, grid, poles, policy):
    out = {}
    for family in FAMILY_ORDER:
        with open(Path(acts_root) / family / "rows.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        tags, alt, self_ = {}, {}, {}
        scored = middle = excluded = 0
        by_wording = {}
        for row in rows:
            game = grid.GRID_BY_ID[row["game_id"]]
            tags[row["tag"]] = tags.get(row["tag"], 0) + 1
            if row["value"] == "":
                continue
            scored += 1
            value = float(row["value"])
            if poles.tag_class(row["tag"], game.scorer) != "direct":
                excluded += 1
                continue
            pole = poles.classify(value, game, policy)
            # focal share is on the value NORMALISED by the cell's scale, so the
            # six stakes of a game can be counted together
            key = value / float(game.pole_scale)
            if pole == poles.ALT:
                alt[key] = alt.get(key, 0) + 1
            elif pole == poles.SELF:
                self_[key] = self_.get(key, 0) + 1
            else:
                middle += 1
            slot = by_wording.setdefault(game.wording, {"n": 0, "alt": 0})
            slot["n"] += 1
            slot["alt"] += 1 if pole == poles.ALT else 0

        def modal(counts):
            if not counts:
                return None
            value, n = max(counts.items(), key=lambda kv: kv[1])
            total = sum(counts.values())
            return {"value_over_scale": value, "n": n, "total": total,
                    "share": n / total}

        unresolved = sum(n for t, n in tags.items() if t in poles.UNRESOLVED_TAGS)
        out[family] = {
            "rows_generated": len(rows), "rows_scored": scored,
            "n_altruistic": sum(alt.values()), "n_self_interested": sum(self_.values()),
            "n_middle": middle, "n_tag_excluded": excluded,
            "unresolved_fraction": unresolved / len(rows),
            "tag_counts": dict(sorted(tags.items(), key=lambda kv: -kv[1])),
            "focal_altruistic": modal(alt), "focal_self_interested": modal(self_),
            "by_wording": by_wording,
        }
    return out


def pd_probe_section(probe_csv, grid, poles):
    rows = list(csv.DictReader(open(probe_csv, encoding="utf-8")))
    by_matrix, cooperations = {}, 0
    for row in rows:
        value = float(row["value"])
        slot = by_matrix.setdefault(row["game_id"], {"n": 0, "cooperate": 0})
        slot["n"] += 1
        if value == 1.0:
            slot["cooperate"] += 1
            cooperations += 1
    return {"n_draws": len(rows), "n_matrices": len(by_matrix),
            "cooperations": cooperations, "by_matrix": by_matrix}


# --- the run ------------------------------------------------------------------

def committed_comparison(recomputed, committed):
    """How a recomputed vector compares with the .pt this directory ships.

    Both norms are taken in float64, but that does NOT make the ratio independent
    of storage: the committed file is float32, and promoting it cannot recover the
    bits rounding threw away. The ratio's floor is that rounding, ~1e-8, which is
    what `--max-norm-drift` has to clear. The cosine is not affected — rounding
    barely turns the vector.

    The ratio is taken on EVERY layer and the worst one is reported: layer 20 is
    where the claims are read, but a scale error on any other layer is just as
    much a failure to reproduce, and no cosine can see it.
    """
    committed = committed.double()
    c = cosines(recomputed, committed)
    mine, theirs = recomputed.norm(dim=1), committed.norm(dim=1)
    ratios = torch.where(theirs > 0, mine / theirs.clamp_min(1e-300),
                         torch.full_like(theirs, float("inf")))
    drift = (ratios - 1.0).abs()
    worst = int(drift.argmax())
    return {
        "cos_layer20": c[20].item(),
        "min_cos_over_layers": c.min().item(),
        "norm_layer20": mine[20].item(),
        "norm_layer20_committed": theirs[20].item(),
        "norm_ratio_layer20": ratios[20].item(),
        "worst_norm_drift": drift[worst].item(),
        "worst_norm_drift_layer": worst,
        "norm_at_worst_drift": mine[worst].item(),
        "norm_at_worst_drift_committed": theirs[worst].item(),
    }


def reproduction_gate(out, min_cosine, max_norm_drift):
    """Pass/fail over every committed vector this run recomputed.

    Everything outside A and A2 is a recomputation with nothing to check it
    against. These two have the committed artifacts, and until this existed a
    cos_layer20 of 0.3 was written into the report and the process exited 0.
    """
    failures = []
    seen_cosines, seen_drifts = [], []
    for section in ("A_per_game_vs_committed", "A2_pooled_vs_committed"):
        for name, entry in sorted(out[section].items()):
            if "error" in entry:
                failures.append("%s.%s: %s" % (section, name, entry["error"]))
                continue
            worst = entry["min_cos_over_layers"]
            seen_cosines.append(worst)
            if not worst >= min_cosine:
                failures.append("%s.%s: worst layer cosine %.12f < %.12f"
                                % (section, name, worst, min_cosine))
            drift = entry["worst_norm_drift"]
            seen_drifts.append(drift)
            if not drift <= max_norm_drift:
                failures.append("%s.%s: layer-%d norm %.6f against the committed "
                                "%.6f (%.3e drift)"
                                % (section, name, entry["worst_norm_drift_layer"],
                                   entry["norm_at_worst_drift"],
                                   entry["norm_at_worst_drift_committed"], drift))
    # a gate over nothing would report `passed` while proving nothing
    if not seen_cosines:
        failures.append("no committed vector was compared, so this gate proves nothing")
    # the observed margins are reported, not just the thresholds: a tolerance nobody
    # can see the distance to is a tolerance nobody notices is set wrong
    return {"min_cosine": min_cosine, "max_norm_drift": max_norm_drift,
            "n_vectors_checked": len(seen_cosines),
            "worst_min_cos_over_layers": min(seen_cosines, default=None),
            "worst_norm_drift_over_layers": max(seen_drifts, default=None),
            "failures": failures, "passed": not failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True,
                    help="directory holding <game>/shard_*.pt and <game>/rows.csv")
    ap.add_argument("--grid", default=str(HERE / "prompting"),
                    help="directory holding crossgame_grid.py and poles.py")
    ap.add_argument("--repo-root", default=None,
                    help="repository root, so crossgame_grid can import audit "
                         "(default: four levels above this file)")
    ap.add_argument("--vectors", default=str(DIR / "vectors"))
    ap.add_argument("--dictator", required=True,
                    help="the archived Dictator-only cell-balanced vector")
    ap.add_argument("--altruism", default=None,
                    help="the repo's shipped altruism vector "
                         "(default: persona_vectors/Qwen2.5-7B-Instruct under the root)")
    ap.add_argument("--probe", default=str(DIR / "evidence"
                                           / "pd_probe_6matrices_16samples.csv"))
    ap.add_argument("--snapshot", default=None,
                    help="local Qwen2.5-7B-Instruct snapshot; omit to skip the "
                         "layer-0 digit share and token decode")
    ap.add_argument("--control-draws", type=int, default=1000)
    ap.add_argument("--split-seed", type=int, default=20260821,
                    help="deliberately not the run's seed; see section J")
    ap.add_argument("--policy", default="strict", choices=("strict", "relaxed"))
    ap.add_argument("--min-cosine", type=float, default=1 - 1e-9,
                    help="floor for every committed vector's worst layer cosine; "
                         "float64 reassociation noise is ~2e-15")
    ap.add_argument("--max-norm-drift", type=float, default=1e-6,
                    help="allowed |1 - recomputed/committed| on the worst layer's "
                         "norm, which a cosine cannot see; the floor is the float32 "
                         "the committed vectors are STORED in, ~1e-8, not float64 "
                         "epsilon")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = (Path(args.repo_root) if args.repo_root else DIR.parents[1]).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(args.grid).resolve()))
    import crossgame_grid as grid  # noqa: E402
    import poles  # noqa: E402

    vec_dir = Path(args.vectors)
    dictator = torch.load(args.dictator, map_location="cpu").double()
    altruism_path = (Path(args.altruism) if args.altruism else
                     root / "persona_vectors/Qwen2.5-7B-Instruct"
                     / "altruism_response_avg_diff.pt")
    altruism = torch.load(altruism_path, map_location="cpu").double()

    # every resolved input, recorded: this is the file that decides whether the
    # committed vectors reproduce, and a verdict is only as readable as its inputs
    inputs = {
        "paths_relative_to": "the repository root, which is not recorded: it is "
                             "wherever this checkout happens to sit",
        "grid": repo_path(args.grid, root),
        "vectors": repo_path(vec_dir, root),
        "dictator": repo_path(args.dictator, root),
        "altruism": repo_path(altruism_path, root),
        "probe": repo_path(args.probe, root),
        "acts": acts_identity(args.acts),
        "snapshot": snapshot_identity(args.snapshot) if args.snapshot else None,
        "control_draws": args.control_draws,
        "split_seed": args.split_seed,
    }

    acts, labels = load(args.acts, grid, poles, args.policy)
    out = {"policy": args.policy, "inputs": inputs, "n_rows": acts.shape[0],
           "hidden_size": acts.shape[2], "n_hidden_states": acts.shape[1],
           "theoretical_random_cosine_sd": 1.0 / math.sqrt(acts.shape[2])}

    # A -- per-game vectors against the committed artifacts
    cells = {f: usable_cells(labels, poles, f) for f in FAMILY_ORDER}
    game_vecs = {f: cell_balanced(acts, cells[f]) for f in FAMILY_ORDER}
    out["A_per_game_vs_committed"] = {}
    for f in FAMILY_ORDER:
        path = vec_dir / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                          % (f, args.policy))
        entry = {"usable_cells": len(cells[f])}
        # "the report is written either way" has to hold for a MISSING file too,
        # or the gate's own error branch is unreachable from here
        if path.is_file():
            entry.update(committed_comparison(game_vecs[f],
                                              torch.load(path, map_location="cpu")))
        else:
            entry["error"] = "not committed: %s" % path.name
        out["A_per_game_vs_committed"][f] = entry

    # B -- the agreement matrix
    out["B_agreement"] = {"layer20": {}, "layer0": {}}
    for i, a in enumerate(FAMILY_ORDER):
        for b in FAMILY_ORDER[i + 1:]:
            c = cosines(game_vecs[a], game_vecs[b])
            out["B_agreement"]["layer20"]["%s|%s" % (a, b)] = c[20].item()
            out["B_agreement"]["layer0"]["%s|%s" % (a, b)] = c[0].item()

    # the pools, all nine weightings
    all_cells = {}
    for f in FAMILY_ORDER:
        all_cells.update({"%s/%s" % (f, c): v for c, v in cells[f].items()})
    pools = {"cell_balanced": cell_balanced(acts, all_cells)}
    for scheme in SCHEMES:
        pools[scheme] = weighted_pool(game_vecs, cells, scheme)
    out["E_weights"] = {scheme: across_game_weights(cells, scheme) for scheme in SCHEMES}

    # A2 -- the committed POOLED vectors against this recomputation. Section A
    # covers the six per-game files; without this the nine pooled files, which is
    # what most of README section 5 is read off, would carry no independent check.
    out["A2_pooled_vs_committed"] = {}
    for scheme in ALL_SCHEMES:
        path = vec_dir / ("decision_pooled_%s_response_avg_diff_%s.pt"
                          % (scheme, args.policy))
        if not path.is_file():
            out["A2_pooled_vs_committed"][scheme] = {
                "error": "not committed: %s" % path.name}
            continue
        out["A2_pooled_vs_committed"][scheme] = committed_comparison(
            pools[scheme], torch.load(path, map_location="cpu"))

    # C, D -- leave-one-game-out AUC
    out["C_logo_cell_balanced"] = {}
    out["D_logo_by_scheme"] = {scheme: {} for scheme in SCHEMES}
    out["K_logo_cos_vs_held_game"] = {scheme: {} for scheme in ALL_SCHEMES}
    logo = {scheme: {} for scheme in ALL_SCHEMES}
    for held in FAMILY_ORDER:
        alt, self_ = pole_positions(labels, poles, held)
        rest = {f: v for f, v in game_vecs.items() if f != held}
        rest_cells_by_family = {f: c for f, c in cells.items() if f != held}
        flat = {}
        for f, c in rest_cells_by_family.items():
            flat.update({"%s/%s" % (f, cell): v for cell, v in c.items()})
        flat_pool = cell_balanced(acts, flat)
        entry = summarise_auc(auc_by_layer(acts, flat_pool, alt, self_))
        entry.update({"n_altruistic": len(alt), "n_self_interested": len(self_)})
        out["C_logo_cell_balanced"][held] = entry
        logo["cell_balanced"][held] = flat_pool
        for scheme in SCHEMES:
            direction = weighted_pool(rest, rest_cells_by_family, scheme)
            logo[scheme][held] = direction
            out["D_logo_by_scheme"][scheme][held] = summarise_auc(
                auc_by_layer(acts, direction, alt, self_))
        for scheme in ALL_SCHEMES:
            c = cosines(logo[scheme][held], game_vecs[held])
            out["K_logo_cos_vs_held_game"][scheme][held] = {
                "layer0": c[0].item(), "layer20": c[20].item()}
    logo_geu = logo["game_equal_unit"]

    # E -- the pooled cosines
    out["E_pool_cosines"] = {"n_usable_cells_total": len(all_cells)}
    for name, pool in pools.items():
        vd, va = cosines(pool, dictator), cosines(pool, altruism)
        vc = cosines(pool, pools["cell_balanced"])
        out["E_pool_cosines"][name] = {
            "vs_dictator_layer0": vd[0].item(), "vs_dictator_layer20": vd[20].item(),
            "vs_dictator_min_layer10_27": min(vd[10:28].tolist()),
            "vs_altruism_layer20": va[20].item(),
            "vs_cell_balanced_pool_layer20": vc[20].item(),
            "norm_layer20": pool[20].norm().item(),
        }
    out["E_pool_cosines"]["dictator_vs_altruism_layer20"] = \
        cosines(dictator, altruism)[20].item()
    out["E_per_game_vs_dictator_layer20"] = {
        f: cosines(game_vecs[f], dictator)[20].item() for f in FAMILY_ORDER}
    out["E_per_game_vs_altruism_layer20"] = {
        f: cosines(game_vecs[f], altruism)[20].item() for f in FAMILY_ORDER}
    out["E_per_game_vs_pool_layer20"] = {
        s: {f: cosines(game_vecs[f], p)[20].item() for f in FAMILY_ORDER}
        for s, p in pools.items()}

    # F -- project the Dictator direction out of each leave-one-out pool
    out["F_dictator_projected_out"] = {}
    u = unit(dictator)
    for held in FAMILY_ORDER:
        alt, self_ = pole_positions(labels, poles, held)
        direction = logo_geu[held]
        residual = direction - (direction * u).sum(dim=1, keepdim=True) * u
        before = out["D_logo_by_scheme"]["game_equal_unit"][held]
        after = summarise_auc(auc_by_layer(acts, residual, alt, self_))
        out["F_dictator_projected_out"][held] = {
            "auc_layer20_before": before["layer20"],
            "auc_layer20_after": after["layer20"],
            "best_after": after["best"], "best_after_layer": after["best_layer"],
            "residual_norm_fraction_layer20":
                (residual[20].norm() / direction[20].norm()).item(),
        }

    # J -- split-half within cell, independent draw
    out["J_split_half_independent_seed"] = split_half_section(
        acts, cells, args.split_seed)

    # H -- census, effective n, focal points
    out["H_census"] = census_section(args.acts, grid, poles, args.policy)
    for f in FAMILY_ORDER:
        out["H_census"][f]["usable_cells"] = len(cells[f])
        out["H_census"][f]["effective_n"] = effective_n(cells[f])
        out["H_census"][f]["rows_in_usable_cells"] = {
            "altruistic": sum(len(a) for a, _s in cells[f].values()),
            "self_interested": sum(len(s) for _a, s in cells[f].values()),
        }

    # I -- the PD payoff-matrix probe
    if Path(args.probe).is_file():
        out["I_pd_probe"] = pd_probe_section(args.probe, grid, poles)

    # G -- the layer-0 digit share and token decode
    if args.snapshot:
        targets = {"pool_%s" % k: v for k, v in pools.items()}
        targets["archived_dictator"] = dictator
        targets["shipped_altruism"] = altruism
        for f in FAMILY_ORDER:
            targets["game_" + f] = game_vecs[f]
        out["G_layer0"] = digit_section(args.snapshot, args.acts, targets,
                                        args.control_draws)
    else:
        out["G_layer0"] = {"skipped": "no --snapshot given"}

    out["gate"] = reproduction_gate(out, args.min_cosine, args.max_norm_drift)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote %s" % repo_path(args.out, root))
    for failure in out["gate"]["failures"]:
        print("FAIL %s" % failure)
    if out["gate"]["failures"]:
        raise SystemExit("%d of the committed vectors do not reproduce from these "
                         "activations" % len(out["gate"]["failures"]))
    gate = out["gate"]
    print("gate: %d committed vectors reproduce; worst layer cosine %.15f (floor "
          "%.15f), worst per-layer norm drift %.3e (allowed %.1e)"
          % (gate["n_vectors_checked"], gate["worst_min_cos_over_layers"],
             args.min_cosine, gate["worst_norm_drift_over_layers"],
             args.max_norm_drift))


if __name__ == "__main__":
    main()
