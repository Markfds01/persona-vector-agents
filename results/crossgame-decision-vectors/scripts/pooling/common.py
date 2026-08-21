"""Shared machinery for the pooled decision vector. CPU only.

A cell is one (family, stake, wording); a game's own vector is the unweighted
mean over its USABLE cells (both poles populated) of alt-mean minus self-mean.
This module then combines the six game vectors under one explicit, stated
weighting - nine of them, including the two that balance the ANSWER-FORMAT axis
rather than the game axis, which is the axis the confound actually lives on.

Nothing here loads a model or touches a GPU: it reads the activation shards
`audit.extract` wrote and does linear algebra on the CPU.

Two paths have to come from the caller, because one of them is 5.3 GB and is not
committed:

  DM_ACTS        the activation root: <family>/{rows.csv, shard_*.pt, meta.json}
  DM_POOLED_OUT  where this stage writes its vectors, JSON and logs

Everything else is derived from this file's own location, so the stage runs out
of a checkout with no absolute path in it.
"""

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

# This box is shared and its load average runs into the hundreds. Grabbing all
# 256 cores makes every run slower, not faster, so cap the thread pool unless the
# caller says otherwise.
torch.set_num_threads(int(os.environ.get("DM_THREADS", "16")))

HERE = Path(__file__).resolve().parent
#: this results directory, and the repository root three levels above it
RESULTS = HERE.parents[1]
REPO = RESULTS.parents[1]
#: where crossgame_grid.py and poles.py live - the grid and pole DEFINITIONS
GRID = HERE.parent / "extraction"


def _required(name, what):
    value = os.environ.get(name)
    if not value:
        raise SystemExit("set %s to %s" % (name, what))
    return Path(value)


ACTS = _required("DM_ACTS", "the activation root written by audit.extract "
                            "(<family>/shard_*.pt and rows.csv)")
OUT = _required("DM_POOLED_OUT", "the directory this stage writes into")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(GRID))

import crossgame_grid  # noqa: E402
import poles  # noqa: E402

FAMILIES = list(crossgame_grid.FAMILIES)

#: the Dictator-only vector section 6 projects out, fit on a separate earlier run
DICTATOR_VECTOR = Path(os.environ.get(
    "DM_DICTATOR_VECTOR",
    REPO / "results/dictator-decision-vector/vectors"
         / "decision_response_avg_diff_cellbalanced.pt"))
#: the repo's shipped trait vector, the negative control
ALTRUISM_VECTOR = Path(os.environ.get(
    "DM_ALTRUISM_VECTOR",
    REPO / "persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt"))


# --- loading -----------------------------------------------------------------

def cache_fingerprint():
    """What the cached labels depend on, so a stale cache cannot be reused silently.

    `labels.json` bakes in the output of `poles.py` and `crossgame_grid.py`. Edit
    a pole rule or a stake and the shards are unchanged, so a cache keyed only on
    their existence would feed the old labels to every stage while the report
    claims the new definitions.
    """
    parts = [ACTS.resolve().as_posix()]
    for module in (poles, crossgame_grid):
        source = Path(module.__file__).read_bytes()
        parts.append("%s=%s" % (Path(module.__file__).name,
                                hashlib.sha256(source).hexdigest()))
    for family in FAMILIES:
        game_dir = ACTS / family
        shards = sorted(game_dir.glob("shard_*.pt"))
        parts.append("%s:%s" % (family, ",".join(
            "%s:%d" % (path.name, path.stat().st_size) for path in shards)))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load_response_avg(cache_dir=OUT / "cache"):
    """(N, 29, 3584) float32 response_avg for all six games + aligned labels.

    Cached, because reading 5.3 GB of shards to keep 1.9 GB of it is the only
    slow step in the whole analysis and a session limit must not cost it twice.
    The cache is keyed on the pole and grid definitions and on the shards, so it
    is rebuilt rather than silently reused when any of them changes.
    """
    cache_dir = Path(cache_dir)
    acts_path = cache_dir / "response_avg.pt"
    labels_path = cache_dir / "labels.json"
    stamp_path = cache_dir / "fingerprint.txt"
    fingerprint = cache_fingerprint()
    if acts_path.exists() and labels_path.exists() and stamp_path.exists():
        if stamp_path.read_text(encoding="utf-8").strip() == fingerprint:
            return (torch.load(acts_path, map_location="cpu"),
                    json.loads(labels_path.read_text()))
        print("cache fingerprint changed; rebuilding %s" % cache_dir, flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    labels, blocks, offset = [], [], 0
    for family in FAMILIES:
        game_dir = ACTS / family
        with open(game_dir / "rows.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        shards = sorted(game_dir.glob("shard_*.pt"))
        if not shards:
            raise SystemExit("no shards for %s" % family)
        index, chunks = [], []
        for path in shards:
            payload = torch.load(path, map_location="cpu")
            index.extend(payload["row_index"].tolist())
            chunks.append(payload["response_avg"].clone())
            del payload
        block = torch.cat(chunks, dim=0)
        del chunks
        if len(index) != block.shape[0]:
            raise SystemExit("%s: shard row count disagrees with activations" % family)
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
                "cell": game.cell,
                "stake": game.stake,
                "wording": game.wording,
                "scorer": game.scorer,
                "value": value,
                "tag": row["tag"],
                "tag_class": poles.tag_class(row["tag"], game.scorer),
                "pole_strict": poles.classify(value, game, "strict"),
                "pole_relaxed": poles.classify(value, game, "relaxed"),
            })
        blocks.append(block)
        offset += block.shape[0]
        print("loaded %s: %d rows" % (family, block.shape[0]), flush=True)

    acts = torch.cat(blocks, dim=0)
    del blocks
    if acts.shape[0] != len(labels):
        raise SystemExit("activation count disagrees with label count")
    torch.save(acts, acts_path)
    labels_path.write_text(json.dumps(labels))
    # written last: a fingerprint present means the two files beside it are complete
    stamp_path.write_text(fingerprint, encoding="utf-8")
    return acts, labels


# --- poles and cells ----------------------------------------------------------

def cell_index(labels, policy, allow_derived=False):
    """{family: {cell: (alt_positions, self_positions)}} for USABLE cells only.

    A cell is usable when it holds rows in both poles; only such a cell can
    contribute a within-cell difference. Unusable cells are counted by the
    caller, never silently dropped.
    """
    ok = ("direct", "derived") if allow_derived else ("direct",)
    key = "pole_" + policy
    seen = {}
    for item in labels:
        if item["tag_class"] not in ok:
            continue
        pole = item[key]
        if pole == poles.MIDDLE:
            continue
        bucket = seen.setdefault(item["family"], {}).setdefault(item["cell"], ([], []))
        bucket[0 if pole == poles.ALT else 1].append(item["position"])
    usable = {}
    for family, cells in seen.items():
        keep = {c: v for c, v in sorted(cells.items()) if v[0] and v[1]}
        if keep:
            usable[family] = keep
    return usable, seen


def pole_rows(labels, policy, family=None, allow_derived=False):
    """(alt_positions, self_positions) over ALL rows of a game, cells ignored."""
    ok = ("direct", "derived") if allow_derived else ("direct",)
    key = "pole_" + policy
    alt, self_ = [], []
    for item in labels:
        if family is not None and item["family"] != family:
            continue
        if item["tag_class"] not in ok:
            continue
        if item[key] == poles.ALT:
            alt.append(item["position"])
        elif item[key] == poles.SELF:
            self_.append(item["position"])
    return alt, self_


# --- vectors ------------------------------------------------------------------

def mean_of(acts, positions):
    """Mean activation over a set of row positions, in float64."""
    idx = torch.tensor(positions, dtype=torch.long)
    return acts.index_select(0, idx).double().mean(dim=0)


def cell_vectors(acts, cells):
    """{cell: (29, 3584) alt-mean minus self-mean} for one game's usable cells."""
    return {c: mean_of(acts, a) - mean_of(acts, s) for c, (a, s) in cells.items()}


def game_vector(acts, cells):
    """One game's CELL-BALANCED vector: the unweighted mean over its usable cells."""
    stack = torch.stack([v for _c, v in sorted(cell_vectors(acts, cells).items())])
    return stack.mean(dim=0)


def unit(v):
    """Row-normalise a (L, H) tensor per layer; a zero row stays zero."""
    n = v.norm(dim=1, keepdim=True)
    return torch.where(n > 0, v / n.clamp_min(1e-12), torch.zeros_like(v))


def effective_n(cells):
    """Effective sample size behind a game's cell-balanced vector.

    The vector is (1/C) * sum_c (mean_alt_c - mean_self_c). Under a common
    per-row variance s^2 the variance of that estimate is
    (s^2 / C^2) * sum_c (1/a_c + 1/s_c), so the precision is proportional to
    C^2 / sum_c (1/a_c + 1/s_c) and that ratio is the number of rows a single
    unpartitioned two-group difference would need to be this precise. It is the
    honest denominator behind a game's contribution: a game weighted 1/6 of the
    pool but resting on 30 effective rows is carrying its sixth on thin evidence.
    """
    c = len(cells)
    inv = sum(1.0 / len(a) + 1.0 / len(s) for a, s in cells.values())
    return (c * c) / inv if inv > 0 else 0.0


#: The four games whose answer is a dollar amount, the one answered in fish, and
#: the one answered with the words Cooperate/Defect. This is the ANSWER-FORMAT
#: partition, which is the axis the confound actually lives on.
DOLLAR_FAMILIES = ("dictator", "trust", "ultimatum", "apology")
FORMAT_FAMILIES = {
    "dollar": DOLLAR_FAMILIES,
    "fish": ("overfishing",),
    "binary": ("prisoners_dilemma",),
}
NON_DOLLAR_FAMILIES = ("overfishing", "prisoners_dilemma")

WEIGHTINGS = ("game_equal_unit", "game_equal_raw",
              "game_precision_raw", "game_precision_unit", "cell_balanced",
              "family_balanced_unit", "family_balanced_raw",
              "non_dollar_unit", "non_dollar_raw")

#: scheme -> (weight kind, whether each game is unit-normalised per layer first)
SCHEME_SPEC = {
    "game_equal_unit": ("equal", True),
    "game_equal_raw": ("equal", False),
    "game_precision_raw": ("precision", False),
    "game_precision_unit": ("precision", True),
    "cell_balanced": ("cells", False),
    "family_balanced_unit": ("family", True),
    "family_balanced_raw": ("family", False),
    "non_dollar_unit": ("non_dollar", True),
    "non_dollar_raw": ("non_dollar", False),
}


def weights_for(kind, cells_by_family):
    """Unnormalised across-game weights. The caller normalises to sum 1.

    equal       one per game. Note this still puts 4/6 of the weight on the
                dollar-format answer, because four of the six games are dollar
                games - balancing the game axis is not balancing the format axis.
    cells       proportional to a game's usable-cell count (the predecessor's).
    precision   proportional to effective_n, i.e. inverse-variance.
    family      one third to each ANSWER FORMAT, split evenly inside it: 1/12 to
                each dollar game, 1/3 to overfishing, 1/3 to the Prisoner's
                Dilemma. The confound being chased is the answer surface, so this
                is the axis to equalise. Chosen AFTER seeing the agreement matrix.
    non_dollar  the dollar games dropped entirely; overfishing and PD half each.
                The most the dataset allows short of having no dollar answers at
                all, and correspondingly the thinnest.
    """
    families = sorted(cells_by_family)
    if kind == "equal":
        return {f: 1.0 for f in families}
    if kind == "cells":
        return {f: float(len(cells_by_family[f])) for f in families}
    if kind == "precision":
        return {f: effective_n(cells_by_family[f]) for f in families}
    if kind == "family":
        present = {name: [f for f in group if f in cells_by_family]
                   for name, group in FORMAT_FAMILIES.items()}
        present = {name: group for name, group in present.items() if group}
        share = 1.0 / len(present)
        return {f: share / len(group)
                for group in present.values() for f in group}
    if kind == "non_dollar":
        keep = [f for f in NON_DOLLAR_FAMILIES if f in cells_by_family]
        if not keep:
            raise ValueError("non_dollar weighting needs at least one non-dollar game")
        return {f: (1.0 / len(keep) if f in keep else 0.0) for f in families}
    raise ValueError("unknown weight kind %r" % (kind,))


def combine(game_vecs, cells_by_family, scheme, cell_vecs_by_family=None):
    """Combine per-game cell-balanced vectors under one explicit weighting.

    `cell_balanced` is the one scheme that is not a weighted sum of the per-game
    vectors - it is the flat mean over every usable cell of every game - so it is
    handled first and separately. Everything else is
    `sum_g w_g * (v_g or v_g/|v_g|) / sum_g w_g`, with the weights from
    `weights_for` and the per-layer unit option from SCHEME_SPEC.
    """
    families = sorted(game_vecs)
    if scheme == "cell_balanced":
        if cell_vecs_by_family is None:
            raise ValueError("cell_balanced needs the per-cell vectors")
        stack = torch.stack([v for f in families
                             for _c, v in sorted(cell_vecs_by_family[f].items())])
        return stack.mean(dim=0)
    kind, unit_first = SCHEME_SPEC[scheme]
    weights = weights_for(kind, {f: cells_by_family[f] for f in families})
    w = torch.tensor([weights[f] for f in families], dtype=torch.double)
    if w.sum() <= 0:
        raise ValueError("%s: no game carries any weight here" % scheme)
    w = w / w.sum()
    stack = torch.stack([unit(game_vecs[f]) if unit_first else game_vecs[f]
                         for f in families])
    return (stack * w.view(-1, 1, 1)).sum(dim=0)


# --- measurement ---------------------------------------------------------------

def cosines(a, b):
    a, b = a.double(), b.double()
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(den > 0, num / den, torch.full_like(num, float("nan")))


def auc(pos, neg):
    """Mann-Whitney AUC with tie correction: P(alt projection > self projection)."""
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


def project(acts, positions, direction):
    idx = torch.tensor(positions, dtype=torch.long)
    x = acts.index_select(0, idx).double()
    u = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (x * u.unsqueeze(0)).sum(dim=2)


def auc_by_layer(acts, direction, alt, self_):
    pa, ps = project(acts, alt, direction), project(acts, self_, direction)
    out = []
    for l in range(pa.shape[1]):
        a, s = pa[:, l], ps[:, l]
        pooled = math.sqrt((a.var(unbiased=True).item()
                            + s.var(unbiased=True).item()) / 2)
        out.append({"layer": l, "auc": auc(a, s),
                    "cohens_d": ((a.mean().item() - s.mean().item()) / pooled)
                    if pooled > 0 else float("nan")})
    return out


def summarize(values):
    v = values[torch.isfinite(values)]
    if v.numel() == 0:
        return {"n": 0}
    return {"n": int(v.numel()), "mean": v.mean().item(),
            "sd": v.std(unbiased=True).item(),
            "min": v.min().item(), "max": v.max().item(),
            "p2.5": v.quantile(0.025).item(), "p97.5": v.quantile(0.975).item()}
