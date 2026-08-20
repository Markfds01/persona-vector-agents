"""Independent recomputation of every load-bearing number in README.md.

Written during packaging, not part of either run. It exists so the README's
tables can be checked by `regenerate and diff` rather than taken on trust, and so
that the check does not go through either run's analysis code: it imports only
`crossgame_grid` and `poles`, which are the grid and pole DEFINITIONS, and does
its own linear algebra.

What it recomputes, all from the activation shards:

  A  the six per-game cell-balanced vectors, against the committed .pt files
  B  the 6x6 agreement matrix at layer 20, and at layer 0
  C  leave-one-game-out AUC under the cell-balanced pool
  D  leave-one-game-out AUC under game_equal_unit
  E  pooled cosines against the archived Dictator vector and the shipped altruism
     vector, per scheme
  F  the Dictator-projected-out collapse
  G  the layer-0 digit-span share, its control bands, and the token decode
  H  the pole census, tag counts, parse coverage, focal shares, effective n
  I  the PD wording breakdown and the payoff-matrix probe

CPU only. No GPU, no model loaded, no forward pass: G reads the embedding matrix
off the safetensors shard because layer 0 of `response_avg` IS an average of rows
of that matrix.

Usage, from a checkout of this repository:

    python results/crossgame-decision-vectors/scripts/verify_committed.py \
        --acts   <dir holding acts/<game>/shard_*.pt and rows.csv> \
        --grid   results/crossgame-decision-vectors/scripts/extraction \
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
    axis = big - emb[tok.encode("0")[0]].double()
    axis = axis / axis.norm()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True,
                    help="directory holding <game>/shard_*.pt and <game>/rows.csv")
    ap.add_argument("--grid", default=str(HERE / "extraction"),
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
    ap.add_argument("--policy", default="strict", choices=("strict", "relaxed"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.repo_root) if args.repo_root else DIR.parents[1]
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

    acts, labels = load(args.acts, grid, poles, args.policy)
    out = {"policy": args.policy, "n_rows": acts.shape[0],
           "hidden_size": acts.shape[2], "n_hidden_states": acts.shape[1],
           "theoretical_random_cosine_sd": 1.0 / math.sqrt(acts.shape[2])}

    # A -- per-game vectors against the committed artifacts
    cells = {f: usable_cells(labels, poles, f) for f in FAMILY_ORDER}
    game_vecs = {f: cell_balanced(acts, cells[f]) for f in FAMILY_ORDER}
    out["A_per_game_vs_committed"] = {}
    for f in FAMILY_ORDER:
        path = vec_dir / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                          % (f, args.policy))
        c = cosines(game_vecs[f], torch.load(path, map_location="cpu"))
        out["A_per_game_vs_committed"][f] = {
            "usable_cells": len(cells[f]),
            "cos_layer20": c[20].item(),
            "min_cos_over_layers": c.min().item(),
            "norm_layer20": game_vecs[f][20].norm().item(),
        }

    # B -- the agreement matrix
    out["B_agreement"] = {"layer20": {}, "layer0": {}}
    for i, a in enumerate(FAMILY_ORDER):
        for b in FAMILY_ORDER[i + 1:]:
            c = cosines(game_vecs[a], game_vecs[b])
            out["B_agreement"]["layer20"]["%s|%s" % (a, b)] = c[20].item()
            out["B_agreement"]["layer0"]["%s|%s" % (a, b)] = c[0].item()

    # the pools
    all_cells = {}
    for f in FAMILY_ORDER:
        all_cells.update({"%s/%s" % (f, c): v for c, v in cells[f].items()})
    pools = {
        "cell_balanced": cell_balanced(acts, all_cells),
        "game_equal_unit": torch.stack([unit(game_vecs[f])
                                        for f in FAMILY_ORDER]).mean(dim=0),
        "game_equal_raw": torch.stack([game_vecs[f]
                                       for f in FAMILY_ORDER]).mean(dim=0),
    }
    w = torch.tensor([effective_n(cells[f]) for f in sorted(FAMILY_ORDER)],
                     dtype=torch.double)
    w = w / w.sum()
    pools["game_precision_raw"] = (
        torch.stack([game_vecs[f] for f in sorted(FAMILY_ORDER)])
        * w.view(-1, 1, 1)).sum(dim=0)
    pools["game_precision_unit"] = (
        torch.stack([unit(game_vecs[f]) for f in sorted(FAMILY_ORDER)])
        * w.view(-1, 1, 1)).sum(dim=0)

    # C, D -- leave-one-game-out AUC
    out["C_logo_cell_balanced"] = {}
    out["D_logo_game_equal_unit"] = {}
    logo_geu = {}
    for held in FAMILY_ORDER:
        alt, self_ = pole_positions(labels, poles, held)
        rest_cells = {}
        for f in FAMILY_ORDER:
            if f != held:
                rest_cells.update({"%s/%s" % (f, c): v for c, v in cells[f].items()})
        entry = summarise_auc(auc_by_layer(acts, cell_balanced(acts, rest_cells),
                                           alt, self_))
        entry.update({"n_altruistic": len(alt), "n_self_interested": len(self_)})
        out["C_logo_cell_balanced"][held] = entry

        direction = torch.stack([unit(game_vecs[f]) for f in FAMILY_ORDER
                                 if f != held]).mean(dim=0)
        logo_geu[held] = direction
        out["D_logo_game_equal_unit"][held] = summarise_auc(
            auc_by_layer(acts, direction, alt, self_))

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
        before = out["D_logo_game_equal_unit"][held]
        after = summarise_auc(auc_by_layer(acts, residual, alt, self_))
        out["F_dictator_projected_out"][held] = {
            "auc_layer20_before": before["layer20"],
            "auc_layer20_after": after["layer20"],
            "best_after": after["best"], "best_after_layer": after["best_layer"],
            "residual_norm_fraction_layer20":
                (residual[20].norm() / direction[20].norm()).item(),
        }

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

    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
