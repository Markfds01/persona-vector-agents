"""Build the shuffled-label null vector from the SAME activations as the real one.

The real vector under test is `decision_response_avg_diff_cellbalanced.pt`: the
mean over prompt cells of (mean altruistic activation - mean self-interested
activation) within that cell. This script rebuilds that construction exactly and
then rebuilds it again with the pole labels PERMUTED WITHIN EACH CELL, sizes
preserved.

Within-cell permutation is the matched null for a cell-balanced vector: it holds
the prompt composition and every per-cell pole count fixed and destroys only the
association between a row's activation and the decision it recorded. A global
permutation would additionally reshuffle composition across cells, which the
cell-balanced construction is specifically built to cancel.

The rebuild of the REAL vector is checked against the shipped file before the
null is trusted: if this script cannot reproduce the artifact, its null is not a
null for that artifact.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

DIRECT_TAGS = ("a2_anchor", "a2_near", "verb_obj", "answer_is", "bare", "bare_int")


def sha16(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_response_avg(act_dir):
    shards = sorted(Path(act_dir).glob("shard_*.pt"))
    if not shards:
        raise SystemExit("no activation shards in %s" % act_dir)
    index, acts = [], []
    for path in shards:
        payload = torch.load(path, map_location="cpu")
        index.extend(payload["row_index"].tolist())
        acts.append(payload["response_avg"])
    return index, torch.cat(acts, dim=0)


def label(rows, index):
    out = []
    for position, row_index in enumerate(index):
        row = rows[row_index]
        wording = row["game_id"].split("/")[1]
        endowment = int(row["game_id"].split("/e")[1])
        value = float(row["value"])
        out.append({"position": position, "cell": "%s/e%d" % (wording, endowment),
                    "fraction": value / endowment, "tag": row["tag"]})
    return out


def poles(labels):
    alt, self_, middle, excluded = [], [], [], []
    for item in labels:
        if item["tag"] not in DIRECT_TAGS:
            excluded.append(item)
        elif item["fraction"] == 0.0:
            self_.append(item)
        elif item["fraction"] >= 0.5:
            alt.append(item)
        else:
            middle.append(item)
    return alt, self_, middle, excluded


def mean_of(acts, positions):
    idx = torch.tensor(positions, dtype=torch.long)
    return acts.index_select(0, idx).double().mean(dim=0)


def balanced(acts, cells):
    """cells: list of (cell, alt_positions, self_positions), only both-pole cells."""
    stack = torch.stack([mean_of(acts, a) - mean_of(acts, s) for _c, a, s in cells])
    return stack.mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--reference", required=True, help="the shipped cell-balanced vector")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    with open(args.rows, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index, acts = load_response_avg(args.acts)
    if len(index) != acts.shape[0]:
        raise SystemExit("shard row count disagrees with activation count")
    labels = label(rows, index)
    alt, self_, middle, excluded = poles(labels)

    by_cell = {}
    for item in alt:
        by_cell.setdefault(item["cell"], ([], []))[0].append(item["position"])
    for item in self_:
        by_cell.setdefault(item["cell"], ([], []))[1].append(item["position"])
    usable = [(cell, a, s) for cell, (a, s) in sorted(by_cell.items()) if a and s]

    real = balanced(acts, usable)
    reference = torch.load(args.reference, map_location="cpu").double()
    if reference.shape != real.shape:
        raise SystemExit("reference shape %s != rebuilt %s" % (tuple(reference.shape),
                                                               tuple(real.shape)))
    delta = (reference - real).norm(dim=1)
    rel = (delta / reference.norm(dim=1).clamp_min(1e-12)).max().item()
    if rel > 1e-6:
        raise SystemExit("rebuild does not reproduce the shipped vector: max relative "
                         "per-layer deviation %.3g" % rel)

    generator = torch.Generator().manual_seed(args.seed)
    shuffled_cells = []
    for cell, a, s in usable:
        pool = a + s
        perm = torch.randperm(len(pool), generator=generator).tolist()
        picked = [pool[i] for i in perm]
        shuffled_cells.append((cell, picked[:len(a)], picked[len(a):]))
    fake = balanced(acts, shuffled_cells)

    torch.save(fake.float(), args.out)

    layer = args.layer
    def cos(x, y):
        return float((x[layer] * y[layer]).sum()
                     / (x[layer].norm() * y[layer].norm()))

    report = {
        "seed": args.seed,
        "layer": layer,
        "n_altruistic": len(alt), "n_self_interested": len(self_),
        "n_middle_discarded": len(middle), "n_tag_excluded": len(excluded),
        "n_cells_total": len(by_cell), "n_cells_usable": len(usable),
        "rebuild_max_relative_layer_deviation_vs_shipped": rel,
        "real_norm_layer": float(real[layer].norm()),
        "reference_norm_layer": float(reference[layer].norm()),
        "null_norm_layer": float(fake[layer].norm()),
        "cos_null_vs_real_layer": cos(fake, real),
        "null_norms_by_layer": fake.norm(dim=1).tolist(),
        "real_norms_by_layer": real.norm(dim=1).tolist(),
        "out": str(Path(args.out).resolve()),
        "out_sha256_16": None,
        "reference_sha256_16": sha16(args.reference),
    }
    report["out_sha256_16"] = sha16(args.out)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if not k.endswith("_by_layer")}, indent=2))


if __name__ == "__main__":
    main()
