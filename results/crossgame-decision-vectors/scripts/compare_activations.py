"""Compare two activation directories row by row. CPU only.

Two runs of `lab.extract` over the SAME rows, at the same revision, dtype,
attention kernel and batch size, should agree bit for bit: the forward pass is
deterministic and nothing here samples. So this reports what was actually
achieved rather than asserting it — the fraction of elements that are exactly
equal, the largest absolute and relative difference, and the smallest per-row
cosine, per pooling.

Rows are joined on `row_index`, the position of the row in its CSV, so the two
directories may be sharded differently and one may hold a subset of the other.
A row present in both but with a different `prompt_len` is a disagreement about
where the answer starts, which is reported separately and is never a tolerance.

Usage:

    python results/crossgame-decision-vectors/scripts/compare_activations.py \
        --a <dir of shard_*.pt> --b <dir of shard_*.pt> [--out report.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# the repo root is four levels up from this directory
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lab.extract import POOLINGS  # noqa: E402

# Shared box: grabbing all 256 cores makes every run slower, not faster.
torch.set_num_threads(int(os.environ.get("DM_THREADS", "16")))


def shard_paths(directory):
    paths = sorted(Path(directory).glob("shard_*.pt"))
    if not paths:
        raise SystemExit("no shard_*.pt in %s" % directory)
    return paths


def row_index_map(directory):
    """{row_index: (path, position)}; only the small index tensor is read."""
    out = {}
    for path in shard_paths(directory):
        payload = torch.load(path, map_location="cpu", mmap=True)
        for position, row_index in enumerate(payload["row_index"].tolist()):
            if row_index in out:
                raise SystemExit("%s: row %d appears twice" % (directory, row_index))
            out[row_index] = (path, position)
    return out


class Stat:
    """Running agreement between two tensors, accumulated a block at a time."""

    def __init__(self):
        self.elements = 0
        self.exactly_equal = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.min_cos = 1.0

    def add(self, a, b):
        a, b = a.double(), b.double()
        diff = (a - b).abs()
        self.elements += a.numel()
        self.exactly_equal += int((a == b).sum())
        self.max_abs = max(self.max_abs, float(diff.max()))
        scale = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
        self.max_rel = max(self.max_rel, float((diff / scale).max()))
        # per row AND per layer: a concatenated cosine is dominated by the
        # largest-norm layer, so one flipped layer would be invisible in it
        flat_a = a.reshape(-1, a.shape[-1])
        flat_b = b.reshape(-1, b.shape[-1])
        den = (flat_a.norm(dim=1) * flat_b.norm(dim=1)).clamp_min(1e-300)
        self.min_cos = min(self.min_cos, float(((flat_a * flat_b).sum(dim=1) / den).min()))

    def as_record(self):
        return {"elements": self.elements, "exactly_equal": self.exactly_equal,
                "exact_fraction": self.exactly_equal / self.elements if self.elements
                else None,
                "max_abs_diff": self.max_abs, "max_rel_diff": self.max_rel,
                "min_cosine_per_row_per_layer": self.min_cos}


def compare(dir_a, dir_b):
    index_a, index_b = row_index_map(dir_a), row_index_map(dir_b)
    shared = sorted(set(index_a) & set(index_b))
    if not shared:
        raise SystemExit("the two directories share no row_index")

    by_shard = {}
    for row_index in shared:
        by_shard.setdefault((index_a[row_index][0], index_b[row_index][0]),
                            []).append(row_index)

    stats = {name: Stat() for name in POOLINGS}
    prompt_len_disagreements = []
    for (path_a, path_b), rows in sorted(by_shard.items()):
        payload_a = torch.load(path_a, map_location="cpu")
        payload_b = torch.load(path_b, map_location="cpu")
        pos_a = torch.tensor([index_a[r][1] for r in rows], dtype=torch.long)
        pos_b = torch.tensor([index_b[r][1] for r in rows], dtype=torch.long)
        for field in ("prompt_len", "total_len"):
            left = payload_a[field].index_select(0, pos_a)
            right = payload_b[field].index_select(0, pos_b)
            for offset in (left != right).nonzero().flatten().tolist():
                prompt_len_disagreements.append(
                    {"row_index": rows[offset], "field": field,
                     "a": int(left[offset]), "b": int(right[offset])})
        for name in POOLINGS:
            stats[name].add(payload_a[name].index_select(0, pos_a),
                            payload_b[name].index_select(0, pos_b))
        del payload_a, payload_b

    return {
        "dir_a": str(Path(dir_a).resolve()), "dir_b": str(Path(dir_b).resolve()),
        "rows_a": len(index_a), "rows_b": len(index_b), "rows_compared": len(shared),
        "rows_only_in_a": sorted(set(index_a) - set(index_b)),
        "rows_only_in_b": sorted(set(index_b) - set(index_a)),
        "token_span_disagreements": prompt_len_disagreements,
        "poolings": {name: stat.as_record() for name, stat in stats.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="directory of shard_*.pt")
    parser.add_argument("--b", required=True, help="directory of shard_*.pt")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = compare(args.a, args.b)
    print("compared %d rows (%d in a, %d in b)"
          % (report["rows_compared"], report["rows_a"], report["rows_b"]))
    if report["token_span_disagreements"]:
        print("TOKEN SPAN DISAGREEMENTS: %d" % len(report["token_span_disagreements"]))
    for name, stat in report["poolings"].items():
        print("%-13s exact %d/%d (%.6f)  max|d| %.3e  max rel %.3e  min cos %.12f"
              % (name, stat["exactly_equal"], stat["elements"], stat["exact_fraction"],
                 stat["max_abs_diff"], stat["max_rel_diff"],
                 stat["min_cosine_per_row_per_layer"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
