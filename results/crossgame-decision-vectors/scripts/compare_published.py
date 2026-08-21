"""Diff a rebuild of this directory against what it published. CPU only.

Round 3 re-extracted the whole corpus with one shared extractor and rebuilt every
derived number from it. This is how the two are compared, mechanically rather
than by eye: every committed vector against its rebuilt counterpart, and every
committed analysis JSON against the file the rebuild wrote in its place, leaf by
leaf.

The headline comparability statistic is the per-layer cosine between an old and a
new vector. The JSON diff is the rest of it — the same schema on both sides, so a
recursive walk reaches every published scalar without a table of names to
maintain.

Differences that are about WHERE a run happened rather than WHAT it computed
(absolute paths, wall-clock seconds, shard counts, library versions) are listed
separately as provenance rather than mixed into the numeric deltas.

Usage:

    python results/crossgame-decision-vectors/scripts/compare_published.py \\
        --old <checkout of the published directory> --new <the rebuild's WORK dir> \\
        --out analysis/rebuild_deltas.json
"""

import argparse
import json
import os
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("DM_THREADS", "8")))

#: committed analysis file -> where the rebuild writes it
JSON_FILES = (
    ("analysis/crossgame_analysis.json", "crossgame/analysis.json"),
    ("analysis/crossgame_layer0_tokens.json", "crossgame/crossgame_layer0_tokens.json"),
    ("analysis/pooled_build.json", "pooled/build.json"),
    ("analysis/pooled_analysis.json", "pooled/analysis.json"),
    ("analysis/pooled_addendum.json", "pooled/addendum.json"),
    ("analysis/pooled_reliability.json", "pooled/reliability.json"),
    ("analysis/pooled_layer0_tokens.json", "pooled/layer0_tokens.json"),
)

#: keys whose value says where a run happened, not what it computed
PROVENANCE_KEYS = frozenset((
    "rows_csv", "acts_dir", "dictator_vector_path", "dictator_vector",
    "their_altruism_vector", "committed_vector", "seconds", "n_shards",
    "shard_rows", "resumed_from_shards", "poolings", "extractor", "device",
    "torch_version", "transformers_version",
))

#: below this the relative delta of a near-zero cosine is noise, not a movement
REL_FLOOR = 1e-3

#: the layer every claim in README.md is read at
LAYER = 20


def walk(old, new, path, numeric, provenance, structural):
    """Recursively pair two JSON trees, sorting every leaf into one of three lists."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            here = "%s.%s" % (path, key) if path else str(key)
            if key in PROVENANCE_KEYS:
                if old.get(key) != new.get(key):
                    provenance.append({"path": here, "old": old.get(key),
                                       "new": new.get(key)})
                continue
            if key not in old or key not in new:
                structural.append({"path": here,
                                   "only_in": "old" if key in old else "new"})
                continue
            walk(old[key], new[key], here, numeric, provenance, structural)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            structural.append({"path": path, "old_len": len(old), "new_len": len(new)})
            return
        for index, (a, b) in enumerate(zip(old, new)):
            walk(a, b, "%s[%d]" % (path, index), numeric, provenance, structural)
        return
    if isinstance(old, bool) or isinstance(new, bool) or old is None or new is None:
        if old != new:
            structural.append({"path": path, "old": old, "new": new})
        return
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        delta = abs(float(new) - float(old))
        scale = max(abs(float(old)), abs(float(new)))
        numeric.append({"path": path, "old": old, "new": new, "abs_delta": delta,
                        "rel_delta": delta / scale if scale > REL_FLOOR else None})
        return
    if old != new:
        structural.append({"path": path, "old": old, "new": new})


def diff_json(old_path, new_path, top):
    numeric, provenance, structural = [], [], []
    walk(json.loads(Path(old_path).read_text(encoding="utf-8")),
         json.loads(Path(new_path).read_text(encoding="utf-8")),
         "", numeric, provenance, structural)
    moved = sorted((n for n in numeric if n["abs_delta"] > 0),
                   key=lambda n: -n["abs_delta"])
    # integer-valued leaves are counts, not measurements: a row count that moves by 1
    # would otherwise dominate every cosine delta the comparison exists to report
    measured = [n for n in moved if not (float(n["old"]).is_integer()
                                         and float(n["new"]).is_integer())]
    return {
        "old": str(old_path), "new": str(new_path),
        "n_numeric_leaves": len(numeric),
        "n_numeric_leaves_identical": len(numeric) - len(moved),
        "max_abs_delta": measured[0]["abs_delta"] if measured else None,
        "max_abs_delta_path": measured[0]["path"] if measured else None,
        "largest_moves": measured[:top],
        "counts_that_moved": [n for n in moved if n not in measured][:top],
        "provenance_differences": provenance,
        "structural_differences": structural[:top],
        "n_structural_differences": len(structural),
    }


def diff_vector(old_path, new_path):
    old = torch.load(old_path, map_location="cpu").double()
    new = torch.load(new_path, map_location="cpu").double()
    if old.shape != new.shape:
        return {"error": "shape %s != %s" % (tuple(old.shape), tuple(new.shape))}
    if old.shape[0] <= LAYER:
        return {"error": "only %d layers; layer %d is what the claims use"
                         % (old.shape[0], LAYER)}
    cos = ((old * new).sum(dim=1)
           / (old.norm(dim=1) * new.norm(dim=1)).clamp_min(1e-300))
    return {
        "bit_identical": bool(torch.equal(old.float(), new.float())),
        "cos_layer0": cos[0].item(), "cos_layer20": cos[LAYER].item(),
        "min_cos_over_layers": cos.min().item(),
        "max_abs_elementwise_diff": float((old - new).abs().max()),
        "norm_layer20_old": old[LAYER].norm().item(),
        "norm_layer20_new": new[LAYER].norm().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True,
                        help="the published directory as committed")
    parser.add_argument("--new", required=True, help="the rebuild's WORK directory")
    parser.add_argument("--new-vectors", default=None,
                        help="default: <new>/pooled/vectors")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    old_root, new_root = Path(args.old), Path(args.new)
    new_vectors = Path(args.new_vectors) if args.new_vectors else new_root / "pooled" / "vectors"

    report = {"old": str(old_root.resolve()), "new": str(new_root.resolve()),
              "vectors": {}, "json": {}, "missing": []}
    committed = sorted((old_root / "vectors").glob("*.pt"))
    if not committed:
        raise SystemExit("no vectors under %s/vectors; --old must point at the "
                         "published directory" % old_root)
    for path in committed:
        rebuilt = new_vectors / path.name
        if not rebuilt.is_file():
            report["missing"].append("vectors/%s" % path.name)
            continue
        report["vectors"][path.name] = diff_vector(path, rebuilt)
    for committed, produced in JSON_FILES:
        old_path, new_path = old_root / committed, new_root / produced
        if not old_path.is_file() or not new_path.is_file():
            report["missing"].append(committed if not old_path.is_file() else produced)
            continue
        report["json"][committed] = diff_json(old_path, new_path, args.top)

    vectors = list(report["vectors"].values())
    comparable = [v for v in vectors if "min_cos_over_layers" in v]
    deltas = [(name, f["max_abs_delta"]) for name, f in report["json"].items()
              if f["max_abs_delta"] is not None]
    worst_name, worst_delta = max(deltas, key=lambda kv: kv[1], default=(None, None))
    report["summary"] = {
        "n_vectors_compared": len(vectors),
        "n_vectors_bit_identical": sum(1 for v in vectors if v.get("bit_identical")),
        "n_vectors_uncomparable": len(vectors) - len(comparable),
        "worst_vector_cosine": min((v["min_cos_over_layers"] for v in comparable),
                                   default=None),
        "n_json_files_compared": len(report["json"]),
        "n_structural_differences": sum(f["n_structural_differences"]
                                        for f in report["json"].values()),
        "n_missing": len(report["missing"]),
        "largest_numeric_delta": worst_delta,
        "largest_numeric_delta_path": ("%s :: %s"
                                       % (worst_name,
                                          report["json"][worst_name]["max_abs_delta_path"])
                                       if worst_name else None),
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print("vectors: %d compared, %d bit-identical, %d uncomparable, worst cosine %s"
          % (summary["n_vectors_compared"], summary["n_vectors_bit_identical"],
             summary["n_vectors_uncomparable"], summary["worst_vector_cosine"]))
    for name, entry in sorted(report["vectors"].items()):
        if "error" in entry:
            print("  UNCOMPARABLE %s: %s" % (name, entry["error"]))
    for name, entry in sorted(report["json"].items()):
        print("%-42s %6d leaves, %6d identical, %3d structural, max |delta| %s at %s"
              % (name, entry["n_numeric_leaves"], entry["n_numeric_leaves_identical"],
                 entry["n_structural_differences"],
                 "none" if entry["max_abs_delta"] is None
                 else "%.3e" % entry["max_abs_delta"], entry["max_abs_delta_path"]))
    if report["missing"]:
        print("MISSING: %s" % ", ".join(report["missing"]))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
