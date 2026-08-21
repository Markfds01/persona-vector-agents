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
import math
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


class Leaves:
    """Where each paired leaf lands.

    `non_finite` is its own bucket rather than part of `numeric`: a NaN satisfies
    no comparison, so counting one as an identical leaf claims a check that did
    not happen.
    """

    def __init__(self):
        self.numeric = []
        self.provenance = []
        self.structural = []
        self.non_finite = []


def walk(old, new, path, leaves):
    """Recursively pair two JSON trees, sorting every leaf into one of four lists."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            here = "%s.%s" % (path, key) if path else str(key)
            if key in PROVENANCE_KEYS:
                if old.get(key) != new.get(key):
                    leaves.provenance.append({"path": here, "old": old.get(key),
                                              "new": new.get(key)})
                continue
            if key not in old or key not in new:
                leaves.structural.append({"path": here,
                                          "only_in": "old" if key in old else "new"})
                continue
            walk(old[key], new[key], here, leaves)
        return
    if isinstance(old, list) and isinstance(new, list):
        # a truncated top-N list (16 entries against 20) is still aligned over the
        # entries both sides have; returning here left the whole subtree with zero
        # numeric leaves and the file "compared" without comparing anything
        if len(old) != len(new):
            leaves.structural.append({"path": path, "old_len": len(old),
                                      "new_len": len(new)})
        for index, (a, b) in enumerate(zip(old, new)):
            walk(a, b, "%s[%d]" % (path, index), leaves)
        return
    if isinstance(old, bool) or isinstance(new, bool) or old is None or new is None:
        if old != new:
            leaves.structural.append({"path": path, "old": old, "new": new})
        return
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        a, b = float(old), float(new)
        # NaN is not a value: `nan > 0` is False, so a NaN that appeared, vanished
        # or stayed all came out the far side counted as an IDENTICAL leaf and
        # excluded from `max_abs_delta`. A NaN on one side only is a real change;
        # one on both sides is a leaf nobody can verify in either direction.
        if not (math.isfinite(a) and math.isfinite(b)):
            same = a == b or (math.isnan(a) and math.isnan(b))
            bucket = leaves.non_finite if same else leaves.structural
            bucket.append({"path": path, "old": old, "new": new})
            return
        delta = abs(b - a)
        scale = max(abs(a), abs(b))
        leaves.numeric.append({"path": path, "old": old, "new": new,
                               "abs_delta": delta,
                               "rel_delta": delta / scale if scale > REL_FLOOR else None})
        return
    if old != new:
        leaves.structural.append({"path": path, "old": old, "new": new})


def diff_json(old_path, new_path, top):
    leaves = Leaves()
    walk(json.loads(Path(old_path).read_text(encoding="utf-8")),
         json.loads(Path(new_path).read_text(encoding="utf-8")),
         "", leaves)
    numeric, provenance, structural = leaves.numeric, leaves.provenance, leaves.structural
    moved = sorted((n for n in numeric if n["abs_delta"] > 0),
                   key=lambda n: -n["abs_delta"])
    # An integer-valued leaf can be a count OR a measurement — an AUC that collapses
    # 1.0 -> 0.0 and a norm that moves 8.0 -> 9.0 are both integer-valued. Filtering
    # them out of the headline printed a catastrophe as "max |delta| none", so the
    # headline is now every moved leaf and the split is only a reading aid.
    integral = [n for n in moved if float(n["old"]).is_integer()
                and float(n["new"]).is_integer()]
    return {
        "old": str(old_path), "new": str(new_path),
        "n_numeric_leaves": len(numeric),
        "n_numeric_leaves_identical": len(numeric) - len(moved),
        "n_numeric_leaves_that_moved": len(moved),
        "max_abs_delta": moved[0]["abs_delta"] if moved else None,
        "max_abs_delta_path": moved[0]["path"] if moved else None,
        "largest_moves": moved[:top],
        "n_integer_valued_moves": len(integral),
        "integer_valued_moves": integral[:top],
        "n_non_finite_leaves": len(leaves.non_finite),
        "non_finite_leaves": leaves.non_finite[:top],
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
    committed = {path.name: path for path in (old_root / "vectors").glob("*.pt")}
    if not committed:
        raise SystemExit("no vectors under %s/vectors; --old must point at the "
                         "published directory" % old_root)
    # both directions: walking only the OLD tree left a vector the rebuild added
    # undiffed while the report still said nothing was missing
    rebuilt = {path.name: path for path in new_vectors.glob("*.pt")}
    if not rebuilt:
        # symmetric with the --old guard above: without it a wrong --new reported
        # 0 vectors compared, a null worst cosine, and exit 0
        raise SystemExit("no vectors under %s; --new must point at the rebuild's "
                         "WORK directory (or pass --new-vectors)" % new_vectors)
    for name in sorted(set(committed) | set(rebuilt)):
        if name not in rebuilt:
            report["missing"].append({"path": "vectors/%s" % name, "only_in": "old"})
            continue
        if name not in committed:
            report["missing"].append({"path": "vectors/%s" % name, "only_in": "new"})
            continue
        report["vectors"][name] = diff_vector(committed[name], rebuilt[name])
    for published, produced in JSON_FILES:
        old_path, new_path = old_root / published, new_root / produced
        if not old_path.is_file() or not new_path.is_file():
            # naming a side it is only in would be a lie when it is in neither
            side = ("neither" if not old_path.is_file() and not new_path.is_file()
                    else "new" if not old_path.is_file() else "old")
            report["missing"].append(
                {"path": published if not old_path.is_file() else produced,
                 "only_in": side})
            continue
        report["json"][published] = diff_json(old_path, new_path, args.top)

    vectors = list(report["vectors"].values())
    comparable = [v for v in vectors if "min_cos_over_layers" in v]
    deltas = [(name, f["max_abs_delta"]) for name, f in report["json"].items()
              if f["max_abs_delta"] is not None]  # None only when nothing moved
    worst_name, worst_delta = max(deltas, key=lambda kv: kv[1], default=(None, None))
    report["summary"] = {
        "n_vectors_compared": len(vectors),
        "n_vectors_bit_identical": sum(1 for v in vectors if v.get("bit_identical")),
        "n_vectors_uncomparable": len(vectors) - len(comparable),
        "worst_vector_cosine": min((v["min_cos_over_layers"] for v in comparable),
                                   default=None),
        "n_json_files_compared": len(report["json"]),
        "json_files_with_no_numeric_leaves": sorted(
            name for name, f in report["json"].items() if not f["n_numeric_leaves"]),
        "n_structural_differences": sum(f["n_structural_differences"]
                                        for f in report["json"].values()),
        "n_numeric_leaves_that_moved": sum(f["n_numeric_leaves_that_moved"]
                                           for f in report["json"].values()),
        "n_non_finite_leaves": sum(f["n_non_finite_leaves"]
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
        print("%-42s %6d leaves, %6d identical, %3d non-finite, %3d structural, "
              "max |delta| %s at %s"
              % (name, entry["n_numeric_leaves"], entry["n_numeric_leaves_identical"],
                 entry["n_non_finite_leaves"], entry["n_structural_differences"],
                 "nothing moved" if entry["max_abs_delta"] is None
                 else "%.3e" % entry["max_abs_delta"], entry["max_abs_delta_path"]))
        if not entry["n_numeric_leaves"]:
            print("  NOTHING NUMERIC COMPARED in %s" % name)
    if report["missing"]:
        print("MISSING: %s" % ", ".join("%s (only in %s)" % (m["path"], m["only_in"])
                                        for m in report["missing"]))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
