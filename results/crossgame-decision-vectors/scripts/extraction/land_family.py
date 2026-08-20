"""Land one game the moment it finishes: its vector, its counts, its manifest entry.

The GPU is shared with another tenant that loads and unloads models on its own
schedule, so this run assumes it will be interrupted. Landing per game means an
interruption costs the game in flight, not the set, and it means partial output
is interpretable: the manifest says which games are present, how many rows each
pole holds, and what the vector's own split-half separation was.

The vector written here is built exactly as the pooled one is - same poles, same
cell balancing, same code path in `analyze_crossgame` - so a per-game vector and
the pooled vector are directly comparable.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch import analyze_crossgame as A
from scratch import poles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else {"families": []})

    entry_manifest = {"family": args.family,
                      "rows_csv": str(Path(args.rows).resolve()),
                      "acts_dir": str(Path(args.acts).resolve())}
    labels, acts, metas, tag_counts = A.load_all({"families": [entry_manifest]})

    record = dict(entry_manifest)
    record["n_generated_rows"] = sum(tag_counts[args.family].values())
    record["tag_counts"] = tag_counts[args.family]
    record["activation_meta"] = metas[args.family]
    record["by_policy"] = {}
    summary = {}
    for policy in poles.POLICIES:
        stats, naive, bal = A.build_family(acts["response_avg"], labels, args.family,
                                           out_dir, args.seed, out_dir, policy)
        record["by_policy"][policy] = stats
        if not stats["usable"]:
            summary[policy] = {"usable": False, "reason": stats["reason"]}
            continue
        shape = tuple(naive.shape)
        if shape != (29, 3584):
            raise SystemExit("%s: vector shape %s is not (29, 3584)"
                             % (args.family, shape))
        sh = stats["split_half"]
        summary[policy] = {
            "alt": stats["n_altruistic"], "self": stats["n_self_interested"],
            "middle": stats["n_middle_discarded"],
            "usable_cells": stats["n_cells_seen"] and len(stats["usable_cells"]),
            "layer_0_auc": sh["by_layer"][0]["auc"] if sh else None,
            "layer_20_auc": sh["by_layer"][20]["auc"] if sh else None,
            "best_layer": max(sh["by_layer"], key=lambda r: r["auc"])["layer"] if sh else None,
            "best_layer_auc": max(r["auc"] for r in sh["by_layer"]) if sh else None,
            "norm_layer_20_unbalanced": stats["norms_unbalanced"][20],
            "norm_layer_20_balanced": (stats["norms_balanced"][20]
                                       if stats["norms_balanced"] else None),
        }
    record["headline"] = summary

    manifest["families"] = [f for f in manifest["families"]
                            if f["family"] != args.family] + [record]
    manifest["families"].sort(key=lambda f: f["family"])
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("landed %s (%d rows): %s"
          % (args.family, record["n_generated_rows"], json.dumps(summary)), flush=True)


if __name__ == "__main__":
    main()
