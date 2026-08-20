"""Dump a stratified sample of pole rows so the labels can be checked by hand.

The vector's labels come only from `audit.parse`. This sampler exists to MEASURE
that labelling's error rate, not to relabel anything: the parser is scored
against a human reading of the same response, and the measured rate goes in the
report beside the result.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--per-tag", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tail", type=int, default=340)
    args = ap.parse_args()

    with open(args.rows, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    buckets = {}
    for i, row in enumerate(rows):
        if row["value"] == "":
            continue
        endowment = int(row["game_id"].split("/e")[1])
        fraction = float(row["value"]) / endowment
        if fraction == 0.0:
            pole = "self"
        elif fraction >= 0.5:
            pole = "altruistic"
        else:
            continue
        buckets.setdefault((pole, row["tag"]), []).append((i, row, endowment, fraction))

    rng = random.Random(args.seed)
    for key in sorted(buckets):
        pool = buckets[key]
        take = pool if len(pool) <= args.per_tag else rng.sample(pool, args.per_tag)
        print("#" * 100)
        print("## pole=%s tag=%s  (%d rows in this bucket, showing %d)"
              % (key[0], key[1], len(pool), len(take)))
        for i, row, endowment, fraction in take:
            print("-" * 100)
            print("row=%d %s value=%s endowment=%d fraction=%.2f"
                  % (i, row["game_id"], row["value"], endowment, fraction))
            print("..." + row["answer"][-args.tail:].replace("\n", " | "))


if __name__ == "__main__":
    main()
