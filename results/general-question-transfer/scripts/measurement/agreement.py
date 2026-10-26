"""Agreement between raters on 0-100 scores: human-human, human-judge, judge-judge.

Checkpoint 2 asks, on the ~200-answer blind calibration sheet, for three numbers per
pair of raters and per score (trait, coherence), each with a bootstrap interval over
items:

* Spearman's rho on the raw 0-100 scores,
* quadratic-weighted Cohen's kappa on BINNED scores,
* Krippendorff's alpha (interval metric) on the same binned scores; it also accepts
  missing ratings, which kappa does not.

The bins are fixed in PREREG.md (`BIN_EDGES`): 0-19, 20-39, 40-59, 60-79, 80-100.

    python agreement.py --a human_labels_ana_2026-11-01.csv --b judge_scores.csv \\
        --score trait --out agreement_trait_ana_vs_judge.json

Each CSV needs `item_id` and a column named by `--score` (the grader page's export
and the judge-reveal export both have them). Items missing from either side, or
with an empty score, are counted and left out of rho and kappa; alpha uses them
as missing. Stdlib + numpy.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_PER_GAME = Path(__file__).resolve().parents[3] / "crossgame-per-game-steering" / "scripts" / "measurement"
sys.path.insert(0, str(_PER_GAME))

import stats  # noqa: E402

BIN_EDGES = (20, 40, 60, 80)          # 5 bins; a score of exactly 20 is bin 1
N_BINS = len(BIN_EDGES) + 1
N_BOOT = 2000
BOOT_SEED = 20261028           # PREREG.md section 13


def to_bin(score: float) -> int:
    if not 0 <= score <= 100:
        raise ValueError("score %r outside 0-100" % score)
    return sum(score >= edge for edge in BIN_EDGES)


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int], n_cat=N_BINS) -> Optional[float]:
    """Cohen's kappa with quadratic disagreement weights, on category indices."""
    if len(a) != len(b):
        raise ValueError("paired ratings required")
    if not a:
        return None
    observed = np.zeros((n_cat, n_cat))
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((n_cat, n_cat))
    weights = (i - j) ** 2 / float((n_cat - 1) ** 2)
    denom = (weights * expected).sum()
    if denom == 0:
        return None
    return float(1.0 - (weights * observed).sum() / denom)


def krippendorff_alpha(units: Sequence[Sequence[Optional[float]]], metric="interval") -> Optional[float]:
    """Krippendorff's alpha; `units[u]` is the ratings of unit u (None = missing).

    Units with fewer than two ratings are not pairable and drop out, as in the
    definition. Metrics: `interval` (squared difference) or `nominal`.
    """
    if metric not in ("interval", "nominal"):
        raise ValueError("metric is interval or nominal")
    pairable = [[v for v in unit if v is not None] for unit in units]
    pairable = [u for u in pairable if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    if n <= 1:
        return None

    def disagreement(values):
        """Sum of delta over ordered pairs i != j, in closed form."""
        m = len(values)
        if metric == "interval":
            x = np.asarray(values, dtype=float)
            return 2.0 * m * float((x * x).sum()) - 2.0 * float(x.sum()) ** 2
        counts: Dict = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return float(m * m - sum(c * c for c in counts.values()))

    d_o = sum(disagreement(u) / (len(u) - 1) for u in pairable) / n
    d_e = disagreement([v for u in pairable for v in u]) / (n * (n - 1))
    if d_e == 0:
        return None
    return 1.0 - d_o / d_e


def pair_stats(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Dict:
    """Point estimates for one pair of raters over the same items (None = missing)."""
    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    xs = [x for x, _ in both]
    ys = [y for _, y in both]
    bins_a = [to_bin(x) for x in xs]
    bins_b = [to_bin(y) for y in ys]
    units = [[None if x is None else to_bin(x), None if y is None else to_bin(y)]
             for x, y in zip(a, b)]
    return {
        "n_items": len(a),
        "n_paired": len(both),
        "spearman": stats.spearman(xs, ys),
        "qw_kappa_binned": quadratic_weighted_kappa(bins_a, bins_b),
        "krippendorff_alpha_interval_binned": krippendorff_alpha(units, "interval"),
        "mean_abs_diff": (sum(abs(x - y) for x, y in both) / len(both)) if both else None,
    }


def bootstrap_pair(a, b, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95) -> Dict:
    """`pair_stats` with percentile intervals from resampling items."""
    point = pair_stats(a, b)
    rng = np.random.default_rng(seed)
    n = len(a)
    draws = {k: [] for k in ("spearman", "qw_kappa_binned", "krippendorff_alpha_interval_binned")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = pair_stats([a[i] for i in idx], [b[i] for i in idx])
        for k in draws:
            if s[k] is not None and not math.isnan(s[k]):
                draws[k].append(s[k])
    alpha = (1 - level) / 2
    for k, values in draws.items():
        if values:
            point[k + "_ci"] = [float(np.quantile(values, alpha)),
                                float(np.quantile(values, 1 - alpha))]
            point[k + "_n_boot_defined"] = len(values)
        else:
            point[k + "_ci"] = None
    point["n_boot"] = n_boot
    point["seed"] = seed
    return point


def read_scores(path, score) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = row["item_id"]
            if item in out:
                raise SystemExit("%s: duplicate item_id %s" % (path, item))
            text = (row.get(score) or "").strip()
            out[item] = float(text) if text else None
    return out


def map_retest(scores, key) -> Dict[str, Optional[float]]:
    """Re-key a test-retest session's scores to the original item ids."""
    unknown = sorted(set(scores) - set(key))
    if unknown:
        raise SystemExit("retest ids not in the key: %s" % unknown[:5])
    return {key[i]: v for i, v in scores.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--score", required=True, choices=("trait", "coherence"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--retest-key", help="test-retest: JSON {new item_id: original item_id}; "
                    "b's ids are mapped back before pairing")
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    a, b = read_scores(args.a, args.score), read_scores(args.b, args.score)
    if args.retest_key:
        b = map_retest(b, json.loads(Path(args.retest_key).read_text()))
    items = sorted(set(a) | set(b))
    result = bootstrap_pair([a.get(i) for i in items], [b.get(i) for i in items],
                            n_boot=args.n_boot, seed=args.seed)
    result.update({"a": Path(args.a).name, "b": Path(args.b).name, "score": args.score,
                   "only_in_a": len(set(a) - set(b)), "only_in_b": len(set(b) - set(a)),
                   "bins": "0-19,20-39,40-59,60-79,80-100"})
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
