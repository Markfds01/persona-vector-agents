"""Judge scores per (arm, k) point, question-clustered intervals, and the verdict.

Built before any judgment exists, so the analysis is fixed by PREREG.md rather than
by what the data turned out to look like. What each function answers:

* `point_summary` - one point's rows (one arm, one k, one question set, one judge):
  how many rows, how many the judge scored, how many the coherence rule excludes,
  the mean trait score filtered and unfiltered, the surface degeneracy counts, and
  whether the point produced a distribution at all.
* `clustered_bootstrap` / `clustered_bootstrap_diff` - the interval. Samples of one
  question are correlated (five draws of the same prompt), so the effective n is
  closer to the number of questions than to the number of rows: questions are
  resampled with replacement, then samples within each drawn question.
* `end_verdict` - "decision - null at matched unit beta", read off whether the
  interval excludes zero in the pre-registered direction, never off a p.
  A point that produced no distribution makes the end `undetermined`.
* `holm` - which members of the pre-registered primary family would survive a Holm
  correction. Reported beside the verdicts, never in place of them.

Exclusion never depends on the trait score: a row is excluded by the PRIMARY judge's
coherence (< 50, or not scored) only.

`stats.py` (Wilson, Newcombe, Spearman, surface degeneracy) and `degradation.py`
(the per-game study's "this point is not a result" thresholds) are imported from
`results/crossgame-per-game-steering/scripts/measurement`, not copied, so the two
studies call the same point unreadable by the same rule.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence

import numpy as np

_PER_GAME = Path(__file__).resolve().parents[3] / "crossgame-per-game-steering" / "scripts" / "measurement"
sys.path.insert(0, str(_PER_GAME))

import degradation  # noqa: E402
import stats  # noqa: E402

#: upstream's coherence convention, by the PRIMARY judge
COHERENCE_THRESHOLD = 50.0
#: bootstrap draws and seed, fixed in PREREG.md sections 8.1 and 13
N_BOOT = 10000
BOOT_SEED = 20261028
#: the per-game study's gates, reused verbatim
DEGENERATE_NON_LATIN = degradation.DEGENERATE_NON_LATIN
COLLAPSED_COVERAGE = degradation.COLLAPSED_COVERAGE

VERDICTS = ("beats", "null", "reversed", "undetermined")


def _is_score(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def kept(row, threshold=COHERENCE_THRESHOLD) -> bool:
    """The exclusion rule: primary-judge coherence scored and >= threshold."""
    return row.get("coherence_status") == "ok" and _is_score(row.get("coherence")) \
        and float(row["coherence"]) >= threshold


def groups_by_question(rows: Iterable[Mapping], value_key="trait", filtered=True,
                       threshold=COHERENCE_THRESHOLD) -> Dict[Hashable, List[float]]:
    """question_id -> the scored values that enter the mean."""
    out: Dict[Hashable, List[float]] = {}
    for row in rows:
        if row.get(value_key + "_status") != "ok" or not _is_score(row.get(value_key)):
            continue
        if filtered and not kept(row, threshold):
            continue
        out.setdefault(row["question_id"], []).append(float(row[value_key]))
    return out


def _boot_means(rng, groups_list, picks):
    """Mean of the pooled values of each bootstrap draw (questions, then samples).

    `picks[b]` is draw b's resample of question indices; every time a question is
    picked, its own values are resampled with replacement, independently.
    """
    n_boot = picks.shape[0]
    sums = np.zeros(n_boot)
    counts = np.zeros(n_boot)
    for q, values in enumerate(groups_list):
        values = np.asarray(values, dtype=float)
        m = len(values)
        times = (picks == q).sum(axis=1)          # how often q was drawn, per draw
        total = int(times.sum())
        if m == 0 or total == 0:
            continue
        resampled = values[rng.integers(0, m, size=(total, m))].sum(axis=1)
        owner = np.repeat(np.arange(n_boot), times)
        np.add.at(sums, owner, resampled)
        np.add.at(counts, owner, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def clustered_bootstrap(groups: Mapping[Hashable, Sequence[float]], n_boot=N_BOOT, seed=BOOT_SEED,
                        level=0.95) -> Dict:
    """Pooled mean with a question-clustered percentile interval."""
    keys = sorted(groups, key=str)
    values = [list(groups[k]) for k in keys]
    n_values = sum(len(v) for v in values)
    if n_values == 0:
        return {"estimate": None, "lo": None, "hi": None, "n_questions": 0, "n_values": 0}
    estimate = sum(sum(v) for v in values) / n_values
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = _boot_means(rng, values, picks)
    boots = boots[~np.isnan(boots)]
    alpha = (1 - level) / 2
    return {"estimate": estimate, "lo": float(np.quantile(boots, alpha)),
            "hi": float(np.quantile(boots, 1 - alpha)),
            "n_questions": sum(1 for v in values if v), "n_values": n_values,
            "n_boot": int(boots.size), "seed": seed, "estimator": "question_clustered_bootstrap"}


def clustered_bootstrap_diff(a: Mapping[Hashable, Sequence[float]],
                             b: Mapping[Hashable, Sequence[float]], n_boot=N_BOOT, seed=BOOT_SEED,
                             level=0.95) -> Dict:
    """mean(a) - mean(b), questions resampled JOINTLY (both arms saw the same questions).

    The same question draw is applied to both arms; samples within a question are
    resampled independently per arm. Also returns a two-sided bootstrap p, used for
    the Holm flag only - the verdict is the interval.
    """
    keys = sorted(set(a) | set(b), key=str)
    va = [list(a.get(k, ())) for k in keys]
    vb = [list(b.get(k, ())) for k in keys]
    na, nb = sum(map(len, va)), sum(map(len, vb))
    if na == 0 or nb == 0:
        return {"diff": None, "lo": None, "hi": None, "excludes_zero": None, "p": None}
    diff = sum(map(sum, va)) / na - sum(map(sum, vb)) / nb
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = _boot_means(rng, va, picks) - _boot_means(rng, vb, picks)
    boots = boots[~np.isnan(boots)]
    alpha = (1 - level) / 2
    lo, hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
    below = (np.sum(boots <= 0) + 1) / (boots.size + 1)
    above = (np.sum(boots >= 0) + 1) / (boots.size + 1)
    return {"diff": diff, "lo": lo, "hi": hi, "excludes_zero": not (lo <= 0.0 <= hi),
            "p": float(min(1.0, 2 * min(below, above))), "n_boot": int(boots.size),
            "seed": seed, "estimator": "question_clustered_bootstrap"}


def point_summary(rows: Sequence[Mapping], threshold=COHERENCE_THRESHOLD, n_boot=N_BOOT,
                  seed=BOOT_SEED) -> Dict:
    """Everything reported for one (arm, k, question set, judge) point."""
    n = len(rows)
    if n == 0:
        return {"n_rows": 0}
    trait_ok = [r for r in rows if r.get("trait_status") == "ok" and _is_score(r.get("trait"))]
    coherence_ok = [r for r in rows if r.get("coherence_status") == "ok"
                    and _is_score(r.get("coherence"))]
    kept_rows = [r for r in rows if kept(r, threshold)]
    usable_filtered = [r for r in kept_rows
                       if r.get("trait_status") == "ok" and _is_score(r.get("trait"))]
    status_counts: Dict[str, Dict[str, int]] = {}
    for field in ("generation_status", "trait_status", "coherence_status"):
        counts: Dict[str, int] = {}
        for r in rows:
            counts[str(r.get(field))] = counts.get(str(r.get(field)), 0) + 1
        status_counts[field] = counts
    surface = stats.degeneracy([r.get("answer") or "" for r in rows])
    filtered = clustered_bootstrap(groups_by_question(rows, "trait", True, threshold),
                                   n_boot=n_boot, seed=seed)
    unfiltered = clustered_bootstrap(groups_by_question(rows, "trait", False, threshold),
                                     n_boot=n_boot, seed=seed)
    coherence = clustered_bootstrap(groups_by_question(rows, "coherence", False, threshold),
                                    n_boot=n_boot, seed=seed)
    summary = {
        "n_rows": n,
        "n_questions": len({r["question_id"] for r in rows}),
        "status_counts": status_counts,
        "share_truncated": sum(r.get("generation_status") == "truncated" for r in rows) / n,
        "share_coherence_unscored": 1 - len(coherence_ok) / n,
        "share_excluded": 1 - len(kept_rows) / n,
        "share_trait_unscored": 1 - len(trait_ok) / n,
        "coverage_filtered": len(usable_filtered) / n,
        "coverage_unfiltered": len(trait_ok) / n,
        "trait_filtered": filtered,
        "trait_unfiltered": unfiltered,
        "coherence": coherence,
        "degeneracy": surface,
    }
    summary["unreadable_filtered"] = produced_no_distribution(summary, filtered=True)
    summary["unreadable_unfiltered"] = produced_no_distribution(summary, filtered=False)
    return summary


def produced_no_distribution(summary: Mapping, filtered=True) -> bool:
    """The per-game study's gate: non-Latin >= 0.5, or under half the rows usable.

    "Usable" is the judge-score analogue of the per-game study's parse coverage: a
    trait score the judge returned, and (filtered analysis) a coherence >= 50. That
    is how a coherence collapse makes a point unreadable.
    """
    coverage = summary["coverage_filtered" if filtered else "coverage_unfiltered"]
    return (summary["degeneracy"].get("share_with_non_latin_script", 0.0) >= DEGENERATE_NON_LATIN
            or coverage < COLLAPSED_COVERAGE)


def end_verdict(diff: Mapping, direction: int, decision_unreadable: bool,
                null_unreadable: bool) -> str:
    """One end of the ladder against one null.

    `direction` is +1 for the positive end (decision expected above null) and -1
    for the negative end. An end where either arm produced no distribution is
    `undetermined` - missing evidence, not evidence of absence.
    """
    if direction not in (1, -1):
        raise ValueError("direction is +1 or -1")
    if decision_unreadable or null_unreadable or diff.get("diff") is None:
        return "undetermined"
    if not diff["excludes_zero"]:
        return "null"
    return "beats" if (diff["diff"] > 0) == (direction > 0) else "reversed"


def holm(pvalues: Mapping[Hashable, Optional[float]], alpha=0.05) -> Dict[Hashable, Dict]:
    """Holm step-down over a declared family. A missing p counts as a member that fails.

    Returns name -> {p, adjusted, survives}. The family size is every key passed,
    including the ones with no p, so dropping an undetermined contrast cannot make
    the rest look stronger.
    """
    m = len(pvalues)
    present = sorted(((p, k) for k, p in pvalues.items() if p is not None),
                     key=lambda t: (t[0], str(t[1])))
    out: Dict[Hashable, Dict] = {k: {"p": None, "adjusted": None, "survives": False,
                                     "family_size": m}
                                 for k, p in pvalues.items() if p is None}
    running = 0.0
    still = True
    for rank, (p, k) in enumerate(present):
        adjusted = min(1.0, max(running, (m - rank) * p))
        running = adjusted
        still = still and adjusted <= alpha
        out[k] = {"p": p, "adjusted": adjusted, "survives": still, "family_size": m}
    return out


def bridge_point(choices: Sequence[Optional[int]]) -> Dict:
    """P(other-regarding) with Wilson, over parsed bridge answers only; unparsed counted."""
    parsed = [c for c in choices if c is not None]
    out = stats.wilson(sum(parsed), len(parsed))
    out["n_rows"] = len(choices)
    out["parse_coverage"] = len(parsed) / len(choices) if choices else None
    return out


def bridge_diff(a: Sequence[Optional[int]], b: Sequence[Optional[int]]) -> Dict:
    """P_a - P_b with Newcombe's hybrid-score interval; verdict off the interval."""
    pa = [c for c in a if c is not None]
    pb = [c for c in b if c is not None]
    return stats.newcombe(sum(pa), len(pa), sum(pb), len(pb))


def auc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUC: P(score of an other-regarding choice > a self-interested one)."""
    if not scores_pos or not scores_neg:
        return None
    wins = 0.0
    for x in scores_pos:
        for y in scores_neg:
            wins += 1.0 if x > y else 0.5 if x == y else 0.0
    return wins / (len(scores_pos) * len(scores_neg))
