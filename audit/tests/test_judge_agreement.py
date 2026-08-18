"""Agreement between the free scorers and the paid gpt-4.1-mini judge.

The labels are the judge values committed in this repo, so the whole thing runs
offline in a couple of seconds. Every figure below is measured on conditions the
scorers were never tuned on; `TUNING_COEFFICIENT` is excluded throughout.

"Agrees" means `round(judge) == ours`, the judge being a soft expectation over
its own token distribution and therefore rarely a clean integer (it is never
more than $0.48 off one in this corpus).

The floors are set below the measured values so the suite is not brittle. Each
one carries the measurement it was derived from; if a change moves a number,
re-measure and say so rather than lowering the floor.
"""

import re
import statistics

from audit.games import by_id, score
from audit.parse import extract_amount
from audit.tests.conftest import TUNING_COEFFICIENT

DICTATOR = by_id("altruism_v2/dictator")
OVERFISHING = by_id("altruism_v3/overfishing")
PRISONERS_DILEMMA = by_id("altruism_v3/prisoners_dilemma")


def agrees(value, judge) -> bool:
    return value is not None and round(judge) == value


def pearson(xs, ys) -> float:
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spread = (sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)) ** 0.5
    return covariance / spread


def _scored(rows, game):
    """[(ours, judge)] for every row, with `ours` None where nothing resolved."""
    return [(score(game, row.answer).value, row.judge) for row in rows]


def test_held_out_amount_coverage(v2_held_out):
    """Measured: 1951/2000 rows resolve = 97.55%. The other 49 are reported, not guessed."""
    scored = [pair for rows in v2_held_out.values() for pair in _scored(rows, DICTATOR)]
    assert len(scored) == 2000
    resolved = [pair for pair in scored if pair[0] is not None]
    assert len(resolved) / len(scored) >= 0.96


def test_held_out_amount_agreement(v2_held_out):
    """Measured: 1865/1951 resolved rows = 95.59% exact, over 10 steering conditions."""
    scored = [pair for rows in v2_held_out.values() for pair in _scored(rows, DICTATOR)]
    resolved = [pair for pair in scored if pair[0] is not None]
    exact = sum(1 for ours, judge in resolved if agrees(ours, judge))
    assert exact / len(resolved) >= 0.945


def test_held_out_condition_means_track_the_judge(v2_held_out):
    """Condition means are what any conclusion rests on. Measured: r = 0.9992, max gap $2.46."""
    ours, theirs = [], []
    for coefficient in sorted(v2_held_out):
        resolved = [pair for pair in _scored(v2_held_out[coefficient], DICTATOR)
                    if pair[0] is not None]
        ours.append(statistics.mean(value for value, _judge in resolved))
        theirs.append(statistics.mean(judge for _value, judge in resolved))

    assert len(ours) == 10
    assert pearson(ours, theirs) >= 0.995
    assert max(abs(a - b) for a, b in zip(ours, theirs)) <= 3.0


def test_scorer_beats_a_naive_last_number_regex(v2_held_out):
    """The reason this package is not three lines of regex.

    Measured on the held-out rows: last-number agrees on 20.8%, the scorer on 93.3%
    (both over all 2,000 rows, so unresolved counts against the scorer).
    """
    last_number = re.compile(r"\$?\s?(\d{1,3})(?:\.\d+)?")

    def naive(answer):
        found = [int(n) for n in last_number.findall(answer or "") if 0 <= int(n) <= 100]
        return float(found[-1]) if found else None

    rows = [row for condition in v2_held_out.values() for row in condition]
    naive_exact = sum(1 for row in rows if agrees(naive(row.answer), row.judge))
    ours_exact = sum(1 for row in rows
                     if agrees(extract_amount(row.answer, 100).value, row.judge))

    assert naive_exact / len(rows) < 0.35
    assert ours_exact / len(rows) >= 0.90


def test_tuned_condition_still_reproduces_the_judge(v2_by_coefficient):
    """Not held out - the scorers were tuned here. Measured: 199/200 = 99.5%."""
    scored = _scored(v2_by_coefficient[TUNING_COEFFICIENT], DICTATOR)
    exact = sum(1 for ours, judge in scored if agrees(ours, judge))
    assert exact / len(scored) >= 0.99


def test_fish_agreement_against_the_per_question_judge(per_question_labels):
    """Measured: 209/210 resolve, 194 of those agree = 92.82%.

    Below the amount scorer, and below the 96.7% the investigation measured on its
    own n=30 run. The residual is trailing-sentence contamination: a response that
    decides on 50 and then explains "the total is then 100" can resolve to 100.
    """
    rows = per_question_labels["q3_fish_caught"]
    scored = _scored(rows, OVERFISHING)
    resolved = [pair for pair in scored if pair[0] is not None]

    assert len(resolved) / len(scored) >= 0.97
    exact = sum(1 for ours, judge in resolved if agrees(ours, judge))
    assert exact / len(resolved) >= 0.90


def test_cooperate_agreement_against_the_per_question_judge(per_question_labels):
    """Measured: 210/210 resolve, 206 agree = 98.10%.

    The binary judge reports p(cooperate) in [0, 1], so its label is thresholded.
    """
    rows = per_question_labels["q4_cooperated"]
    scored = _scored(rows, PRISONERS_DILEMMA)
    resolved = [pair for pair in scored if pair[0] is not None]

    assert len(resolved) == len(scored)
    exact = sum(1 for ours, judge in resolved if (ours >= 0.5) == (judge >= 0.5))
    assert exact / len(resolved) >= 0.95
