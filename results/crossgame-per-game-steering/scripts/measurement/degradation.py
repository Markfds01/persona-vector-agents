"""When a steered point stopped being a measurement, and how badly.

Two DIFFERENT kinds of degraded point, kept apart because they mean different
things and carry different consequences:

* `degenerate` — the model stopped answering in English. Every answer at the
  Prisoner's Dilemma's k=+4 and k=+5 contains non-Latin script; the scorer reads
  them correctly, but what it is reading is no longer this game being played in
  the language the prompt was written in. Those points are not results.
* `low_coverage` — the scorer could not read enough of it. A caveat drawn on the
  point, never a disqualification: Overfishing's k=-4 and k=-5 resolved 89 and 86
  answers of 100 and carry a large, graded, real effect.

Below `COLLAPSED_COVERAGE` the point did not produce a distribution at all, and a
contrast computed against it is not a comparison in either direction. That is the
gate, and it lives here rather than in either consumer because the figures and the
pooled tables must call the same point unreadable or the package contradicts
itself in print.

Stdlib only, so the tables can import it under an interpreter with no matplotlib.
"""

#: Every answer non-Latin means the measurement is no longer about the game. Half
#: is far past any rate the healthy points show (they sit at 0.00-0.13).
DEGENERATE_NON_LATIN = 0.5
#: Below this the point is drawn hollow with its parsed count: still measured,
#: measured on less.
LOW_COVERAGE = 0.90
#: Below this the point did not produce a distribution at all - most answers never
#: state a decision - so a comparison against it is not a comparison.
COLLAPSED_COVERAGE = 0.50


def point_flags(point):
    """`(degenerate, low_coverage)` for one beta point.

    Degenerate is the hard one - the answers are not in English any more and the
    point is not a result. Low coverage is a caveat: fewer rows resolved.
    """
    degeneracy = point["degeneracy"]
    return (degeneracy["share_with_non_latin_script"] >= DEGENERATE_NON_LATIN,
            point["parse_coverage"] < LOW_COVERAGE)


def produced_no_distribution(point):
    """Did this point fail to produce readable decisions at all?

    Degenerate, or under half the answers resolved. A merely thin point is a
    caveat and does NOT land here - that is what keeps `LOW_COVERAGE` and
    `COLLAPSED_COVERAGE` two different numbers.
    """
    return (point_flags(point)[0]
            or point["parse_coverage"] < COLLAPSED_COVERAGE)
