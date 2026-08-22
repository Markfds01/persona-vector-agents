"""The figure's judgement calls, tested away from matplotlib.

`make_figures.py` decides four things from the data rather than from a caption:
which points are a degeneration, which arm has no room left, which rungs the
supported band covers, and whether an end beat its null. Each is a plain function
and each can be wrong in a way that produces a plausible picture, so each is
pinned here on hand-built inputs, and then checked once against the committed
analysis so the rules and the published verdicts cannot drift apart.

No matplotlib figure is rendered; nothing here needs a display or a GPU.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


# the interpreter that produces the numbers carries torch, not matplotlib; the
# figures are drawn under a plain one, so this module skips rather than fails there
pytest.importorskip("matplotlib")

make_figures = _load("make_figures", SCRIPTS / "figures" / "make_figures.py")


def _point(coverage=1.0, non_latin=0.0):
    return {"parse_coverage": coverage,
            "degeneracy": {"share_with_non_latin_script": non_latin}}


def _pole(p, lo, hi):
    return {"p": p, "lo": lo, "hi": hi}


# --- the two kinds of degraded point ------------------------------------------

def test_answers_that_left_english_are_degenerate_not_merely_thin():
    degenerate, low = make_figures.point_flags(_point(coverage=0.86, non_latin=1.0))
    assert degenerate and low


def test_a_thin_point_in_english_is_a_caveat_and_not_a_degeneration():
    degenerate, low = make_figures.point_flags(_point(coverage=0.86, non_latin=0.13))
    assert not degenerate
    assert low


def test_a_healthy_point_carries_neither_flag():
    assert make_figures.point_flags(_point()) == (False, False)


# --- ceiling and floor --------------------------------------------------------

def test_a_baseline_inside_its_own_noise_of_the_bound_has_no_room():
    # Overfishing's own baseline: 0.959, Wilson [0.900, 0.984]
    assert make_figures.no_room(_pole(0.959, 0.900, 0.984), +1)


def test_a_baseline_at_the_floor_has_no_room_below_it():
    assert make_figures.no_room(_pole(0.0, 0.0, 0.037), -1)


def test_a_mid_range_baseline_has_room_in_both_directions():
    baseline = _pole(0.400, 0.309, 0.498)
    assert not make_figures.no_room(baseline, +1)
    assert not make_figures.no_room(baseline, -1)


def test_a_low_baseline_still_has_room_below_when_it_is_outside_its_own_noise():
    # the Dictator starts at 0.140 and really does fall to 0.020; that is not a floor
    assert not make_figures.no_room(_pole(0.140, 0.085, 0.221), -1)


# --- the supported band -------------------------------------------------------

def _arm(entries):
    """`entries` maps k -> (moved_significantly, non_latin_share)."""
    points, moves = {}, {}
    for k, (moved, non_latin) in entries.items():
        points[str(k)] = _point(non_latin=non_latin)
        moves[str(k)] = {"altruistic": {"excludes_zero": moved}}
    return {"points": points, "move_from_zero": moves}


def test_the_band_stops_where_the_arm_stopped_being_significant():
    arm = _arm({0: (False, 0.0), 1: (True, 0.0), 2: (True, 0.0), 3: (True, 0.0),
                4: (True, 0.0), 5: (False, 0.0)})
    assert make_figures.supported_range(arm, +1) == (1, 4)


def test_a_degenerate_rung_is_never_inside_the_supported_band():
    # the Prisoner's Dilemma: +4 and +5 move furthest of all and are not English
    arm = _arm({0: (False, 0.0), 1: (False, 0.03), 2: (True, 0.03), 3: (True, 0.03),
                4: (True, 1.0), 5: (True, 1.0)})
    assert make_figures.supported_range(arm, +1) == (2, 3)


def test_a_side_that_never_moved_gets_no_band_rather_than_a_band_at_zero():
    arm = _arm({-2: (False, 0.0), -1: (False, 0.0), 0: (False, 0.0), 1: (True, 0.0)})
    assert make_figures.supported_range(arm, -1) is None


def test_the_two_sides_are_measured_separately_so_an_asymmetric_arm_draws_asymmetric():
    arm = _arm({-2: (True, 0.0), -1: (True, 0.0), 0: (False, 0.0), 1: (False, 0.0),
                2: (False, 0.0)})
    assert make_figures.supported_range(arm, -1) == (-1, -2)
    assert make_figures.supported_range(arm, +1) is None


def test_a_single_significant_rung_is_a_band_of_one():
    arm = _arm({0: (False, 0.0), 1: (False, 0.0), 5: (True, 0.0)})
    assert make_figures.supported_range(arm, +1) == (5, 5)


def test_contiguous_degenerate_rungs_are_reported_as_one_run():
    arm = _arm({3: (True, 0.03), 4: (True, 1.0), 5: (True, 1.0)})
    assert make_figures.degenerate_runs(arm) == [[4, 5]]


# --- beating the null ---------------------------------------------------------

def _game(negative, positive, decision=None, null=None):
    """`negative`/`positive` are `(diff, excludes_zero)` at k=-5 and k=+5."""
    decision = decision or {"-5": _point(), "5": _point()}
    null = null or {"-5": _point(), "5": _point()}
    return {
        "decision_vs_null": {
            "-5": {"altruistic_decision_minus_null":
                   {"diff": negative[0], "excludes_zero": negative[1]}},
            "5": {"altruistic_decision_minus_null":
                  {"diff": positive[0], "excludes_zero": positive[1]}},
        },
        "arms": {"decision": {"points": decision},
                 "shuffled-null": {"points": null}},
    }


def test_beating_the_null_needs_the_difference_to_point_the_steered_way():
    verdict = make_figures.null_verdict(_game((-0.155, True), (+0.116, True)))
    assert verdict == {-1: "beats", +1: "beats"}


def test_a_significant_difference_the_wrong_way_is_not_a_win():
    # significant, but the real arm is BELOW its null at k=+5
    verdict = make_figures.null_verdict(_game((-0.9, True), (-0.5, True)))
    assert verdict[+1] == "null"


def test_an_interval_containing_zero_on_healthy_points_is_a_null():
    # the Ultimatum at k=+5: both arms healthy, +0.021 [-0.117, +0.157]
    verdict = make_figures.null_verdict(_game((-0.9, True), (+0.021, False)))
    assert verdict[+1] == "null"


def test_an_interval_containing_zero_where_an_arm_collapsed_is_undetermined():
    # the Ultimatum at k=-5: its NULL arm parsed 8 answers of 100
    game = _game((-0.179, False), (+0.021, False),
                 null={"-5": _point(coverage=0.08), "5": _point()})
    verdict = make_figures.null_verdict(game)
    assert verdict[-1] == "undetermined"
    assert verdict[+1] == "null"


def test_a_significant_result_on_a_thin_point_still_counts():
    # Trust's null at k=-5 resolved 87 of 100 and the difference is overwhelming
    game = _game((-0.691, True), (+0.498, True),
                 null={"-5": _point(coverage=0.87), "5": _point()})
    assert make_figures.null_verdict(game)[-1] == "beats"


# --- a comparison with nothing behind it --------------------------------------

def test_a_comparison_against_an_arm_that_answered_in_another_language_is_degraded():
    # true even when the interval clears zero: the PD's own k=+5 decision arm
    game = _game((-0.02, False), (+0.697, True),
                 decision={"-5": _point(), "5": _point(coverage=0.86, non_latin=1.0)})
    assert make_figures.comparison_degraded(game, +1)
    assert not make_figures.comparison_degraded(game, -1)
    assert make_figures.degraded_ends(game) == [+1]


def test_a_comparison_against_an_arm_that_barely_parsed_is_degraded():
    game = _game((-0.179, False), (+0.021, False),
                 null={"-5": _point(coverage=0.08), "5": _point()})
    assert make_figures.degraded_ends(game) == [-1]


def test_a_merely_thin_arm_does_not_degrade_the_comparison():
    # Trust's null resolved 87 of 100 and the difference is overwhelming
    game = _game((-0.691, True), (+0.498, True),
                 null={"-5": _point(coverage=0.87), "5": _point()})
    assert make_figures.degraded_ends(game) == []


def test_two_healthy_arms_leave_the_comparison_alone():
    assert make_figures.degraded_ends(_game((-0.5, True), (+0.5, True))) == []


def test_a_game_that_clears_the_bar_nowhere_says_so_in_its_own_headline():
    line, colour = make_figures.verdict_line({-1: "null", +1: "null"})
    assert "does NOT beat its own null at either end" == line
    assert colour == make_figures.BAD


def test_an_undetermined_end_is_not_reported_as_a_null():
    line, _colour = make_figures.verdict_line({-1: "undetermined", +1: "null"})
    assert "undetermined" in line and "does not beat its null" in line


# --- against the committed analysis -------------------------------------------

@pytest.fixture(scope="module")
def analysis():
    path = PACKAGE / "analysis" / "steering.json"
    if not path.exists():
        pytest.skip("no committed analysis to check the rules against")
    return json.loads(path.read_text())


def test_the_rules_reproduce_the_published_per_game_verdicts(analysis):
    """README section 5, derived rather than transcribed."""
    expected = {
        "dictator": {-1: "beats", +1: "beats"},
        "trust": {-1: "beats", +1: "beats"},
        "apology": {-1: "beats", +1: "beats"},
        "ultimatum": {-1: "undetermined", +1: "null"},
        "overfishing": {-1: "beats", +1: "null"},
        "prisoners_dilemma": {-1: "null", +1: "beats"},
    }
    for game, want in expected.items():
        assert make_figures.null_verdict(analysis["games"][game]) == want, game


def test_exactly_the_two_games_the_report_calls_pinned_have_no_room(analysis):
    ceilings, floors = [], []
    for game in make_figures.GAMES:
        baseline = (analysis["games"][game]["arms"]["decision"]["points"]["0"]
                    ["poles"]["strict"]["altruistic"])
        if make_figures.no_room(baseline, +1):
            ceilings.append(game)
        if make_figures.no_room(baseline, -1):
            floors.append(game)
    assert ceilings == ["overfishing"]
    assert floors == ["prisoners_dilemma"]


def test_the_only_degenerate_points_in_the_run_are_the_two_the_report_names(analysis):
    found = []
    for game in make_figures.GAMES:
        for arm_name, arm in analysis["games"][game]["arms"].items():
            for key, point in arm["points"].items():
                if make_figures.point_flags(point)[0]:
                    found.append((game, arm_name, int(key)))
    assert sorted(found) == [("prisoners_dilemma", "decision", 4),
                            ("prisoners_dilemma", "decision", 5)]


def test_the_prisoners_dilemmas_defensible_band_is_the_one_the_report_claims(analysis):
    arm = analysis["games"]["prisoners_dilemma"]["arms"]["decision"]
    assert make_figures.supported_range(arm, +1) == (2, 3)
    assert make_figures.supported_range(arm, -1) is None


def test_only_the_two_ends_the_report_qualifies_are_degraded_comparisons(analysis):
    """README section 3 qualifies exactly these two, for exactly this reason."""
    found = []
    for game in make_figures.GAMES:
        for side in make_figures.degraded_ends(analysis["games"][game]):
            found.append((game, side))
    assert sorted(found) == [("prisoners_dilemma", +1), ("ultimatum", -1)]
