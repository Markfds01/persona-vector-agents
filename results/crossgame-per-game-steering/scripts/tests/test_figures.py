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


def test_a_significant_contrast_on_a_struck_out_point_is_not_a_win():
    """The Prisoner's Dilemma at k=+5: +0.697, and every answer non-Latin.

    The interval clears zero by a distance and points the steered way. It is
    still not a win: the figure strikes that point out as "not a result", and a
    contrast computed across it cannot be the evidence for a win any more than
    for a loss.
    """
    game = _game((-0.02, False), (+0.697, True),
                 decision={"-5": _point(), "5": _point(coverage=0.86, non_latin=1.0)})
    assert make_figures.null_verdict(game)[+1] == "undetermined"


def test_a_significant_contrast_against_an_arm_that_barely_parsed_is_not_a_win():
    """The relaxed Ultimatum at k=+5: its null stated an offer in 7 answers of 100."""
    game = _game((-0.324, True), (+0.697, True),
                 null={"-5": _point(), "5": _point(coverage=0.07, non_latin=1.0)})
    assert make_figures.null_verdict(game)[+1] == "undetermined"
    assert make_figures.null_verdict(game)[-1] == "beats"


def test_a_thin_arm_does_not_turn_a_measured_null_into_undetermined():
    """Low coverage is a caveat; only a COLLAPSED arm makes a comparison say nothing.

    0.87 is under LOW_COVERAGE and well over COLLAPSED_COVERAGE, so an interval
    containing zero there is a real "does not beat", not missing evidence.
    """
    game = _game((-0.02, False), (+0.02, False),
                 null={"-5": _point(coverage=0.87), "5": _point()})
    assert make_figures.null_verdict(game)[-1] == "null"


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


# --- the headline count -------------------------------------------------------

def _analysis(**games):
    return {"games": games}


def test_the_headline_does_not_count_an_end_that_could_not_be_compared():
    analysis = _analysis(
        trust=_game((-0.7, True), (+0.5, True)),
        prisoners_dilemma=_game(
            (-0.02, False), (+0.697, True),
            decision={"-5": _point(), "5": _point(coverage=0.86, non_latin=1.0)}))
    line = make_figures._headline(analysis, ["trust", "prisoners_dilemma"])
    assert line.startswith("1 of 2 clear that bar at both ends")
    assert "1 with an end that could not be compared" in line


def test_a_game_measured_at_both_ends_and_beating_at_neither_is_counted_as_neither():
    analysis = _analysis(ultimatum=_game((-0.02, False), (+0.02, False)))
    assert make_figures._headline(analysis, ["ultimatum"]) == (
        "0 of 1 clear that bar at both ends, 0 at one end, 1 at neither")


def test_a_game_beating_at_one_end_is_counted_and_not_dropped():
    analysis = _analysis(overfishing=_game((-0.94, True), (-0.04, False)))
    assert make_figures._headline(analysis, ["overfishing"]) == (
        "0 of 1 clear that bar at both ends, 1 at one end, 0 at neither")


def test_an_end_that_could_not_be_compared_still_leaves_the_game_in_a_total():
    """The counts partition the games: no game falls out of all three."""
    analysis = _analysis(
        trust=_game((-0.7, True), (+0.5, True)),
        overfishing=_game((-0.94, True), (-0.04, False)),
        ultimatum=_game(
            (-0.179, False), (+0.021, False),
            null={"-5": _point(coverage=0.08), "5": _point()}),
        prisoners_dilemma=_game(
            (-0.02, False), (+0.697, True),
            decision={"-5": _point(), "5": _point(coverage=0.86, non_latin=1.0)}))
    games = ["trust", "overfishing", "ultimatum", "prisoners_dilemma"]
    line = make_figures._headline(analysis, games)
    assert line == ("1 of 4 clear that bar at both ends, 1 at one end, 2 at "
                    "neither; 2 with an end that could not be compared")


# --- a point nothing could be read off ----------------------------------------

def _pole_point(p, lo, hi, coverage=1.0, non_latin=0.0):
    point = _point(coverage=coverage, non_latin=non_latin)
    point["poles"] = {"strict": {"altruistic": _pole(p, lo, hi)}}
    point["n_parsed"] = int(round(100 * coverage))
    return point


def test_a_point_that_parsed_nothing_is_dropped_from_the_line_not_crashed_on():
    getter = make_figures._pole_getter("strict")
    assert getter(_pole_point(None, None, None, coverage=0.0)) is None
    assert getter(_pole_point(0.5, 0.4, 0.6)) == (0.4, 0.6, 0.5)


def test_a_series_skips_the_unreadable_point_and_keeps_the_rest():
    arm = {"points": {"0": _pole_point(0.5, 0.4, 0.6),
                      "5": _pole_point(None, None, None, coverage=0.0)}}
    series = make_figures._series(arm, make_figures._pole_getter("strict"))
    assert [entry["k"] for entry in series] == [0]


def test_a_baseline_that_parsed_nothing_is_refused_by_name():
    """Every band and every dashed line on the panel is measured against k=0."""
    game = _game((-0.5, True), (+0.5, True))
    game["arms"]["decision"]["points"]["0"] = _pole_point(None, None, None,
                                                          coverage=0.0)
    with pytest.raises(SystemExit) as excinfo:
        make_figures._panel(None, "trust", game, "strict",
                            make_figures._pole_getter("strict"), "y", (0, 1), True)
    assert "trust" in str(excinfo.value) and "k=0" in str(excinfo.value)


def test_the_note_names_the_arm_that_tripped_the_gate_not_the_thinner_one():
    """The Prisoner's Dilemma's k=+5: the decision arm left English at 86 parsed,
    while its null parsed all 100. Blaming the smaller count would still name the
    decision arm there; blaming it when the healthy arm is the thinner one would
    name the wrong one."""
    game = _game((-0.02, False), (+0.697, True),
                 decision={"-5": _point(), "5": _point(coverage=0.95,
                                                       non_latin=1.0)},
                 null={"-5": _point(), "5": _point(coverage=0.60)})
    for arm, points in game["arms"].items():
        for key, point in points["points"].items():
            point["n_parsed"] = int(round(100 * point["parse_coverage"]))
            point["n_rows"] = 100
    note = make_figures._comparison_notes(_analysis(prisoners_dilemma=game),
                                          ["prisoners_dilemma"])
    assert "its decision arm parsed 95" in note


# --- a run that did not cover the whole ladder --------------------------------

def test_the_ends_are_the_outermost_rungs_the_run_actually_shares():
    game = _game((-0.5, True), (+0.5, True))
    game["decision_vs_null"]["3"] = game["decision_vs_null"]["5"]
    assert make_figures.ladder_ends(game) == {-1: "-5", +1: "5"}
    del game["decision_vs_null"]["5"]
    assert make_figures.ladder_ends(game) == {-1: "-5", +1: "3"}


def test_a_run_that_never_left_one_side_of_zero_is_refused_by_name():
    game = _game((-0.5, True), (+0.5, True))
    del game["decision_vs_null"]["5"]
    game["family"] = "dictator"
    with pytest.raises(SystemExit) as excinfo:
        make_figures.ladder_ends(game)
    assert "dictator" in str(excinfo.value)


def test_an_analysis_without_a_null_arm_is_refused_rather_than_half_drawn():
    game = _game((-0.5, True), (+0.5, True))
    game["arms"]["decision"]["points"]["0"] = _point()
    del game["arms"]["shuffled-null"]
    with pytest.raises(SystemExit) as excinfo:
        make_figures._require_drawable("trust", game)
    assert "shuffled-null" in str(excinfo.value)


def test_a_ladder_without_its_shared_no_op_is_refused_rather_than_half_drawn():
    game = _game((-0.5, True), (+0.5, True))
    with pytest.raises(SystemExit) as excinfo:
        make_figures._require_drawable("trust", game)
    assert "k=0" in str(excinfo.value)


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
        # +5 is where every answer is non-Latin: struck out on the figure,
        # so the +0.697 contrast computed across it is not a verdict either way
        "prisoners_dilemma": {-1: "null", +1: "undetermined"},
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


# --- the filenames ------------------------------------------------------------

def test_the_policy_is_in_every_filename_because_one_directory_holds_both():
    """`figures/` holds both policies, the way `vectors/` holds both."""
    assert [name for _build, name in make_figures.figure_files("strict")] == [
        "steering_pole_shares_strict.png",
        "steering_own_measure_strict.png",
        "steering_vs_null_strict.png",
    ]
    assert [name for _build, name in make_figures.figure_files("relaxed")] == [
        "steering_pole_shares_relaxed.png",
        "steering_own_measure_relaxed.png",
        "steering_vs_null_relaxed.png",
    ]


def test_no_filename_is_shared_between_the_two_policies():
    """The collision the one-directory layout would otherwise reintroduce."""
    strict = {name for _build, name in make_figures.figure_files("strict")}
    relaxed = {name for _build, name in make_figures.figure_files("relaxed")}
    assert not strict & relaxed


# --- against the committed relaxed analysis -----------------------------------

def _drawn(analysis):
    """The games an analysis carries, in the canonical order."""
    return [game for game in make_figures.GAMES if game in analysis["games"]]


@pytest.fixture(scope="module")
def relaxed():
    path = PACKAGE / "analysis" / "steering_relaxed.json"
    if not path.exists():
        pytest.skip("no committed relaxed analysis to check the rules against")
    return json.loads(path.read_text())


def test_the_rules_reproduce_the_published_relaxed_per_game_verdicts(relaxed):
    """README section 11.1, derived rather than transcribed.

    Five games. The Prisoner's Dilemma was not re-run: its relaxed vector is
    element-wise identical to its strict one, so section 3 already holds it.
    """
    expected = {
        "dictator": {-1: "beats", +1: "null"},
        "trust": {-1: "beats", +1: "beats"},
        "apology": {-1: "beats", +1: "beats"},
        # the negative end is the win the round existed to settle; at +5 the
        # NULL arm parsed 7 of 100, so that end is not evidence either way
        "ultimatum": {-1: "beats", +1: "undetermined"},
        "overfishing": {-1: "beats", +1: "null"},
    }
    assert set(relaxed["games"]) == set(expected)
    for game, want in expected.items():
        assert make_figures.null_verdict(relaxed["games"][game]) == want, game


def test_only_overfishing_has_no_room_on_the_relaxed_arm(relaxed):
    ceilings, floors = [], []
    for game in _drawn(relaxed):
        baseline = (relaxed["games"][game]["arms"]["decision"]["points"]["0"]
                    ["poles"]["relaxed"]["altruistic"])
        if make_figures.no_room(baseline, +1):
            ceilings.append(game)
        if make_figures.no_room(baseline, -1):
            floors.append(game)
    assert ceilings == ["overfishing"]
    assert floors == []


def test_the_relaxed_runs_only_degenerate_point_is_the_ultimatums_null(relaxed):
    found = []
    for game in _drawn(relaxed):
        for arm_name, arm in relaxed["games"][game]["arms"].items():
            for key, point in arm["points"].items():
                if make_figures.point_flags(point)[0]:
                    found.append((game, arm_name, int(key)))
    assert sorted(found) == [("ultimatum", "shuffled-null", 5)]


def test_only_the_ultimatums_positive_end_is_degraded_on_the_relaxed_arm(relaxed):
    """README section 11.2 qualifies exactly this one end."""
    found = []
    for game in _drawn(relaxed):
        for side in make_figures.degraded_ends(relaxed["games"][game]):
            found.append((game, side))
    assert sorted(found) == [("ultimatum", +1)]


# --- the own-measure axis, which is shared across the two policies ------------

def _measure_game(ci_highs):
    """One game whose single arm carries exactly these interval upper bounds."""
    points = {str(k): {"measure": {"ci_high": hi}}
              for k, hi in enumerate(ci_highs)}
    return {"arms": {"decision": {"points": points}}}


def test_the_top_spans_both_policies_and_not_the_file_it_was_drawn_from():
    strict = {"games": {"trust": _measure_game([10.0, 20.0])}}
    relaxed = {"games": {"trust": _measure_game([10.0, 50.0])}}
    assert (make_figures.measure_tops(strict, ["trust"], relaxed)["trust"]
            == pytest.approx(50.0 * make_figures.MEASURE_HEADROOM))
    assert (make_figures.measure_tops(strict, ["trust"], relaxed)
            == make_figures.measure_tops(relaxed, ["trust"], strict))


def test_a_point_that_produced_no_interval_does_not_set_the_top():
    game = {"games": {"trust": _measure_game([10.0, None])}}
    assert (make_figures.measure_tops(game, ["trust"], None)["trust"]
            == pytest.approx(10.0 * make_figures.MEASURE_HEADROOM))


def test_a_game_the_other_policy_never_ran_keeps_its_own_top():
    strict = {"games": {"trust": _measure_game([10.0]),
                        "apology": _measure_game([40.0])}}
    relaxed = {"games": {"trust": _measure_game([90.0])}}
    tops = make_figures.measure_tops(strict, ["trust", "apology"], relaxed)
    assert tops["apology"] == pytest.approx(40.0 * make_figures.MEASURE_HEADROOM)


def test_the_top_is_per_game_and_never_one_limit_across_games():
    """Dollars given and fish caught are not one scale; §4 says so."""
    analysis = {"games": {"trust": _measure_game([10.0]),
                          "overfishing": _measure_game([100.0])}}
    tops = make_figures.measure_tops(analysis, ["trust", "overfishing"], None)
    assert tops["trust"] != tops["overfishing"]


# --- finding the other policy's analysis --------------------------------------

def test_the_sibling_is_the_other_policys_file_beside_this_one():
    strict = Path("/somewhere/analysis/steering.json")
    assert make_figures.sibling_analysis_path(strict, "strict") == (
        Path("/somewhere/analysis/steering_relaxed.json"))
    assert make_figures.sibling_analysis_path(
        Path("/somewhere/analysis/steering_relaxed.json"), "relaxed") == strict


def test_an_analysis_under_another_name_has_no_twin_rather_than_a_guessed_one():
    """Guessing would risk sharing an axis between two different sweeps."""
    assert make_figures.sibling_analysis_path(Path("/x/mine.json"), "strict") is None


def test_a_missing_sibling_is_stated_and_not_silently_a_per_file_axis(tmp_path):
    path = tmp_path / "steering.json"
    path.write_text("{}")
    sibling, why = make_figures.load_sibling(path, "strict")
    assert sibling is None
    assert "steering_relaxed.json" in why


def test_an_unreadable_sibling_is_stated_rather_than_crashed_on(tmp_path):
    (tmp_path / "steering.json").write_text("{}")
    (tmp_path / "steering_relaxed.json").write_text("{not json")
    sibling, why = make_figures.load_sibling(tmp_path / "steering.json", "strict")
    assert sibling is None
    assert "could not be read" in why


def test_a_sibling_carrying_the_same_policy_is_refused(tmp_path):
    for name in ("steering.json", "steering_relaxed.json"):
        (tmp_path / name).write_text(json.dumps({"policy": "strict"}))
    sibling, why = make_figures.load_sibling(tmp_path / "steering.json", "strict")
    assert sibling is None
    assert "'relaxed'" in why


def test_an_unshared_axis_says_so_on_the_figure_and_a_shared_one_says_it_is():
    shared, colour = make_figures.measure_axis_note(
        "strict", make_figures.MeasureLimits({}, None))
    assert "BOTH pole policies" in shared and "relaxed twin" in shared
    assert colour != make_figures.BAD
    warned, colour = make_figures.measure_axis_note(
        "strict", make_figures.MeasureLimits(
            {}, "steering_relaxed.json is not beside this analysis"))
    assert "NOT shared" in warned and "steering_relaxed.json" in warned
    assert colour == make_figures.BAD


# --- the shared axis on the committed pair ------------------------------------

def test_the_committed_pair_locate_each_other(analysis, relaxed):
    for name, policy in (("steering.json", "strict"),
                         ("steering_relaxed.json", "relaxed")):
        sibling, why = make_figures.load_sibling(
            PACKAGE / "analysis" / name, policy)
        assert why is None
        assert sibling["policy"] == make_figures.OTHER_POLICY[policy]


def test_a_game_gets_the_same_own_measure_axis_in_both_policies_figures(analysis,
                                                                        relaxed):
    """The property the operator asked for: same game, both policies, one axis."""
    strict_tops = make_figures.measure_tops(analysis, _drawn(analysis), relaxed)
    relaxed_tops = make_figures.measure_tops(relaxed, _drawn(relaxed), analysis)
    for game in _drawn(relaxed):
        assert strict_tops[game] == relaxed_tops[game], game


def test_every_shared_game_would_have_had_a_different_axis_per_file(analysis,
                                                                    relaxed):
    """Without this the axes really did differ - the test is not a no-op."""
    differ = [game for game in _drawn(relaxed)
              if make_figures.measure_top([analysis["games"][game]])
              != make_figures.measure_top([relaxed["games"][game]])]
    assert sorted(differ) == sorted(_drawn(relaxed))


def test_the_prisoners_dilemma_axis_does_not_move(analysis, relaxed):
    """Only strict ran it, and its own measure is already a proportion."""
    assert "prisoners_dilemma" not in relaxed["games"]
    tops = make_figures.measure_tops(analysis, _drawn(analysis), relaxed)
    assert tops["prisoners_dilemma"] == make_figures.measure_top(
        [analysis["games"]["prisoners_dilemma"]])
    assert 1.0 < tops["prisoners_dilemma"] < 1.2


# --- both policies on one set of axes -----------------------------------------

def test_the_combined_files_are_a_third_sibling_of_the_two_policy_names():
    names = [name for _build, name in make_figures.combined_figure_files()]
    assert names == ["steering_pole_shares_both_policies.png",
                     "steering_own_measure_both_policies.png",
                     "steering_vs_null_both_policies.png"]
    for policy in ("strict", "relaxed"):
        assert not set(names) & {name for _build, name
                                 in make_figures.figure_files(policy)}


def test_the_policies_are_drawn_in_a_fixed_order_whichever_run_drew_them():
    """Either invocation writes the combined figures, so they must not differ."""
    strict = {"games": {"trust": "strict game"}}
    relaxed = {"games": {"trust": "relaxed game"}}
    forward = make_figures.drawn_policies("trust", {"strict": strict,
                                                    "relaxed": relaxed})
    backward = make_figures.drawn_policies("trust", {"relaxed": relaxed,
                                                     "strict": strict})
    assert forward == backward == [("strict", "strict game"),
                                   ("relaxed", "relaxed game")]


def test_a_game_only_one_policy_ran_still_gets_a_panel(analysis, relaxed):
    by_policy = {"strict": analysis, "relaxed": relaxed}
    assert make_figures.combined_games(by_policy) == list(make_figures.GAMES)
    assert make_figures.drawn_policies("prisoners_dilemma", by_policy) == [
        ("strict", analysis["games"]["prisoners_dilemma"])]


def test_the_combined_own_measure_axis_is_the_per_policy_shared_one(analysis,
                                                                    relaxed):
    by_policy = {"strict": analysis, "relaxed": relaxed}
    tops = make_figures.measure_tops(analysis, _drawn(analysis), relaxed)
    for game in make_figures.combined_games(by_policy):
        assert make_figures.combined_measure_top(game, by_policy) == tops[game]


def test_a_combined_panel_draws_both_arms_of_both_policies(analysis, relaxed):
    """Four series, because dropping the nulls would hide which arm moved."""
    import matplotlib.pyplot as plt
    by_policy = {"strict": analysis, "relaxed": relaxed}
    fig, ax = plt.subplots()
    make_figures._combined_panel(
        ax, "dictator", make_figures.drawn_policies("dictator", by_policy),
        make_figures._pole_getter, "P(altruistic pole)", (-0.05, 1.08))
    assert len(ax.containers) == 4
    plt.close(fig)


def test_the_prisoners_dilemma_panel_draws_one_pair_and_says_why(analysis,
                                                                 relaxed):
    """Two coincident pairs would read as a plotting fault; it is vector reuse."""
    import matplotlib.pyplot as plt
    by_policy = {"strict": analysis, "relaxed": relaxed}
    fig, ax = plt.subplots()
    make_figures._combined_panel(
        ax, "prisoners_dilemma",
        make_figures.drawn_policies("prisoners_dilemma", by_policy),
        make_figures._pole_getter, "P(altruistic pole)", (-0.05, 1.08))
    assert len(ax.containers) == 2
    assert any(make_figures.REUSED_NOTE in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_each_policys_null_keeps_its_own_degradation_marks(analysis, relaxed):
    """The Ultimatum's strict null fails at k=-5 and its relaxed null at k=+5.

    A panel-level mark would say the game is unreadable at both ends, which is
    the opposite of what these two runs show.
    """
    by_policy = {"strict": analysis, "relaxed": relaxed}
    flagged = {}
    for policy, game in make_figures.drawn_policies("ultimatum", by_policy):
        series = make_figures._series(game["arms"]["shuffled-null"],
                                      make_figures._pole_getter(policy))
        flagged[policy] = sorted(point["k"] for point in series
                                 if point["degenerate"] or point["low_coverage"])
    assert flagged == {"strict": [-5], "relaxed": [5]}


def test_the_policies_wear_different_colours_in_the_combined_figures():
    assert (make_figures.POLICY_COLOUR["strict"]
            != make_figures.POLICY_COLOUR["relaxed"])
    assert make_figures.POLICY_COLOUR["strict"] == make_figures.DECISION
