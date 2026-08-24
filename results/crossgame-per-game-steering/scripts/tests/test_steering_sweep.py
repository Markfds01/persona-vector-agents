"""The parts of this package that can be wrong without anything noticing.

No GPU, no weights, no activations. What is exercised here is the code that
decides WHICH points run, WHICH rows count as a pole, and WHETHER a curve is
called monotone — three places where a silent mistake produces a plausible
number rather than an error.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "prompting"))
sys.path.insert(0, str(SCRIPTS / "measurement"))
sys.path.insert(0, str(SCRIPTS.parents[2]))

import eval_games  # noqa: E402
import stats  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


run_sweep = _load("run_sweep", SCRIPTS / "prompting" / "run_sweep.py")
build_nulls = _load("build_nulls", SCRIPTS / "measurement" / "build_nulls.py")
analyze_game = _load("analyze_game", SCRIPTS / "measurement" / "analyze_game.py")


# --- the ladder ---------------------------------------------------------------

def test_the_ladder_is_symmetric_and_starts_at_zero():
    assert run_sweep.REAL_KS[0] == 0
    assert sorted(run_sweep.REAL_KS) == list(range(-5, 6))
    assert set(run_sweep.NULL_KS) <= set(run_sweep.REAL_KS)


def test_selected_ks_keeps_the_ladders_own_order():
    assert run_sweep._selected_ks("5,-1,0", run_sweep.REAL_KS, "--ks") == [0, -1, 5]


def test_selected_ks_defaults_to_the_whole_ladder():
    assert run_sweep._selected_ks("  ", run_sweep.REAL_KS, "--ks") == list(run_sweep.REAL_KS)


def test_a_k_off_the_ladder_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        run_sweep._selected_ks("6", run_sweep.REAL_KS, "--ks")
    assert "6" in str(excinfo.value)


def test_the_unit_beta_of_k_is_k_times_the_reference_norm(tmp_path):
    vector = torch.zeros(29, 4)
    vector[20] = torch.tensor([3.0, 4.0, 0.0, 0.0])
    path = tmp_path / "reference.pt"
    torch.save(vector, path)
    assert run_sweep.reference_norm(path, 20) == pytest.approx(5.0)


def test_a_subset_spec_that_names_no_rung_is_refused_rather_than_run_empty():
    """An arm with no points fails inside run_arm, after the other arm's hours."""
    with pytest.raises(SystemExit) as excinfo:
        run_sweep._selected_ks(",", run_sweep.REAL_KS, "--ks")
    assert "names no rung" in str(excinfo.value)


# --- resume -------------------------------------------------------------------

#: The sha the rows in these fixtures were "steered with", and another one.
VECTOR = "0123456789abcdef"
OTHER_VECTOR = "fedcba9876543210"


def _write_rows(path, n, sha=VECTOR):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["value", "tag", "steer_vector_sha256"])
        writer.writeheader()
        for index in range(n):
            writer.writerow({"value": index, "tag": "bare",
                             "steer_vector_sha256": sha})


def test_a_finished_point_is_skipped_and_a_short_one_is_not(tmp_path):
    full = tmp_path / "full.csv"
    short = tmp_path / "short.csv"
    _write_rows(full, 20)
    _write_rows(short, 19)
    assert run_sweep.completed_rows(full, 20, VECTOR) is True
    assert run_sweep.completed_rows(short, 20, VECTOR) is False
    assert run_sweep.completed_rows(tmp_path / "absent.csv", 20, VECTOR) is False


def test_a_point_with_extra_rows_is_not_treated_as_finished(tmp_path):
    """A file that is not exactly this run's point is regenerated, not trusted."""
    path = tmp_path / "long.csv"
    _write_rows(path, 21)
    assert run_sweep.completed_rows(path, 20, VECTOR) is False


def test_an_unreadable_csv_is_regenerated_rather_than_reused(tmp_path):
    path = tmp_path / "nul.csv"
    path.write_bytes(b"value,tag\n1,bare\n\x00\n")
    assert run_sweep.completed_rows(path, 2, VECTOR) is False


def test_a_whole_point_from_another_vector_is_refused_not_resumed(tmp_path):
    """The relaxed round over the strict rows directory: the path cannot tell them
    apart, so the rows are asked instead."""
    path = tmp_path / "strict_rows.csv"
    _write_rows(path, 20, sha=OTHER_VECTOR)
    with pytest.raises(SystemExit) as excinfo:
        run_sweep.completed_rows(path, 20, VECTOR)
    message = str(excinfo.value)
    assert OTHER_VECTOR in message and VECTOR in message


def test_rows_that_never_named_a_vector_are_refused_too(tmp_path):
    path = tmp_path / "unnamed.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["value", "tag"])
        writer.writeheader()
        writer.writerows([{"value": 1, "tag": "bare"}] * 20)
    with pytest.raises(SystemExit):
        run_sweep.completed_rows(path, 20, VECTOR)


def test_the_coefficient_manifest_is_replaced_whole_or_not_at_all(tmp_path):
    """It is rewritten after every game and read by every stage downstream."""
    path = tmp_path / "coefficients.csv"
    path.write_text("the previous game's manifest\n")
    record = {"games": {"dictator": {"coefficients": [{"family": "dictator",
                                                       "k": 0}]}}}
    assert run_sweep.write_coefficients(path, record) == [{"family": "dictator",
                                                           "k": 0}]
    assert path.read_text().splitlines()[0] == "family,k"
    assert not list(tmp_path.glob("*.tmp"))


def test_the_steering_columns_still_carry_the_sha_the_resume_check_reads():
    """`completed_rows` reads a column `audit.steer` owns; a rename would make
    every resumed point look like another vector's."""
    assert "steer_vector_sha256" in run_sweep.steer.STEERING_FIELDS


# --- the poles ----------------------------------------------------------------

@pytest.mark.parametrize("family", ["dictator", "trust", "ultimatum", "apology"])
def test_an_amount_games_poles_are_zero_and_half_under_strict(family):
    assert eval_games.classify(family, 0.0, "strict") == "self_interested"
    assert eval_games.classify(family, 50.0, "strict") == "altruistic"
    assert eval_games.classify(family, 49.0, "strict") == "middle"
    assert eval_games.classify(family, 1.0, "strict") == "middle"


@pytest.mark.parametrize("family", ["dictator", "trust", "ultimatum", "apology"])
def test_relaxed_widens_the_self_pole_below_a_tenth_of_the_endowment(family):
    assert eval_games.classify(family, 9.0, "relaxed") == "self_interested"
    assert eval_games.classify(family, 11.0, "relaxed") == "middle"
    assert eval_games.classify(family, 9.0, "strict") == "middle"


@pytest.mark.parametrize("family", ["dictator", "trust", "ultimatum", "apology"])
def test_exactly_a_tenth_falls_outside_the_relaxed_self_pole(family):
    """Recorded, not corrected: `1.0 - 0.9` is 0.09999999999999998 in binary, so a
    transfer of exactly a tenth misses a threshold whose prose includes it. It is
    the boundary the committed relaxed vectors were built under."""
    assert eval_games.classify(family, 10.0, "relaxed") == "middle"


def test_overfishing_splits_at_the_sustainable_half_not_at_the_answer_bound():
    # the question is sustainable while the TWO agents together take at most 100,
    # so one agent's restrained share is 50 and the maximum catch is 100
    assert eval_games.classify("overfishing", 50.0, "strict") == "altruistic"
    assert eval_games.classify("overfishing", 51.0, "strict") == "middle"
    assert eval_games.classify("overfishing", 100.0, "strict") == "self_interested"
    assert eval_games.classify("overfishing", 51.0, "relaxed") == "self_interested"


def test_the_prisoners_dilemma_has_two_points_and_no_middle():
    for policy in ("strict", "relaxed"):
        assert eval_games.classify("prisoners_dilemma", 1.0, policy) == "altruistic"
        assert eval_games.classify("prisoners_dilemma", 0.0, policy) == "self_interested"


def test_overfishing_is_the_one_game_whose_own_measure_falls_with_restraint():
    low = [f for f in eval_games.FAMILIES if not eval_games.ALTRUISTIC_IS_HIGH[f]]
    assert low == ["overfishing"]


def test_every_steered_game_is_a_shipped_altruism_v3_question():
    from audit import games

    for game in eval_games.GAMES:
        assert games.by_id(game.id).question_sha256 == game.question_sha256
    # the shipped registry is not shadowed by this module
    assert type(games.by_id("altruism_v3/dictator")) is games.Game


def test_an_amount_games_pole_scale_must_be_its_declared_endowment(monkeypatch):
    monkeypatch.setitem(eval_games.POLE_SCALES, "dictator", (50.0, "wrong"))
    with pytest.raises(ValueError, match="not the endowment"):
        eval_games._build()


# --- the null -----------------------------------------------------------------

def _labels(cell, count, position_start):
    return [{"cell": cell, "position": position_start + i} for i in range(count)]


def test_the_permutation_keeps_every_cells_two_pole_counts():
    alt = _labels("a", 3, 0) + _labels("b", 1, 10)
    self_ = _labels("a", 2, 20) + _labels("b", 4, 30)
    generator = torch.Generator().manual_seed(7)
    new_alt, new_self = build_nulls.permute_within_cells(alt, self_, generator)
    for cell, expected_alt, expected_self in (("a", 3, 2), ("b", 1, 4)):
        assert sum(1 for i in new_alt if i["cell"] == cell) == expected_alt
        assert sum(1 for i in new_self if i["cell"] == cell) == expected_self
    assert ({i["position"] for i in new_alt + new_self}
            == {i["position"] for i in alt + self_})


def test_the_corpus_is_named_by_its_leaf_and_never_by_where_it_sat():
    assert build_nulls.acts_label("/home/someone/private/acts_v3", "dictator") == \
        "acts_v3/dictator"
    assert build_nulls.acts_label("/", "dictator") == "./dictator"


def test_no_message_this_module_raises_carries_an_absolute_path():
    """The driver tees stdout into the committed provenance log, so a raise is a
    write into a public repo."""
    root = "/home/someone/private/acts_v3"
    committed = torch.zeros(29, 8)
    committed[20, 0] = 1.0
    rebuilt = committed.clone()
    rebuilt[20, 0] = 5.0
    with pytest.raises(SystemExit) as excinfo:
        build_nulls.check_rebuild("dictator", build_nulls.acts_label(root, "dictator"),
                                  committed, rebuilt)
    message = str(excinfo.value)
    assert "acts_v3/dictator" in message
    assert root not in message and "/home/" not in message


def test_the_permutation_moves_rows_ACROSS_the_two_poles():
    """The property that makes the null a null.

    A shuffle WITHIN each pole keeps every count, leaves single-pole cells alone
    and is a function of its seed - it passes the other three tests here - while
    reproducing the treatment arm exactly. Only a row changing pole separates the
    two, so that is what is asserted. Twenty-five against twenty-five: a uniform
    permutation that happened to keep the poles apart has probability 1/C(50,25).
    """
    alt, self_ = _labels("a", 25, 0), _labels("a", 25, 100)
    altruistic_before = {item["position"] for item in alt}
    for seed in (1, 2, 3):
        generator = torch.Generator().manual_seed(seed)
        new_alt, new_self = build_nulls.permute_within_cells(alt, self_, generator)
        assert any(item["position"] in altruistic_before for item in new_self), seed


def test_a_single_pole_cell_is_carried_through_untouched():
    alt = _labels("a", 2, 0) + _labels("only", 3, 50)
    self_ = _labels("a", 2, 20)
    generator = torch.Generator().manual_seed(1)
    new_alt, new_self = build_nulls.permute_within_cells(alt, self_, generator)
    assert [i["position"] for i in new_alt if i["cell"] == "only"] == [50, 51, 52]
    assert not [i for i in new_self if i["cell"] == "only"]


def test_the_permutation_is_a_function_of_its_seed():
    alt, self_ = _labels("a", 4, 0), _labels("a", 4, 10)
    runs = []
    for seed in (3, 3, 4):
        generator = torch.Generator().manual_seed(seed)
        new_alt, _ = build_nulls.permute_within_cells(alt, self_, generator)
        runs.append([i["position"] for i in new_alt])
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]


# --- statistics ---------------------------------------------------------------

def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    for successes, n in ((0, 200), (200, 200)):
        interval = stats.wilson(successes, n)
        assert 0.0 <= interval["lo"] <= interval["hi"] <= 1.0
    assert stats.wilson(0, 200)["p"] == 0.0
    assert stats.wilson(0, 0)["p"] is None


def test_newcombe_brackets_the_difference_and_reports_whether_it_clears_zero():
    same = stats.newcombe(100, 200, 100, 200)
    assert same["diff"] == 0.0
    assert same["excludes_zero"] is False
    apart = stats.newcombe(198, 200, 20, 200)
    assert apart["diff"] == pytest.approx(0.89)
    assert apart["excludes_zero"] is True
    assert apart["lo"] < apart["diff"] < apart["hi"]


def test_the_newcombe_interval_and_the_score_test_are_two_procedures():
    """They can disagree, so no verdict may be read off the p.

    Newcombe's hybrid-score interval is not the inversion of the pooled score
    test. The Prisoner's Dilemma's k=+1 move is one of the sixteen (k1, k2) pairs
    at n=100 where they part, and every one of the sixteen has the interval as the
    more conservative side - which is the side this package reads.
    """
    interval = stats.newcombe(4, 100, 0, 100)
    assert interval["excludes_zero"] is False
    assert interval["lo"] == pytest.approx(-0.0042808, abs=1e-6)
    assert stats.score_test(4, 100, 0, 100) == pytest.approx(0.0434, abs=1e-4)

    disagreeing = []
    for k1 in range(101):
        for k2 in range(101):
            p_value = stats.score_test(k1, 100, k2, 100)
            if p_value is None:
                continue
            clears = stats.newcombe(k1, 100, k2, 100)["excludes_zero"]
            if clears != (p_value < 0.05):
                disagreeing.append((k1, k2, clears))
    assert len(disagreeing) == 16
    assert all(clears is False for _k1, _k2, clears in disagreeing)


def test_welch_reports_no_difference_between_two_identical_samples():
    a = stats.describe([1.0, 2.0, 3.0, 4.0])
    result = stats.welch(a, a)
    assert result["diff"] == 0.0
    assert result["p"] == pytest.approx(1.0)


def test_the_t_quantile_matches_published_values():
    assert stats.t_ppf975(1) == pytest.approx(12.706, abs=1e-3)
    assert stats.t_ppf975(10) == pytest.approx(2.228, abs=1e-3)
    assert stats.t_ppf975(1000) == pytest.approx(1.962, abs=1e-3)


def test_spearman_handles_ties_and_a_constant_series():
    assert stats.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 2, 3, 4], [1, 1, 2, 2]) == pytest.approx(0.8944272, abs=1e-6)
    assert stats.spearman([1, 2, 3], [5, 5, 5]) is None


def test_degeneracy_counts_repetition_rather_than_rating_it():
    healthy = stats.degeneracy(["one two three four five six seven eight"])
    assert healthy["share_trigram_repeated_5x"] == 0.0
    looped = stats.degeneracy(["a b c " * 8])
    assert looped["share_trigram_repeated_5x"] == 1.0


# --- monotonicity -------------------------------------------------------------

def _point(share, n=200):
    successes = int(round(share * n))
    return {
        "measure": stats.describe([1.0] * successes + [0.0] * (n - successes)),
        "poles": {"strict": {"n_parsed": n,
                             "altruistic": stats.wilson(successes, n),
                             "self_interested": stats.wilson(n - successes, n),
                             "middle": stats.wilson(0, n)}},
    }


def test_a_clean_rising_curve_is_monotone_with_no_reversals():
    points = {k: _point(0.5 + 0.08 * k) for k in range(-5, 6)}
    result = analyze_game.monotonicity(points, "strict")
    assert result["altruistic"]["direction"] == "increasing"
    assert result["altruistic"]["strictly_monotone"] is True
    assert result["monotone_up_to_noise"] is True


def test_a_large_backward_step_is_reported_as_a_real_reversal():
    shares = {-1: 0.10, 0: 0.50, 1: 0.90, 2: 0.20, 3: 0.95}
    points = {k: _point(v) for k, v in shares.items()}
    result = analyze_game.monotonicity(points, "strict")
    assert result["monotone_up_to_noise"] is False
    assert [(r["from_k"], r["to_k"])
            for r in result["significant_reversals_on_altruistic_share"]] == [(1, 2)]


def test_a_wobble_inside_its_own_interval_is_not_called_a_reversal():
    shares = {-1: 0.10, 0: 0.40, 1: 0.42, 2: 0.41, 3: 0.80}
    points = {k: _point(v) for k, v in shares.items()}
    result = analyze_game.monotonicity(points, "strict")
    assert result["altruistic"]["strictly_monotone"] is False
    assert result["monotone_up_to_noise"] is True


def test_a_flat_curve_reports_no_direction_rather_than_a_false_one():
    points = {k: _point(0.5) for k in range(-2, 3)}
    result = analyze_game.monotonicity(points, "strict")
    assert result["altruistic"]["spearman_rho_vs_k"] is None
    assert result["monotone_up_to_noise"] is True


def test_a_point_the_scorer_could_not_read_at_all_reports_rather_than_crashes():
    """At a large beta a game can degenerate until nothing parses. That point has
    no mean and no share, and it must not take the rest of the curve with it."""
    empty = {"measure": stats.describe([]),
             "poles": {"strict": {"n_parsed": 0,
                                  "altruistic": stats.wilson(0, 0),
                                  "self_interested": stats.wilson(0, 0),
                                  "middle": stats.wilson(0, 0)}}}
    points = {k: _point(0.5 + 0.08 * k) for k in (-1, 0, 1)}
    points[2] = empty
    result = analyze_game.monotonicity(points, "strict")
    assert result["altruistic"]["available"] is False
    assert result["monotone_up_to_noise"] is True
    moves = analyze_game.move_from_zero(points, "strict")
    assert moves[2]["mean"]["diff"] is None
    assert moves[2]["altruistic"]["diff"] is None


# --- a 0/1 measure is a share, whatever the column is called -------------------

def test_the_prisoners_dilemma_is_the_one_game_whose_own_measure_is_binary():
    binary = [f for f in eval_games.FAMILIES if eval_games.measure_is_binary(f)]
    assert binary == ["prisoners_dilemma"]


def test_a_t_interval_on_a_zero_one_measure_runs_below_zero_and_wilson_does_not():
    """The shipped defect: P(Cooperate) at k=-5 was 0.030, CI [-0.0040, 0.0640]."""
    values = [1.0] * 3 + [0.0] * 97
    gaussian = stats.describe(values)
    binomial = stats.describe_binary(values)
    assert gaussian["ci_low"] < 0.0
    assert binomial["ci_low"] > 0.0
    assert binomial["mean"] == gaussian["mean"] == 0.03


def test_a_measure_every_answer_agrees_on_gets_an_interval_rather_than_a_point():
    """At k=0 the PD never cooperated; a t interval there has width exactly zero."""
    values = [0.0] * 100
    assert stats.describe(values)["ci_high"] == stats.describe(values)["ci_low"]
    binomial = stats.describe_binary(values)
    assert binomial["ci_high"] > 0.0
    assert binomial["ci_high"] == pytest.approx(0.036993, abs=1e-6)


def test_a_binary_summary_carries_no_wald_standard_error_to_be_misused():
    summary = stats.describe_binary([1.0, 0.0, 1.0])
    assert summary["se"] is None
    assert summary["estimator"] == "wilson"
    assert summary["successes"] == 2


def test_a_value_that_is_neither_zero_nor_one_is_refused_not_rounded():
    with pytest.raises(ValueError):
        stats.describe_binary([0.0, 0.5, 1.0])


def test_the_sample_sd_is_the_same_number_under_both_summaries():
    values = [1.0] * 24 + [0.0] * 76
    assert stats.describe_binary(values)["sd"] == stats.describe(values)["sd"]


# --- and its difference goes down the same path -------------------------------

def test_a_difference_of_two_shares_is_newcombe_and_a_score_test():
    a = stats.describe_binary([1.0] * 24 + [0.0] * 76)
    b = stats.describe_binary([0.0] * 100)
    contrast = stats.difference(a, b)
    assert contrast["estimator"] == "newcombe"
    assert contrast["diff"] == pytest.approx(0.24)
    assert contrast["ci_low"] > 0.0 and contrast["excludes_zero"]
    assert contrast["p"] == pytest.approx(1.7668604e-07, rel=1e-6)


def test_a_difference_of_two_means_is_still_welch():
    a = stats.describe([1.0, 2.0, 3.0, 4.0])
    b = stats.describe([2.0, 3.0, 4.0, 5.0])
    assert stats.difference(a, b)["estimator"] == "welch"


def test_differencing_a_share_against_a_mean_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        stats.difference(stats.describe_binary([1.0, 0.0]),
                         stats.describe([1.0, 2.0, 3.0]))


def test_welch_refuses_a_share_outright_rather_than_reporting_a_t_test_on_it():
    with pytest.raises(ValueError):
        stats.welch(stats.describe_binary([1.0, 0.0]),
                    stats.describe_binary([0.0, 0.0]))


def test_the_score_test_has_nothing_to_say_when_neither_sample_varies():
    assert stats.score_test(0, 100, 0, 100) is None
    assert stats.score_test(50, 50, 50, 50) is None


def test_a_binary_own_measure_and_its_altruistic_pole_are_the_same_number():
    """For the PD they must be: `poles._binary_pole` calls 1.0 the altruistic pole."""
    values = [1.0] * 62 + [0.0] * 38
    classified = [eval_games.classify("prisoners_dilemma", v, "strict") for v in values]
    share = stats.wilson(classified.count("altruistic"), len(classified))
    measure = stats.describe_binary(values)
    assert (measure["mean"], measure["ci_low"], measure["ci_high"]) == (
        share["p"], share["lo"], share["hi"])


# --- the shared no-op at beta = 0 ---------------------------------------------

def _beta0_pair(tmp_path, decision_rows, null_rows):
    coefficients = []
    for arm, rows in (("decision", decision_rows), ("shuffled-null", null_rows)):
        name = "%s_coef0.csv" % arm
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["continuation", "answer",
                                                        "value", "tag", "steer_vector"])
            writer.writeheader()
            writer.writerows(rows)
        coefficients.append({"family": "dictator", "arm": arm, "k": 0,
                             "rows_csv": name})
    return coefficients


def _row(continuation="I will give $50.", answer="50", vector="decision.pt"):
    return {"continuation": continuation, "answer": answer, "value": answer,
            "tag": "bare", "steer_vector": vector}


def test_the_two_arms_at_beta_zero_may_differ_only_in_which_vector_was_named(tmp_path):
    coefficients = _beta0_pair(tmp_path, [_row()], [_row(vector="null.pt")])
    assert analyze_game._beta0_generations_identical(tmp_path, coefficients,
                                                     "dictator") is True


def test_a_different_generation_at_beta_zero_is_caught_even_when_it_scores_the_same(
        tmp_path):
    """The old check compared the scored answer alone, so this pair passed it."""
    coefficients = _beta0_pair(
        tmp_path, [_row()], [_row(continuation="Fifty dollars, I think.")])
    assert analyze_game._beta0_generations_identical(tmp_path, coefficients,
                                                     "dictator") is False


def test_one_arm_missing_beta_zero_reports_nothing_rather_than_a_pass(tmp_path):
    coefficients = _beta0_pair(tmp_path, [_row()], [_row()])
    coefficients = [entry for entry in coefficients if entry["arm"] == "decision"]
    assert analyze_game._beta0_generations_identical(tmp_path, coefficients,
                                                     "dictator") is None


# --- the null-vector rebuild gate, made to fire --------------------------------

def test_the_rebuild_gate_passes_a_vector_that_reproduces():
    committed = torch.randn(29, 8, dtype=torch.float64)
    assert build_nulls.check_rebuild("dictator", "acts", committed,
                                     committed.clone()) == 0.0


def test_the_rebuild_gate_refuses_a_vector_that_does_not_reproduce():
    committed = torch.randn(29, 8, dtype=torch.float64)
    drifted = committed.clone()
    drifted[20] *= 1.001                      # one layer, a thousand times the floor
    with pytest.raises(SystemExit) as excinfo:
        build_nulls.check_rebuild("dictator", "acts", committed, drifted)
    assert "would not be a null of" in str(excinfo.value)


def test_the_rebuild_gate_is_per_layer_and_not_over_the_whole_tensor():
    """A cosine is scale-invariant and a whole-tensor norm drowns one layer."""
    committed = torch.ones(29, 64, dtype=torch.float64)
    drifted = committed.clone()
    drifted[3] *= 2.0
    assert build_nulls.rebuild_deviation(committed, drifted) == pytest.approx(1.0)
    with pytest.raises(SystemExit):
        build_nulls.check_rebuild("dictator", "acts", committed, drifted)


def test_a_rebuild_of_a_different_shape_stops_the_run_before_the_deviation():
    with pytest.raises(SystemExit) as excinfo:
        build_nulls.check_rebuild("dictator", "acts",
                                  torch.ones(29, 8, dtype=torch.float64),
                                  torch.ones(29, 16, dtype=torch.float64))
    assert "committed vector is" in str(excinfo.value)


def test_the_gate_tolerance_is_the_one_the_reports_were_accepted_under():
    assert build_nulls.REBUILD_TOLERANCE == 1e-6
