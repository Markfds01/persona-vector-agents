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


# --- resume -------------------------------------------------------------------

def _write_rows(path, n):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["value", "tag"])
        writer.writeheader()
        for index in range(n):
            writer.writerow({"value": index, "tag": "bare"})


def test_a_finished_point_is_skipped_and_a_short_one_is_not(tmp_path):
    full = tmp_path / "full.csv"
    short = tmp_path / "short.csv"
    _write_rows(full, 20)
    _write_rows(short, 19)
    assert run_sweep.completed_rows(full, 20) is True
    assert run_sweep.completed_rows(short, 20) is False
    assert run_sweep.completed_rows(tmp_path / "absent.csv", 20) is False


def test_a_point_with_extra_rows_is_not_treated_as_finished(tmp_path):
    """A file that is not exactly this run's point is regenerated, not trusted."""
    path = tmp_path / "long.csv"
    _write_rows(path, 21)
    assert run_sweep.completed_rows(path, 20) is False


def test_an_unreadable_csv_is_regenerated_rather_than_reused(tmp_path):
    path = tmp_path / "nul.csv"
    path.write_bytes(b"value,tag\n1,bare\n\x00\n")
    assert run_sweep.completed_rows(path, 2) is False


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
