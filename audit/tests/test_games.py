"""The declarations in `audit.games` are checked against the upstream data files."""

import csv
import json
import re

import pytest

from audit.games import (GAMES, AnswerSpace, DataMismatch, Game, IntegerRange,
                         by_id, fingerprint, for_question_set, score)
from audit.paths import QUESTION_DIR
from audit.tests.conftest import V2_DIR

#: "(0-100)" as the question's own answer prompt writes its range.
_STATED_RANGE_RX = re.compile(r"\((\d+)\s*-\s*(\d+)\)")


def test_game_ids_are_unique():
    ids = [game.id for game in GAMES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("game", GAMES, ids=lambda g: g.id)
def test_question_text_still_matches_the_declared_fingerprint(game):
    assert fingerprint(game.question()) == game.question_sha256


@pytest.mark.parametrize("game", GAMES, ids=lambda g: g.id)
def test_declared_range_is_the_range_the_question_states(game):
    if not isinstance(game.answer_space.stated, IntegerRange):
        pytest.skip("discrete answer space")
    stated = game.answer_space.stated
    in_question = _STATED_RANGE_RX.findall(game.question())
    assert (str(stated.low), str(stated.high)) in in_question, (
        "%s declares %s but the question writes %s" % (game.id, stated, in_question))


@pytest.mark.parametrize("game", GAMES, ids=lambda g: g.id)
def test_contradictory_game_keeps_both_readings(game):
    space = game.answer_space
    if not space.contradictory:
        assert [name for name, _s in space.readings()] == ["stated"]
        return
    assert [name for name, _s in space.readings()] == ["stated", "implied"]
    assert space.conflict
    assert space.implied != space.stated
    # the losing reading has to be traceable to the question, not invented here
    assert "$%d" % space.implied.high in game.question()


def test_the_v1_dictator_is_the_known_self_contradictory_question():
    contradictory = [game.id for game in GAMES if game.answer_space.contradictory]
    assert contradictory == ["altruism_v1/dictator"]

    space = by_id("altruism_v1/dictator").answer_space
    assert space.stated == IntegerRange(0, 100)     # "any whole-dollar amount from $0 to $100"
    assert space.implied == IntegerRange(0, 10)     # "an endowment of $10"
    assert space.contains(100) and space.contains(10)


def test_scoring_is_bounded_by_the_chosen_reading():
    game = by_id("altruism_v1/dictator")
    answer = "I would give $60 to Agent 2."
    assert score(game, answer, "stated").value == 60.0
    # unpayable under the $10 endowment, so it resolves to nothing rather than a guess
    assert score(game, answer, "implied").value is None


def test_unknown_reading_is_rejected():
    with pytest.raises(KeyError):
        score(by_id("altruism_v2/dictator"), "I give $5.", "implied")


def test_scorer_must_fit_its_answer_space():
    with pytest.raises(ValueError):
        # a discrete-choice scorer cannot own a numeric range
        Game("x/y", "y", "altruism_v2", 0, "binary_choice",
             AnswerSpace(IntegerRange(0, 100)), "0" * 16)
    with pytest.raises(ValueError):
        Game("x/y", "y", "altruism_v2", 0, "no_such_scorer",
             AnswerSpace(IntegerRange(0, 100)), "0" * 16)


def test_a_changed_question_fails_loudly():
    game = by_id("altruism_v2/dictator")
    moved = Game(game.id, game.family, game.question_set, game.question_index,
                 game.scorer, game.answer_space, "deadbeefdeadbeef")
    with pytest.raises(DataMismatch):
        moved.question()


def test_v2_result_files_carry_exactly_the_declared_questions():
    declared = {game.question_id for game in for_question_set("altruism_v2")}
    path = next(iter(sorted(V2_DIR.glob("altruism_steer_*_game_v2.csv"))))
    with path.open(encoding="utf-8", newline="") as handle:
        found = {row["question_id"] for row in csv.DictReader(handle)}
    assert found == declared


def test_per_question_judge_rows_reuse_the_v3_questions():
    """`altruism.json`'s Agent-1 questions are byte-identical to `altruism_v3`.

    That identity is why the fish and cooperate agreement tests may score
    `*_compare_with_expectation.csv` rows against the `altruism_v3` declarations.
    """
    with (QUESTION_DIR / "altruism.json").open(encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    for game in for_question_set("altruism_v3"):
        assert questions[game.question_index]["text"] == game.question()
