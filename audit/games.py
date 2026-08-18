"""One declaration per game: where its question lives, and what its answers may be.

The upstream repo never states a game's answer space anywhere a program can read.
The range exists only as prose inside the question text, which is how
`altruism_v1.json`'s Dictator game shipped saying "$10 endowment" and "from $0 to
$100" in the same sentence. Every scorer needs an upper bound, so that bound has
to be declared somewhere — here, pinned to the exact question text it was read
off (`question_sha256`), so an edit upstream fails loudly instead of silently
re-scoring against the wrong bound.

Where a question contradicts itself, both readings are kept
(`AnswerSpace.stated` and `.implied`) and the game is flagged. Nothing here
picks a winner: a later robustness pass will want to score both ways.

Question text is read from the upstream JSON at call time, never copied here.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple, Union

from audit import parse
from audit.paths import QUESTION_DIR


class DataMismatch(RuntimeError):
    """An upstream data file no longer matches what this module declares."""


@dataclass(frozen=True)
class IntegerRange:
    """An inclusive integer answer space, e.g. dollars 0-100."""

    low: int
    high: int

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("empty range: %d > %d" % (self.low, self.high))

    def contains(self, value) -> bool:
        return value is not None and self.low <= value <= self.high

    def __str__(self) -> str:
        return "%d-%d" % (self.low, self.high)


@dataclass(frozen=True)
class ActionSet:
    """A discrete answer space: each action and the value a scorer emits for it."""

    actions: Tuple[Tuple[str, float], ...]

    def contains(self, value) -> bool:
        return any(value == action_value for _name, action_value in self.actions)

    def __str__(self) -> str:
        return "/".join(name for name, _v in self.actions)


@dataclass(frozen=True)
class AnswerSpace:
    """What counts as a valid answer, under every reading the question supports.

    `stated` is the space the question's own answer prompt names. `implied` is
    set only when some other part of the question implies a different space; then
    `conflict` says how, and neither reading is preferred.
    """

    stated: Union[IntegerRange, ActionSet]
    implied: Optional[IntegerRange] = None
    conflict: str = ""

    def __post_init__(self):
        if bool(self.implied) != bool(self.conflict):
            raise ValueError("`implied` and `conflict` must be set together")

    @property
    def contradictory(self) -> bool:
        return self.implied is not None

    def readings(self) -> Tuple[Tuple[str, Union[IntegerRange, ActionSet]], ...]:
        """Every reading, as (name, space). One entry unless the question conflicts."""
        if self.implied is None:
            return (("stated", self.stated),)
        return (("stated", self.stated), ("implied", self.implied))

    def reading(self, name: str) -> Union[IntegerRange, ActionSet]:
        for candidate, space in self.readings():
            if candidate == name:
                return space
        raise KeyError("no reading %r; have %s"
                       % (name, [n for n, _s in self.readings()]))

    def contains(self, value) -> bool:
        """True when `value` is valid under at least one reading."""
        return any(space.contains(value) for _name, space in self.readings())


#: Scorer name -> the function in `audit.parse` that implements it.
SCORERS = {
    "amount": parse.extract_amount,
    "fish": parse.extract_fish,
    "binary_choice": parse.extract_cooperate,
}

#: Scorers that take an upper bound; `binary_choice` does not.
_BOUNDED_SCORERS = ("amount", "fish")


@dataclass(frozen=True)
class Game:
    """One question, scored one way, inside one declared answer space."""

    id: str
    family: str
    question_set: str
    question_index: int
    scorer: str
    answer_space: AnswerSpace
    question_sha256: str = field(repr=False)

    def __post_init__(self):
        if self.scorer not in SCORERS:
            raise ValueError("unknown scorer %r for %s" % (self.scorer, self.id))
        if (self.scorer in _BOUNDED_SCORERS) != isinstance(self.answer_space.stated, IntegerRange):
            raise ValueError("%s: scorer %r does not fit its answer space"
                             % (self.id, self.scorer))

    @property
    def question_id(self) -> str:
        """The `question_id` this game's rows carry in the result CSVs."""
        return "altruism_%d" % self.question_index

    @property
    def source_file(self):
        return QUESTION_DIR / ("%s.json" % self.question_set)

    def question(self) -> str:
        """The upstream question text, checked against the fingerprint declared here."""
        questions = _load_questions(self.question_set)
        if self.question_index >= len(questions):
            raise DataMismatch("%s: %s has only %d questions"
                               % (self.id, self.source_file, len(questions)))
        text = questions[self.question_index]
        digest = fingerprint(text)
        if digest != self.question_sha256:
            raise DataMismatch(
                "%s: question text changed upstream (%s != declared %s); re-read the "
                "question and re-declare its answer space"
                % (self.id, digest, self.question_sha256))
        return text


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=None)
def _load_questions(question_set: str) -> Tuple[str, ...]:
    path = QUESTION_DIR / ("%s.json" % question_set)
    if not path.exists():
        raise DataMismatch("missing question set: %s" % path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(payload["questions"])


def score(game: Game, answer: str, reading: str = "stated") -> parse.Extraction:
    """Score one raw response with this game's scorer, bounded by its answer space.

    `reading` selects which answer space to bound by; it only matters for a game
    whose question contradicts itself.
    """
    space = game.answer_space.reading(reading)
    scorer = SCORERS[game.scorer]
    if game.scorer not in _BOUNDED_SCORERS:
        return scorer(answer)
    if space.low != 0:
        raise NotImplementedError(
            "%s: scorers assume a 0-based answer space, got %s" % (game.id, space))
    return scorer(answer, space.high)


def _amount_game(question_set, index, family, sha, high, implied=None, conflict=""):
    space = AnswerSpace(IntegerRange(0, high),
                        IntegerRange(0, implied) if implied is not None else None,
                        conflict)
    return Game("%s/%s" % (question_set, family), family, question_set, index,
                "amount", space, sha)


#: Cooperate/defect, as the scorer reports them.
_COOPERATE_OR_DEFECT = AnswerSpace(ActionSet((("defect", 0.0), ("cooperate", 1.0))))

_V1_DICTATOR_CONFLICT = (
    "question gives Agent 1 an endowment of $10 but asks for an amount from $0 to "
    "$100; only $0-$10 is payable"
)

GAMES = (
    # altruism_v1 - the earlier draft, $10/$5 stakes. Used by the committed
    # eval_persona_eval/.../deprecated/v1 results (n=10).
    _amount_game("altruism_v1", 0, "dictator", "fa72234d97e5c4b6", 100,
                 implied=10, conflict=_V1_DICTATOR_CONFLICT),
    _amount_game("altruism_v1", 1, "trust", "3b48abe395260df9", 10),
    _amount_game("altruism_v1", 2, "ultimatum", "03afd280bfe3461c", 10),
    Game("altruism_v1/overfishing", "overfishing", "altruism_v1", 3, "fish",
         AnswerSpace(IntegerRange(0, 10)), "687021210b32e5c5"),
    Game("altruism_v1/prisoners_dilemma", "prisoners_dilemma", "altruism_v1", 4,
         "binary_choice", _COOPERATE_OR_DEFECT, "17a525066d4cd042"),
    _amount_game("altruism_v1", 5, "apology", "dc585c072e014538", 5),

    # altruism_v2 - four games, all $0-100. Its eval_prompt is the only upstream
    # judge prompt that asks for a dollar amount, so its `altruism` column is the
    # labelled set the scorers are measured against.
    _amount_game("altruism_v2", 0, "dictator", "ac8f242ed284c309", 100),
    _amount_game("altruism_v2", 1, "trust", "e2b5ded85cf30a5a", 100),
    _amount_game("altruism_v2", 2, "ultimatum", "88db37ffdd939d93", 100),
    _amount_game("altruism_v2", 3, "apology", "d037b79a3c33348b", 100),

    # altruism_v3 - the paper's Table 1 suite, all $0-100 / 0-100. Committed
    # under eval_persona_eval/.../v1 despite the directory name.
    _amount_game("altruism_v3", 0, "dictator", "ac8f242ed284c309", 100),
    _amount_game("altruism_v3", 1, "trust", "e2b5ded85cf30a5a", 100),
    _amount_game("altruism_v3", 2, "ultimatum", "88db37ffdd939d93", 100),
    Game("altruism_v3/overfishing", "overfishing", "altruism_v3", 3, "fish",
         AnswerSpace(IntegerRange(0, 100)), "c20287ba2d576e04"),
    Game("altruism_v3/prisoners_dilemma", "prisoners_dilemma", "altruism_v3", 4,
         "binary_choice", _COOPERATE_OR_DEFECT, "17a525066d4cd042"),
    _amount_game("altruism_v3", 5, "apology", "d037b79a3c33348b", 100),
)

GAMES_BY_ID = {game.id: game for game in GAMES}


def by_id(game_id: str) -> Game:
    try:
        return GAMES_BY_ID[game_id]
    except KeyError:
        raise KeyError("unknown game %r; have %s" % (game_id, sorted(GAMES_BY_ID)))


def for_question_set(question_set: str) -> Tuple[Game, ...]:
    """Every declared game from one question file, in question order."""
    return tuple(g for g in GAMES if g.question_set == question_set)
