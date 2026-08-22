"""The six evaluation prompts this sweep steers, and the pole scale each needs.

The prompts are the paper's Table-1 suite, `altruism_v3`, one question per game,
mode `free` — the same prompt the Dictator-only sweep steered, so the six arms
here extend that run rather than starting a second convention. They are declared
in `audit.games` and are read from there untouched: nothing in this module edits,
re-registers or shadows a shipped game, so `audit.generate.plan` sees exactly the
declaration it always sees.

What is added is one number per game. `poles.classify` needs a `pole_scale` —
the endowment for an amount game, the season capacity for Overfishing, 1.0 for
the Prisoner's Dilemma, whose score is already 0/1 — and the shipped `Game` has
no such field, because the pole definition belongs to this investigation and not
to the upstream data. It is declared here, checked against the question's own
declared answer space at import, and used only when a finished row CSV is scored
into poles. Generation never sees it.

The scale is not free choice: it is read off the question text, and the question
text is fingerprinted by `audit.games`, so an upstream edit fails loudly in
`Game.question()` instead of silently re-scoring against the wrong bound.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

HERE = Path(__file__).resolve().parent
# poles.py lives with the vector study that built these vectors; the repo root is
# five levels up from this file
_VECTORS = HERE.parents[2] / "crossgame-decision-vectors" / "scripts" / "prompting"
sys.path.insert(0, str(_VECTORS))
sys.path.insert(0, str(HERE.parents[3]))

from audit import games  # noqa: E402

import poles  # noqa: E402

#: The question set the sweep runs on, and the games in the order they are run.
QUESTION_SET = "altruism_v3"
FAMILIES = ("dictator", "trust", "ultimatum", "apology", "overfishing",
            "prisoners_dilemma")

#: family -> (pole scale, the phrase in the question text it was read off).
#:
#: Overfishing is the one that is not simply the answer bound: the question makes
#: the season sustainable when the two agents' catches TOTAL at most 100, so the
#: capacity is 100 and one agent's sustainable share is 50, while the maximum
#: catch is the full 100. `poles._fish_pole` halves the scale to get that share.
POLE_SCALES = {
    "dictator": (100.0, "Agent 1 receives an endowment of $100"),
    "trust": (100.0, "Agent 1 receives $100"),
    "ultimatum": (100.0, "Agent 1 receives $100"),
    "apology": (100.0, "may transfer any whole-dollar amount from $0 to $100"),
    "overfishing": (100.0, "If the total caught in a season is <= 100"),
    "prisoners_dilemma": (1.0, "the binary score is already 0/1"),
}


@dataclass(frozen=True)
class EvalGame(games.Game):
    """A shipped `altruism_v3` game plus the pole scale its family is scored on.

    A subclass rather than a wrapper so `question()` keeps checking the upstream
    fingerprint and every scorer path stays the shipped one; `pole_scale` is the
    only thing this study adds.
    """

    pole_scale: float = 0.0
    #: where in the question text the scale was read
    scale_source: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.pole_scale <= 0:
            raise ValueError("%s: pole_scale must be positive" % self.id)
        if not self.scale_source:
            raise ValueError("%s: say where the pole scale was read from" % self.id)


def _build() -> Tuple[EvalGame, ...]:
    built = []
    for family in FAMILIES:
        shipped = games.by_id("%s/%s" % (QUESTION_SET, family))
        scale, source = POLE_SCALES[family]
        if shipped.scorer == "amount" and scale != shipped.answer_space.stated.high:
            raise ValueError(
                "%s: pole scale %g is not the endowment the answer space declares "
                "(%s); an amount game's scale IS its endowment"
                % (shipped.id, scale, shipped.answer_space.stated))
        built.append(EvalGame(
            id=shipped.id, family=shipped.family, question_set=shipped.question_set,
            question_index=shipped.question_index, scorer=shipped.scorer,
            answer_space=shipped.answer_space, question_sha256=shipped.question_sha256,
            pole_scale=scale, scale_source=source))
    return tuple(built)


GAMES = _build()
BY_FAMILY = {game.family: game for game in GAMES}


def by_family(family: str) -> EvalGame:
    try:
        return BY_FAMILY[family]
    except KeyError:
        raise KeyError("unknown family %r; have %s" % (family, list(BY_FAMILY)))


def classify(family: str, value: float, policy: str) -> str:
    """`poles.SELF` / `ALT` / `MIDDLE` for one scored value of this game."""
    return poles.classify(value, by_family(family), policy)


def measure_is_binary(family: str) -> bool:
    """Is this game's OWN measure a 0/1 outcome rather than an amount?

    The scorer decides it, so there is nothing extra to keep in step: a
    `binary_choice` game resolves to 0.0 or 1.0 and to nothing else
    (`poles._binary_pole` raises otherwise), which makes its mean a share and a
    t interval on it undefined at the ends of the ladder.
    """
    return by_family(family).scorer == "binary_choice"


def pairs(families=FAMILIES) -> Tuple[Tuple[str, str], ...]:
    """The `(game_id, mode)` pairs `audit.generate.run` takes, mode `free`."""
    return tuple((by_family(f).id, "free") for f in families)


#: Which direction of this game's own measure is the altruistic one. Amount games
#: and the PD score go up with restraint; a fish catch goes DOWN with it, so a
#: mean read without this is a sign error waiting to happen. Pole shares carry no
#: such trap and are what the cross-game tables compare.
ALTRUISTIC_IS_HIGH = {
    "dictator": True, "trust": True, "ultimatum": True, "apology": True,
    "overfishing": False, "prisoners_dilemma": True,
}

#: The unit each game's raw measure is in, for labelling a table.
MEASURE_UNITS = {
    "dictator": "dollars given", "trust": "dollars sent",
    "ultimatum": "dollars offered", "apology": "dollars transferred",
    "overfishing": "fish caught", "prisoners_dilemma": "P(Cooperate)",
}

for _family in FAMILIES:
    if _family not in ALTRUISTIC_IS_HIGH or _family not in MEASURE_UNITS:
        raise ImportError("%s: every family needs a measure direction and a unit"
                          % _family)
