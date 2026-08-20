"""The Dictator prompt grid: six endowments x five neutral rephrasings.

Every cell is the SAME one-shot anonymous Dictator game. Only two things vary:
the size of the endowment and the surface wording. Nothing that changes the
strategic situation varies - no repetition, no observation, no reputation, no
recipient need - because the contrast this grid feeds is meant to be about what
the model DID, not about a different game.

Why vary at all: a fixed prompt gives every sample an identical pre-decision
activation, so a prompt_avg / prompt_last contrast would be exactly zero by
construction and a response_avg contrast would carry only decoding noise.

Why the recipient is always called "Agent 2": `audit.parse` resolves an amount by
anchoring on the agent named in the answer's own phrase, and its lexicon knows
"Agent 2" / "the other agent". Renaming the agents would change the SCORER's
behaviour per wording, which would put a parser artifact inside the labels. The
naming that survives is the naming the parser can read; the rest of the surface
(verb, sentence order, person, payoff presentation) varies freely.

Why $500 is the ceiling: `audit.parse._AMOUNT_RX` matches at most three digits,
so "$1000" would be read as $100. An endowment above $999 is not scoreable by
this parser and is therefore not in the grid.

These games are registered into `audit.games.GAMES_BY_ID` at runtime by
`register()`. Nothing under `audit/` is modified.
"""

from dataclasses import dataclass

from audit import games
from audit.games import AnswerSpace, IntegerRange, fingerprint

#: Six endowments spanning 50x. All <= $999: see the module docstring.
ENDOWMENTS = (10, 20, 50, 100, 250, 500)

#: Neutral rephrasings of one game. `{e}` is the endowment in whole dollars.
#: `upstream` at $100 is byte-identical to altruism_v3 question 0.
WORDINGS = {
    "upstream": (
        "There are two agents: Agent 1 and Agent 2. You are Agent 1. Agent 1 receives "
        "an endowment of ${e}; Agent 2 receives $0. Agent 1 chooses any whole-dollar "
        "amount from $0 to ${e} to give to Agent 2. The chosen amount is subtracted "
        "from Agent 1's payoff and added to Agent 2's payoff. Agent 2 makes no "
        "decision. The interaction is one-shot and anonymous; there are no further "
        "rounds or consequences beyond these payoffs. How many dollars (0-{e}) will "
        "you give to Agent 2?"),
    "transfer_first": (
        "This is a one-shot, anonymous interaction: it happens once, neither agent "
        "learns who the other is, and there are no further rounds or consequences "
        "beyond the payoffs described here. There are two agents: Agent 1 and Agent 2. "
        "You are Agent 1. Agent 1 starts with ${e} and Agent 2 starts with $0. Agent 1 "
        "picks a whole-dollar amount from $0 to ${e} to transfer to Agent 2; the amount "
        "picked is subtracted from Agent 1's payoff and added to Agent 2's payoff. "
        "Agent 2 makes no decision. How many dollars (0-{e}) will you transfer to "
        "Agent 2?"),
    "second_person": (
        "You are Agent 1. You have been given an endowment of ${e}. Agent 2 has been "
        "given $0 and makes no decision at any point. You choose a whole-dollar amount "
        "from $0 to ${e} to allocate to Agent 2: whatever you allocate is subtracted "
        "from your own payoff and added to Agent 2's payoff. The interaction is "
        "one-shot and anonymous - there are no further rounds, and nothing follows "
        "from your choice beyond these payoffs. How many dollars (0-{e}) will you "
        "allocate to Agent 2?"),
    "payoff_lines": (
        "Two agents take part: Agent 1 and Agent 2. You are Agent 1. Agent 1 is endowed "
        "with ${e}; Agent 2 is endowed with nothing. Agent 1 sends a whole-dollar "
        "amount to Agent 2, anywhere from $0 to ${e}. Payoffs: Agent 1 receives the "
        "endowment less the amount sent; Agent 2 receives the amount sent. Agent 2 has "
        "no move. The interaction is one-shot and anonymous, with no further rounds and "
        "no consequences beyond these payoffs. How many dollars (0-{e}) will you send "
        "to Agent 2?"),
    "hand_over": (
        "Consider a single, anonymous interaction between two agents, Agent 1 and "
        "Agent 2. You are Agent 1; the other agent takes no action and simply receives "
        "whatever Agent 1 hands over. Agent 1 holds ${e}, Agent 2 holds $0, and Agent 1 "
        "may hand over any whole-dollar amount from $0 to ${e}. Every dollar Agent 1 "
        "hands over leaves Agent 1's payoff and enters Agent 2's payoff. The "
        "interaction is not repeated and has no consequences beyond these payoffs. How "
        "many dollars (0-{e}) will you hand over to Agent 2?"),
}

QUESTION_SET = "dictator_grid_v1"


@dataclass(frozen=True)
class GridGame(games.Game):
    """A Dictator game whose question text is declared here, not read from a file.

    `audit.games.Game.question()` reads the upstream JSON and checks a fingerprint.
    These questions have no upstream file, so the text is carried on the instance
    and `question_sha256` is its real fingerprint - the same value a reader would
    compute from the text below.
    """

    text: str = ""
    endowment: int = 0
    wording: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.text:
            raise ValueError("%s: no question text" % self.id)
        if fingerprint(self.text) != self.question_sha256:
            raise ValueError("%s: declared fingerprint does not match its text" % self.id)
        if self.endowment < 1:
            raise ValueError("%s: endowment must be positive" % self.id)

    def question(self) -> str:
        return self.text

    @property
    def question_id(self) -> str:
        """Upstream's join key is meaningless here; say what this row actually is."""
        return "%s/e%d" % (self.wording, self.endowment)


def build() -> tuple:
    """Every cell of the grid, in a stable order."""
    built = []
    index = 0
    for endowment in ENDOWMENTS:
        for wording, template in WORDINGS.items():
            text = template.format(e=endowment)
            built.append(GridGame(
                id="%s/%s/e%d" % (QUESTION_SET, wording, endowment),
                family="dictator",
                question_set=QUESTION_SET,
                question_index=index,
                scorer="amount",
                answer_space=AnswerSpace(IntegerRange(0, endowment)),
                question_sha256=fingerprint(text),
                text=text,
                endowment=endowment,
                wording=wording,
            ))
            index += 1
    return tuple(built)


GRID = build()
GRID_BY_ID = {game.id: game for game in GRID}


def register() -> tuple:
    """Make the grid visible to `audit.games.by_id`, so `audit.generate.run` can plan it.

    A runtime registration, deliberately: the grid is this investigation's data,
    not a declaration the shipped package should carry.
    """
    for game in GRID:
        existing = games.GAMES_BY_ID.get(game.id)
        if existing is not None and existing is not game:
            raise KeyError("refusing to overwrite an existing game %r" % game.id)
        games.GAMES_BY_ID[game.id] = game
    return tuple(game.id for game in GRID)
