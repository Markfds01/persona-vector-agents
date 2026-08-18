"""Turn a game into the exact string the model sees.

Elicitation is an experimental variable, not an implementation detail. The same
model on the same Dictator question gives an expectation near $40 or near $16
depending only on how the answer is asked for. So a mode is a named value that
travels with every result row, never a code path someone edits: add one to
`MODES` and no call site changes.

Nothing here loads a model, a tokenizer or a network resource. The chat template
arrives as a parameter, so the exact bytes are assertable against a fake.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

from audit.games import DataMismatch, Game, fingerprint
from audit.paths import QUESTION_DIR

#: The `Wording` fields a mode may pre-fill into the assistant turn.
PREFILL_KINDS = ("stub", "stub_explicit")


@dataclass(frozen=True)
class Wording:
    """The game-specific text the modes need.

    Modes differ in shape, games differ in wording: a stub has to name that game's
    recipient and unit, so "I will give $" is the Dictator's, not everyone's.

    A stub may not end in whitespace. BPE wants the space attached to the answer
    (" 12" is one token), so a stub ending on a lone space token is off
    distribution — and it would be so for some families and not others, which is
    exactly the artifact this module exists to measure.
    """

    strict_instruction: str
    stub: str
    stub_explicit: str

    def __post_init__(self):
        for kind in PREFILL_KINDS:
            text = getattr(self, kind)
            if text != text.rstrip():
                raise ValueError("%s stub ends in whitespace: %r" % (kind, text))

    def prefill(self, kind: str) -> str:
        if kind not in PREFILL_KINDS:
            raise KeyError("no prefill %r; have %s" % (kind, list(PREFILL_KINDS)))
        return getattr(self, kind)


_WHOLE_DOLLARS = "Respond with only the whole-dollar amount as a number, and nothing else."

#: Game family -> its wording. Only the Dictator pair is measured (the $40-vs-$16
#: result); the rest follow the same shape against each question's own verb.
WORDING = {
    "dictator": Wording(
        _WHOLE_DOLLARS,
        "I will give $",
        "The amount I will give to Agent 2 is $"),
    "trust": Wording(
        _WHOLE_DOLLARS,
        "I will send $",
        "The amount I will send to Agent 2 is $"),
    "ultimatum": Wording(
        _WHOLE_DOLLARS,
        "I will propose to give $",
        "The amount I will propose to give to Agent 2 is $"),
    "apology": Wording(
        _WHOLE_DOLLARS,
        "I will transfer $",
        "The amount I will transfer to Agent 2 is $"),
    "overfishing": Wording(
        "Respond with only the number of fish as a number, and nothing else.",
        "I will catch",
        "The number of fish I will catch in the first season is"),
    "prisoners_dilemma": Wording(
        "Respond with only C or D, and nothing else.",
        "I will choose",
        "My choice between Cooperate (C) and Defect (D) is"),
}


def wording_for(game: Game) -> Wording:
    try:
        return WORDING[game.family]
    except KeyError:
        raise KeyError("%s: no elicitation wording for family %r; declare one in "
                       "audit.elicit.WORDING" % (game.id, game.family))


@dataclass(frozen=True)
class Mode:
    """One way of asking for the answer — a value, not a code path.

    `instruct` appends the game's strict instruction to the user turn; `prefill`
    names the `Wording` field pre-filled into the assistant turn.
    """

    name: str
    instruct: bool = False
    prefill: Optional[str] = None

    def __post_init__(self):
        if self.prefill is not None and self.prefill not in PREFILL_KINDS:
            raise ValueError("%s: unknown prefill %r" % (self.name, self.prefill))


MODES = {mode.name: mode for mode in (
    #: the paper's prompt verbatim, answered in prose
    Mode("free"),
    #: plus "respond with only the number/action"
    Mode("strict", instruct=True),
    #: prose prompt, assistant turn opened for the model
    Mode("stub", prefill="stub"),
    #: same, with an opener that names the recipient and the unit
    Mode("stub_explicit", prefill="stub_explicit"),
)}


def mode_names() -> Tuple[str, ...]:
    """Every elicitation mode, in declaration order."""
    return tuple(MODES)


def by_name(name: str) -> Mode:
    try:
        return MODES[name]
    except KeyError:
        raise KeyError("unknown elicitation mode %r; have %s" % (name, list(MODES)))


@dataclass(frozen=True)
class Persona:
    """The upstream `persona_instruction_type` prefix, as a system turn.

    Off by default: `render` takes `persona=None`. `polarity` is "pos" or "neg",
    `index` selects one of the trait file's instruction pairs.
    """

    polarity: str
    index: int = 0
    assistant_name: Optional[str] = None

    def __post_init__(self):
        if self.polarity not in ("pos", "neg"):
            raise ValueError("persona polarity must be 'pos' or 'neg', got %r" % (self.polarity,))
        if self.index < 0:
            raise ValueError("persona index must be >= 0, got %d" % self.index)

    @property
    def label(self) -> str:
        """What the result row records, e.g. `pos_0`."""
        return "%s_%d" % (self.polarity, self.index)

    def system_prompt(self, question_set: str) -> str:
        """The system turn upstream builds for this polarity, byte for byte.

        Upstream names the assistant after the trait *file*, so `altruism_v3`/pos
        yields "an altruism_v3 assistant". Kept for parity; `assistant_name`
        overrides it.
        """
        instructions = _load_instructions(question_set)
        if self.index >= len(instructions):
            raise DataMismatch("%s declares %d persona instructions; asked for index %d"
                               % (question_set, len(instructions), self.index))
        instruction = instructions[self.index].get(self.polarity)
        if not instruction:
            raise DataMismatch("%s instruction %d has no %r text"
                               % (question_set, self.index, self.polarity))
        name = self.assistant_name
        if name is None:
            name = question_set if self.polarity == "pos" else "helpful"
        return "You are %s %s assistant. %s" % (_a_or_an(name), name, instruction)


@dataclass(frozen=True)
class RenderedPrompt:
    """The exact string the model sees, and where its answer begins.

    `text` is the chat template's output followed by whatever the mode pre-filled
    into the assistant turn, so the answer begins at the end of `text` by
    construction. `with_answer` is the seam a logits pass scores through.
    """

    game_id: str
    mode: str
    persona: str
    messages: Tuple[Dict[str, str], ...]
    generation_prompt: str
    assistant_prefill: str

    @property
    def text(self) -> str:
        return self.generation_prompt + self.assistant_prefill

    @property
    def answer_char_offset(self) -> int:
        """Where the answer starts, as a CHARACTER index. Not a token index.

        A token-level pass cannot use this directly: see `with_answer`.
        """
        return len(self.text)

    def with_answer(self, answer: str) -> str:
        """`text` continued by `answer` — the string a logits pass scores.

        Tokenizing a candidate on its own is wrong, and so is assuming the
        prompt's tokens are a prefix of this string's tokens: the tokenizer merges
        across the boundary (measured — a stub ending in a space plus "C" merges
        into a single " C"). A caller must check prefix stability per candidate,
        or work from the tokenizer's offset mapping, and report the candidates
        where it fails rather than scoring them anyway.
        """
        return self.text + answer

    @property
    def sha256(self) -> str:
        """Fingerprint of `text`, so a row can prove which string produced it."""
        return fingerprint(self.text)


def render(game: Game, mode: str, tokenizer, persona: Optional[Persona] = None) -> RenderedPrompt:
    """Render one (game, mode) into the string the model sees.

    `tokenizer` need only provide
    `apply_chat_template(messages, tokenize=False, add_generation_prompt=True) -> str`;
    nothing is downloaded here. A stubbed assistant turn is appended after the
    generation prompt rather than passed as an assistant message, because the
    chat template would close that turn with `<|im_end|>`.
    """
    selected = by_name(mode)
    wording = wording_for(game)

    user = game.question()
    if selected.instruct:
        user = "%s\n\n%s" % (user, wording.strict_instruction)

    messages = []
    if persona is not None:
        messages.append({"role": "system", "content": persona.system_prompt(game.question_set)})
    messages.append({"role": "user", "content": user})

    generation_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(generation_prompt, str):
        raise TypeError("apply_chat_template returned %s, not a string; this module "
                        "renders text only" % type(generation_prompt).__name__)

    prefill = wording.prefill(selected.prefill) if selected.prefill else ""
    return RenderedPrompt(game.id, selected.name, persona.label if persona else "",
                          tuple(messages), generation_prompt, prefill)


def chat_template_fingerprint(tokenizer) -> str:
    """Fingerprint the tokenizer's chat template.

    A fake tokenizer cannot notice that an upstream template changed; recording
    this beside every result row is what makes such a change visible.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("tokenizer exposes no chat_template; the rendered prompt "
                         "would be unattributable")
    if not isinstance(template, str):
        template = json.dumps(template, sort_keys=True)
    return fingerprint(template)


def _a_or_an(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


@lru_cache(maxsize=None)
def _load_instructions(question_set: str) -> Tuple[Dict[str, str], ...]:
    path = QUESTION_DIR / ("%s.json" % question_set)
    if not path.exists():
        raise DataMismatch("missing question set: %s" % path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    instructions = payload.get("instruction")
    if not instructions:
        raise DataMismatch("%s declares no persona instructions" % path)
    return tuple(instructions)
