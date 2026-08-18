"""Deterministic scorers for the game responses — a free stand-in for the paid judge.

Each scorer reads one raw generation and returns an `Extraction(value, tag)`:

* `value` is the action the response commits to — dollars given to Agent 2, fish
  caught in season one, or 1.0/0.0 for cooperate/defect.
* `tag` names how that value was resolved, so a downstream analysis can weigh or
  drop a resolution path.

An unresolvable response returns `value=None` with a tag saying why. It is never
a silent zero and never a guess — that is the whole advantage over an LLM judge,
which always answers with a plausible-looking number.

Ported unchanged in behavior from the baseline-parity investigation, whose
measured agreement with the paid gpt-4.1-mini judge is the reason this code is
here. Do not change a pattern without re-running `audit/tests/test_judge_agreement.py`.
"""

import re
from typing import NamedTuple, Optional


class Extraction(NamedTuple):
    """A scored response. `value is None` exactly when `tag` is unresolved."""

    value: Optional[float]
    tag: str


#: Nothing was resolved; these are the only tags that carry `value=None`.
UNRESOLVED_TAGS = frozenset({"empty", "refusal", "unparsed"})

#: How `extract_amount` resolved a dollar figure.
AMOUNT_TAGS = frozenset({
    "a2_anchor",   # the amount's own phrase names Agent 2
    "a2_near",     # Agent 2 is only the nearest mention, not in-phrase
    "complement",  # the clause named Agent 1's share; we took pot minus it
    "verb_obj",    # object of a give/send/transfer verb
    "keep",        # a keep-amount; we took pot minus it
    "answer_is",   # "the answer is $x"
    "bare",        # a short clause that is just the amount
    "bare_int",    # a trailing clause that is just an integer
}) | UNRESOLVED_TAGS

#: How `extract_fish` resolved a catch count.
FISH_TAGS = frozenset({"fish", "fish_bare"}) | UNRESOLVED_TAGS

#: How `extract_cooperate` resolved a choice. `_decl` = the clause declared a decision.
CHOICE_TAGS = frozenset({
    "coop", "coop_decl", "defect", "defect_decl", "mixed",
}) | UNRESOLVED_TAGS

TAGS = AMOUNT_TAGS | FISH_TAGS | CHOICE_TAGS


# --- normalisation -----------------------------------------------------------
# Responses are markdown with occasional LaTeX. Anchors like "to Agent 2" sit
# behind "**" and "\)" often enough to break naive matching, so flatten first.
_STRIP = [
    (re.compile(r"\\(?:text|boxed|mathrm|mathbf)\s*\{"), " "),
    (re.compile(r"\\[\[\]()]"), " "),
    (re.compile(r"[*_`{}]"), " "),
    (re.compile(r"\\,|\;|\\!"), " "),
    (re.compile(r"[ \t]+"), " "),
]


def normalize(text):
    """Flatten markdown and LaTeX decoration so the anchors below can match."""
    for rx, replacement in _STRIP:
        text = rx.sub(replacement, text)
    return text


# --- amount lexicon ----------------------------------------------------------
# Action space is 0-100 in every v2/v3 question; larger numbers are payoff
# arithmetic (3x, $150, $300), not a commitment.
_AMOUNT_RX = re.compile(
    r"\$\s?(\d{1,3})(?:\.\d+)?(?!\s?%)|\b(\d{1,3})(?:\.\d+)?\s?(?:dollars?|USD)\b", re.I)

_GIVE_RX = re.compile(
    r"\b(?:give[ns]?|giving|send[s]?|sending|sent|transfer(?:s|red|ring)?|"
    r"propos(?:e|es|ed|ing|al)|offer(?:s|ed|ing)?|allocat(?:e|es|ed|ing)|"
    r"shar(?:e|es|ed|ing)|contribut(?:e|es|ed|ing)|donat(?:e|es|ed|ing)|"
    r"split(?:s|ting)?|hand(?:s|ing)? over|pass(?:es|ing)? on)\b", re.I)
_KEEP_RX = re.compile(r"\b(?:keep[s]?|keeping|kept|retain(?:s|ed|ing)?)\b", re.I)

_AGENT2_RX = re.compile(r"\bagent\s*2\b|\bthe (?:responder|trustee|other agent)\b", re.I)
_AGENT1_RX = re.compile(r"\bagent\s*1\b|\bmyself\b|\bfor me\b|\byourself\b|"
                        r"\bmy own\b|\bthe (?:proposer|investor|dictator)\b", re.I)

# Left context that always makes a number arithmetic or a payoff readout,
# never an offer.
_ARITH_LEFT_RX = re.compile(
    r"(?:payoff (?:of|is|at|will be|would be|remains?)|gets? back|return(?:s|ed|ing)?|"
    r"(?:leav(?:e|es|ing)|keep(?:s|ing)?|maintain(?:s|ing)?|end(?:s|ing)? up)"
    r"[^$]{0,25}?(?:with|at)|"
    r"earn(?:s|ed|ing)?|worth|up to|maximum of|guaranteed|"
    r"(?:sends?|sending|sent|gives?|giving) back[^$]{0,12}|"
    r"[\w)][ ]?[-+*/=×]|\bx\b)"
    r"\s*(?:the\s+)?\$?\s?$",
    re.I,
)
# Left context that restates the rules. Only suppresses the pot value itself:
# "Agent 1 receives $100" is the endowment, "Agent 2 receives $30" is an offer.
_ENDOWMENT_LEFT_RX = re.compile(
    r"(?:receiv(?:e|es|ed)|endowment of|out of|initial|starting|total of|sum of|"
    r"los(?:e|es|t|ing)|the|this|that|entire|whole|full|original|available|remaining)"
    r"\s+(?:an? )?(?:endowment of )?\$?\s?$"
    r"|(?:loss|endowment|pot|stake)[^$]{0,25}?(?:was|were|is|are|of)\s*\$?\s?$",
    re.I,
)
# A minus only means arithmetic when it is spaced ("$100 - $40"); "33-34" is a range.
_ARITH_RIGHT_RX = re.compile(r"^(?:\s*[+*/=×]|\s*-\s|\s?\bx\b)")
# Comparatives name the option NOT taken ("$1 is better than $0").
_COUNTERFACTUAL_LEFT_RX = re.compile(
    r"(?:better|worse|more|less|greater|higher|larger|lower|smaller|rather|"
    r"instead of|other than|than|compared to|versus|vs\.?|risk of|"
    r"avoid(?:ing)? (?:getting|receiving|a)|walk(?:ing)? away with)"
    r"\s*(?:getting |receiving |having |only |just |a )?\$?\s?$",
    re.I,
)
# Phrase separators: "$20 for Agent 2 and $80 for myself" is two allocations.
_PHRASE_SPLIT_RX = re.compile(r",|;|\band\b|\bwhile\b|\bwhereas\b|\bbut\b|\bso that\b", re.I)

_REFUSAL_RX = re.compile(
    r"\b(?:i (?:cannot|can't|can not|won't|will not|am unable to)\s+"
    r"(?:provide|answer|make|participate|engage|play|decide|choose|give|send|transfer)|"
    r"i don't (?:have|make) (?:a |any )?(?:preference|decision|choice))",
    re.I,
)

_BARE_INT_RX = re.compile(r"^\W{0,4}(\d{1,3})\W{0,4}$")
_RANGE_GAP_RX = re.compile(r"^\s?(?:-|\u2013|\u2014|to|or)\s?\$?\s?$", re.I)


def _clause_spans(text):
    """(start, end) for each sentence/line — the unit every scorer resolves within."""
    spans, start = [], 0
    for m in re.finditer(r"\n+|(?<=[.!?:])\s+", text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _amounts(clause, pot):
    """[(start, end, value)] for every commitment-shaped amount in the clause."""
    found = []
    for m in _AMOUNT_RX.finditer(clause):
        groups = [g for g in m.groups() if g is not None]
        value = int(groups[-1])
        if not 0 <= value <= pot:
            continue
        digit = re.search(r"\d", m.group(0))
        start = m.start() + digit.start()
        end = m.start() + len(m.group(0))
        left = clause[max(0, start - 34):start]
        if _ARITH_LEFT_RX.search(left) or _COUNTERFACTUAL_LEFT_RX.search(left):
            continue
        if value == pot and _ENDOWMENT_LEFT_RX.search(left):
            continue
        if _ARITH_RIGHT_RX.match(clause[end:end + 3]):
            continue
        found.append((start, end, float(value)))
    return found


def _merge_ranges(amounts, clause):
    """Collapse "$20 to $30" into one candidate at its midpoint."""
    merged, i = [], 0
    while i < len(amounts):
        if i + 1 < len(amounts) and _RANGE_GAP_RX.match(clause[amounts[i][1]:amounts[i + 1][0]]):
            low, high = amounts[i], amounts[i + 1]
            merged.append((low[0], high[1], (low[2] + high[2]) / 2.0))
            i += 2
        else:
            merged.append(amounts[i])
            i += 1
    return merged


def _phrase_spans(clause):
    """Each allocation in "X for Agent 1 and Y for Agent 2" gets its own span."""
    spans, prev = [], 0
    for m in _PHRASE_SPLIT_RX.finditer(clause):
        if m.start() > prev:
            spans.append((prev, m.start()))
        prev = m.end()
    spans.append((prev, len(clause)))
    return spans


def _owner(pos, clause):
    """(owner, explicit) for the amount at pos.

    explicit=True when the amount's own phrase names the agent ("$25 to Agent 2");
    False when ownership was only inferred from the nearest mention elsewhere in
    the clause, which is much weaker evidence.
    """
    for low, high in _phrase_spans(clause):
        if not low <= pos < high:
            continue
        phrase = clause[low:high]
        has2, has1 = bool(_AGENT2_RX.search(phrase)), bool(_AGENT1_RX.search(phrase))
        if has2 and not has1:
            return "a2", True
        if has1 and not has2:
            return "a1", True
        if not has2 and not has1 and _KEEP_RX.search(phrase):
            return "a1", True
        break
    at2 = [m.start() for m in _AGENT2_RX.finditer(clause)]
    at1 = [m.start() for m in _AGENT1_RX.finditer(clause)]
    if not at2 and not at1:
        return None, False
    nearest2 = min((abs(pos - a) for a in at2), default=10 ** 6)
    nearest1 = min((abs(pos - a) for a in at1), default=10 ** 6)
    return ("a2" if nearest2 <= nearest1 else "a1"), False


def _resolve_amount(clause, pot, near_end):
    """(value, tag) this clause commits to giving Agent 2, or None."""
    amounts = _merge_ranges(_amounts(clause, pot), clause)
    if not amounts:
        bare = _BARE_INT_RX.match(clause.strip())
        if near_end and bare and 0 <= int(bare.group(1)) <= pot:
            return float(bare.group(1)), "bare_int"
        return None

    if _AGENT2_RX.search(clause) or _AGENT1_RX.search(clause):
        owned = [(amount, _owner(amount[0], clause)) for amount in amounts]
        named = [a for a, (who, explicit) in owned if who == "a2" and explicit]
        theirs = named or [a for a, (who, _e) in owned if who == "a2"]
        if theirs:
            return max(theirs, key=lambda a: a[0])[2], "a2_anchor" if named else "a2_near"
        ours = [a for a, (who, _e) in owned if who == "a1"]
        if ours and _AGENT2_RX.search(clause):
            return pot - max(ours, key=lambda a: a[0])[2], "complement"

    give_ends = [m.end() for m in _GIVE_RX.finditer(clause)]
    with_object = [g for g in give_ends if any(a[0] >= g for a in amounts)]
    if with_object:
        last_give = max(with_object)
        return min((a for a in amounts if a[0] >= last_give), key=lambda a: a[0])[2], "verb_obj"

    keep_ends = [m.end() for m in _KEEP_RX.finditer(clause)]
    kept = [a for a in amounts if keep_ends and a[0] >= min(keep_ends)]
    if kept:
        return pot - min(kept, key=lambda a: a[0])[2], "keep"

    if re.search(r"\b(?:answer|amount|choice|decision|offer|proposal|split|transfer)\b"
                 r"[^.\n]{0,25}?(?:\bis\b|\bbe\b|\bare\b|:)\s*\$?\s?\d", clause, re.I):
        return amounts[0][2], "answer_is"
    if len(clause.strip()) <= 14:
        return amounts[0][2], "bare"
    return None


def extract_amount(answer, pot=100):
    """Dollars the response commits to giving Agent 2, within [0, pot].

    A response settles on its decision last, so clauses are scanned last-to-first
    and the last one that resolves wins.
    """
    if not isinstance(answer, str) or not answer.strip():
        return Extraction(None, "empty")
    text = normalize(answer.strip())
    spans = _clause_spans(text)

    for i in range(len(spans) - 1, -1, -1):
        low, high = spans[i]
        resolved = _resolve_amount(text[low:high], pot, near_end=(i >= len(spans) - 3))
        if resolved is not None:
            return Extraction(resolved[0], resolved[1])

    if _REFUSAL_RX.search(text):
        return Extraction(None, "refusal")
    return Extraction(None, "unparsed")


# --- Overfishing: the count Agent 1 harvests, not a dollar amount ---------------

_CATCH_RX = re.compile(
    r"\b(?:catch|catches|catching|caught|harvest(?:s|ing|ed)?|take|takes|taking|"
    r"fish for|limit myself to|go with|choose)\b", re.I)
_FISH_NUM_RX = re.compile(r"\b(\d{1,3})(?:\.\d+)?\s*(?:fish\b)?", re.I)
# Numbers that restate the rules (the 100 cap, the 0-100 range) rather than a choice.
_FISH_RULE_RX = re.compile(
    r"(?:total|combined|together|exceeds?|exceed|>|<=|≤|≥|less than|more than|greater than|"
    r"up to|limit(?: of)?|cap(?: of)?|capacity|maximum|between|range|sum|out of|stock of|"
    r"lake (?:has|holds)|both|each other|-|\bor\b)\s*(?:the\s+)?$",
    re.I,
)
_FISH_RANGE_RX = re.compile(r"\b(\d{1,3})\s?(?:-|\u2013|to|or)\s?(\d{1,3})\b")


def _fish_candidates(clause, cap):
    """[(start, count)] for every catch-shaped number in the clause."""
    midpoints = {}
    for m in _FISH_RANGE_RX.finditer(clause):
        low, high = int(m.group(1)), int(m.group(2))
        if 0 <= low < high <= cap:
            midpoints[m.start()] = (low + high) / 2.0
            midpoints[m.start(2)] = None      # suppress the upper end as its own candidate

    candidates = []
    for m in _FISH_NUM_RX.finditer(clause):
        if m.start() in midpoints and midpoints[m.start()] is None:
            continue
        midpoint = midpoints.get(m.start())
        value = midpoint if midpoint is not None else int(m.group(1))
        if not 0 <= value <= cap:
            continue
        left = clause[max(0, m.start() - 26):m.start()]
        if (_FISH_RULE_RX.search(left) or _ARITH_LEFT_RX.search(left)
                or _COUNTERFACTUAL_LEFT_RX.search(left)):
            continue
        if _ARITH_RIGHT_RX.match(clause[m.end():m.end() + 3]):
            continue
        has_unit = bool(re.match(r"\s*fish\b", clause[m.end():m.end() + 6], re.I)) or \
            "fish" in m.group(0).lower()
        near_verb = bool(_CATCH_RX.search(clause[max(0, m.start() - 45):m.start()]))
        if has_unit or near_verb:
            candidates.append((m.start(), float(value)))
    return candidates


def extract_fish(answer, cap=100):
    """Fish the response commits to catching in season one, within [0, cap]."""
    if not isinstance(answer, str) or not answer.strip():
        return Extraction(None, "empty")
    text = normalize(answer.strip())

    for low, high in reversed(_clause_spans(text)):
        clause = text[low:high]
        candidates = _fish_candidates(clause, cap)
        if candidates:
            # the count a clause settles on is the last one it names
            return Extraction(max(candidates, key=lambda c: c[0])[1], "fish")
        bare = _BARE_INT_RX.match(clause.strip())
        if bare and high >= len(text) - 60 and 0 <= int(bare.group(1)) <= cap:
            return Extraction(float(bare.group(1)), "fish_bare")

    if _REFUSAL_RX.search(text):
        return Extraction(None, "refusal")
    return Extraction(None, "unparsed")


# --- Prisoner's Dilemma: cooperate (1.0) or defect (0.0) ------------------------

_DECISION_RX = re.compile(
    r"\b(?:i will|i would|i'll|i choose|i chose|my choice|my decision|"
    r"i decide|therefore|thus|so,|hence|final(?:ly)?|conclusion|answer|"
    r"i am going to|i'd|i opt|opt for|going with|go with)\b", re.I)
_COOP_RX = re.compile(r"\bcooperate\b|\bcooperation\b|\(\s*c\s*\)|(?<![a-z])c(?![a-z])", re.I)
_DEFECT_RX = re.compile(r"\bdefect(?:ion|ing)?\b|\(\s*d\s*\)|(?<![a-z])d(?![a-z])", re.I)
_NOT_COOP_RX = re.compile(
    r"\bnot? (?:to )?cooperate\b|\brather than cooperat|\binstead of cooperat", re.I)
_NOT_DEFECT_RX = re.compile(
    r"\bnot? (?:to )?defect\b|\brather than defect|\binstead of defect", re.I)


def extract_cooperate(answer):
    """1.0 cooperate / 0.0 defect for the Prisoner's Dilemma choice.

    Two passes: clauses that declare a decision first, then any clause. Within a
    pass, later clauses win.
    """
    if not isinstance(answer, str) or not answer.strip():
        return Extraction(None, "empty")
    text = normalize(answer.strip())
    spans = _clause_spans(text)

    for declared_only in (True, False):
        for low, high in reversed(spans):
            clause = text[low:high]
            if declared_only and not _DECISION_RX.search(clause):
                continue
            coop = [m.start() for m in _COOP_RX.finditer(clause) if not _NOT_COOP_RX.search(
                clause[max(0, m.start() - 22):m.start() + 4])]
            defect = [m.start() for m in _DEFECT_RX.finditer(clause) if not _NOT_DEFECT_RX.search(
                clause[max(0, m.start() - 22):m.start() + 4])]
            suffix = "_decl" if declared_only else ""
            if coop and not defect:
                return Extraction(1.0, "coop" + suffix)
            if defect and not coop:
                return Extraction(0.0, "defect" + suffix)
            if coop and defect:
                # both named: the later mention is the one the clause lands on
                return Extraction(1.0 if max(coop) > max(defect) else 0.0, "mixed")

    if _REFUSAL_RX.search(text):
        return Extraction(None, "refusal")
    return Extraction(None, "unparsed")
