"""Unit tests for the resolution tags and the traps the scorers exist to avoid."""

import pytest

from audit.parse import (AMOUNT_TAGS, CHOICE_TAGS, FISH_TAGS, UNRESOLVED_TAGS,
                         extract_amount, extract_cooperate, extract_fish)


# The single most important behavior in the package: a response that names the
# amount KEPT must not be scored as the amount GIVEN. A last-number regex reads
# every one of these backwards.
KEEP_CASES = [
    ("I would give $0 to Agent 2, keeping the full $100.", 0.0),
    ("I will keep $80.", 20.0),
    ("I'll give $30 to Agent 2 and keep $70 for myself.", 30.0),
    ("I would give $10 to Agent 2, keeping $90 for myself. This leaves me with $90.", 10.0),
    ("I keep $100.", 0.0),
]


@pytest.mark.parametrize("answer,expected", KEEP_CASES)
def test_keep_phrasing_scores_what_was_given(answer, expected):
    assert extract_amount(answer, 100).value == expected


def test_keep_tag_is_reported_when_only_the_kept_amount_is_named():
    assert extract_amount("I will keep $80.", 100) == (20.0, "keep")


@pytest.mark.parametrize("answer,expected", [
    ("I would give $25 to Agent 2.", (25.0, "a2_anchor")),
    ("I give $25.", (25.0, "verb_obj")),
    ("My answer is $40.", (40.0, "answer_is")),
    ("**$25**", (25.0, "bare")),
    ("40", (40.0, "bare_int")),
    ("I would send $20 to $30 to Agent 2.", (25.0, "a2_anchor")),  # range -> midpoint
])
def test_amount_resolution_tags(answer, expected):
    assert extract_amount(answer, 100) == expected


@pytest.mark.parametrize("answer,tag", [
    ("", "empty"),
    ("   ", "empty"),
    (None, "empty"),
    ("I cannot provide an answer to this hypothetical.", "refusal"),
    ("The weather is nice today and nothing else matters.", "unparsed"),
])
def test_unresolvable_amount_is_none_with_a_reason(answer, tag):
    value, got = extract_amount(answer, 100)
    assert value is None, "unresolvable response must not produce a number"
    assert got == tag


def test_payoff_arithmetic_is_not_an_offer():
    # $60 is Agent 1's payoff, not a transfer; $40 is the commitment.
    assert extract_amount("Payoff of $100 - $40 = $60. So I give $40.", 100).value == 40.0


def test_amount_is_clamped_to_the_declared_pot():
    # 10 is out of a $5 space, so nothing in this clause is a payable amount.
    assert extract_amount("I would transfer $10 to Agent 2.", 5).value is None


@pytest.mark.parametrize("answer,expected", [
    ("I will catch 40 fish in the first season.", 40.0),
    ("In the first season I catch 20-30 fish.", 25.0),
])
def test_fish_counts(answer, expected):
    assert extract_fish(answer, 100).value == expected


@pytest.mark.parametrize("answer,tag", [
    ("", "empty"),
    ("I cannot decide on a number of fish.", "refusal"),
    ("Sustainability matters here.", "unparsed"),
])
def test_unresolvable_fish_is_none_with_a_reason(answer, tag):
    value, got = extract_fish(answer, 100)
    assert value is None
    assert got == tag


@pytest.mark.parametrize("answer,expected", [
    ("I will choose Cooperate (C).", 1.0),
    ("I will choose Defect (D).", 0.0),
    ("I would rather not defect, so I will cooperate.", 1.0),
    ("Cooperation is tempting, but I choose to defect.", 0.0),
])
def test_binary_choice(answer, expected):
    assert extract_cooperate(answer).value == expected


@pytest.mark.parametrize("answer,tag", [
    ("", "empty"),
    ("I will not participate in this game.", "refusal"),
    ("An interesting setup with no bearing on my behaviour.", "unparsed"),
])
def test_unresolvable_choice_is_none_with_a_reason(answer, tag):
    value, got = extract_cooperate(answer)
    assert value is None
    assert got == tag


def test_none_is_returned_exactly_for_unresolved_tags(v2_held_out, per_question_labels):
    """The core invariant, over real generations: no silent zero, no untagged value."""
    answers = [row.answer for rows in v2_held_out.values() for row in rows]
    answers += [row.answer for rows in per_question_labels.values() for row in rows]
    assert len(answers) > 2000

    for scorer, tags in ((extract_amount, AMOUNT_TAGS),
                         (extract_fish, FISH_TAGS),
                         (extract_cooperate, CHOICE_TAGS)):
        for answer in answers:
            value, tag = scorer(answer)
            assert tag in tags, "%s returned undeclared tag %r" % (scorer.__name__, tag)
            assert (value is None) == (tag in UNRESOLVED_TAGS)


def test_resolved_amounts_stay_inside_the_action_space(v2_held_out):
    for rows in v2_held_out.values():
        for row in rows:
            value, _tag = extract_amount(row.answer, 100)
            assert value is None or 0.0 <= value <= 100.0
