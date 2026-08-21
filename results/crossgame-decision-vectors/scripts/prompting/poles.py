"""Outcome-defined poles, normalised per game family, under two stated policies.

The contrast is always the same one: what the model DID, read off the scorer's
resolved value - never off a rating, never off the wording of the answer. What
differs per family is what "the payoff-maximising choice" and "restraint" mean in
that game's own units, so each family declares both explicitly.

Two policies are carried, and every result is reported under both, because they
disagree about one game and that disagreement is a finding rather than a knob.

  STRICT - the brief's definition, and the one the Dictator-only run used, so a
  vector built under it is directly comparable with the archived Dictator vector.
      amount games      self = transferred exactly 0
                        alt  = transferred at least half the endowment
      overfishing       self = caught the maximum M
                        alt  = caught at most the sustainable share K/2
      prisoner's dil.   self = Defect, alt = Cooperate

  RELAXED - the same poles widened to the choice that actually maximises own
  payoff in each game, which for one game is not the strict pole at all:
      amount games      self = kept at least 90% of the endowment (fraction <= 0.1)
                        alt  = unchanged
      overfishing       self = caught MORE than the sustainable share K/2 - the
                               defection in a commons dilemma, whether or not it
                               is the maximum
                        alt  = unchanged
      prisoner's dil.   unchanged; the answer space has two points and no middle

Why RELAXED exists, and why it is not tuning. In the Dictator and the Apology the
own-payoff-maximising transfer is exactly $0, so strict and relaxed agree about
where the pole is. In the Ultimatum they do not: the proposer's subgame-perfect
offer is the SMALLEST POSITIVE amount a rational responder accepts, because an
offer of exactly $0 leaves the responder indifferent and free to reject. A pole
pinned at exactly $0 therefore excludes precisely the self-interested play, and
the calibration pass shows what that costs - 1 self-interested row in 48, against
47 rows sitting just above the pole. RELAXED applies one uniform threshold to all
four amount games rather than a different rule for the Ultimatum, so no game gets
its own hand-fitted boundary.

Tag policy. `audit.parse` names the path it resolved a value along. Paths that
read the number straight off the response are the primary poles. Paths that
DERIVE it - the amount scorer's `complement` and `keep` subtract a read amount
from the pot, and the binary scorer's `mixed` picks between two choices the
response both named, by position - can land a row in the wrong pole, and land it
exactly on the boundary. Those are excluded from the poles, counted, and re-run
as a sensitivity check rather than silently included or silently dropped.

Two known `audit.parse` bugs bear on the counts and are reported, not corrected
here: a refusal misclassification that discards some $0 answers, and an `a2_near`
case that can read a figure out of a payoff sentence as the transfer. Both are
directional and both are named in the report.
"""

#: Values read directly off the response, per scorer.
DIRECT_TAGS = {
    "amount": ("a2_anchor", "a2_near", "verb_obj", "answer_is", "bare", "bare_int"),
    "fish": ("fish", "fish_bare"),
    "binary_choice": ("coop_decl", "defect_decl", "coop", "defect"),
}

#: Values derived rather than read. Sensitivity only, never a primary pole.
DERIVED_TAGS = {
    "amount": ("complement", "keep"),
    "fish": (),
    "binary_choice": ("mixed",),
}

#: Values the scorer could not resolve at all. Never a pole, always reported.
UNRESOLVED_TAGS = ("empty", "refusal", "unparsed")

SELF, ALT, MIDDLE = "self_interested", "altruistic", "middle"
POLICIES = ("strict", "relaxed")

#: RELAXED keeps-at-least-this-much of the endowment still counts as self-interested.
RELAXED_KEEP_FRACTION = 0.9


def _amount_pole(value, game, policy):
    fraction = value / float(game.pole_scale)
    if fraction >= 0.5:
        return ALT
    if policy == "strict":
        return SELF if fraction == 0.0 else MIDDLE
    return SELF if fraction <= (1.0 - RELAXED_KEEP_FRACTION) else MIDDLE


def _fish_pole(value, game, policy):
    max_catch = game.answer_space.stated.high
    sustainable = game.pole_scale / 2.0        # pole_scale is the season capacity K
    if sustainable >= max_catch:
        raise ValueError("%s: sustainable share %g is not below the maximum catch %g; "
                         "the poles would overlap" % (game.id, sustainable, max_catch))
    if value <= sustainable:
        return ALT
    if policy == "strict":
        return SELF if value == max_catch else MIDDLE
    return SELF


def _binary_pole(value, game, policy):
    if value == 0.0:
        return SELF
    if value == 1.0:
        return ALT
    raise ValueError("%s: binary_choice resolved to %r, which is neither pole"
                     % (game.id, value))


_POLE_BY_SCORER = {
    "amount": _amount_pole,
    "fish": _fish_pole,
    "binary_choice": _binary_pole,
}


def classify(value, game, policy):
    """SELF / ALT / MIDDLE for one resolved value, in this game's own units."""
    if policy not in POLICIES:
        raise ValueError("unknown pole policy %r" % (policy,))
    return _POLE_BY_SCORER[game.scorer](value, game, policy)


def tag_class(tag, scorer):
    """'direct' / 'derived' / 'unresolved' / 'unknown' for one parser tag."""
    if tag in DIRECT_TAGS[scorer]:
        return "direct"
    if tag in DERIVED_TAGS[scorer]:
        return "derived"
    if tag in UNRESOLVED_TAGS:
        return "unresolved"
    return "unknown"
