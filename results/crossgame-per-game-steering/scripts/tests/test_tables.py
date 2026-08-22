"""The pooled log's one judgement call: which contrasts are not evidence at all.

`provenance/tables.log` is what `README.md` section 8 tells a reproducer to run,
and two of its eighteen contrasts are computed across an arm that never produced a
distribution. Printed bare, one of them reads as a win and the other as a measured
null. The rule that marks them is `degradation.py`, shared with the figures, so
these tests pin both the rule and the marking.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "measurement"))

import degradation  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


tables = _load("crossgame_tables", SCRIPTS / "pooling" / "crossgame_tables.py")


def _point(coverage=1.0, non_latin=0.0):
    return {"parse_coverage": coverage,
            "degeneracy": {"share_with_non_latin_script": non_latin}}


# --- the rule -----------------------------------------------------------------

def test_a_thin_point_is_a_caveat_and_still_a_distribution():
    """Overfishing's k=-5 resolved 86 of 100 and carries a real, graded effect."""
    assert degradation.point_flags(_point(coverage=0.86))[1] is True
    assert degradation.produced_no_distribution(_point(coverage=0.86)) is False


def test_a_point_that_left_english_produced_no_distribution_however_well_parsed():
    assert degradation.produced_no_distribution(
        _point(coverage=0.86, non_latin=1.0)) is True


def test_a_point_below_half_coverage_produced_no_distribution():
    assert degradation.produced_no_distribution(_point(coverage=0.08)) is True
    assert degradation.produced_no_distribution(_point(coverage=0.50)) is False


# --- the marking --------------------------------------------------------------

def _game(decision, null):
    return {"arms": {"decision": {"points": decision},
                     "shuffled-null": {"points": null}}}


def test_a_contrast_across_two_healthy_arms_is_marked_readable():
    game = _game({"5": _point()}, {"5": _point()})
    assert tables.unreadable_arms(game, "5") == []


def test_the_arm_that_did_not_answer_is_the_one_named():
    game = _game({"-5": _point()}, {"-5": _point(coverage=0.08)})
    assert tables.unreadable_arms(game, "-5") == ["null"]
    game = _game({"5": _point(coverage=0.86, non_latin=1.0)}, {"5": _point()})
    assert tables.unreadable_arms(game, "5") == ["decision"]


def test_both_arms_are_named_when_both_collapsed():
    game = _game({"5": _point(coverage=0.1)}, {"5": _point(coverage=0.1)})
    assert tables.unreadable_arms(game, "5") == ["decision", "null"]


# --- against the committed analysis -------------------------------------------

@pytest.fixture(scope="module")
def analysis():
    path = PACKAGE / "analysis" / "steering.json"
    if not path.exists():
        pytest.skip("no committed analysis to check the rule against")
    return json.loads(path.read_text())


def test_exactly_the_two_contrasts_the_report_qualifies_are_marked(analysis):
    """README sections 3, 10 and 11.2 name these two and no others."""
    found = []
    for family, game in analysis["games"].items():
        for key in game.get("decision_vs_null", {}):
            for arm in tables.unreadable_arms(game, key):
                found.append((family, int(key), arm))
    assert sorted(found) == [("prisoners_dilemma", 5, "decision"),
                             ("ultimatum", -5, "null")]
