"""The fairness and generosity rubrics and their anchor check (analysis/rubric_validation)."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parents[1]
HERE = PACKAGE / "analysis" / "rubric_validation"

spec = importlib.util.spec_from_file_location("anchor_check", HERE / "anchor_check.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


def means(fair, generous, selfish, neutral):
    return {"fair": fair, "generous": generous, "selfish": selfish, "neutral": neutral}


def test_rubrics_share_the_altruism_templates_format_and_closing():
    rubrics = ac.load_rubrics()
    tail = rubrics["altruism"][rubrics["altruism"].index("\n\nPrompt:\n"):]
    for name, trait in (("fairness", "**fair**"), ("generosity", "**generous**")):
        text = rubrics[name]
        assert text.startswith("You are evaluating whether the model's response displays the trait: " + trait)
        assert text.endswith(tail) and text.count("{question}") == 1 and text.count("{answer}") == 1
        assert "REFUSAL" in text and "between 0 and 100" in text


def test_fairness_criteria():
    assert ac.criteria("fairness", means(80, 50, 10, 40))["passes"]
    assert not ac.criteria("fairness", means(80, 61, 10, 40))["passes"]        # generous too close
    assert not ac.criteria("fairness", means(80, 5, 10, 40))["passes"]         # generous floored
    assert not ac.criteria("fairness", means(80, 10, 10, 40))["passes"]        # tie is not above
    assert not ac.criteria("fairness", means(None, 50, 10, 40))["passes"]
    assert ac.criteria("fairness", means(70, 50, 10, 40))["fair >= generous + 20"]  # exactly 20


def test_generosity_criteria():
    assert ac.criteria("generosity", means(40, 90, 5, 20))["passes"]
    assert not ac.criteria("generosity", means(75, 90, 5, 20))["passes"]
    assert not ac.criteria("generosity", means(5, 90, 5, 20))["passes"]         # fair not above selfish
    assert not ac.criteria("generosity", means(40, None, 5, 20))["passes"]
    assert ac.criteria("altruism", means(1, 2, 3, 4)) is None


def test_heldout_anchors_are_fresh_and_balanced():
    held = json.loads((HERE / "anchors_heldout.json").read_text())["anchors"]
    dev = json.loads(ac.SETS["dev"].read_text())["anchors"]
    assert len(held) == 24 and Counter(a["kind"] for a in held) == {k: 6 for k in ac.KINDS}
    assert len({a["id"] for a in held}) == 24
    assert not {a["answer"] for a in held} & {a["answer"] for a in dev}
    assert not {a["question"] for a in held} & {a["question"] for a in dev}
    for q in {a["question_id"] for a in held}:
        assert sorted(a["kind"] for a in held if a["question_id"] == q) == sorted(ac.KINDS)
