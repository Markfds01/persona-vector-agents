"""The checkpoint-1 drafts: shape, counterbalancing, provenance and checksums."""

import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "questions"))
sys.path.insert(0, str(SCRIPTS / "measurement"))

import bridge_parse  # noqa: E402
import build_bridge  # noqa: E402

GAME_WORDS = re.compile(r"dictator|ultimatum|prisoner|trust game|overfish|apology game|"
                        r"game theory|\$\s?\d|dollar|agent 1|agent 2|payoff", re.I)


def load(name):
    return json.loads((PACKAGE / "questions" / name).read_text(encoding="utf-8"))


def test_heldout_draft_shape_and_authorship():
    doc = load("heldout_draft.json")
    qs = [q["question"] for q in doc["questions"]]
    assert len(qs) == 40 and len(set(qs)) == 40
    assert len({q["id"] for q in doc["questions"]}) == 40
    assert "Claude" in doc["authorship"] and "2026-10-23" in doc["authorship"]
    assert doc["status"].startswith("DRAFT")
    for q in qs:
        assert not GAME_WORDS.search(q), q
        assert q.rstrip().endswith("?")


def test_bridge_counterbalanced_exactly_half_by_recorded_seed():
    doc = load("bridge_draft.json")
    items = doc["items"]
    assert len(items) == 40
    assert sum(it["other_regarding_option"] == "A" for it in items) == 20
    assert doc["counterbalance"]["seed"] == build_bridge.SEED
    assert json.dumps(build_bridge.build(), indent=1, ensure_ascii=False) + "\n" == \
        (PACKAGE / "questions" / "bridge_draft.json").read_text(encoding="utf-8")


def test_bridge_items_render_their_options_in_the_assigned_order():
    for it, (scenario, other, selfish) in zip(load("bridge_draft.json")["items"],
                                              build_bridge.ITEMS):
        a, b = (other, selfish) if it["other_regarding_option"] == "A" else (selfish, other)
        assert (it["option_a"], it["option_b"]) == (a, b)
        assert "\nA: %s\nB: %s\n" % (a, b) in it["prompt"]
        assert not GAME_WORDS.search(it["prompt"]), it["id"]


def test_bridge_prompt_asks_for_what_the_parser_reads():
    for it in load("bridge_draft.json")["items"]:
        assert '"Answer: A"' in it["prompt"] and '"Answer: B"' in it["prompt"]
        # the prompt itself contains both answer strings; the parser must not be
        # fed the prompt, only the model's continuation
        assert bridge_parse.parse_choice(it["prompt"]).status == "conflicting"


def test_sha256sums_match():
    sums = (PACKAGE / "questions" / "SHA256SUMS").read_text().split("\n")
    entries = [line.split() for line in sums if line.strip()]
    assert {name for _, name in entries} == {"heldout_draft.json", "bridge_draft.json"}
    for digest, name in entries:
        data = (PACKAGE / "questions" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_near_duplicate_report_covers_the_current_drafts():
    report = json.loads((PACKAGE / "questions" / "near_duplicates.json").read_text())
    for name in ("heldout_draft.json", "bridge_draft.json"):
        data = (PACKAGE / "questions" / name).read_bytes()
        assert report["inputs_sha256"][name] == hashlib.sha256(data).hexdigest()
    assert report["pairs_above_threshold"] == []
    assert all(c["caught_embedding"] for c in report["positive_controls"])
