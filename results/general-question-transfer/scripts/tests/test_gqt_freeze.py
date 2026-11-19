"""The freeze: PREREG.md is never edited after the freezing commit."""

import hashlib
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "provenance"))

import freeze  # noqa: E402

#: The sha256 of PREREG.md as frozen on 2026-10-23. Changing PREREG.md breaks this test
#: on purpose: later changes go to DEVIATIONS.md, never to PREREG.md.
PREREG_SHA256 = "72a9cb77d5be934984e0dced42dd5e437f35f67d55eeb2f140393ca869633302"


def prereg_label():
    return "results/general-question-transfer/PREREG.md"


def test_prereg_sha256_is_the_frozen_one():
    digest = hashlib.sha256((PACKAGE / "PREREG.md").read_bytes()).hexdigest()
    assert digest == PREREG_SHA256, "PREREG.md was edited after the freeze; record it in DEVIATIONS.md"


def test_freeze_record_holds_the_same_prereg_hash():
    assert freeze.recorded()[prereg_label()] == PREREG_SHA256


def test_freeze_record_reproduces_every_hash():
    now = {label: digest for _, label, digest in freeze.entries()[0]}
    assert now == freeze.recorded()


def test_freeze_record_names_the_parent_and_the_header():
    text = freeze.OUT.read_text(encoding="utf-8")
    assert re.search(r"\*\*parent\*\* is `[0-9a-f]{40}`", text)
    assert "Today Date: 23 Oct 2026" in text and "26 Jul 2024" not in text


def test_prereg_is_marked_frozen_and_has_no_open_decision_left():
    text = (PACKAGE / "PREREG.md").read_text(encoding="utf-8")
    assert "STATUS: FROZEN, 2026-10-23" in text
    assert "DRAFT" not in text and "operator to confirm" not in text and "OPEN DECISION" not in text
    assert "NOT pre-registered here" in text          # D12, deferred to checkpoint 2
    assert "heldout_draft" not in text and "bridge_draft" not in text


def test_llama_header_is_the_explicit_date_not_the_template_default():
    header = freeze.llama_header()
    assert header.endswith("<|eot_id|>")
    assert "Today Date: %s\n" % freeze.LLAMA_DATE_STRING in header
    assert freeze.LLAMA_TEMPLATE_DEFAULT_DATE not in header


def test_pinned_template_hashes_match_the_ones_prereg_states():
    prereg = (PACKAGE / "PREREG.md").read_text(encoding="utf-8")
    rec = freeze.recorded()
    for label_part in ("trait template", "coherence template"):
        digest = next(v for k, v in rec.items() if k.startswith(label_part))
        assert digest in prereg, label_part
