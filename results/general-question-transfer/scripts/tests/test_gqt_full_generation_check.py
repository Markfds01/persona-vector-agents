"""The full run's plan and per-point integrity check (scripts/validation/full_generation_check.py)."""

import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "validation"))

import full_generation_check as fgc  # noqa: E402


def test_plan_is_div8_main_plus_the_unhooked_repeat():
    plan = fgc.planned_points()
    assert len(plan) == len(set(plan)) == 111
    rows = Counter()
    for arm, qset, _ in plan:
        rows[qset] += 120
    assert rows == {"heldout": 6120, "in_distribution": 1800, "bridge": 5400}
    assert sum(rows.values()) - 120 == 13200          # the study's rows, without unhooked_repeat
    assert len(fgc.ALL22) == len(set(fgc.ALL22)) == 22
    assert sorted(k for a, s, k in plan if a == fgc.TRAIT and s == "heldout") == [-5, -3, -1, 1, 3, 5]


def good_rows(**over):
    rows = []
    for q in range(40):
        for s in range(3):
            r = {"question_id": "q%02d" % q, "sample_index": s, "arm": "a", "question_set": "heldout",
                 "k": 3.0, "question_file_sha256": "Q", "torch_version": fgc.TORCH,
                 "engine_version": fgc.TRANSFORMERS, "max_new_tokens": 1000, "seed": 0,
                 "use_model_defaults": False, "vector_sha256": "V"}
            r.update(over)
            rows.append(r)
    return rows


def test_check_point_accepts_a_whole_point_and_names_what_is_wrong():
    assert fgc.check_point(good_rows(), "a", "heldout", 3.0, "Q", "V") == []
    assert fgc.check_point(good_rows()[:-1], "a", "heldout", 3.0, "Q", "V")
    bad = fgc.check_point(good_rows(torch_version="2.7.1+cu126"), "a", "heldout", 3.0, "Q", "V")
    assert any(p.startswith("torch_version") for p in bad)
    assert any(p.startswith("vector_sha256") for p in fgc.check_point(good_rows(), "a", "heldout", 3.0, "Q", "W"))
