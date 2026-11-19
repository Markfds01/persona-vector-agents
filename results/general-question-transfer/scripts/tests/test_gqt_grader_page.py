"""The grader page is blind by construction: what goes in, and what never does."""

import csv
import json
import re
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import grader_page as gp  # noqa: E402


def write_sheet(path, rows, columns=gp.SHEET_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def embedded(html):
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',
                                html, re.S).group(1).replace("<\\/", "</"))


def test_page_embeds_every_item_and_nothing_else(tmp_path):
    rows = [{"item_id": "x%d" % i, "question": "Q%d" % i, "answer": "A%d" % i} for i in range(5)]
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, rows)
    html = gp.write_page(sheet, tmp_path / "p.html").read_text()
    data = embedded(html)
    assert data["items"] == rows
    assert set(data) == {"items", "key", "columns", "rubrics", "title"}
    assert data["columns"] == list(gp.EXPORT_COLUMNS)


@pytest.mark.parametrize("columns", [
    ("item_id", "question", "answer", "arm"),
    ("item_id", "question", "answer", "trait"),
    ("item_id", "answer", "question"),
    ("question", "answer"),
])
def test_sheet_with_anything_but_question_and_answer_is_refused(tmp_path, columns):
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, [{c: "v" for c in columns}], columns)
    with pytest.raises(gp.SheetError):
        gp.read_sheet(sheet)


def test_duplicate_or_empty_ids_refused(tmp_path):
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, [{"item_id": "a", "question": "q", "answer": "x"}] * 2)
    with pytest.raises(gp.SheetError):
        gp.read_sheet(sheet)
    write_sheet(sheet, [{"item_id": " ", "question": "q", "answer": "x"}])
    with pytest.raises(gp.SheetError):
        gp.read_sheet(sheet)


def test_script_breakout_is_escaped():
    rows = [{"item_id": "a", "question": "</script><b>", "answer": "<!-- </SCRIPT> -->"}]
    html = gp.render(rows, title='<x> & "y"')
    script = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert "</" not in script.group(1)
    assert embedded(html)["items"] == rows
    assert "<title>&lt;x&gt; &amp; &quot;y&quot;</title>" in html


def test_storage_key_depends_on_the_sheet():
    a = [{"item_id": "a", "question": "q", "answer": "x"}]
    b = [{"item_id": "a", "question": "q", "answer": "y"}]
    assert gp.sheet_key(a) != gp.sheet_key(b)
    assert gp.sheet_key(a) == gp.sheet_key([dict(a[0])])


def test_demo_is_synthetic_and_judge_scores_stay_out_of_the_page(tmp_path):
    sheet, page = gp.demo(tmp_path, n_items=8, seed=1)
    html = page.read_text()
    judge = json.loads((tmp_path / "judge_reveal_SYNTHETIC.json").read_text())
    items = embedded(html)["items"]
    assert len(items) == 8 and set(items[0]) == set(gp.SHEET_COLUMNS)
    assert all(i["answer"].startswith("[SYNTHETIC PLACEHOLDER]") for i in items)
    assert "SYNTHETIC" in judge["judge"]
    for detail in (v["trait_detail"] for v in judge["items"].values()):
        assert detail not in html
    assert "trait_status" not in json.dumps(embedded(html))


def test_page_has_the_controls_the_operator_asked_for():
    html = gp.render([{"item_id": "a", "question": "q", "answer": "x"}])
    for needle in ('id="reviewer"', 'id="export"', 'id="filter"', 'id="judgefile"',
                   "localStorage", "keydown", "human_labels_", 'type="range"',
                   'data-flag="refusal"', 'data-flag="off_topic"', "revealed_at"):
        assert needle in html


def test_two_reviewers_get_separate_state_and_separate_exports():
    html = gp.render([{"item_id": "a", "question": "q", "answer": "x"}])
    # ratings are namespaced by reviewer, a rating needs a name, the export is named by it
    assert "const stateKey = () => KEY + ':' + slug(reviewer.value)" in html
    assert "if (!named())" in html
    assert "localStorage.getItem(stateKey())" in html and "localStorage.setItem(stateKey()" in html
    assert "human_labels_${fileSlug}_" in html
    # the legacy shared namespace is gone: nothing reads or writes the bare sheet key
    assert "localStorage.getItem(KEY)" not in html and "localStorage.setItem(KEY," not in html


def test_committed_demo_is_reproducible():
    demo_dir = SCRIPTS.parent / "grader_demo"
    committed = (demo_dir / "grader_SYNTHETIC.html").read_text()
    rows = gp.read_sheet(demo_dir / "blind_sheet_SYNTHETIC.csv")
    assert committed == gp.render(rows, "SYNTHETIC DEMO - placeholder items, not model outputs")


def test_retest_subset_is_hidden_shuffled_and_keyed(tmp_path):
    rows = [{"item_id": "x%d" % i, "question": "Q%d" % i, "answer": "A%d" % i} for i in range(40)]
    out, key = gp.make_retest(rows, 0.25, seed=7)
    assert len(out) == len(key) == 10
    assert set(key.values()) <= {r["item_id"] for r in rows}
    assert len(set(key.values())) == 10
    assert not {r["item_id"] for r in out} & {r["item_id"] for r in rows}   # fresh ids
    by_orig = {r["item_id"]: r for r in rows}
    for r in out:
        orig = by_orig[key[r["item_id"]]]
        assert (r["question"], r["answer"]) == (orig["question"], orig["answer"])
    assert gp.make_retest(rows, 0.25, seed=7) == (out, key)                  # seeded
    assert gp.make_retest(rows, 0.25, seed=8)[1] != key


def test_retest_cli_writes_a_sheet_the_page_accepts(tmp_path):
    rows = [{"item_id": "x%d" % i, "question": "Q%d" % i, "answer": "A%d" % i} for i in range(8)]
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, rows)
    gp.main(["retest", "--sheet", str(sheet), "--out-sheet", str(tmp_path / "r.csv"),
             "--out-key", str(tmp_path / "k.json")])
    assert len(gp.read_sheet(tmp_path / "r.csv")) == 2
    assert len(json.loads((tmp_path / "k.json").read_text())) == 2


# --- fairness and generosity (DEVIATIONS 2026-11-04) ---------------------------------

ONE = [{"item_id": "a", "question": "q", "answer": "x"}]


def test_four_scores_all_required_and_exported():
    html = gp.render(ONE)
    assert gp.SCORE_FIELDS == ("trait", "coherence", "fairness", "generosity")
    assert "const SCORES = ['trait', 'coherence', 'fairness', 'generosity']" in html
    assert "SCORES.every(" in html                       # an item is rated only with all four
    assert "Altruism (0-100)" in html and "Fairness (0-100)" in html and "Generosity (0-100)" in html
    assert embedded(html)["columns"] == list(gp.EXPORT_COLUMNS)
    assert {"trait", "fairness", "generosity"} <= set(gp.EXPORT_COLUMNS)
    assert "D.columns.map(c => q(v[c]))" in html          # export follows EXPORT_COLUMNS


def test_definitions_are_the_frozen_rubric_wording():
    import json as _json
    rubrics = embedded(gp.render(ONE))["rubrics"]
    assert set(rubrics) == {"trait", "fairness", "generosity"}
    for field, name in gp.RUBRIC_FILES.items():
        prompt = _json.loads((gp.RUBRIC_DIR / name).read_text())["eval_prompt"]
        assert rubrics[field] and rubrics[field] in prompt
        assert "{question}" not in rubrics[field] and "You are evaluating" not in rubrics[field]
    assert "non-altruistic" in rubrics["trait"] and "Generosity is not fairness" in rubrics["generosity"]


def test_old_format_export_reads_with_empty_new_columns(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("item_id,trait,coherence,refusal,off_topic,note,reviewer,rated_at,judge_revealed_at\n"
                   'a,40,90,0,0,"hi",ana,2026-11-01T10:00:00Z,\n')
    rows = gp.read_export(old)
    assert rows[0]["trait"] == "40" and rows[0]["fairness"] == "" and rows[0]["generosity"] == ""
    assert set(rows[0]) == set(gp.EXPORT_COLUMNS)


def test_judge_reveal_stays_gated_and_new_scores_optional():
    html = gp.render(ONE)
    # revealed only for a rated item, only from the loaded reveal file, never embedded
    assert "if (!judge || !s.revealed_at) return ''" in html
    assert "btn.disabled = !judge || !isRated(s) || !!s.revealed_at" in html
    assert "filter(f => f in j)" in html                  # no fairness/generosity in file -> nothing shown
    assert "trait_status" not in json.dumps(embedded(html)) and "\"judge\"" not in json.dumps(embedded(html))


def test_synthetic_reveal_fixture_has_no_new_scores(tmp_path):
    gp.demo(tmp_path, n_items=4, seed=2)
    judge = json.loads((tmp_path / "judge_reveal_SYNTHETIC.json").read_text())
    assert all("fairness" not in v and "generosity" not in v for v in judge["items"].values())


def test_regenerated_page_keeps_items_ids_and_order():
    sheet = SCRIPTS.parent / "human" / "calibration_sheet.csv"
    page = SCRIPTS.parent / "human" / "grader.html"
    rows = gp.read_sheet(sheet)
    data = embedded(page.read_text())
    assert data["items"] == [{c: r[c] for c in gp.SHEET_COLUMNS} for r in rows]
    assert len(rows) == 164
    assert data["columns"] == list(gp.EXPORT_COLUMNS) and "fairness" in data["rubrics"]
