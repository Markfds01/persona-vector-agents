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
    assert set(data) == {"items", "key", "columns", "title"}
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
