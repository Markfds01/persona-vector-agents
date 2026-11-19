"""The checkpoint-2 validation numbers and the human calibration export, on synthetic inputs."""

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ("validation", "human", "judging", "measurement"):
    sys.path.insert(0, str(SCRIPTS / sub))
sys.path.insert(0, str(SCRIPTS))

import export_calibration as ec  # noqa: E402
import gqt_judging as gj  # noqa: E402
import pilot_validation as pv  # noqa: E402


# --- AUC ----------------------------------------------------------------------------

def test_auc_known_values():
    assert pv.auc_vectorised([3, 4], [1, 2]) == 1.0
    assert pv.auc_vectorised([1, 2], [3, 4]) == 0.0
    assert pv.auc_vectorised([1, 1], [1, 1]) == 0.5
    assert pv.auc_vectorised([], [1]) is None


def test_clustered_auc_interval_covers_a_perfect_separator_and_resamples_questions():
    items = [("q%d" % q, 80 + q, 1) for q in range(20)] + [("q%d" % q, 20 + q, 0) for q in range(20)]
    result = pv.clustered_auc(items, n_boot=300)
    assert result["auc"] == 1.0 and result["lo"] >= 0.99 and result["n_questions"] == 20
    noisy = [("q%d" % (i % 10), (i * 37) % 100, i % 2) for i in range(200)]
    wide = pv.clustered_auc(noisy, n_boot=300)
    assert wide["lo"] < wide["auc"] < wide["hi"]


def test_clustered_auc_with_one_class_has_no_value():
    assert pv.clustered_auc([("q", 5, 1), ("r", 6, 1)], n_boot=10)["auc"] is None


# --- the byte-identity controls ----------------------------------------------------------

def rows_for_point(name, ids):
    return [{"question_id": "q%d" % i, "sample_index": 0, "output_ids_sha256": h} for i, h in enumerate(ids)]


def points(base, unhooked, repeat, steered):
    return {"baseline k+0|heldout": rows_for_point("b", base),
            "unhooked k+0|heldout": rows_for_point("u", unhooked),
            "unhooked_repeat k+0|heldout": rows_for_point("r", repeat),
            "decision/dictator k+3|heldout": rows_for_point("s", steered)}


HOOKS = {"baseline k+0|heldout": {"hooked": True, "hook_fires": 10, "hook_fires_expected": 10},
         "unhooked k+0|heldout": {"hooked": False, "hook_fires": 0, "hook_fires_expected": 10}}


def verdict(base, unhooked, repeat, steered, hooks=HOOKS):
    return pv.control_verdict(pv.identity_checks(points(base, unhooked, repeat, steered), []), hooks)


def test_identity_passes_only_with_all_three_positive_controls():
    got = verdict(["a", "b"], ["a", "b"], ["a", "b"], ["a", "x"])
    assert got["verdict"] == "pass" and got["check_is_valid"]


def test_identity_is_void_when_the_hook_never_fired():
    hooks = {"baseline k+0|heldout": {"hooked": True, "hook_fires": 0, "hook_fires_expected": 10}}
    got = verdict(["a", "b"], ["a", "b"], ["a", "b"], ["a", "x"], hooks)
    assert got["control_b_hook_fired_exactly_as_expected"] is False and got["verdict"] == "fail"


def test_identity_is_void_when_a_steered_point_changes_nothing():
    assert verdict(["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"])["verdict"] == "fail"


def test_identity_is_void_when_the_unhooked_run_does_not_reproduce():
    got = verdict(["a", "b"], ["a", "b"], ["a", "z"], ["a", "x"])
    assert got["control_a_unhooked_reproducible"] is False and got["verdict"] == "fail"


def test_identity_fails_when_k0_differs_from_unhooked():
    assert verdict(["a", "b"], ["a", "q"], ["a", "q"], ["a", "x"])["verdict"] == "fail"


def test_identity_reports_not_run_when_a_control_is_missing():
    pts = points(["a"], ["a"], ["a"], ["x"])
    del pts["unhooked_repeat k+0|heldout"]
    got = pv.control_verdict(pv.identity_checks(pts, []), HOOKS)
    assert got["verdict"] == "not_run" and got["control_a_unhooked_reproducible"] is None


# --- determinism -------------------------------------------------------------------------

def test_determinism_summary(tmp_path):
    rows = [{"payload_sha256": "a", "key": "k1", "kind": "trait", "exact_match": True, "same_status": True,
             "score_delta": 0.0, "score_original": 5.0, "score_resend": 5.0},
            {"payload_sha256": "b", "key": "k2", "kind": "trait", "exact_match": False, "same_status": True,
             "score_delta": -10.0, "score_original": 80.0, "score_resend": 70.0},
            {"payload_sha256": "c", "key": "k3", "kind": "pairwise", "exact_match": True, "same_status": True,
             "score_delta": None, "score_original": None, "score_resend": None}]
    (tmp_path / "determinism_resend.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = pv.determinism(tmp_path)
    assert got["n"] == 3 and got["exact_match"] == 2 and got["max_abs_delta"] == 10.0
    assert got["n_delta_nonzero"] == 1 and got["mismatches"][0]["key"] == "k2"
    assert pv.determinism(tmp_path / "nowhere") == {"n": 0}


# --- the human calibration sample ---------------------------------------------------------

def gen_rows(n_per_point=40):
    arms = [("decision/dictator", (-3, 3)), ("trait/altruism", (-3, 3)), ("shuffled_null/dictator", (-3, 3)),
            ("orthogonal_null/dictator", (-3, 3)), ("baseline", (0,))]
    out = []
    for arm, ks in arms:
        for k in ks:
            for q in range(n_per_point):
                out.append({"row_id": "%s|heldout|k%+g|q%02d|s0" % (arm, k, q), "arm": arm, "k": float(k),
                            "question_set": "heldout", "question_id": "q%02d" % q,
                            "sample_index": 0, "question": "Q%d" % q, "answer": "reply " + gj.sha256_text("%s%s%d" % (arm, k, q))[:10]})
    out.append({"row_id": "x", "arm": "baseline", "k": 0.0, "question_set": "bridge", "question_id": "b",
                "sample_index": 0, "question": "B", "answer": "bridge"})
    return out


def test_calibration_strata_and_quotas():
    items, key, strata = ec.draw(gen_rows())
    drawn = {(s["role"], s["k"]): s["drawn"] for s in strata}
    assert drawn[("decision", -3.0)] == drawn[("trait", 3.0)] == 18 and drawn[("baseline", 0.0)] == 20
    assert drawn[("relaxed_dictator_only", 3.0)] == 0          # no such arm in the pilot: not topped up
    assert len(items) == len(key) == 8 * 18 + 20
    assert not any(k["row_id"] == "x" for k in key.values())     # bridge rows are never sampled


def test_calibration_draw_is_seeded_and_ids_are_opaque_and_unique():
    a = ec.draw(gen_rows())
    b = ec.draw(gen_rows())
    assert a[0] == b[0] and a[1] == b[1]
    ids = [i["item_id"] for i in a[0]]
    assert len(set(ids)) == len(ids)
    for i in a[0]:
        assert "decision" not in i["item_id"] and "k+" not in i["item_id"]


def test_the_sheet_carries_only_item_id_question_answer(tmp_path):
    rows = gen_rows()
    rowdir = tmp_path / "rows"
    for r in rows:
        path = rowdir / r["arm"].replace("/", "-") / ("%s_k%+g.jsonl" % (r["question_set"], r["k"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(r) + "\n")
    ec.main(["--rows-dir", str(rowdir), "--human-dir", str(tmp_path / "human"),
             "--key", str(tmp_path / "key.json")])
    with open(tmp_path / "human" / "calibration_sheet.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["item_id", "question", "answer"]
        sheet = list(reader)
    html = (tmp_path / "human" / "grader.html").read_text()
    key = json.loads((tmp_path / "key.json").read_text())
    assert len(sheet) == 164 and len(key["items"]) == 164
    # no arm, k or row id reaches the page; the stratum key stays out of it
    for banned in ("decision/dictator", "orthogonal_null", "trait/altruism", "shuffled_null", "stratum"):
        assert banned not in html


def test_role_mapping():
    assert [ec.role_of(a) for a in ("baseline", "decision/trust", "relaxed/ultimatum", "dictator_only",
                                    "trait/altruism", "shuffled_null/trust", "orthogonal_null/trust")] == [
        "baseline", "decision", "relaxed_dictator_only", "relaxed_dictator_only", "trait", "shuffled_null",
        "orthogonal_null"]
    assert ec.role_of("unhooked") is None


# --- bridge and the report ------------------------------------------------------------------

def bridge_rows():
    out = []
    for q in range(6):
        for arm, k in (("baseline", 0.0), ("trait/altruism", 3.0)):
            ans = "Answer: A\nbecause" if (q % 2 == 0) else "Answer: B\nbecause"
            out.append({"row_id": "%s|bridge|k%+g|b%d|s0" % (arm, k, q), "arm": arm, "k": k,
                        "question_set": "bridge", "question_id": "b%d" % q, "sample_index": 0,
                        "question": "Q", "answer": ans})
    out.append({"row_id": "u", "arm": "baseline", "k": 0.0, "question_set": "bridge", "question_id": "b0",
                "sample_index": 1, "question": "Q", "answer": "I would think about it."})
    return out


def test_bridge_summary_counts_unparseable_and_scores_the_other_regarding_option():
    labels = {"b%d" % q: "A" for q in range(6)}
    points, parsed = pv.bridge_summary(bridge_rows(), labels)
    base = points["baseline k+0"]
    assert base["n_rows"] == 7 and base["parse_coverage"] == pytest.approx(6 / 7)
    assert base["p"] == pytest.approx(0.5)               # A at even questions
    assert parsed["u"][0] == "unparseable" and parsed["u"][1] is None


def test_bridge_auc_uses_only_parsed_rows_with_a_judge_score():
    rows = bridge_rows()
    labels = {"b%d" % q: "A" for q in range(6)}
    _, parsed = pv.bridge_summary(rows, labels)
    judge = gj.JUDGE_DIRS["primary"]
    trait = {}
    for r in rows:
        value = parsed[r["row_id"]][1]
        if value is not None:
            trait[r["row_id"] + "|trait"] = {"row_id": r["row_id"], "status": "ok",
                                              "score": 80.0 if value else 20.0}
    result = pv.bridge_auc(rows, parsed, {judge: {"trait": trait}}, judge)
    assert result["auc"] == 1.0 and result["n"] == 12 and result["n_candidate_rows"] == 13
    assert set(result["by_k"]) == {"-3", "+0", "+3"} and result["by_k"]["+3"]["n"] == 6


def test_report_renders_from_a_minimal_validation_and_says_not_run(tmp_path):
    import write_report as wr
    v = {"generation": {"points": {}, "hook_points": {}, "identity": {
             "k0_hooked_vs_unhooked_identical_rows": None, "unhooked_vs_unhooked_repeat_identical_rows": None,
             "k_nonzero_vs_k0_identical_rows": {}},
         "controls": {"verdict": "not_run", "control_b_hook_fired_exactly_as_expected": None}},
         "judge_status": {}, "throughput": {}, "determinism": {}, "points_primary": None,
         "bridge": {"points": {}, "auc": {}}, "judge_judge": None, "pairwise": {}}
    (tmp_path / "v.json").write_text(json.dumps(v))
    wr.main(["--validation", str(tmp_path / "v.json"), "--human-key", str(tmp_path / "none.json"),
             "--out", str(tmp_path / "r.md")])
    text = (tmp_path / "r.md").read_text()
    assert "NOT RUN" in text and "not exported" in text and "not run (no primary-judge rows)" in text
    assert "Human-human" not in text or "not run" in text


def pw_row(rid, order, score, pos, status="ok"):
    return {"key": "%s|pairwise|%s" % (rid, order), "row_id": rid, "order": order, "score": score,
            "steered_position": pos, "status": status}


def test_pairwise_summary_order_consistency_and_position_bias():
    judge = gj.JUDGE_DIRS["primary"]
    rows = [{"row_id": "r1", "arm": "decision/dictator", "k": 3.0},
            {"row_id": "r2", "arm": "decision/dictator", "k": 3.0}]
    got = {}
    for r in (pw_row("r1", "baseline_first", 2.0, 2), pw_row("r1", "steered_first", 1.0, 1),   # steered both times
              pw_row("r2", "baseline_first", 1.0, 2), pw_row("r2", "steered_first", 1.0, 1)):  # position 1 both times
        got[r["key"]] = r
    out = pv.pairwise_summary({judge: {"pairwise": got}}, judge, rows)
    assert out["order_consistency"] == 0.5 and out["n_rows_both_orders"] == 2
    assert out["p_steered_chosen_by_point"]["decision/dictator k+3"] == pytest.approx(0.75)
    assert out["position_bias_p_first_minus_half"] == pytest.approx(0.25)


def test_pairwise_agreement_counts_only_calls_both_judges_answered():
    a, b = gj.JUDGE_DIRS["primary"], gj.JUDGE_DIRS["secondary"]
    ja = {"k1": pw_row("r", "x", 1.0, 1), "k2": pw_row("r", "y", 2.0, 1), "k3": pw_row("r", "z", 2.0, 1, "unparseable")}
    jb = {"k1": pw_row("r", "x", 1.0, 1), "k2": pw_row("r", "y", 1.0, 1), "k3": pw_row("r", "z", 2.0, 1)}
    got = pv.pairwise_agreement({a: {"pairwise": ja}, b: {"pairwise": jb}})
    assert got == {"n_both_ok": 2, "n_agree": 1, "share": 0.5}
