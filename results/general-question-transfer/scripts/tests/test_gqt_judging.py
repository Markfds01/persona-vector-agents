"""The judging stage's plan, prompts, resume and stop-with-nothing-written behaviour.

Offline: a fake llama-server on localhost (the same shape as the client's own tests) and
synthetic generation rows. No model, no network beyond 127.0.0.1.
"""

import json
import math
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "judging"))

import gqt_judging as gj  # noqa: E402

TEMPLATES = gj.load_templates()


def row(arm, k, question_set, qid, s, answer=None):
    return {"row_id": "%s|%s|k%+g|%s|s%d" % (arm, question_set, k, qid, s), "arm": arm, "k": float(k),
            "question_set": question_set, "question_id": qid, "sample_index": s,
            "question": "Q %s?" % qid, "answer": answer or "reply " + gj.sha256_text("%s|%s|%s|%s|%s" % (arm, k, question_set, qid, s))[:12],
            "generation_status": "ok"}


def rows_for(n_questions=10, samples=3, arms=("decision/dictator", "trait/altruism")):
    out = []
    for qs in ("heldout", "bridge"):
        for arm, ks in [("baseline", (0,))] + [(a, (-3, 3)) for a in arms]:
            for k in ks:
                for q in range(n_questions):
                    for s in range(samples):
                        out.append(row(arm, k, qs, "%s_%02d" % (qs, q), s))
    return out


def test_templates_are_the_frozen_ones():
    freeze = (SCRIPTS.parent / "provenance" / "freeze.md").read_text()
    for kind in gj.KINDS:
        assert gj.sha256_text(TEMPLATES[kind]) in freeze, kind


def test_primary_plan_counts_follow_the_design():
    rows = rows_for()
    plan = gj.build_plan("primary", rows, TEMPLATES)
    count = lambda kind: sum(t.kind == kind for t in plan)  # noqa: E731
    heldout = sum(r["question_set"] == "heldout" for r in rows)
    bridge = sum(r["question_set"] == "bridge" for r in rows)
    steered = sum(r["question_set"] == "heldout" and r["k"] != 0 for r in rows)
    auc = sum(gj.bridge_auc_row(r) for r in rows)
    assert count("coherence") == heldout + bridge
    assert count("trait") == heldout + auc
    assert count("pairwise") == 2 * steered
    assert auc == 3 * 10 * 3


def test_bridge_auc_subset_is_baseline_and_trait_vector_only():
    assert gj.bridge_auc_row(row("baseline", 0, "bridge", "b", 0))
    assert gj.bridge_auc_row(row("trait/altruism", 3, "bridge", "b", 0))
    assert not gj.bridge_auc_row(row("decision/dictator", 3, "bridge", "b", 0))
    assert not gj.bridge_auc_row(row("trait/altruism", 3, "heldout", "b", 0))


def test_secondary_plan_is_the_pre_registered_subset_and_scores_no_bridge_coherence():
    rows = rows_for(n_questions=40)
    plan = gj.build_plan("secondary", rows, TEMPLATES)
    by_id = {r["row_id"]: r for r in rows}
    heldout_trait = [t for t in plan if t.kind == "trait" and by_id[t.row_id]["question_set"] == "heldout"]
    points = {(r["arm"], r["k"]) for r in rows if r["question_set"] == "heldout"}
    assert len(heldout_trait) == 30 * len(points)
    # one fixed subset of (question, sample) pairs for every point
    subsets = {frozenset((by_id[t.row_id]["question_id"], by_id[t.row_id]["sample_index"])
                         for t in heldout_trait
                         if (by_id[t.row_id]["arm"], by_id[t.row_id]["k"]) == p) for p in points}
    assert len(subsets) == 1
    assert not [t for t in plan if t.kind == "coherence" and by_id[t.row_id]["question_set"] == "bridge"]
    assert sum(t.kind == "pairwise" for t in plan) == 2 * 30 * 4


def test_subset_matches_the_stated_formula():
    import random
    pairs = [("q%02d" % q, s) for q in range(40) for s in range(3)]
    assert gj.secondary_subset(pairs) == set(random.Random(20261024).sample(sorted(pairs), 30))


def test_judge_never_sees_the_arm_k_or_row_id():
    rows = rows_for(n_questions=10)
    for judge in ("primary", "secondary"):
        for task in gj.build_plan(judge, rows, TEMPLATES):
            for banned in ("decision/dictator", "trait/altruism", "baseline", "k+3", "k-3",
                           "heldout|", "row_id", "unit_beta"):
                assert banned not in task.prompt, (banned, task.key)


def test_text_mode_appends_the_format_instruction_and_logprob_mode_does_not():
    rows = rows_for(n_questions=10)
    primary = [t for t in gj.build_plan("primary", rows, TEMPLATES) if t.kind == "trait"][0]
    secondary = [t for t in gj.build_plan("secondary", rows, TEMPLATES) if t.kind == "trait"][0]
    assert gj.TEXT_FORMAT_INSTRUCTION not in primary.prompt
    assert secondary.prompt.endswith(gj.TEXT_FORMAT_INSTRUCTION)
    assert primary.instruction_sha256 is None and secondary.instruction_sha256


def test_pairwise_goes_in_both_orders_with_the_baseline_answer_of_the_same_q_and_s():
    rows = rows_for(n_questions=2)
    by_id = {r["row_id"]: r for r in rows}
    tasks = [t for t in gj.build_plan("primary", rows, TEMPLATES) if t.kind == "pairwise"]
    steered_row = "decision/dictator|heldout|k+3|heldout_00|s1"
    mine = [t for t in tasks if t.row_id == steered_row]
    assert sorted(t.order for t in mine) == ["baseline_first", "steered_first"]
    steered = by_id[steered_row]["answer"]
    base = by_id["baseline|heldout|k+0|heldout_00|s1"]["answer"]
    for t in mine:
        first, second = (base, steered) if t.order == "baseline_first" else (steered, base)
        assert t.prompt.index(first) < t.prompt.index(second)
        assert t.steered_position == (2 if t.order == "baseline_first" else 1)


def test_shuffle_is_seeded_and_a_permutation():
    plan = gj.build_plan("primary", rows_for(n_questions=2), TEMPLATES)
    a, b = gj.shuffled(plan), gj.shuffled(plan)
    assert [t.key for t in a] == [t.key for t in b] != [t.key for t in plan]
    assert sorted(t.key for t in a) == sorted(t.key for t in plan)


def test_control_arms_are_not_loaded_for_judging(tmp_path):
    for arm in ("unhooked", "baseline"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "heldout_k+0.jsonl").write_text(
            json.dumps(row(arm, 0, "heldout", "heldout_00", 0)) + "\n")
    assert [r["arm"] for r in gj.load_generation_rows(tmp_path)] == ["baseline"]


def test_resend_selection_is_seeded_five_percent_of_sorted_distinct_payloads():
    shas = ["%064x" % i for i in range(200)] + ["%064x" % 3]
    chosen = gj.resend_selection(shas)
    assert len(chosen) == 10 and chosen == gj.resend_selection(reversed(shas))
    assert gj.resend_selection([]) == [] and len(gj.resend_selection(["a" * 64])) == 1


def test_latest_row_supersedes_a_transport_error():
    old = {"key": "k", "status": "transport_error"}
    new = {"key": "k", "status": "ok"}
    assert gj.done_keys([old]) == set() and gj.done_keys([old, new]) == {"k"}


def test_comparable_reads_top_logprobs_or_content():
    resp = {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "7", "logprob": -0.1}]}]},
                         "message": {"content": "7"}}]}
    assert gj.comparable(resp, "logprob_0_100") == [["7", -0.1]]
    assert gj.comparable(resp, "text_0_100") == "7"
    assert gj.comparable(None, "text_0_100") is None


# --- the CLI against a fake server ---------------------------------------------------

class FakeJudge:
    def __init__(self, model):
        outer = self
        self.model = model
        self.posts = 0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._json({"data": [{"id": outer.model}]} if self.path == "/v1/models" else {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.posts += 1
                top = [{"token": str(50 + i), "logprob": math.log(0.05)} for i in range(20)]
                self._json({"model": outer.model, "choices": [{
                    "finish_reason": "length", "message": {"content": "50"},
                    "logprobs": {"content": [{"token": "50", "logprob": -1, "top_logprobs": top}]}}]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def write_rows(directory):
    for r in rows_for(n_questions=2, samples=1, arms=("trait/altruism",)):
        path = directory / r["arm"].replace("/", "-") / ("%s_k%+g.jsonl" % (r["question_set"], r["k"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(r) + "\n")


def cli(env_url, rows, out, *extra):
    import os
    env = dict(os.environ, PIMIENTOROJO_BASE_URL=env_url)
    return subprocess.run([sys.executable, str(SCRIPTS / "judging" / "run_judging.py"), "--judge",
                           "primary", "--rows-dir", str(rows), "--out-dir", str(out), "--kinds",
                           "coherence"] + list(extra), capture_output=True, text=True, env=env,
                          cwd=str(SCRIPTS.parents[2]))


def test_cli_writes_one_row_per_call_and_a_rerun_sends_nothing(tmp_path):
    server = FakeJudge(gj.SERVED_IDS["primary"])
    try:
        rows, out = tmp_path / "rows", tmp_path / "judgments"
        write_rows(rows)
        first = cli(server.url, rows, out)
        assert first.returncode == 0, first.stderr
        written = gj.read_jsonl(out / gj.JUDGE_DIRS["primary"] / "coherence.jsonl")
        n_rows = sum(1 for _ in gj.load_generation_rows(rows))
        assert len(written) == n_rows and server.posts == n_rows
        assert {r["status"] for r in written} == {"ok"} and written[0]["score"] == pytest.approx(59.5)
        assert written[0]["served_model"] == gj.SERVED_IDS["primary"]
        assert written[0]["decoding"]["cache_prompt"] is False
        second = cli(server.url, rows, out)
        assert second.returncode == 0 and server.posts == n_rows
        assert len(gj.read_jsonl(out / gj.JUDGE_DIRS["primary"] / "coherence.jsonl")) == n_rows
    finally:
        server.close()


def test_cli_with_the_tunnel_down_stops_and_writes_nothing(tmp_path):
    server = FakeJudge(gj.SERVED_IDS["primary"])
    url = server.url
    server.close()
    rows, out = tmp_path / "rows", tmp_path / "judgments"
    write_rows(rows)
    result = cli(url, rows, out)
    assert result.returncode != 0 and "nothing written" in result.stderr
    assert not out.exists()


def test_cli_aborts_on_a_served_model_mismatch_and_writes_nothing(tmp_path):
    server = FakeJudge("some-other-model")
    try:
        rows, out = tmp_path / "rows", tmp_path / "judgments"
        write_rows(rows)
        result = cli(server.url, rows, out)
        assert result.returncode != 0 and "ModelMismatch" in result.stderr
        assert not out.exists() and server.posts == 0
    finally:
        server.close()


def test_pairwise_render_never_resubstitutes_a_field_name_found_in_an_answer():
    out = gj.render_pairwise("Q={question} 1={answer_1} 2={answer_2}", "what {answer_2}?", "A1 {question}", "A2")
    assert out == "Q=what {answer_2}? 1=A1 {question} 2=A2"


def test_plan_refuses_a_steered_row_without_its_baseline_pair():
    rows = [r for r in rows_for(n_questions=2) if not (r["arm"] == "baseline" and r["question_set"] == "heldout"
                                                        and r["question_id"] == "heldout_00")]
    with pytest.raises(ValueError, match="no baseline row"):
        gj.build_plan("primary", rows, TEMPLATES)


def test_plan_refuses_a_question_set_it_has_no_plan_for():
    rows = rows_for(n_questions=2) + [row("baseline", 0, "in_distribution", "q", 0)]
    with pytest.raises(ValueError, match="in_distribution"):
        gj.build_plan("primary", rows, TEMPLATES)


def test_fairness_template_sits_next_to_the_trait_one_and_is_pinned_in_the_prereg():
    prereg = (gj.REPO / "docs" / "fairness_exploratory_prereg.md").read_text(encoding="utf-8")
    fields = lambda text: sorted(set(re.findall(r"\{([^{}]*)\}", text)))  # noqa: E731
    assert fields(TEMPLATES["fairness"]) == fields(TEMPLATES["trait"]) == ["answer", "question"]
    assert TEMPLATES["fairness"] != TEMPLATES["trait"]
    assert gj.sha256_text(TEMPLATES["fairness"]) in prereg


def test_fairness_is_planned_only_when_asked_and_leaves_the_rest_of_the_plan_alone():
    rows = rows_for()
    default = gj.build_plan("primary", rows, TEMPLATES)
    assert not any(t.kind == "fairness" for t in default)
    plan = gj.build_plan("primary", rows, TEMPLATES, with_fairness=True)
    fair = [t for t in plan if t.kind == "fairness"]
    assert len(fair) == sum(r["question_set"] == "heldout" for r in rows)
    assert all("displays the trait: **fair**" in t.prompt for t in fair)
    assert [t.key for t in plan if t.kind != "fairness"] == [t.key for t in default]


def test_secondary_fairness_follows_the_same_pre_registered_subset():
    rows = rows_for(n_questions=12)
    plan = gj.build_plan("secondary", rows, TEMPLATES, with_fairness=True)
    scored_heldout = [t for t in plan if t.kind == "coherence"]
    fair = [t for t in plan if t.kind == "fairness"]
    assert len(fair) == len(scored_heldout) == 5 * gj.SUBSET_SIZE
