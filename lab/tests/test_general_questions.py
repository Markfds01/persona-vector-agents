"""`lab.general_questions` with a fake engine: no GPU, no model, no network.

The generation path's own guarantees are asserted on rows the fake engine produces:
ordering and seeding, the `truncated` status, whole-point resume checked against the vector
sha256, and the arithmetic of the hook-fire expectation.
"""

import json
from pathlib import Path

import pytest

from audit import generate
from lab import general_questions as gq

PACKAGE = Path(__file__).resolve().parents[2] / "results" / "general-question-transfer"
STOP = (151645, 151643)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        assert [m["role"] for m in messages] == ["user"]
        return "<sys/><user>%s</user><assistant>" % messages[0]["content"]


class FakeEngine:
    """Answers with scripted id lists; records every (prompts, seed) it was asked for."""

    columns = {"steer_coeff": 0.0}
    fired = 0

    def __init__(self, ids_for):
        self.ids_for = ids_for
        self.calls = []
        self.last_ids = []

    def generate(self, prompts, sampling, seed):
        self.calls.append((list(prompts), seed))
        self.last_ids = [self.ids_for(len(self.calls) - 1, i) for i in range(len(prompts))]
        return ["answer %d" % i for i in range(len(prompts))]


def record():
    description = generate.EngineDescription(
        engine="fake", engine_version="0", torch_version="0", model_id="m",
        model_revision="a" * 40, dtype="torch.bfloat16", attn_implementation="sdpa",
        stop_token_ids=STOP)
    return generate.Provenance(description, "t" * 64, "c" * 40, False)


def questions(n=3):
    return [gq.Question("heldout", "q%02d" % i, "question %d?" % i) for i in reversed(range(n))]


def point(engine, tmp_path, *, k=3.0, samples=2, batch_size=4, sha="v" * 64, name="arm/x",
          sampling=None):
    sampling = sampling or generate.preset("neutral").sampling.with_seed(7).__class__(
        temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
        max_new_tokens=5, min_new_tokens=1, seed=7)
    return gq.generate_point(
        engine, FakeTokenizer(), questions(), arm=gq.Arm(name, "decision", "/v.pt"), k=k,
        samples=samples, sampling=sampling, batch_size=batch_size, stop_ids=STOP,
        out_path=tmp_path / "p.jsonl", question_file_sha256="f" * 64, record=record(),
        vector_sha256=sha)


def stops_after(n):
    return lambda call, i: [1] * (n - 1) + [STOP[0]] + [STOP[1]] * (6 - n)


def test_rows_are_ordered_by_question_then_sample_and_batches_are_seeded(tmp_path):
    engine = FakeEngine(stops_after(3))
    summary = point(engine, tmp_path)
    rows = gq.read_rows(tmp_path / "p.jsonl")
    assert [(r["question_id"], r["sample_index"]) for r in rows] == [
        (q, s) for q in ("q00", "q01", "q02") for s in (0, 1)]
    assert [seed for _, seed in engine.calls] == [7, 8]
    assert [len(p) for p, _ in engine.calls] == [4, 2]
    assert [r["batch_index"] for r in rows] == [0, 0, 0, 0, 1, 1]
    assert summary["n_rows"] == 6 and summary["n_batches"] == 2
    # prompts are the question verbatim in one user turn, repeated per sample
    assert engine.calls[0][0][0] == "<sys/><user>question 0?</user><assistant>"
    assert engine.calls[0][0][0] == engine.calls[0][0][1]


def test_row_provenance_fields(tmp_path):
    point(FakeEngine(stops_after(3)), tmp_path, k=-3.0)
    row = gq.read_rows(tmp_path / "p.jsonl")[0]
    assert row["unit_beta"] == pytest.approx(-31.5249) and row["k"] == -3.0
    assert row["use_model_defaults"] is False and row["hooked"] is True
    assert row["model_revision"] == "a" * 40 and row["vector_sha256"] == "v" * 64
    assert row["max_new_tokens"] == 5 and row["seed"] == 7
    assert row["row_id"] == "arm/x|heldout|k-3|q00|s0"
    assert row["stop_token_ids"] == "151645|151643"
    assert len(row["prompt_sha256"]) == 64 and row["question"] == "question 0?"


def test_stop_found_is_ok_and_none_found_at_the_limit_is_truncated(tmp_path):
    def ids(call, i):
        return [1, 2, 3, 4, 5] if i % 2 else [1, 2, STOP[0], STOP[1], STOP[1]]
    summary = point(FakeEngine(ids), tmp_path, samples=1)
    rows = gq.read_rows(tmp_path / "p.jsonl")
    assert [r["generation_status"] for r in rows] == ["ok", "truncated", "ok"]
    assert [r["n_generated_tokens"] for r in rows] == [3, 5, 3]
    assert summary["n_truncated"] == 1


def test_a_stop_token_on_the_last_position_is_not_truncated(tmp_path):
    point(FakeEngine(lambda c, i: [1, 1, 1, 1, STOP[0]]), tmp_path, samples=1)
    assert {r["generation_status"] for r in gq.read_rows(tmp_path / "p.jsonl")} == {"ok"}


def test_trim_ids_cuts_at_the_first_stop_even_when_padding_follows():
    assert gq.trim_ids([5, 6, STOP[1], STOP[1], STOP[1]], STOP) == ([5, 6, STOP[1]], True)
    assert gq.trim_ids([5, 6, 7], STOP) == ([5, 6, 7], False)


def test_identical_ids_have_identical_hashes_and_different_ids_do_not(tmp_path):
    point(FakeEngine(stops_after(3)), tmp_path)
    first = [r["output_ids_sha256"] for r in gq.read_rows(tmp_path / "p.jsonl")]
    point(FakeEngine(stops_after(3)), tmp_path)
    assert first == [r["output_ids_sha256"] for r in gq.read_rows(tmp_path / "p.jsonl")]
    point(FakeEngine(stops_after(4)), tmp_path)
    assert first != [r["output_ids_sha256"] for r in gq.read_rows(tmp_path / "p.jsonl")]


def test_hook_fire_expectation_is_the_longest_row_per_batch(tmp_path):
    def ids(call, i):
        return stops_after(2 if i else 4)(call, i)
    summary = point(FakeEngine(ids), tmp_path)
    # two batches, each with one row of 4 tokens: 4 + 4 forward passes
    assert summary["hook_fires_expected"] == 8
    assert gq.expected_hook_fires([4, 2]) == 6


def test_unhooked_point_is_marked_and_has_no_steering_columns(tmp_path):
    class Plain(FakeEngine):
        columns = None
    point(Plain(stops_after(3)), tmp_path, sha=None, name="unhooked")
    row = gq.read_rows(tmp_path / "p.jsonl")[0]
    assert row["hooked"] is False and row["unit_beta"] == 0.0
    assert "steer_coeff" not in row and row["vector_sha256"] is None


def test_point_write_is_atomic_and_leaves_no_staging_file(tmp_path):
    point(FakeEngine(stops_after(3)), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.jsonl"]


def complete(path, **kw):
    args = dict(arm="arm/x", max_new_tokens=5, question_file_sha256="f" * 64)
    args.update(kw)
    return gq.point_complete(path, 6, args.pop("sha", "v" * 64), **args)


def test_resume_accepts_a_whole_point_of_this_vector(tmp_path):
    point(FakeEngine(stops_after(3)), tmp_path)
    assert complete(tmp_path / "p.jsonl") is True


def test_resume_regenerates_a_missing_or_short_point(tmp_path):
    assert complete(tmp_path / "p.jsonl") is False
    point(FakeEngine(stops_after(3)), tmp_path)
    lines = (tmp_path / "p.jsonl").read_text().splitlines()
    (tmp_path / "p.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert complete(tmp_path / "p.jsonl") is False


@pytest.mark.parametrize("kw", [{"sha": "w" * 64}, {"arm": "arm/y"}, {"max_new_tokens": 1000},
                                {"question_file_sha256": "0" * 64}])
def test_resume_refuses_a_whole_point_made_with_something_else(tmp_path, kw):
    point(FakeEngine(stops_after(3)), tmp_path)
    with pytest.raises(gq.GeneralQuestionError):
        complete(tmp_path / "p.jsonl", **kw)


def test_frozen_question_sets_load_with_their_sha():
    held, sha = gq.load_questions(PACKAGE, "heldout")
    bridge, bsha = gq.load_questions(PACKAGE, "bridge")
    assert len(held) == 40 and len(bridge) == 40 and sha != bsha
    assert held[0].question_id == "heldout_00" and "Answer: A" in bridge[0].text
    with pytest.raises(gq.GeneralQuestionError):
        gq.load_questions(PACKAGE, "nope")


def test_point_slug_and_row_id_carry_the_coefficient():
    assert gq.point_slug("orthogonal_null/dictator", "bridge", 3.0) == \
        "orthogonal_null-dictator/bridge_k+3"
    assert gq.row_id("baseline", "heldout", 0.0, "q1", 2) == "baseline|heldout|k+0|q1|s2"


def test_rows_are_valid_json_lines(tmp_path):
    point(FakeEngine(stops_after(3)), tmp_path)
    for line in (tmp_path / "p.jsonl").read_text().splitlines():
        assert isinstance(json.loads(line), dict)
