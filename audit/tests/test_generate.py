"""Generation config, provenance, row schema and writer — all without a model.

Nothing here loads weights. What the tests can prove is the CPU-side contract:
that a run cannot be configured with a missing sampling parameter, that every row
carries what it takes to reproduce it, and that an unparsed answer survives.
"""

import csv
import re
import subprocess
import sys

import pytest

from audit import elicit, generate
from audit.games import by_id
from audit.generate import (ROW_FIELDS, ConfigError, EngineDescription, ProvenanceError,
                            Row, Sampling)
from audit.paths import REPO_ROOT
from audit.tests.fakes import (FAKE_ENGINE_DESCRIPTION, QwenLikeTokenizer, ScriptedEngine,
                               SeededEngine, TemplatelessTokenizer)

DICTATOR = "altruism_v3/dictator"
TRUST = "altruism_v3/trust"

VALID_SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                      max_new_tokens=64, seed=7)


@pytest.fixture
def tokenizer():
    return QwenLikeTokenizer()


@pytest.fixture
def sampling():
    return Sampling(**VALID_SAMPLING)


# --- sampling config: nothing may default --------------------------------------

@pytest.mark.parametrize("omitted", generate.SAMPLING_FIELDS)
def test_a_missing_sampling_parameter_is_an_error(omitted):
    """Silence here is how `generation_config.json` gets to set temperature to 0.7."""
    mapping = {k: v for k, v in VALID_SAMPLING.items() if k != omitted}
    with pytest.raises(ConfigError) as excinfo:
        Sampling.from_mapping(mapping)
    assert omitted in str(excinfo.value)


def test_an_unrecognised_sampling_parameter_is_an_error():
    with pytest.raises(ConfigError) as excinfo:
        Sampling.from_mapping(dict(VALID_SAMPLING, min_p=0.1))
    assert "min_p" in str(excinfo.value)


@pytest.mark.parametrize("omitted", generate.SAMPLING_FIELDS)
def test_sampling_cannot_be_constructed_partially(omitted):
    kwargs = {k: v for k, v in VALID_SAMPLING.items() if k != omitted}
    with pytest.raises(TypeError):
        Sampling(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("temperature", -0.1),
    ("temperature", "hot"),
    ("top_p", 0.0),
    ("top_p", 1.5),
    ("top_k", -1),
    ("top_k", 0.5),
    ("repetition_penalty", 0.0),
    ("max_new_tokens", 0),
    ("max_new_tokens", True),
    ("seed", -1),
])
def test_an_out_of_range_sampling_parameter_is_rejected(field, value):
    with pytest.raises(ConfigError):
        Sampling.from_mapping(dict(VALID_SAMPLING, **{field: value}))


def test_generation_kwargs_state_every_knob(sampling):
    kwargs = sampling.generation_kwargs()
    assert set(kwargs) == {"do_sample", "temperature", "top_p", "top_k",
                           "repetition_penalty", "max_new_tokens"}
    assert kwargs["do_sample"] is True
    assert Sampling(**dict(VALID_SAMPLING, temperature=0.0)).generation_kwargs()[
        "do_sample"] is False


def test_presets_are_declared_in_one_place_and_none_is_recommended_yet():
    assert generate.PRESETS == {}
    with pytest.raises(ConfigError):
        generate.preset("paper")


# --- provenance ----------------------------------------------------------------

def test_an_incomplete_engine_description_is_refused():
    with pytest.raises(ProvenanceError):
        EngineDescription("hf", "4.0", "2.0", "qwen", "", "torch.bfloat16")


def test_provenance_needs_a_fingerprintable_chat_template():
    with pytest.raises(ValueError):
        generate.provenance(ScriptedEngine(["x"]), TemplatelessTokenizer())


def test_an_engine_that_does_not_describe_itself_is_refused(tokenizer):
    class Mute:
        def describe(self):
            return {"engine": "hf"}

    with pytest.raises(ProvenanceError):
        generate.provenance(Mute(), tokenizer)


def test_the_repo_commit_is_recorded():
    assert re.fullmatch(r"[0-9a-f]{40}", generate.repo_commit())
    assert isinstance(generate.repo_is_dirty(), bool)


# --- row schema ----------------------------------------------------------------

def test_the_row_schema_is_pinned():
    """A downstream analysis reads these columns; changing one is a breaking change."""
    assert ROW_FIELDS == (
        "game_id", "question_id", "question_set", "family", "mode", "persona", "reading",
        "sample_index", "batch_index", "seed", "continuation", "answer", "value", "tag",
        "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens",
        "engine", "engine_version", "torch_version", "model_id", "model_revision",
        "dtype", "chat_template_sha256", "question_sha256", "prompt_sha256",
        "repo_commit", "repo_dirty")


def test_every_row_carries_its_whole_provenance(tokenizer, sampling):
    engine = ScriptedEngine(["I will give $30 to Agent 2."])
    rows = generate.run([(DICTATOR, "free")], engine, sampling, tokenizer,
                        samples_per_prompt=1)
    row = rows[0]
    assert row.game_id == DICTATOR
    assert row.question_id == "altruism_0"
    assert row.question_set == "altruism_v3" and row.family == "dictator"
    assert row.mode == "free" and row.persona == "" and row.reading == "stated"
    assert row.seed == 7 and row.sample_index == 0 and row.batch_index == 0
    assert (row.temperature, row.top_p, row.top_k) == (1.0, 1.0, 0)
    assert (row.repetition_penalty, row.max_new_tokens) == (1.0, 64)
    assert row.engine == "fake" and row.engine_version == "0.0.0"
    assert row.torch_version == "0.0.0" and row.dtype == "torch.float32"
    assert row.model_id == "fake/model" and row.model_revision == "0" * 40
    assert len(row.chat_template_sha256) == 16
    assert row.question_sha256 == "ac8f242ed284c309"
    assert row.prompt_sha256 == elicit.render(
        by_id(DICTATOR), "free", tokenizer).sha256
    assert re.fullmatch(r"[0-9a-f]{40}", row.repo_commit)
    # nothing a reproduction needs may be blank
    for name in ROW_FIELDS:
        if name not in ("persona", "value", "repo_dirty", "sample_index", "batch_index"):
            assert getattr(row, name) != "", name


# --- running -------------------------------------------------------------------

def test_one_row_per_sample_per_pair(tokenizer, sampling):
    engine = ScriptedEngine(["I will give $30 to Agent 2."])
    rows = generate.run([(DICTATOR, "free"), (TRUST, "stub")], engine, sampling,
                        tokenizer, samples_per_prompt=3)
    assert len(rows) == 6
    assert [r.game_id for r in rows] == [DICTATOR] * 3 + [TRUST] * 3
    assert [r.sample_index for r in rows] == [0, 1, 2, 0, 1, 2]


def test_the_engine_is_handed_exactly_the_rendered_prompts(tokenizer, sampling):
    engine = ScriptedEngine(["$30."])
    generate.run([(DICTATOR, "stub")], engine, sampling, tokenizer, samples_per_prompt=2)
    prompts, _seed = engine.calls[0]
    expected = elicit.render(by_id(DICTATOR), "stub", tokenizer).text
    assert prompts == [expected, expected]


def test_the_scored_answer_includes_the_modes_prefill(tokenizer, sampling):
    """"40 to Agent 2." only names a recipient once the stub is put back in front."""
    engine = ScriptedEngine(["40 to Agent 2."])
    row = generate.run([(DICTATOR, "stub")], engine, sampling, tokenizer,
                       samples_per_prompt=1)[0]
    assert row.continuation == "40 to Agent 2."
    assert row.answer == "I will give $40 to Agent 2."
    assert row.value == 40.0 and row.tag == "a2_anchor"


def test_an_unparsed_answer_keeps_its_row_and_its_tag(tokenizer, sampling):
    engine = ScriptedEngine(["I will give $30 to Agent 2.",
                             "I cannot answer that.",
                             "Interesting question."])
    rows = generate.run([(DICTATOR, "free")], engine, sampling, tokenizer,
                        samples_per_prompt=3, batch_size=3)
    assert len(rows) == 3
    assert [(r.value, r.tag) for r in rows] == [
        (30.0, "a2_anchor"), (None, "refusal"), (None, "unparsed")]


def test_the_chosen_reading_bounds_the_score(tokenizer, sampling):
    engine = ScriptedEngine(["I would give $60 to Agent 2."])
    contradictory = "altruism_v1/dictator"
    stated = generate.run([(contradictory, "free")], engine, sampling, tokenizer,
                          samples_per_prompt=1)[0]
    implied = generate.run([(contradictory, "free")], engine, sampling, tokenizer,
                           samples_per_prompt=1, reading="implied")[0]
    assert (stated.value, stated.reading) == (60.0, "stated")
    # $60 is unpayable out of a $10 endowment, so it resolves to nothing, not a guess
    assert (implied.value, implied.reading) == (None, "implied")


def test_a_reading_the_game_does_not_declare_fails_before_generating(tokenizer, sampling):
    engine = ScriptedEngine(["I will give $30 to Agent 2."])
    with pytest.raises(KeyError):
        generate.run([(DICTATOR, "free")], engine, sampling, tokenizer,
                     samples_per_prompt=1, reading="implied")
    assert engine.calls == []


def test_batches_are_seeded_from_the_run_seed_and_recorded(tokenizer, sampling):
    engine = ScriptedEngine(["I will give $30 to Agent 2."])
    rows = generate.run([(DICTATOR, "free")], engine, sampling, tokenizer,
                        samples_per_prompt=5, batch_size=2)
    assert [seed for _prompts, seed in engine.calls] == [7, 8, 9]
    assert [r.batch_index for r in rows] == [0, 0, 1, 1, 2]
    assert {r.seed for r in rows} == {7}


def test_a_run_is_reproducible_from_its_seed(tokenizer, sampling):
    def once(seed):
        return generate.run([(DICTATOR, "stub")], SeededEngine(),
                            Sampling(**dict(VALID_SAMPLING, seed=seed)), tokenizer,
                            samples_per_prompt=4, batch_size=2)

    assert once(7) == once(7)
    assert [r.answer for r in once(7)] != [r.answer for r in once(11)]


def test_batch_size_is_part_of_the_configuration(tokenizer, sampling):
    """Batch composition changes which draws a prompt gets — same seed, other rows."""
    def once(batch_size):
        return [r.answer for r in generate.run(
            [(DICTATOR, "stub")], SeededEngine(), sampling, tokenizer,
            samples_per_prompt=4, batch_size=batch_size)]

    assert once(4) != once(2)


def test_an_engine_that_returns_the_wrong_number_of_answers_fails_loudly(
        tokenizer, sampling):
    class Short:
        def describe(self):
            return FAKE_ENGINE_DESCRIPTION

        def generate(self, prompts, sampling, seed):
            return ["only one"]

    with pytest.raises(RuntimeError):
        generate.run([(DICTATOR, "free")], Short(), sampling, tokenizer,
                     samples_per_prompt=2)


@pytest.mark.parametrize("kwargs", [
    dict(samples_per_prompt=0),
    dict(samples_per_prompt=1, batch_size=0),
])
def test_a_run_that_would_produce_nothing_is_refused(tokenizer, sampling, kwargs):
    with pytest.raises(ConfigError):
        generate.run([(DICTATOR, "free")], ScriptedEngine(["x"]), sampling, tokenizer,
                     **kwargs)


@pytest.mark.parametrize("pair", [("altruism_v3/chess", "free"), (DICTATOR, "cot")])
def test_an_undeclared_game_or_mode_is_refused(tokenizer, pair):
    with pytest.raises(KeyError):
        generate.plan([pair], tokenizer, samples_per_prompt=1)


# --- writer --------------------------------------------------------------------

def test_the_writer_round_trips_every_column(tmp_path, tokenizer, sampling):
    engine = ScriptedEngine(["I will give $30 to Agent 2.", "I cannot answer that."])
    rows = generate.run([(DICTATOR, "free")], engine, sampling, tokenizer,
                        samples_per_prompt=2, batch_size=2)
    path = generate.write_rows(tmp_path / "runs" / "out.csv", rows)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == ROW_FIELDS
        written = list(reader)
    assert len(written) == 2
    assert written[0]["value"] == "30.0" and written[0]["tag"] == "a2_anchor"
    # an unresolved answer is an empty cell, never a zero
    assert written[1]["value"] == "" and written[1]["tag"] == "refusal"
    assert written[1]["answer"] == "I cannot answer that."


def test_the_writer_takes_rows_only(tmp_path):
    with pytest.raises(TypeError):
        generate.write_rows(tmp_path / "out.csv", [{"game_id": "x"}])


def test_a_row_cannot_be_built_with_a_missing_column():
    with pytest.raises(TypeError):
        Row(game_id="x")


# --- boundaries ----------------------------------------------------------------

def test_importing_generate_pulls_in_neither_torch_nor_transformers():
    """The engine imports them when constructed; the module must not need them."""
    probe = ("import sys, audit.generate; "
             "assert 'torch' not in sys.modules, 'imported torch'; "
             "assert 'transformers' not in sys.modules, 'imported transformers'")
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO_ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
