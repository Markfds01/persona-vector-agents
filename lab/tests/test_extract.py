"""The shared teacher-forced capture, without weights.

What these prove is the whole CPU-side contract of `lab.extract`: that a row is
refused unless its prompt re-renders to the bytes it was generated from, that the
three poolings read the positions they claim to, that the reduction happens in the
activation's own dtype before the cast, and that a row the tokenizer cannot split
cleanly is dropped and named rather than pooled with the wrong slice.

The model is a fake whose hidden states encode their own (layer, position), so a
slice that is off by one is visible in the assertion rather than plausible.

`torch` is imported outright rather than skipped past: the whole contract here is
about tensors, `requirements.txt` pins it, and a module-scope skip turned a
torch-less environment into a green run that had checked nothing.
"""

import csv
import errno
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from audit import elicit  # noqa: E402
from audit.games import by_id  # noqa: E402
from audit.generate import ProvenanceError  # noqa: E402
from audit.tests import fakes  # noqa: E402
from audit.tests.fakes import QwenLikeTokenizer  # noqa: E402
from lab import extract  # noqa: E402
from lab.extract import (EMPTY_RESPONSE, NON_FINITE, POOLINGS,  # noqa: E402
                         PREFIX_UNSTABLE, Dropped, PromptMismatch, Seam,
                         ShardWriter, ShardsUnusable, captured_row_indices,
                         capture, load_grid, load_model, non_finite_poolings,
                         pool, scored, seam_of, shards_cover, sweep_staging,
                         write_json_atomically, Prompts)

DICTATOR = "altruism_v3/dictator"
TRUST = "altruism_v3/trust"
N_LAYERS = 3


class CharTokenizer(QwenLikeTokenizer):
    """One id per character, no special tokens, no padding — batch-of-one only.

    Enough of the HF tokenizer surface for the capture loop: `encode` for the
    prompt alone and `__call__(return_tensors="pt")` for prompt+answer, which is
    the single tokenisation the forward pass is fed.
    """

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "the chat template already wrote them"
        return [ord(character) for character in text]

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        ids = self.encode(text, add_special_tokens=False)
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


class PositionCodedModel:
    """Hidden states that spell out their own coordinates: state[l][0, p] == l*1000+p.

    A pooling that reads the wrong span produces a wrong number, not a plausible
    one, so the assertions below are about the slice and not about the arithmetic.
    """

    def __init__(self, n_layers=N_LAYERS, hidden=4, dtype=torch.float32,
                 n_states=None):
        self.config = types.SimpleNamespace(num_hidden_layers=n_layers,
                                            hidden_size=hidden)
        self.device = "cpu"
        self.dtype = dtype
        self._hidden = hidden
        #: what the pass returns, which a broken model can disagree with its config about
        self.n_states = n_states if n_states is not None else n_layers + 1
        self.calls = []

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        assert output_hidden_states is True
        assert input_ids.shape[0] == 1, "batch size 1 is the method, not a default"
        self.calls.append(int(input_ids.shape[1]))
        total = input_ids.shape[1]
        states = tuple(
            torch.full((1, total, self._hidden), 0.0, dtype=self.dtype)
            + torch.arange(total, dtype=self.dtype).view(1, -1, 1)
            + layer * 1000.0
            for layer in range(self.n_states))
        return types.SimpleNamespace(hidden_states=states)


class EvictedModel(PositionCodedModel):
    """Dies partway through, the way a shared card takes a run away from you."""

    def __init__(self, rows_before_eviction, **kwargs):
        super().__init__(**kwargs)
        self._limit = rows_before_eviction

    def __call__(self, **kwargs):
        if len(self.calls) >= self._limit:
            raise RuntimeError("evicted")
        return super().__call__(**kwargs)


@pytest.fixture
def tokenizer():
    return CharTokenizer()


def make_row(tokenizer, game_id=DICTATOR, mode="free", answer="50", value="50.0",
             prompt_sha256=None):
    rendered = elicit.render(by_id(game_id), mode, tokenizer)
    return {"game_id": game_id, "mode": mode, "answer": answer, "value": value,
            "prompt_sha256": prompt_sha256 or rendered.sha256}


# --- which rows are captured at all ---------------------------------------------

def test_only_scored_rows_are_captured():
    """An unscored row carries no pole label; guessing one would invent data."""
    rows = [{"value": "1.0"}, {"value": ""}, {"value": "0.0"}]
    assert [index for index, _row in scored(rows)] == [0, 2]


def test_row_indices_survive_the_unscored_gaps():
    """The index is the row's position in its CSV — the shards join back on it."""
    rows = [{"value": ""}, {"value": ""}, {"value": "3.0"}]
    assert scored(rows) == [(2, rows[2])]


# --- the seam: where the prompt stops and the answer starts ----------------------

def test_a_stable_seam_reports_where_the_answer_starts():
    seam, reason = seam_of([1, 2, 3], [1, 2, 3, 4, 5])
    assert reason == ""
    assert seam == Seam((1, 2, 3, 4, 5), 3)


def test_a_merged_seam_is_dropped_and_named():
    """BPE merges ' ' + 'C' into one ' C': the response slice would hold prompt tokens."""
    seam, reason = seam_of([1, 2, 9], [1, 2, 77])
    assert seam is None
    assert reason == PREFIX_UNSTABLE


def test_an_empty_response_is_dropped_and_named():
    """`response_avg` over an empty span is a NaN, which must never reach a mean."""
    seam, reason = seam_of([1, 2, 3], [1, 2, 3])
    assert seam is None
    assert reason == EMPTY_RESPONSE


@pytest.mark.parametrize("prompt_len", [0, 3])
def test_a_seam_cannot_be_constructed_outside_its_ids(prompt_len):
    with pytest.raises(ValueError):
        Seam((1, 2, 3), prompt_len)


# --- the prompt has to be the one the row was generated from ---------------------

def test_the_prompt_is_re_rendered_and_returned(tokenizer):
    row = make_row(tokenizer)
    assert Prompts(tokenizer).for_row(row) == elicit.render(
        by_id(DICTATOR), "free", tokenizer).text


def test_a_prompt_that_no_longer_renders_the_same_is_refused(tokenizer):
    row = make_row(tokenizer, prompt_sha256="0" * 16)
    with pytest.raises(PromptMismatch) as excinfo:
        Prompts(tokenizer).for_row(row)
    assert DICTATOR in str(excinfo.value)


def test_the_fingerprint_is_checked_per_row_not_per_cache_miss(tokenizer):
    """A cache that skipped the check would let one bad row through per cell."""
    prompts = Prompts(tokenizer)
    prompts.for_row(make_row(tokenizer))
    with pytest.raises(PromptMismatch):
        prompts.for_row(make_row(tokenizer, prompt_sha256="0" * 16))


def test_two_games_do_not_share_a_cached_prompt(tokenizer):
    prompts = Prompts(tokenizer)
    assert prompts.for_row(make_row(tokenizer, game_id=DICTATOR)) != \
        prompts.for_row(make_row(tokenizer, game_id=TRUST))


# --- the three poolings ----------------------------------------------------------

def hidden_states(total, n_layers=N_LAYERS, hidden=2, dtype=torch.float32):
    return tuple(torch.arange(total, dtype=dtype).view(1, -1, 1).expand(1, total, hidden)
                 .contiguous() + layer * 1000.0 for layer in range(n_layers + 1))


def test_pooling_reads_the_positions_it_claims():
    states = hidden_states(total=10)
    pooled = pool(states, prompt_len=4)
    assert set(pooled) == set(POOLINGS)
    for layer in range(N_LAYERS + 1):
        base = layer * 1000.0
        assert pooled["prompt_avg"][layer].tolist() == pytest.approx([base + 1.5] * 2)
        assert pooled["prompt_last"][layer].tolist() == pytest.approx([base + 3.0] * 2)
        assert pooled["response_avg"][layer].tolist() == pytest.approx([base + 6.5] * 2)


def test_pooling_covers_every_hidden_state():
    """29 states for 28 layers: the embedding output is layer 0 and is not optional."""
    pooled = pool(hidden_states(total=6, n_layers=28), prompt_len=2)
    assert all(tensor.shape[0] == 29 for tensor in pooled.values())


def test_pooling_is_float32_whatever_the_pass_ran_in():
    pooled = pool(hidden_states(total=6, dtype=torch.bfloat16), prompt_len=2)
    assert all(tensor.dtype is torch.float32 for tensor in pooled.values())


def test_the_mean_is_taken_before_the_cast():
    """Upstream's order. Casting first gives different bits, so it is pinned here."""
    states = tuple(torch.tensor([[[0.1, 0.3], [0.2, 0.7], [0.4, 0.9]]],
                                dtype=torch.bfloat16) for _ in range(N_LAYERS + 1))
    got = pool(states, prompt_len=1)["response_avg"][0]
    expected = states[0][0, 1:, :].mean(dim=0).float()
    assert torch.equal(got, expected)
    assert not torch.equal(got, states[0][0, 1:, :].float().mean(dim=0))


def test_a_non_finite_activation_is_named_rather_than_raised():
    """A property of ONE row's activations, so it is a drop reason, not a raise.

    Raising here killed the run and took every completed forward pass still
    queued in the writer with it.
    """
    states = list(hidden_states(total=4))
    states[0] = states[0].clone()
    states[0][0, 2, 0] = float("nan")
    assert non_finite_poolings(pool(tuple(states), prompt_len=2)) == ["response_avg"]


def test_a_finite_pooling_names_nothing():
    assert non_finite_poolings(pool(hidden_states(total=4), prompt_len=2)) == []


@pytest.mark.parametrize("prompt_len", [0, 4])
def test_pooling_refuses_a_span_it_cannot_split(prompt_len):
    with pytest.raises(ValueError):
        pool(hidden_states(total=4), prompt_len)


def test_pooling_refuses_a_batch(tokenizer):
    """Padding would make the position slices mean something else."""
    states = tuple(torch.zeros(2, 5, 2) for _ in range(N_LAYERS + 1))
    with pytest.raises(ValueError) as excinfo:
        pool(states, prompt_len=2)
    assert "batch" in str(excinfo.value)


# --- shards ----------------------------------------------------------------------

def add_rows(writer, count, first=0, total=6, prompt_len=2):
    for offset in range(count):
        writer.add(first + offset, Seam(tuple(range(total)), prompt_len),
                   pool(hidden_states(total=total), prompt_len))


def test_a_shard_is_written_only_once_it_is_complete(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=3)
    add_rows(writer, 2)
    assert list(tmp_path.glob("shard_*.pt")) == []
    add_rows(writer, 1, first=2)
    assert [path.name for path in sorted(tmp_path.glob("shard_*.pt"))] == ["shard_0000.pt"]


def test_flush_writes_the_remainder(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=3)
    add_rows(writer, 4)
    assert writer.flush() == 1
    assert writer.written == 4
    assert len(list(tmp_path.glob("shard_*.pt"))) == 2


def test_a_shard_carries_the_join_key_and_every_pooling(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=2)
    add_rows(writer, 2, first=7, total=6, prompt_len=2)
    payload = torch.load(tmp_path / "shard_0000.pt", map_location="cpu")
    assert payload["row_index"].tolist() == [7, 8]
    assert payload["prompt_len"].tolist() == [2, 2]
    assert payload["total_len"].tolist() == [6, 6]
    for name in POOLINGS:
        assert payload[name].shape == (2, N_LAYERS + 1, 2)


def test_shard_numbering_continues_from_where_a_run_stopped(tmp_path):
    writer = ShardWriter(tmp_path, first_index=2, shard_rows=1)
    add_rows(writer, 1)
    assert (tmp_path / "shard_0002.pt").is_file()


def test_resume_reads_the_captured_rows_off_the_shards(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=2)
    add_rows(writer, 5, first=10)
    writer.flush()
    seen, n_shards = captured_row_indices(tmp_path)
    assert seen == {10, 11, 12, 13, 14}
    assert n_shards == 3


def test_a_shard_still_lands_where_hard_links_are_unavailable(tmp_path, monkeypatch):
    """FUSE, exFAT and some network mounts have no os.link. Refusing to write there
    would cost a corpus; the fallback rename costs only the EEXIST detection."""
    def no_links(src, dst):
        raise OSError(errno.EPERM, "no hard links here")

    monkeypatch.setattr(extract.os, "link", no_links)
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    assert (tmp_path / "shard_0000.pt").is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert captured_row_indices(tmp_path)[0] == {0}


def test_a_shard_another_run_already_claimed_is_refused(tmp_path):
    (tmp_path / "shard_0000.pt").write_bytes(b"someone else's shard")
    with pytest.raises(ShardsUnusable):
        add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    assert (tmp_path / "shard_0000.pt").read_bytes() == b"someone else's shard"
    assert not list(tmp_path.glob("*.tmp"))


def test_a_shard_another_run_claimed_is_refused_without_hard_links(tmp_path,
                                                                   monkeypatch):
    """The fallback must still not overwrite: os.replace would clobber it silently."""
    def no_links(src, dst):
        raise OSError(errno.EPERM, "no hard links here")

    monkeypatch.setattr(extract.os, "link", no_links)
    (tmp_path / "shard_0000.pt").write_bytes(b"someone else's shard")
    with pytest.raises(ShardsUnusable):
        add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    assert (tmp_path / "shard_0000.pt").read_bytes() == b"someone else's shard"
    assert not list(tmp_path.glob("*.tmp"))


def test_a_kill_during_the_shard_write_leaks_no_staging_file(tmp_path, monkeypatch):
    """A shard stages at ~312 MB and nothing else sweeps one; the budget is ~8 GB
    against a ~5.3 GB corpus, so a leak per interrupted run is a real ceiling."""
    staged = []

    def die(payload, path):
        staged.append(Path(path))
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(torch, "save", die)
    with pytest.raises(OSError):
        add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    assert staged and not list(tmp_path.glob("*.tmp"))


def test_two_shard_writes_do_not_share_a_staging_name(tmp_path, monkeypatch):
    """os.getpid() is not unique across PID namespaces sharing one volume."""
    staged = []
    real_save = torch.save

    def record(payload, path):
        staged.append(Path(path).name)
        return real_save(payload, path)

    monkeypatch.setattr(torch, "save", record)
    add_rows(ShardWriter(tmp_path, shard_rows=1), 2)
    assert len(staged) == 2 and staged[0] != staged[1]


def test_staging_files_an_interrupted_run_left_are_swept(tmp_path):
    (tmp_path / "shard_0000.pt.12345.tmp").write_bytes(b"half a shard")
    (tmp_path / "meta.json.12345.tmp").write_bytes(b"{")
    keep = tmp_path / "shard_0000.pt"
    keep.write_bytes(b"a landed shard")
    assert sweep_staging(tmp_path) == 2
    assert not list(tmp_path.glob("*.tmp"))
    assert keep.read_bytes() == b"a landed shard"


def test_a_shard_that_cannot_be_read_back_is_named(tmp_path):
    """torch.load's UnpicklingError went straight past main and printed a traceback."""
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    (tmp_path / "shard_0000.pt").write_bytes(b"not a torch archive")
    with pytest.raises(ShardsUnusable) as excinfo:
        captured_row_indices(tmp_path)
    assert "shard_0000.pt" in str(excinfo.value)


def test_the_fallback_claims_the_name_before_it_renames(tmp_path, monkeypatch):
    """Where there are no hard links the claim is an O_EXCL create, which is atomic
    on every such filesystem. A check-then-rename let two runs both pass the same
    `exists()` and land their own blocks under one numbering."""
    def no_links(src, dst):
        raise OSError(errno.EPERM, "no hard links here")

    monkeypatch.setattr(extract.os, "link", no_links)
    claimed = {}
    real_replace = extract.os.replace

    def watch(src, dst):
        target = Path(dst)
        claimed["before_rename"] = target.is_file() and target.stat().st_size == 0
        return real_replace(src, dst)

    monkeypatch.setattr(extract.os, "replace", watch)
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    assert claimed["before_rename"], "the shard name must be taken before the rename"
    assert captured_row_indices(tmp_path)[0] == {0}


def test_a_hole_in_the_shard_numbering_is_refused(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=1)
    add_rows(writer, 3)
    (tmp_path / "shard_0001.pt").unlink()
    with pytest.raises(ShardsUnusable):
        captured_row_indices(tmp_path)


# --- the loop end to end ----------------------------------------------------------

def test_capture_pools_every_scored_row(tmp_path, tokenizer):
    rows = [make_row(tokenizer, answer="50"), make_row(tokenizer, answer="0"),
            dict(make_row(tokenizer, answer="25"), value="")]
    model = PositionCodedModel()
    done = capture(rows, model, tokenizer, tmp_path, shard_rows=2)
    assert done.n_scored == 2
    assert done.n_written_now == 2
    assert done.dropped == ()
    assert len(model.calls) == 2
    payload = torch.load(tmp_path / "shard_0000.pt", map_location="cpu")
    assert payload["row_index"].tolist() == [0, 1]


def test_capture_pools_the_answer_it_was_given(tmp_path, tokenizer):
    """`response_avg` must cover exactly the answer's positions, not one more."""
    row = make_row(tokenizer, answer="50")
    prompt_len = len(tokenizer.encode(
        elicit.render(by_id(DICTATOR), "free", tokenizer).text, add_special_tokens=False))
    capture([row], PositionCodedModel(), tokenizer, tmp_path, shard_rows=1)
    payload = torch.load(tmp_path / "shard_0000.pt", map_location="cpu")
    assert payload["prompt_len"].tolist() == [prompt_len]
    assert payload["total_len"].tolist() == [prompt_len + 2]
    assert payload["response_avg"][0, 0].tolist() == pytest.approx(
        [prompt_len + 0.5] * 4)


def test_an_empty_answer_is_dropped_counted_and_named(tmp_path, tokenizer):
    rows = [make_row(tokenizer, answer=""), make_row(tokenizer, answer="50")]
    done = capture(rows, PositionCodedModel(), tokenizer, tmp_path, shard_rows=4)
    assert done.dropped == (Dropped(0, DICTATOR, EMPTY_RESPONSE),)
    assert done.n_scored == 2
    assert done.n_written_now == 1


def test_capture_skips_what_a_previous_invocation_already_wrote(tmp_path, tokenizer):
    rows = [make_row(tokenizer, answer="50"), make_row(tokenizer, answer="0")]
    model = PositionCodedModel()
    done = capture(rows, model, tokenizer, tmp_path, skip={0}, first_shard=1,
                   shard_rows=1)
    assert model.calls == [len(tokenizer.encode(
        Prompts(tokenizer).for_row(rows[1]), add_special_tokens=False)) + 1]
    assert done.n_scored == 2
    assert done.n_written_now == 1
    assert torch.load(tmp_path / "shard_0001.pt",
                      map_location="cpu")["row_index"].tolist() == [1]


def test_capture_refuses_a_row_whose_prompt_moved(tmp_path, tokenizer):
    rows = [make_row(tokenizer, prompt_sha256="0" * 16)]
    with pytest.raises(PromptMismatch):
        capture(rows, PositionCodedModel(), tokenizer, tmp_path)


def test_capture_refuses_a_model_with_the_wrong_state_count(tmp_path, tokenizer):
    """29 states for 28 layers is a fact about the checkpoint, not an assumption."""
    model = PositionCodedModel(n_states=N_LAYERS)
    with pytest.raises(RuntimeError) as excinfo:
        capture([make_row(tokenizer)], model, tokenizer, tmp_path)
    assert "hidden states" in str(excinfo.value)


class PoisonedModel(PositionCodedModel):
    """Returns a NaN on the nth forward pass; every other row is clean."""

    def __init__(self, poison_call=1, **kwargs):
        super().__init__(**kwargs)
        self._poison_call = poison_call

    def __call__(self, **kwargs):
        out = super().__call__(**kwargs)
        if len(self.calls) != self._poison_call:
            return out
        states = list(out.hidden_states)
        states[0] = states[0].clone()
        states[0][0, -1, 0] = float("nan")
        out.hidden_states = tuple(states)
        return out


def test_a_non_finite_activation_is_dropped_counted_and_named(tmp_path, tokenizer):
    rows = [make_row(tokenizer, answer="50"), make_row(tokenizer, answer="0")]
    done = capture(rows, PoisonedModel(poison_call=1), tokenizer, tmp_path,
                   shard_rows=4)
    assert len(done.dropped) == 1
    dropped = done.dropped[0]
    assert (dropped.row_index, dropped.game_id) == (0, DICTATOR)
    assert dropped.reason.startswith(NON_FINITE) and "response_avg" in dropped.reason
    assert done.n_written_now == 1


def test_a_failing_row_still_lands_the_rows_before_it(tmp_path, tokenizer):
    """One raise used to throw away up to `shard_rows` - 1 completed forward passes.

    A shard was only ever flushed by a full block or by the clean exit, so the rows
    already captured behind a mid-block failure went with it — and since `--resume`
    reads the SHARDS, the documented recovery re-ran them forever.
    """
    rows = [make_row(tokenizer, answer=str(index)) for index in range(9)]
    evicted = EvictedModel(7, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        capture(rows, evicted, tokenizer, tmp_path, shard_rows=10)
    seen, n_shards = captured_row_indices(tmp_path)
    assert seen == {0, 1, 2, 3, 4, 5, 6} and n_shards == 1


# --- the grid is data, named on the command line ----------------------------------

def test_a_grid_module_is_imported_and_registered(tmp_path):
    path = tmp_path / "toy_grid.py"
    path.write_text("REGISTERED = []\n\n\ndef register():\n"
                    "    REGISTERED.append(1)\n    return ()\n", encoding="utf-8")
    assert load_grid(path).REGISTERED == [1]


def test_a_module_that_registers_nothing_is_refused(tmp_path):
    path = tmp_path / "not_a_grid.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ImportError) as excinfo:
        load_grid(path)
    assert "register()" in str(excinfo.value)


# --- what the meta has to say ------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def pin(model, commit="a" * 40, attn="sdpa", name_or_path=MODEL_ID):
    """Give a fake model the provenance `meta_record` reads off a real one."""
    model.config._commit_hash = commit
    model.config._attn_implementation = attn
    model.config._name_or_path = name_or_path
    model.dtype = torch.bfloat16
    return model


def pinned_model(**provenance):
    return pin(PositionCodedModel(dtype=torch.bfloat16), **provenance)


def write_rows_csv(path, rows):
    """The rows a run reads, in the CSV shape `main` parses."""
    fields = ["game_id", "mode", "persona", "answer", "value", "prompt_sha256"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def test_the_meta_records_what_ran_not_what_was_asked_for(tokenizer, monkeypatch,
                                                          tmp_path):
    monkeypatch.setitem(sys.modules, "transformers", fakes.fake_transformers("4.52.3"))
    rows_csv = write_rows_csv(tmp_path / "rows.csv", [make_row(tokenizer)])
    meta = extract.meta_record(pinned_model(), tokenizer, rows_csv, 1)
    assert meta["model_id"] == MODEL_ID
    assert meta["model_revision"] == "a" * 40
    assert meta["attn_implementation"] == "sdpa"
    assert meta["dtype"] == "torch.bfloat16"
    assert meta["device"] == "cuda:1"
    assert meta["batch_size"] == 1 and meta["padding"] == "none"
    assert meta["n_hidden_states"] == meta["num_hidden_layers"] + 1
    assert meta["transformers_version"] == "4.52.3"
    assert meta["torch_version"] == torch.__version__
    assert meta["chat_template_sha256"] == elicit.chat_template_fingerprint(tokenizer)
    assert json.loads(json.dumps(meta))["poolings"] == list(POOLINGS)


def test_the_meta_refuses_a_revision_the_weights_cannot_support(tokenizer, monkeypatch,
                                                                tmp_path):
    """A resolver that echoed the request back would let this file invent provenance."""
    monkeypatch.setitem(sys.modules, "transformers", fakes.fake_transformers())
    rows_csv = write_rows_csv(tmp_path / "rows.csv", [make_row(tokenizer)])
    with pytest.raises(ProvenanceError):
        extract.meta_record(pinned_model(commit="main"), tokenizer, rows_csv, 0)


def test_the_meta_refuses_a_model_with_no_checkpoint_name(tokenizer, monkeypatch,
                                                          tmp_path):
    """An empty `model_id` is no provenance at all, so it is refused rather than written."""
    monkeypatch.setitem(sys.modules, "transformers", fakes.fake_transformers())
    rows_csv = write_rows_csv(tmp_path / "rows.csv", [make_row(tokenizer)])
    with pytest.raises(ProvenanceError):
        extract.meta_record(pinned_model(name_or_path=""), tokenizer, rows_csv, 0)


def test_model_id_is_the_request_and_the_revision_is_the_resolution(tokenizer,
                                                                    monkeypatch,
                                                                    tmp_path):
    """`from_pretrained` copies its own argument onto the config, so `model_id` is a
    label that cannot disagree with the caller. The revision is the real identity."""
    monkeypatch.setitem(sys.modules, "transformers", fakes.fake_transformers())
    rows_csv = write_rows_csv(tmp_path / "rows.csv", [make_row(tokenizer)])
    meta = extract.meta_record(pinned_model(name_or_path="/some/local/snapshot"),
                               tokenizer, rows_csv, 0)
    assert meta["model_id"] == "/some/local/snapshot"
    assert meta["model_revision"] == "a" * 40


def test_the_meta_fingerprints_the_rows_csv_by_content(tokenizer, monkeypatch, tmp_path):
    """Two CSVs at different paths with the same bytes are the same input."""
    monkeypatch.setitem(sys.modules, "transformers", fakes.fake_transformers())
    rows = [make_row(tokenizer)]
    a = extract.meta_record(pinned_model(), tokenizer,
                            write_rows_csv(tmp_path / "a.csv", rows), 0)
    b = extract.meta_record(pinned_model(), tokenizer,
                            write_rows_csv(tmp_path / "b.csv", rows), 0)
    assert a["rows_csv"] != b["rows_csv"]
    assert a["rows_csv_sha256"] == b["rows_csv_sha256"]


# --- loading the weights: the two guards that make the meta true -----------------

def fake_auto_transformers(model, tokenizer, ooms=0):
    """`transformers` with only the two `from_pretrained` entry points load_model calls.

    `model_calls` records what each load was ASKED for, which is the other half of
    the pin: the guards check what came back, these check what went in.
    """
    module = fakes.fake_transformers()
    module.model_calls = []
    module.tokenizer_calls = []
    remaining = [ooms]

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            module.model_calls.append(dict(kwargs, model_id=model_id))
            if remaining[0]:
                remaining[0] -= 1
                raise torch.OutOfMemoryError("the other tenant took the card")
            return model

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            module.tokenizer_calls.append(dict(kwargs, model_id=model_id))
            return tokenizer

    module.AutoModelForCausalLM = AutoModelForCausalLM
    module.AutoTokenizer = AutoTokenizer
    return module


#: what a run pins, and what `load_with` uses unless a test overrides it
PINS = {"model_id": MODEL_ID, "revision": "a" * 40, "device": 0,
        "dtype": "bfloat16", "attn": "sdpa"}


def load_with(monkeypatch, model, tokenizer, ooms=0, **overrides):
    """`load_model` against a fake `transformers`. Returns ((model, tokenizer), fake)."""
    fake = fake_auto_transformers(model, tokenizer, ooms)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    pins = dict(PINS)
    retry = {}
    for key, value in overrides.items():
        (pins if key in pins else retry)[key] = value
    got = load_model(pins["model_id"], pins["revision"], pins["device"], pins["dtype"],
                     pins["attn"], **retry)
    return got, fake


def test_load_model_pins_the_kernel_dtype_and_card_on_the_request(monkeypatch,
                                                                  tokenizer):
    """Nothing about the configuration is relaxed to make a load fit."""
    model = pinned_model()
    (got_model, got_tokenizer), fake = load_with(monkeypatch, model, tokenizer)
    assert got_model is model and got_tokenizer is tokenizer
    assert fake.model_calls == [{"model_id": MODEL_ID, "revision": "a" * 40,
                                 "torch_dtype": torch.bfloat16,
                                 "attn_implementation": "sdpa",
                                 "device_map": {"": 0}}]
    assert fake.tokenizer_calls == [{"model_id": MODEL_ID, "revision": "a" * 40}]


def test_load_model_refuses_a_kernel_the_model_did_not_load_with(monkeypatch,
                                                                 tokenizer):
    """sdpa and eager diverge at bf16, so a vector built under one is not
    comparable with one built under the other. Resolving from the model is the
    only thing that makes the meta's `attn_implementation` a fact."""
    with pytest.raises(RuntimeError) as excinfo:
        load_with(monkeypatch, pinned_model(attn="eager"), tokenizer, attn="sdpa")
    assert "sdpa" in str(excinfo.value) and "eager" in str(excinfo.value)


def test_load_model_refuses_weights_from_another_revision(monkeypatch, tokenizer):
    """A silent mismatch builds every future vector off different weights while the
    meta asserts the pinned sha."""
    with pytest.raises(RuntimeError) as excinfo:
        load_with(monkeypatch, pinned_model(commit="b" * 40), tokenizer,
                  revision="a" * 40)
    assert "b" * 40 in str(excinfo.value)


def test_load_model_refuses_weights_whose_revision_is_not_a_sha(monkeypatch,
                                                                tokenizer):
    with pytest.raises(ProvenanceError):
        load_with(monkeypatch, pinned_model(commit="main"), tokenizer)


def no_card(monkeypatch):
    """Take CUDA out of the retry path, so these run the same with or without one."""
    monkeypatch.setattr(extract.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda device: (1 << 30, 44 << 30))


def test_load_model_retries_an_oom_because_the_card_is_shared(monkeypatch, tokenizer):
    """The LOAD is retried, never a partially captured run, and every attempt reported."""
    no_card(monkeypatch)
    seen = []
    _got, fake = load_with(monkeypatch, pinned_model(), tokenizer, ooms=2, attempts=3,
                           retry_seconds=0,
                           on_retry=lambda *reported: seen.append(reported))
    assert len(fake.model_calls) == 3
    assert [(attempt, attempts) for attempt, attempts, _f, _t, _e in seen] == [(1, 3),
                                                                               (2, 3)]
    assert all(isinstance(exc, torch.OutOfMemoryError) for *_rest, exc in seen)


def test_load_model_gives_up_after_the_last_attempt(monkeypatch, tokenizer):
    no_card(monkeypatch)
    with pytest.raises(torch.OutOfMemoryError):
        load_with(monkeypatch, pinned_model(), tokenizer, ooms=5, attempts=2,
                  retry_seconds=0)


def test_load_model_refuses_a_nonsense_attempt_count(monkeypatch, tokenizer):
    with pytest.raises(ValueError):
        load_with(monkeypatch, pinned_model(), tokenizer, attempts=0)


# --- meta.json is the resume gate, so it is replaced rather than truncated --------

def test_a_failed_meta_write_leaves_the_previous_record_intact(tmp_path):
    """Truncate-in-place is the one write here that costs a corpus, not a shard."""
    path = tmp_path / "meta.json"
    write_json_atomically(path, {"complete": False, "n_shards": 3})
    with pytest.raises(TypeError):
        write_json_atomically(path, {"complete": True, "device": object()})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": False,
                                                            "n_shards": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_the_meta_is_replaced_in_one_step(tmp_path):
    path = tmp_path / "meta.json"
    write_json_atomically(path, {"complete": False})
    write_json_atomically(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_meta_that_will_not_parse_is_a_named_refusal(tmp_path):
    """A kill mid-write used to leave truncated JSON and a raw JSONDecodeError."""
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    (tmp_path / "meta.json").write_text('{"rows_csv": "/a/b.csv"', encoding="utf-8")
    with pytest.raises(ShardsUnusable) as excinfo:
        extract.plan_resume(tmp_path, {key: "x" for key in extract.PINNED}, True)
    assert "meta.json" in str(excinfo.value)


# --- what landed is measured, not counted ----------------------------------------

def test_shards_cover_measures_the_rows_that_actually_landed(tmp_path):
    add_rows(ShardWriter(tmp_path, shard_rows=2), 4, first=0)
    todo = [(index, {}) for index in range(4)]
    assert shards_cover(tmp_path, todo, ()) == (4, 2)


def test_shards_cover_refuses_a_corpus_the_shards_do_not_hold(tmp_path):
    """`n_captured` was arithmetic over the row list: it could not see a lost shard."""
    add_rows(ShardWriter(tmp_path, shard_rows=2), 4, first=0)
    todo = [(index, {}) for index in range(6)]
    with pytest.raises(ShardsUnusable) as excinfo:
        shards_cover(tmp_path, todo, ())
    assert "missing [4, 5]" in str(excinfo.value)


def test_shards_cover_does_not_expect_a_dropped_row(tmp_path):
    add_rows(ShardWriter(tmp_path, shard_rows=2), 2, first=0)
    todo = [(index, {}) for index in range(3)]
    assert shards_cover(tmp_path, todo, (Dropped(2, DICTATOR, EMPTY_RESPONSE),)) == (2, 1)


# --- resuming a corpus ------------------------------------------------------------

def test_a_resume_that_changes_a_pinned_field_is_refused():
    """sdpa and eager diverge at bf16; two kernels in one directory is the failure."""
    before = {"model_revision": "a" * 40, "attn_implementation": "sdpa", "dtype": "bf16"}
    after = dict(before, attn_implementation="eager")
    assert extract.resume_conflicts(before, after) == [
        ("attn_implementation", "sdpa", "eager")]


def test_a_resume_that_changes_nothing_pinned_is_allowed():
    meta = {key: "x" for key in extract.PINNED}
    assert extract.resume_conflicts(meta, dict(meta, seconds=12.0)) == []


def test_every_field_that_moves_a_number_is_pinned():
    for field in ("rows_csv", "model_revision", "dtype", "attn_implementation",
                  "chat_template_sha256", "batch_size", "padding"):
        assert field in extract.PINNED


def test_a_shard_is_never_written_over_another_run(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=1)
    add_rows(writer, 1)
    again = ShardWriter(tmp_path, shard_rows=1)
    with pytest.raises(ShardsUnusable):
        add_rows(again, 1)


def test_no_partial_shard_is_left_behind(tmp_path):
    writer = ShardWriter(tmp_path, shard_rows=1)
    add_rows(writer, 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shard_0000.pt",
                                                          "shard_0001.pt"]


def test_a_duplicated_row_across_shards_is_refused(tmp_path):
    """A duplicate would be double-weighted in every pole mean, silently."""
    ShardWriter(tmp_path, shard_rows=1).add(
        7, Seam(tuple(range(6)), 2), pool(hidden_states(total=6), 2))
    ShardWriter(tmp_path, first_index=1, shard_rows=1).add(
        7, Seam(tuple(range(6)), 2), pool(hidden_states(total=6), 2))
    with pytest.raises(ShardsUnusable) as excinfo:
        captured_row_indices(tmp_path)
    assert "row 7" in str(excinfo.value)


def write_meta(tmp_path, **overrides):
    meta = {key: "x" for key in extract.PINNED}
    meta.update(overrides)
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_a_fresh_run_skips_nothing(tmp_path):
    assert extract.plan_resume(tmp_path, write_meta(tmp_path), False) == (set(), 0)


def test_a_resume_picks_up_after_the_last_whole_shard(tmp_path):
    meta = write_meta(tmp_path)
    writer = ShardWriter(tmp_path, shard_rows=2)
    add_rows(writer, 4, first=10)
    skip, first_shard = extract.plan_resume(tmp_path, meta, True)
    assert skip == {10, 11, 12, 13}
    assert first_shard == 2


def test_a_resume_into_a_directory_with_no_run_record_is_refused(tmp_path):
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    with pytest.raises(ShardsUnusable) as excinfo:
        extract.plan_resume(tmp_path, {key: "x" for key in extract.PINNED}, True)
    assert "meta.json" in str(excinfo.value)


def test_a_resume_under_a_different_kernel_is_refused(tmp_path):
    """The failure this module exists to prevent: two kernels in one directory."""
    meta = write_meta(tmp_path, attn_implementation="sdpa")
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    with pytest.raises(ShardsUnusable) as excinfo:
        extract.plan_resume(tmp_path, dict(meta, attn_implementation="eager"), True)
    assert "attn_implementation" in str(excinfo.value)


def test_a_resume_against_a_different_rows_csv_is_refused(tmp_path):
    """`row_index` joins into the CSV; a different one mis-joins the labels."""
    meta = write_meta(tmp_path, rows_csv="/a/dictator.csv")
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    with pytest.raises(ShardsUnusable):
        extract.plan_resume(tmp_path, dict(meta, rows_csv="/a/trust.csv"), True)


def test_a_resume_against_a_rewritten_rows_csv_is_refused(tmp_path):
    """Regenerating the CSV in place keeps the path and moves every answer.

    The per-row prompt fingerprint cannot see it — the prompt is per cell and did
    not change — so the corpus would hold activations for one set of answers at
    row indices whose CSV entries hold another.
    """
    meta = write_meta(tmp_path, rows_csv="/a/dictator.csv", rows_csv_sha256="0" * 64)
    add_rows(ShardWriter(tmp_path, shard_rows=1), 1)
    with pytest.raises(ShardsUnusable) as excinfo:
        extract.plan_resume(tmp_path, dict(meta, rows_csv_sha256="1" * 64), True)
    assert "rows_csv_sha256" in str(excinfo.value)


# --- a persona is part of the prompt, so it is part of the cache key -------------

def test_a_persona_row_re_renders_with_its_persona_turn(tokenizer):
    """The CSVs carry a `persona` column; ignoring it re-renders a different prompt."""
    persona = elicit.Persona("pos", 0)
    rendered = elicit.render(by_id(DICTATOR), "free", tokenizer, persona)
    row = dict(make_row(tokenizer), persona=persona.label,
               prompt_sha256=rendered.sha256)
    assert Prompts(tokenizer).for_row(row) == rendered.text


def test_two_personas_do_not_share_a_cached_prompt(tokenizer):
    prompts = Prompts(tokenizer)
    texts = []
    for polarity in ("pos", "neg"):
        persona = elicit.Persona(polarity, 0)
        rendered = elicit.render(by_id(DICTATOR), "free", tokenizer, persona)
        texts.append(prompts.for_row(dict(make_row(tokenizer), persona=persona.label,
                                          prompt_sha256=rendered.sha256)))
    assert texts[0] != texts[1]


def test_a_persona_label_that_is_not_a_label_is_refused(tokenizer):
    with pytest.raises(ValueError):
        extract.persona_of({"persona": "altruistic"})


# --- the command-line run, end to end --------------------------------------------

def write_grid(tmp_path):
    """A grid module whose games this package already declares."""
    path = tmp_path / "grid.py"
    path.write_text("def register():\n    return ()\n", encoding="utf-8")
    return path


def run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer, model=None, extra=()):
    """Drive `main()` the way the command line does, with no weights."""
    monkeypatch.setitem(sys.modules, "transformers", fakes.fake_transformers())
    model = model if model is not None else pinned_model()
    monkeypatch.setattr(extract, "load_model",
                        lambda *args, **kwargs: (model, tokenizer))
    rows_csv = write_rows_csv(tmp_path / "rows.csv", rows)
    argv = ["extract", "--rows", str(rows_csv), "--out-dir", str(out_dir),
            "--grid", str(write_grid(tmp_path)), "--device", "0"]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    extract.main()
    return json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))


def test_main_finishes_a_run_and_records_that_it_did(monkeypatch, tmp_path, tokenizer):
    """The whole point of the run: a complete meta beside the shards it describes.

    `main` used to raise NameError here, after every shard had landed, so the
    corpus was never the thing the meta said it was.
    """
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer="50"), make_row(tokenizer, answer="0"),
            make_row(tokenizer, answer="25")]
    meta = run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer,
                    extra=["--shard-rows", "2"])
    assert meta["complete"] is True
    assert meta["n_captured"] == 3
    assert meta["n_scored_rows"] == 3
    assert meta["n_shards"] == 2
    assert meta["dropped"] == []
    assert meta["resumed_from_shards"] == 0
    assert meta["seconds"] >= 0.0
    assert meta["extractor"] == "lab.extract"
    assert meta["rows_csv_sha256"] == extract.file_sha256(tmp_path / "rows.csv")
    assert sorted(p.name for p in out_dir.glob("shard_*")) == ["shard_0000.pt",
                                                               "shard_0001.pt"]
    seen, n_shards = captured_row_indices(out_dir)
    assert seen == {0, 1, 2} and n_shards == meta["n_shards"]


def test_main_names_the_rows_it_could_not_pool(monkeypatch, tmp_path, tokenizer):
    """"dropped, counted and named" is an invariant about the FILE, not the loop."""
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer=""), make_row(tokenizer, answer="50")]
    meta = run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer)
    assert meta["n_captured"] == 1
    assert meta["dropped"] == [{"row_index": 0, "game_id": DICTATOR,
                                "reason": EMPTY_RESPONSE}]


def test_main_resumes_from_the_shards_a_killed_run_left(monkeypatch, tmp_path,
                                                        tokenizer):
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer="50"), make_row(tokenizer, answer="0")]
    evicted = pin(EvictedModel(1, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):
        run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer, model=evicted,
                 extra=["--shard-rows", "1"])
    interrupted = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert interrupted["complete"] is False

    meta = run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer,
                    extra=["--resume", "--shard-rows", "1"])
    assert meta["resumed_from_shards"] == 1
    assert meta["complete"] is True
    assert meta["n_captured"] == 2
    assert captured_row_indices(out_dir)[0] == {0, 1}


def test_main_resumes_after_a_kill_inside_a_block(monkeypatch, tmp_path, tokenizer):
    """The reported failure, end to end: `--resume` used to advance by nothing.

    A shard was flushed only by a full block or by a clean exit, so a run killed
    seven rows into a ten-row block left ZERO shards — and `--resume` reads the
    shards, so `RESUME=1`, which `run_extraction.sh` advertises as the recovery
    lever, re-ran the same seven forward passes forever.
    """
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer=str(index)) for index in range(20)]
    evicted = pin(EvictedModel(7, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):
        run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer, model=evicted,
                 extra=["--shard-rows", "10"])
    assert captured_row_indices(out_dir)[0] == set(range(7))

    meta = run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer,
                    extra=["--resume", "--shard-rows", "10"])
    assert meta["complete"] is True
    assert meta["n_captured"] == 20
    assert meta["resumed_from_shards"] == 1


def test_main_drops_a_poison_row_and_captures_the_rest(monkeypatch, tmp_path,
                                                       tokenizer):
    """A non-finite pooling is one row's problem; it used to be the corpus's."""
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer=str(index)) for index in range(12)]
    model = pin(PoisonedModel(poison_call=8, dtype=torch.bfloat16))
    meta = run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer, model=model,
                    extra=["--shard-rows", "10"])
    assert meta["complete"] is True
    assert meta["n_captured"] == 11
    assert [d["row_index"] for d in meta["dropped"]] == [7]
    assert meta["dropped"][0]["reason"].startswith(NON_FINITE)
    assert captured_row_indices(out_dir)[0] == set(range(12)) - {7}


def test_main_never_truncates_the_meta_in_place(monkeypatch, tmp_path, tokenizer):
    """`meta.json` is the resume gate and was the one file here written by
    truncation. Nothing in a run may take that path."""
    real_write_text = Path.write_text

    def guard(self, *args, **kwargs):
        assert self.name != "meta.json", "meta.json must be replaced, not truncated"
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guard)
    meta = run_main(monkeypatch, tmp_path, [make_row(tokenizer, answer="50")],
                    tmp_path / "acts", tokenizer, extra=["--shard-rows", "1"])
    assert meta["complete"] is True


def test_main_refuses_a_meta_the_shards_do_not_support(monkeypatch, tmp_path,
                                                       tokenizer):
    """`n_captured` was arithmetic over the row list, so a shard that never landed
    still came out the far side as a complete corpus."""
    real_flush = extract.ShardWriter.flush

    def lose_the_trailing_block(self):
        if 0 < len(self._pending) < self._shard_rows:
            self._pending = []
            return 0
        return real_flush(self)

    monkeypatch.setattr(extract.ShardWriter, "flush", lose_the_trailing_block)
    rows = [make_row(tokenizer, answer=str(index)) for index in range(3)]
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, tmp_path, rows, tmp_path / "acts", tokenizer,
                 extra=["--shard-rows", "2"])
    assert "the shards are not" in str(excinfo.value)


def test_main_captures_only_the_first_n_rows_under_limit_rows(monkeypatch, tmp_path,
                                                              tokenizer):
    """An equivalence probe, never a result — so the meta records the limit."""
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer=str(index)) for index in range(5)]
    meta = run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer,
                    extra=["--limit-rows", "2", "--shard-rows", "10"])
    assert meta["limit_rows"] == 2
    assert meta["n_scored_rows"] == 2 and meta["n_captured"] == 2
    assert captured_row_indices(out_dir)[0] == {0, 1}


def test_limit_rows_and_resume_together_are_refused(monkeypatch, tmp_path, tokenizer):
    """A probe and a continuation would leave a directory that is neither."""
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, tmp_path, [make_row(tokenizer)], tmp_path / "acts",
                 tokenizer, extra=["--limit-rows", "1", "--resume"])
    assert "--limit-rows" in str(excinfo.value)


def test_a_directory_that_already_holds_shards_is_refused_without_resume(monkeypatch,
                                                                         tmp_path,
                                                                         tokenizer):
    out_dir = tmp_path / "acts"
    rows = [make_row(tokenizer, answer="50")]
    run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer, extra=["--shard-rows", "1"])
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, tmp_path, rows, out_dir, tokenizer,
                 extra=["--shard-rows", "1"])
    assert "--resume" in str(excinfo.value)
    assert captured_row_indices(out_dir)[1] == 1


def test_main_sweeps_the_staging_a_killed_run_abandoned(monkeypatch, tmp_path,
                                                        tokenizer):
    out_dir = tmp_path / "acts"
    out_dir.mkdir()
    (out_dir / "shard_0000.pt.999.tmp").write_bytes(b"half a shard")
    meta = run_main(monkeypatch, tmp_path, [make_row(tokenizer, answer="50")],
                    out_dir, tokenizer, extra=["--shard-rows", "1"])
    assert meta["complete"] is True
    assert not list(out_dir.glob("*.tmp"))


def test_main_refuses_to_resume_a_rows_csv_that_was_regenerated(monkeypatch, tmp_path,
                                                                tokenizer):
    """Same path, new answers: the resume is the only place this can still be caught."""
    out_dir = tmp_path / "acts"
    run_main(monkeypatch, tmp_path, [make_row(tokenizer, answer="50")], out_dir,
             tokenizer, extra=["--shard-rows", "1"])
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, tmp_path, [make_row(tokenizer, answer="0")], out_dir,
                 tokenizer, extra=["--resume", "--shard-rows", "1"])
    assert "rows_csv_sha256" in str(excinfo.value)
