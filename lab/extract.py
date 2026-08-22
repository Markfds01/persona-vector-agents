"""Teacher-forced activation capture — one implementation for every prompt grid.

This is `lab/`, the experimental pipeline. It borrows `audit`'s prompt rendering
and game declarations, which are the clean-room reimplementation of the upstream
paper, and nothing in `audit` depends on this file.

Upstream's `generate_vec.py:28-38` in a form that can be audited. For each
generated row: re-render the prompt it was generated from, verify the rendered
bytes against the fingerprint the row carries, run ONE forward pass over
prompt+response with `output_hidden_states=True`, and pool three ways per layer.

**Nothing here is per game.** A grid declares its own games and registers them
into `audit.games.GAMES_BY_ID`; every row names its own `game_id` and `mode`, so
which prompt is re-rendered, which stakes it carried and which answer space
scored it are all read off the data. The capture loop sees rows, a tokenizer and
a model, and nothing else. Adding a grid is a new declaration, never a second
copy of this file.

The invariants, each checkable against the code below:

  * the prompt is re-rendered and its sha256 checked against the row's
    `prompt_sha256` — a mismatch raises, so a row can never get activations for
    a different prompt;
  * ONE forward pass per row, over prompt+response, `output_hidden_states=True`;
  * three poolings per layer — `prompt_avg` over `[0, prompt_len)`, `prompt_last`
    at `prompt_len-1`, `response_avg` over `[prompt_len, end)`;
  * batch size 1 and no padding, which is the only way those position slices mean
    what they say (and is what upstream does);
  * dtype and attention implementation are pinned and resolved FROM the loaded
    model into the meta: this model's sdpa and eager kernels diverge at bf16, so
    a vector built under one is not comparable with one built under the other;
  * the model revision is recorded as the sha the weights actually came from,
    never a moving ref;
  * prefix instability, empty responses and a non-finite pooling are dropped,
    counted and named — never absorbed into a mean, and never fatal to the rows
    already captured.

Only `response_avg` carries decision information for an outcome-conditioned
contrast: causal masking makes every prompt-side activation identical within a
prompt cell. All three are captured anyway so that stays checkable rather than
asserted.

Shards are written in complete blocks and a run is resumable from them, so an
interruption costs a shard rather than a corpus. What is in flight when a row
fails is flushed on the way out, and `meta.json` — the file the resume reads —
is replaced rather than truncated, for the same reason.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from audit import elicit, games
from audit.generate import (ProvenanceError, resolve_attn_implementation,
                            resolve_revision)

#: Rows per shard. 250 is what the archived runs used; at hidden size 3584 it is
#: ~300 MB per file and bounds what an eviction costs to about a minute.
SHARD_ROWS = 250

#: What is pooled per layer. The three names are the method, not an option: the
#: prompt-side pair exists so "only response_avg carries decision information"
#: can be checked instead of asserted.
POOLINGS = ("prompt_avg", "prompt_last", "response_avg")

#: Why a scored row can still not be pooled. The reason is recorded per row.
PREFIX_UNSTABLE = "prompt tokens are not a prefix of prompt+answer tokens"
EMPTY_RESPONSE = "empty response"
NON_FINITE = "non-finite pooled activation"

#: Staging names this module writes and nothing else reads. An interrupted run
#: leaves them behind at ~312 MB each, so the next run into the directory sweeps
#: them — which assumes one run per directory at a time, as the rest of the module
#: does.
STAGING_GLOBS = ("shard_*.pt.*.tmp", "meta.json.*.tmp")


class PromptMismatch(RuntimeError):
    """A row's prompt no longer re-renders to the bytes the row was generated from."""


class ShardsUnusable(RuntimeError):
    """An output directory cannot be resumed from or safely written into."""


@dataclass(frozen=True)
class Seam:
    """One row's token ids, and where the answer starts in them."""

    ids: Tuple[int, ...]
    prompt_len: int

    def __post_init__(self):
        if not 0 < self.prompt_len < len(self.ids):
            raise ValueError("prompt_len %d does not split %d tokens into a prompt and "
                             "a non-empty response" % (self.prompt_len, len(self.ids)))


@dataclass(frozen=True)
class Dropped:
    """A scored row that could not be pooled, and why."""

    row_index: int
    game_id: str
    reason: str

    def as_record(self) -> Dict[str, object]:
        return {"row_index": self.row_index, "game_id": self.game_id,
                "reason": self.reason}


def scored(rows: Sequence[dict]) -> List[Tuple[int, dict]]:
    """(index, row) for every row the scorer resolved.

    Only a scored row carries a pole label. The rest are counted by the caller
    and never guessed at.
    """
    return [(index, row) for index, row in enumerate(rows) if row["value"] != ""]


def seam_of(prompt_ids: Sequence[int], ids: Sequence[int]):
    """(Seam, "") when a row can be pooled, (None, reason) when it cannot.

    The prompt's own tokens must be a prefix of the prompt+answer tokens. BPE
    merges across the boundary (" " + "C" becomes one " C"), and a row where it
    does has the wrong `prompt_len`, which would put prompt tokens in the
    response slice. An empty response would make `response_avg` a NaN. Both are
    expected outcomes, so both are values here rather than exceptions.
    """
    prompt_len = len(prompt_ids)
    if list(ids[:prompt_len]) != list(prompt_ids):
        return None, PREFIX_UNSTABLE
    if len(ids) <= prompt_len:
        return None, EMPTY_RESPONSE
    return Seam(tuple(ids), prompt_len), ""


def persona_of(row: dict):
    """The persona prefix a row was generated under, or None.

    A grid may prepend upstream's persona system turn, and the row records only
    its label (`pos_0`), not the assistant name a caller could have overridden.
    So this rebuilds the default form and lets the per-row prompt fingerprint
    catch anything else.
    """
    label = row.get("persona", "")
    if not label:
        return None
    polarity, _, index = label.rpartition("_")
    if not index.isdigit():
        raise ValueError("row records persona %r, which is not <polarity>_<index>"
                         % (label,))
    return elicit.Persona(polarity, int(index))


class Prompts:
    """Re-renders each row's prompt and proves it is the one the row came from.

    Cached per (game_id, mode, persona) — everything `render` reads — because a
    grid has a few hundred cells and thousands of rows; the fingerprint check
    runs per ROW, not per cache miss.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._cache = {}

    def for_row(self, row: dict) -> str:
        key = (row["game_id"], row["mode"], row.get("persona", ""))
        rendered = self._cache.get(key)
        if rendered is None:
            rendered = elicit.render(games.by_id(row["game_id"]), row["mode"],
                                     self._tokenizer, persona_of(row))
            self._cache[key] = rendered
        if rendered.sha256 != row["prompt_sha256"]:
            raise PromptMismatch(
                "%s/%s: re-rendered prompt %s but the row records %s; the grid or the "
                "chat template has changed since the row was generated"
                % (row["game_id"], row["mode"], rendered.sha256, row["prompt_sha256"]))
        return rendered.text


def pool(hidden_states, prompt_len: int) -> Dict[str, object]:
    """{pooling: (n_hidden_states, hidden_size) float32 on the CPU} for one row.

    The reduction order is load-bearing and is upstream's: mean first, in the
    activation's own dtype, then cast. Casting first would give different bits.
    """
    import torch

    total_len = hidden_states[0].shape[1]
    if hidden_states[0].shape[0] != 1:
        raise ValueError("expected a batch of 1, got %d" % hidden_states[0].shape[0])
    if not 0 < prompt_len < total_len:
        raise ValueError("prompt_len %d does not split %d positions"
                         % (prompt_len, total_len))
    pooled = {
        "prompt_avg": torch.stack([s[0, :prompt_len, :].mean(dim=0).float().cpu()
                                   for s in hidden_states]),
        "prompt_last": torch.stack([s[0, prompt_len - 1, :].float().cpu()
                                    for s in hidden_states]),
        "response_avg": torch.stack([s[0, prompt_len:, :].mean(dim=0).float().cpu()
                                     for s in hidden_states]),
    }
    if set(pooled) != set(POOLINGS):
        raise ValueError("pooled %s but POOLINGS declares %s; the shards and the meta "
                         "are written from POOLINGS" % (sorted(pooled), list(POOLINGS)))
    return pooled


def non_finite_poolings(pooled: Dict[str, object]) -> List[str]:
    """Which poolings hold a NaN or an inf, in POOLINGS order.

    Not a raise inside `pool`: this is a property of one row's activations, so it
    is a drop reason like the other two, and killing the run over it would throw
    away every completed forward pass still queued in the writer.
    """
    import torch

    return [name for name in POOLINGS if not torch.isfinite(pooled[name]).all()]


def claim_shard(staging, path):
    """Move a finished shard to its final name, refusing another run's.

    `os.link` is the claim: it is atomic and fails on EEXIST, so two runs cannot
    both pass an `exists()` check and interleave into one inode. Not every
    filesystem has hard links (FUSE mounts, exFAT, some network mounts), and there
    the claim is an O_EXCL create of the same name — also atomic, on all of them —
    after which the rename only ever overwrites the empty file this run owns. A
    plain `exists()` check would leave two runs free to interleave their blocks
    under one numbering while `meta.json` still claimed the full count.
    """
    try:
        os.link(staging, path)
        return
    except FileExistsError:
        raise ShardsUnusable("%s already exists; another run is writing into this "
                             "directory" % path) from None
    except OSError:
        pass
    try:
        # 0600 to match the staging inode `os.replace` swaps in below: the claim
        # file is transient, and a mode it cannot keep only misleads
        os.close(os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        raise ShardsUnusable("%s already exists; another run is writing into this "
                             "directory" % path) from None
    try:
        os.replace(staging, path)
    except BaseException:
        # the empty claim is this run's; left behind it reads as a corrupt shard
        # and costs the whole directory instead of one flush
        os.unlink(path)
        raise


class ShardWriter:
    """Writes captures to `shard_%04d.pt`, a whole block at a time.

    A shard exists only once it is complete, so the directory an interrupted run
    leaves behind is exactly what `captured_row_indices` can read back.
    """

    def __init__(self, out_dir, first_index: int = 0, shard_rows: int = SHARD_ROWS):
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.out_dir = Path(out_dir)
        self.shard_index = first_index
        self.written = 0
        self._shard_rows = shard_rows
        self._pending = []

    def add(self, row_index: int, seam: Seam, pooled: Dict[str, object]) -> bool:
        """Queue one row. True when that completed a shard and flushed it."""
        self._pending.append((row_index, seam, pooled))
        if len(self._pending) < self._shard_rows:
            return False
        self.flush()
        return True

    def flush(self) -> int:
        """Write whatever is queued as one shard. Returns the rows written."""
        import torch

        if not self._pending:
            return 0
        payload = {
            "row_index": torch.tensor([r for r, _s, _p in self._pending],
                                      dtype=torch.long),
            "prompt_len": torch.tensor([s.prompt_len for _r, s, _p in self._pending],
                                       dtype=torch.long),
            "total_len": torch.tensor([len(s.ids) for _r, s, _p in self._pending],
                                      dtype=torch.long),
        }
        for name in POOLINGS:
            payload[name] = torch.stack([p[name] for _r, _s, p in self._pending])
        path = self.out_dir / ("shard_%04d.pt" % self.shard_index)
        # written under a name no other run can be using, then claimed: a kill
        # during the write leaves a temp file the resume path ignores, never a
        # truncated shard it would trust. mkstemp rather than the pid, which
        # repeats across PID namespaces sharing a volume; and the save is INSIDE
        # the try, so a kill or ENOSPC mid-write does not leak 312 MB.
        handle, name = tempfile.mkstemp(dir=str(self.out_dir),
                                        prefix=path.name + ".", suffix=".tmp")
        os.close(handle)
        staging = Path(name)
        try:
            torch.save(payload, staging)
            claim_shard(staging, path)
        finally:
            if staging.exists():
                os.unlink(staging)
        count = len(self._pending)
        self.shard_index += 1
        self.written += count
        self._pending = []
        return count


def captured_row_indices(out_dir):
    """Row indices already on disk, read from the shards themselves.

    The shards are the single source of truth — no sidecar index to drift out of
    sync — and `mmap=True` means only the small index tensor is actually read.
    Refuses a directory whose shard numbering has a hole, because then "the next
    shard index" is not defined.
    """
    import torch

    shards = sorted(Path(out_dir).glob("shard_*.pt"))
    expected = [Path(out_dir) / ("shard_%04d.pt" % i) for i in range(len(shards))]
    if shards != expected:
        raise ShardsUnusable("%s: shard numbering has a hole (%s); move the directory "
                             "aside rather than resuming into it"
                             % (out_dir, [p.name for p in shards]))
    seen = set()
    for path in shards:
        try:
            payload = torch.load(path, map_location="cpu", mmap=True)
        except Exception as exc:
            raise ShardsUnusable("%s cannot be read back (%s: %s); it is truncated "
                                 "or corrupt — move it aside rather than resuming "
                                 "into it" % (path, type(exc).__name__, exc)) from exc
        for index in payload["row_index"]:
            index = int(index)
            if index in seen:
                raise ShardsUnusable("%s: row %d is already in an earlier shard; a "
                                     "duplicated row would be double-weighted in every "
                                     "pole mean" % (path, index))
            seen.add(index)
    return seen, len(shards)


def requested_model_id(model) -> str:
    """The checkpoint the run was pointed at, as transformers recorded it.

    NOT a resolver: `from_pretrained` copies its own argument onto the config, so
    this reads the request back and cannot disagree with it. It is a label. The
    thing that actually identifies these weights is the revision sha, which IS
    resolved. Refused when empty, because a nameless label is no provenance.
    """
    label = getattr(getattr(model, "config", None), "_name_or_path", None)
    if not isinstance(label, str) or not label:
        raise ProvenanceError("the loaded model records no checkpoint name, so "
                              "`model_id` would be empty")
    return label


def file_sha256(path) -> str:
    """Content fingerprint of a file, read a block at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(model_id: str, revision: str, device: int, dtype: str, attn: str,
               attempts: int = 1, retry_seconds: int = 30, on_retry=None):
    """Load the weights onto ONE named device, with the kernel and dtype pinned.

    A load that OOMs is transient rather than a result, so the LOAD is retried,
    never a partially captured run, and every attempt is reported. Nothing about
    the configuration is relaxed to make it fit — that would silently break
    comparability with every vector already built.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = None
    for attempt in range(1, attempts + 1):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=getattr(torch, dtype),
                attn_implementation=attn, device_map={"": device})
            break
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if attempt == attempts:
                raise
            if on_retry is not None:
                free, total = torch.cuda.mem_get_info(device)
                on_retry(attempt, attempts, free, total, exc)
            time.sleep(retry_seconds)
    model.eval()
    # resolved from the model with NO fallback to what was asked for: a helper that
    # echoes the request back would make this check vacuous
    resolved = resolve_attn_implementation(model)
    if resolved != attn:
        raise RuntimeError("asked for attn %r, model loaded %r" % (attn, resolved))
    got = resolve_revision(model)
    if got != revision:
        raise RuntimeError("asked for revision %r, weights came from %r"
                           % (revision, got))
    return model, tokenizer


def meta_record(model, tokenizer, rows_csv, device: int) -> Dict[str, object]:
    """The run-level provenance, read off the loaded model rather than the request.

    `model_revision`, `dtype` and `attn_implementation` are RESOLVED: each is read
    from the weights and each resolver is called without a fallback, because one
    that accepted the caller's value would let this file assert a provenance the
    weights do not support. `load_model` has already checked the revision against
    what was asked for. `model_id` is not in that class and is not presented as if
    it were — see `requested_model_id`.

    The rows CSV is recorded by CONTENT as well as by path: the path alone says
    nothing about a file that was regenerated in place, and a resume that read a
    different set of answers at the same row indices would mislabel every pole.
    """
    import torch
    import transformers

    return {
        "rows_csv": str(Path(rows_csv).resolve()),
        "rows_csv_sha256": file_sha256(rows_csv),
        "model_id": requested_model_id(model),
        "model_revision": resolve_revision(model),
        "dtype": str(model.dtype),
        "attn_implementation": resolve_attn_implementation(model),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "n_hidden_states": model.config.num_hidden_layers + 1,
        "chat_template_sha256": elicit.chat_template_fingerprint(tokenizer),
        "device": "cuda:%d" % device,
        "batch_size": 1,
        "padding": "none",
        "poolings": list(POOLINGS),
        "extractor": "lab.extract",
    }


@dataclass(frozen=True)
class Captured:
    """What one invocation of `capture` did."""

    n_scored: int
    n_shards: int
    n_written_now: int
    dropped: Tuple[Dropped, ...]


def capture(rows: Sequence[dict], model, tokenizer, out_dir, *, skip=(),
            first_shard: int = 0, on_shard=None, shard_rows: int = SHARD_ROWS) -> Captured:
    """Capture every scored row not in `skip`.

    One forward pass per row, batch 1, no padding. `skip` is the set of row
    indices already on disk; a dropped row is not in it and is re-attempted,
    which is deterministic and lands in the same place.
    """
    import torch

    prompts = Prompts(tokenizer)
    n_states = model.config.num_hidden_layers + 1
    todo = scored(rows)
    writer = ShardWriter(out_dir, first_shard, shard_rows)
    dropped: List[Dropped] = []

    try:
        for done, (row_index, row) in enumerate(todo):
            if row_index in skip:
                continue
            prompt = prompts.for_row(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            inputs = tokenizer(prompt + row["answer"], return_tensors="pt",
                               add_special_tokens=False)
            seam, reason = seam_of(prompt_ids, inputs["input_ids"][0].tolist())
            if seam is None:
                dropped.append(Dropped(row_index, row["game_id"], reason))
                continue

            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            states = out.hidden_states
            if len(states) != n_states:
                raise RuntimeError("expected %d hidden states, got %d"
                                   % (n_states, len(states)))
            try:
                pooled = pool(states, seam.prompt_len)
            except ValueError as exc:
                raise RuntimeError("row %d (%s): %s"
                                   % (row_index, row["game_id"], exc)) from exc
            del out, states
            bad = non_finite_poolings(pooled)
            if bad:
                dropped.append(Dropped(row_index, row["game_id"],
                                       "%s: %s" % (NON_FINITE, ", ".join(bad))))
                continue
            if writer.add(row_index, seam, pooled) and on_shard is not None:
                on_shard(done + 1, len(todo), writer)
    finally:
        # every completed forward pass still queued is worth a shard: without this
        # one failing row discarded up to `shard_rows` - 1 of them and left the
        # advertised `--resume` nothing to resume from
        writer.flush()
    return Captured(len(todo), writer.shard_index, writer.written, tuple(dropped))


#: the meta fields that must not change between the invocations of one run. A resume
#: that changed any of them would put incomparable activations in one directory, which
#: is the failure this module exists to prevent. The rows CSV is pinned by content as
#: well as by path — regenerating it in place keeps the path and moves every answer.
PINNED = ("rows_csv", "rows_csv_sha256", "model_id", "model_revision", "dtype",
          "attn_implementation", "chat_template_sha256", "batch_size", "padding",
          "poolings", "shard_rows", "limit_rows")


def resume_conflicts(previous: Dict[str, object], current: Dict[str, object]):
    """The pinned fields on which a resume disagrees with the run it is continuing."""
    return [(key, previous.get(key), current.get(key)) for key in PINNED
            if previous.get(key) != current.get(key)]


def plan_resume(out_dir, meta: Dict[str, object], resuming: bool):
    """(rows to skip, next shard index) for this invocation.

    The configuration is written BEFORE any capture precisely so that this can
    check itself against it: continuing a corpus under a different kernel, dtype,
    revision or rows CSV would put incomparable activations in one directory, and
    the final meta would then assert one configuration for all of them.
    """
    out_dir = Path(out_dir)
    if not resuming:
        return set(), 0
    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        raise ShardsUnusable("%s holds shards but no meta.json, so there is nothing "
                             "to check this run against; move it aside" % out_dir)
    try:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ShardsUnusable("%s is not readable JSON (%s); it is the record this "
                             "run would be checked against — move the directory "
                             "aside rather than resuming into it"
                             % (meta_path, exc)) from exc
    conflicts = resume_conflicts(previous, meta)
    if conflicts:
        raise ShardsUnusable("refusing to resume %s: %s"
                             % (out_dir, "; ".join("%s was %r, now %r" % c
                                                   for c in conflicts)))
    return captured_row_indices(out_dir)


def write_json_atomically(path, payload: Dict[str, object]):
    """Replace `path` in one step, so a kill leaves the old file or the new one.

    Every shard in this directory is staged and claimed; `meta.json` is the file
    the resume path is gated on and it was the one truncated in place. It is
    rewritten on every invocation, carrying the whole `dropped` list, so the
    window is not small.
    """
    path = Path(path)
    handle, name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2)
        os.replace(name, path)
    except BaseException:
        if os.path.exists(name):
            os.unlink(name)
        raise


def sweep_staging(out_dir) -> int:
    """Delete staging files an interrupted run abandoned. Returns how many.

    A shard stages at ~312 MB and nothing else ever removes one, while
    `run_extraction.sh` budgets barely above the corpus itself.
    """
    removed = 0
    for pattern in STAGING_GLOBS:
        for path in sorted(Path(out_dir).glob(pattern)):
            os.unlink(path)
            removed += 1
    return removed


def shards_cover(out_dir, todo, dropped) -> Tuple[int, int]:
    """(rows on disk, shards) once the shards are read back and checked.

    `n_captured` was arithmetic over the row list, which cannot see a shard that
    never landed. This measures it — mmap means only the small index tensor is
    read — and it is also the only caller of `captured_row_indices` outside the
    resume path, so it is where a row duplicated across shards is caught.
    """
    seen, n_shards = captured_row_indices(out_dir)
    expected = {index for index, _row in todo} - {d.row_index for d in dropped}
    if seen != expected:
        raise ShardsUnusable(
            "%s holds %d captured rows, this run accounts for %d (missing %s, "
            "unexpected %s); the meta would describe a corpus the shards are not"
            % (out_dir, len(seen), len(expected), sorted(expected - seen)[:10],
               sorted(seen - expected)[:10]))
    return len(seen), n_shards


def load_grid(path):
    """Import a grid module from a file path and register its games.

    A grid is an investigation's data, not a declaration this package ships, so
    it is named on the command line rather than imported by name. The module must
    expose `register()`.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import a grid module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    register = getattr(module, "register", None)
    if register is None:
        raise ImportError("%s exposes no register(); a grid must register its games "
                          "into audit.games.GAMES_BY_ID" % path)
    register()
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="CSV of scored generations")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grid", required=True,
                        help="path to the grid module that declares these rows' games")
    parser.add_argument("--device", type=int, required=True,
                        help="CUDA device index; pinned, never chosen for us")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--limit-rows", type=int, default=0,
                        help="capture only the first N scored rows (an equivalence "
                             "probe, never a result)")
    parser.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    parser.add_argument("--resume", action="store_true",
                        help="continue into a directory that already holds shards")
    parser.add_argument("--load-attempts", type=int, default=60)
    parser.add_argument("--load-retry-seconds", type=int, default=30)
    args = parser.parse_args()

    if args.limit_rows and args.resume:
        raise SystemExit("--limit-rows is a probe and --resume continues a corpus; "
                         "together they would leave a directory that is neither")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    swept = sweep_staging(out_dir)
    if swept:
        print("swept %d staging file(s) an interrupted run left in %s"
              % (swept, out_dir), flush=True)
    resuming = bool(list(out_dir.glob("shard_*.pt")))
    if resuming and not args.resume:
        raise SystemExit("%s already holds shards; pass --resume to continue into it, "
                         "or point --out-dir somewhere else" % out_dir)

    with open(args.rows, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    load_grid(args.grid)
    todo = scored(rows)
    if args.limit_rows:
        keep = {index for index, _row in todo[:args.limit_rows]}
        rows = [row if index in keep else dict(row, value="")
                for index, row in enumerate(rows)]
        todo = scored(rows)
    print("rows=%d scored=%d unscored=%d" % (len(rows), len(todo), len(rows) - len(todo)),
          flush=True)

    def on_retry(attempt, attempts, free, total, exc):
        print("load attempt %d/%d OOMed (%.1f GiB free of %.1f): %s"
              % (attempt, attempts, free / 2 ** 30, total / 2 ** 30, exc), flush=True)

    model, tokenizer = load_model(args.model, args.revision, args.device, args.dtype,
                                  args.attn, args.load_attempts,
                                  args.load_retry_seconds, on_retry)
    meta = meta_record(model, tokenizer, args.rows, args.device)
    meta.update({"shard_rows": args.shard_rows, "limit_rows": args.limit_rows,
                 "n_scored_rows": len(todo), "complete": False})
    print("loaded %s @ %s %s/%s on %s" % (meta["model_id"], meta["model_revision"][:12],
                                          meta["dtype"], meta["attn_implementation"],
                                          meta["device"]), flush=True)

    try:
        skip, first_shard = plan_resume(out_dir, meta, resuming)
    except ShardsUnusable as exc:
        raise SystemExit(str(exc))
    if resuming:
        print("resuming: %d shards, %d rows already captured"
              % (first_shard, len(skip)), flush=True)
    write_json_atomically(meta_path, meta)

    started = time.time()

    def on_shard(done, total, writer):
        print("%5d/%5d  %6.1fs  shard %d" % (done, total, time.time() - started,
                                             writer.shard_index - 1), flush=True)

    try:
        done = capture(rows, model, tokenizer, out_dir, skip=skip,
                       first_shard=first_shard, on_shard=on_shard,
                       shard_rows=args.shard_rows)
        n_captured, n_shards = shards_cover(out_dir, todo, done.dropped)
    except ShardsUnusable as exc:
        # a collision with a concurrent run is an operator mistake, not a crash
        raise SystemExit(str(exc))
    meta.update({
        "n_shards": n_shards,
        "dropped": [d.as_record() for d in done.dropped],
        "n_captured": n_captured,
        "seconds": time.time() - started,
        "resumed_from_shards": first_shard,
        "complete": True,
    })
    write_json_atomically(meta_path, meta)
    print("captured %d of %d scored rows, dropped %d, %.1f min"
          % (meta["n_captured"], done.n_scored, len(done.dropped),
             meta["seconds"] / 60.0), flush=True)


if __name__ == "__main__":
    main()
