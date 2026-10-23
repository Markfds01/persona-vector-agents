"""Generate answers to general (non-game) questions, one user turn each, steered or not.

The generation half of `results/general-question-transfer` (PREREG sections 4 and 14).
It is the `audit` path with a different prompt type: a question is rendered as ONE user
turn through the model's own chat template (no explicit system message, so Qwen inserts
its default one, as in the per-game study) and generated through `audit.generate`'s
engine, sampling and provenance and `audit.steer`'s hook. No new hook, no new sampling.

What is fixed here and why it is not a parameter:

* **Batching and seeds** (PREREG section 4): batches of `batch_size` prompts, each seeded
  `seed + batch_index`; each point (arm, k, question set) is an independent run seeded
  from `seed`, so every arm at a given k draws with the same seeds and the same batch
  composition. Rows are ordered by question id, then sample index.
* **Statuses**: `ok`, or `truncated` when the generation reached `max_new_tokens` without
  emitting a stop token. Truncated rows are kept (PREREG D14). Nothing is dropped.
* **Token ids**: the engine here is `audit.generate.HuggingFaceEngine` plus one thing, the
  generated ids of the last call, kept so a row can carry `output_ids_sha256` and
  `n_generated_tokens`. That is what makes the k = 0 byte-identity check a comparison of
  rows and not of re-tokenised text.
* **Hook fire count**: `CountingSteering` is `audit.steer.ActivationSteering` with a counter.
  Each forward pass through the hooked layer is one fire, so a batch that generated `n`
  tokens at most fired exactly `n` times (one prefill, `n - 1` decode steps); the count is
  checked against that, not just against "> 0".
* **Resume** is per point and refuses a point whose rows name another vector
  (`point_complete`): the file name carries the arm and the coefficient, not the vector.

Importing this module needs neither torch nor transformers.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from audit import generate, steer

#: the trait vector's layer-20 norm, the study's unit (PREREG section 3)
REFERENCE_NORM = 10.5083
QUESTION_SETS = ("heldout", "bridge")

STATUSES = ("ok", "truncated")


class GeneralQuestionError(RuntimeError):
    """A run configured or resumed in a way that cannot give a trustworthy row."""


@dataclass(frozen=True)
class Question:
    question_set: str
    question_id: str
    text: str                       # the user turn, verbatim


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions(package_dir, question_set: str) -> "tuple[List[Question], str]":
    """The frozen question set's items in file order, and the file's sha256."""
    if question_set not in QUESTION_SETS:
        raise GeneralQuestionError("question set %r; have %s" % (question_set, QUESTION_SETS))
    path = Path(package_dir) / "questions" / ("%s.json" % question_set)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if question_set == "heldout":
        items = [Question("heldout", q["id"], q["question"]) for q in data["questions"]]
    else:
        items = [Question("bridge", q["id"], q["prompt"]) for q in data["items"]]
    ids = [q.question_id for q in items]
    if len(set(ids)) != len(ids) or not items:
        raise GeneralQuestionError("%s: question ids missing or not unique" % path)
    return items, hashlib.sha256(raw).hexdigest()


def render_user_turn(tokenizer, text: str) -> str:
    """One user turn through the model's chat template, generation prompt appended."""
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True)


def ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("ascii")).hexdigest()


class IdRecordingEngine(generate.HuggingFaceEngine):
    """`HuggingFaceEngine` that also keeps the generated ids of its last call.

    `generate` is the base class's, line for line, with one addition: the new-token ids
    of every row are stored in `last_ids` (padding included; `trim_ids` cuts at the stop).
    """

    last_ids: List[List[int]]

    def generate(self, prompts, sampling, seed):
        torch = self._torch
        torch.manual_seed(seed)
        config = self._transformers.GenerationConfig(
            **generate.generation_config_fields(sampling, self._tokenizer.pad_token_id,
                                                self._stop_token_ids))
        call = {"generation_config": config}
        if self._pass_use_model_defaults:
            call["use_model_defaults"] = False
        with self._left_padding():
            batch = self._tokenizer(list(prompts), return_tensors="pt", padding=True,
                                    add_special_tokens=False)
        batch = {key: value.to(self._model.device) for key, value in batch.items()}
        with torch.no_grad():
            output = self._model.generate(**batch, **call)
        prompt_len = batch["input_ids"].shape[1]
        self.last_ids = [[int(t) for t in row[prompt_len:]] for row in output]
        return [self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                for row in output]


def trim_ids(ids: Sequence[int], stop_ids: Sequence[int]) -> "tuple[List[int], bool]":
    """The generated ids up to and including the first stop token, and whether one was found.

    After a sequence stops, a batched `generate` pads it with the pad token, which for
    Qwen is itself a stop token; cutting at the FIRST stop is what makes the count exact.
    """
    stops = set(stop_ids)
    for index, token in enumerate(ids):
        if token in stops:
            return list(ids[:index + 1]), True
    return list(ids), False


class CountingSteering(steer.ActivationSteering):
    """The upstream-matched hook, counting how many times it ran."""

    fired = 0

    def _hook(self, module, inputs, output):
        self.fired += 1
        return super()._hook(module, inputs, output)


class CountingSteeredEngine:
    """`audit.steer.SteeredEngine` with a fire counter and the engine's ids exposed."""

    def __init__(self, engine, model, steering: steer.Steering):
        self._engine = engine
        self._steering = CountingSteering(model, steering)

    @property
    def steering(self) -> steer.Steering:
        return self._steering.steering

    @property
    def columns(self) -> Dict[str, object]:
        return self._steering.columns

    @property
    def fired(self) -> int:
        return self._steering.fired

    @property
    def last_ids(self):
        return self._engine.last_ids

    def describe(self):
        return self._engine.describe()

    def generate(self, prompts, sampling, seed):
        with self._steering:
            return self._engine.generate(prompts, sampling, seed)


class UnhookedEngine:
    """No hook at all: the engine itself, with the same surface as the steered one."""

    columns = None
    fired = 0

    def __init__(self, engine):
        self._engine = engine

    @property
    def last_ids(self):
        return self._engine.last_ids

    def describe(self):
        return self._engine.describe()

    def generate(self, prompts, sampling, seed):
        return self._engine.generate(prompts, sampling, seed)


@dataclass(frozen=True)
class Arm:
    """One steered (or unsteered) arm: a name, a role and the vector it loads."""

    name: str                       # e.g. "decision/dictator", "baseline", "unhooked"
    role: str
    vector_path: Optional[str]      # None only for the unhooked control
    unit_beta_scale: float = REFERENCE_NORM


def point_slug(arm: str, question_set: str, k: float) -> str:
    return "%s/%s_k%+g" % (arm.replace("/", "-"), question_set, k)


def row_id(arm: str, question_set: str, k: float, question_id: str, sample_index: int) -> str:
    return "%s|%s|k%+g|%s|s%d" % (arm, question_set, k, question_id, sample_index)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path) -> List[Dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def point_complete(path, expected: int, vector_sha256: Optional[str], *, arm: str,
                   max_new_tokens: int, question_file_sha256: str) -> bool:
    """True when `path` already holds THIS arm's whole point, generated with this vector.

    A short file is reported absent (a point is regenerated whole, never appended to:
    the missing rows would land in differently composed batches). A whole file whose rows
    name another vector, arm or generation length is the wrong directory, not a stale
    point, so this refuses instead of rebuilding one point of a set whose others would
    still be wrong.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        rows = read_rows(path)
    except (OSError, ValueError):
        return False
    if len(rows) != expected:
        return False
    for field, want in (("vector_sha256", vector_sha256), ("arm", arm),
                        ("max_new_tokens", max_new_tokens),
                        ("question_file_sha256", question_file_sha256)):
        wrong = sorted({repr(r.get(field)) for r in rows if r.get(field) != want})
        if wrong:
            raise GeneralQuestionError(
                "%s holds %d rows with %s %s, not the %r this run uses. The path carries the "
                "arm and the coefficient but not the vector, so resuming here would call "
                "another vector's answers complete. Use this study's own rows directory, or "
                "run without resume." % (path, len(rows), field, ", ".join(wrong), want))
    return True


def expected_hook_fires(batches_max_tokens: Sequence[int]) -> int:
    """A batch that generated `n` tokens ran the layer `n` times: prefill + (n - 1) steps."""
    return int(sum(batches_max_tokens))


def generate_point(engine, tokenizer, questions: Sequence[Question], *, arm: Arm, k: float,
                   samples: int, sampling: generate.Sampling, batch_size: int,
                   stop_ids: Sequence[int], out_path, question_file_sha256: str,
                   record: generate.Provenance, vector_sha256: Optional[str],
                   log: Callable[[str], None] = lambda message: None) -> Dict:
    """Generate one whole point and write it atomically. Returns its summary.

    `engine` is a `CountingSteeredEngine` or an `UnhookedEngine`. Prompts are the
    questions in id order, each repeated `samples` times, in batches of `batch_size`.
    """
    if samples < 1 or batch_size < 1:
        raise GeneralQuestionError("samples and batch_size must be >= 1")
    ordered = sorted(questions, key=lambda q: q.question_id)
    tasks = [(q, s) for q in ordered for s in range(samples)]
    rendered = {q.question_id: render_user_turn(tokenizer, q.text) for q in ordered}
    unit_beta = k * arm.unit_beta_scale
    steered = engine.columns
    rows: List[Dict] = []
    batch_max: List[int] = []
    for batch_index, start in enumerate(range(0, len(tasks), batch_size)):
        batch = tasks[start:start + batch_size]
        texts = list(engine.generate([rendered[q.question_id] for q, _ in batch], sampling,
                                     sampling.seed + batch_index))
        if len(texts) != len(batch):
            raise RuntimeError("engine returned %d continuations for %d prompts"
                               % (len(texts), len(batch)))
        trimmed = [trim_ids(ids, stop_ids) for ids in engine.last_ids]
        if len(trimmed) != len(batch):
            raise RuntimeError("engine returned ids for %d rows, expected %d"
                               % (len(trimmed), len(batch)))
        batch_max.append(max(len(ids) for ids, _ in trimmed))
        for position, ((q, s), text, (ids, stopped)) in enumerate(zip(batch, texts, trimmed)):
            truncated = (not stopped) and len(ids) >= sampling.max_new_tokens
            description = record.engine
            row = {
                "row_id": row_id(arm.name, q.question_set, k, q.question_id, s),
                "question_set": q.question_set,
                "question_id": q.question_id,
                "sample_index": s,
                "arm": arm.name,
                "arm_role": arm.role,
                "k": k,
                "unit_beta": unit_beta if steered is not None else 0.0,
                "hooked": steered is not None,
                "question": q.text,
                "answer": text,
                "generation_status": "truncated" if truncated else "ok",
                "n_generated_tokens": len(ids),
                "output_ids_sha256": ids_sha256(ids),
                "seed": sampling.seed,
                "batch_index": batch_index,
                "batch_size": batch_size,
                "position_in_batch": position,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "repetition_penalty": sampling.repetition_penalty,
                "max_new_tokens": sampling.max_new_tokens,
                "min_new_tokens": sampling.min_new_tokens,
                "stop_token_ids": "|".join(str(t) for t in description.stop_token_ids),
                "use_model_defaults": False,
                "engine": description.engine,
                "engine_version": description.engine_version,
                "torch_version": description.torch_version,
                "model_id": description.model_id,
                "model_revision": description.model_revision,
                "dtype": description.dtype,
                "attn_implementation": description.attn_implementation,
                "chat_template_sha256": record.chat_template_sha256,
                "prompt_sha256": sha256_text(rendered[q.question_id]),
                "question_file_sha256": question_file_sha256,
                "vector_sha256": vector_sha256,
                "repo_commit": record.repo_commit,
                "repo_dirty": record.repo_dirty,
            }
            if steered is not None:
                row.update(steered)
            rows.append(row)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = out_path.with_suffix(out_path.suffix + ".tmp")
    with staging.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, out_path)
    summary = {
        "arm": arm.name, "question_set": questions[0].question_set, "k": k,
        "unit_beta": unit_beta if steered is not None else 0.0, "hooked": steered is not None,
        "n_rows": len(rows), "n_batches": len(batch_max),
        "n_truncated": sum(r["generation_status"] == "truncated" for r in rows),
        "hook_fires": engine.fired, "hook_fires_expected": expected_hook_fires(batch_max),
        "vector_sha256": vector_sha256, "rows": str(out_path.name),
    }
    log("%s k=%+g n=%d truncated=%d hook_fires=%d (expected %d)"
        % (arm.name, k, len(rows), summary["n_truncated"], summary["hook_fires"],
           summary["hook_fires_expected"]))
    return summary
