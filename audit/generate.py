"""Run (game, mode) pairs through one model and write scored, self-describing rows.

The engine and every sampling parameter are the caller's explicit choice. Nothing
here picks either from an incidental condition: upstream's `eval/eval_persona.py`
selects the inference engine with `if vector is not None` (line 269), so its
steered runs go through HF `generate()` and its unsteered runs through vLLM, with
different sampling configuration — its beta=0 point is not comparable with its
own steered points. At coef 0 it takes the vLLM branch and sets `vector = None`
(lines 347-354), which makes the HF branch unreachable there: their published
baseline is a vLLM number. One path, always.

Forensics measured that HF alone reproduces that baseline, so this module needs
no vLLM arm — see `NEUTRAL`, and see `run` for how much of "reproduces" is
resolvable at a given sample size.

Every row carries the whole configuration that produced it, because a result
whose sampling settings, model revision and code commit are unrecorded cannot be
reproduced or contradicted.

Keeping the model's `generation_config.json` out of the result takes two things,
not one: an explicit `GenerationConfig`, and `generate(use_model_defaults=False)`.
Without the second, transformers >= 4.50 replaces any field of ours that happens
to equal the global default with the model's own value — see
`supports_use_model_defaults`.

Importing this module needs neither torch nor transformers; the HF engine imports
them when it is constructed.
"""

import csv
import dataclasses
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Sequence, Tuple)

from audit import elicit, games
from audit.games import Game
from audit.paths import REPO_ROOT


class ConfigError(ValueError):
    """A run was configured in a way that would produce an untrustworthy result."""


class ProvenanceError(RuntimeError):
    """Something a result row must record could not be determined."""


def _require_number(name, value, low=None, high=None, exclusive_low=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("%s must be a number, got %r" % (name, value))
    if low is not None and (value <= low if exclusive_low else value < low):
        raise ConfigError("%s must be %s %s, got %r"
                          % (name, ">" if exclusive_low else ">=", low, value))
    if high is not None and value > high:
        raise ConfigError("%s must be <= %s, got %r" % (name, high, value))


def _require_int(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("%s must be an integer, got %r" % (name, value))
    if low is not None and value < low:
        raise ConfigError("%s must be >= %d, got %r" % (name, low, value))


#: Every knob that moves a number. All required, none defaulted.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                   "max_new_tokens", "min_new_tokens", "seed")


@dataclass(frozen=True)
class Sampling:
    """The full sampling configuration of a run.

    Nothing defaults, at either layer. `Qwen2.5-7B-Instruct` ships
    `generation_config.json` with `temperature: 0.7, top_p: 0.8, top_k: 20,
    repetition_penalty: 1.05`, so `generation_kwargs` states all of them and the
    engine hands `generate()` a config built only from them. (The task brief put
    the shift from inheriting those at $7-13 per game; that has not been measured
    in this repo.)
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    max_new_tokens: int
    min_new_tokens: int
    seed: int

    def __post_init__(self):
        _require_number("temperature", self.temperature, low=0.0)
        _require_number("top_p", self.top_p, low=0.0, high=1.0, exclusive_low=True)
        _require_int("top_k", self.top_k, low=0)
        _require_number("min_p", self.min_p, low=0.0, high=1.0)
        _require_number("repetition_penalty", self.repetition_penalty, low=0.0,
                        exclusive_low=True)
        _require_int("max_new_tokens", self.max_new_tokens, low=1)
        _require_int("min_new_tokens", self.min_new_tokens, low=0)
        _require_int("seed", self.seed, low=0)
        if self.min_new_tokens > self.max_new_tokens:
            raise ConfigError("min_new_tokens %d exceeds max_new_tokens %d"
                              % (self.min_new_tokens, self.max_new_tokens))

    def with_seed(self, seed: int) -> "Sampling":
        """The same configuration at another seed — what a preset is varied by."""
        return dataclasses.replace(self, seed=seed)

    @classmethod
    def from_mapping(cls, mapping: Dict) -> "Sampling":
        """Build from a config dict, naming everything missing or unrecognised."""
        missing = [name for name in SAMPLING_FIELDS if name not in mapping]
        unknown = sorted(set(mapping) - set(SAMPLING_FIELDS))
        if missing or unknown:
            raise ConfigError(
                "sampling config is not usable: missing %s, unknown %s; every one of "
                "%s must be stated" % (missing, unknown, list(SAMPLING_FIELDS)))
        return cls(**{name: mapping[name] for name in SAMPLING_FIELDS})

    def generation_kwargs(self) -> Dict:
        """Exactly what the engine passes to `generate()` — nothing else is inherited.

        `do_sample` is derived rather than stored: `temperature=0` means greedy,
        as it does in vLLM. Storing the two separately would let a caller ask for
        sampling at temperature zero, which is not a configuration.
        """
        return {
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
        }


#: transformers >= 4.50 overwrites a field of the caller's `GenerationConfig`
#: with the model's own value whenever that field equals the *global* default,
#: unless `generate(use_model_defaults=False)` is passed. It is gated on the
#: `transformers_version` the model's own generation_config.json declares, so
#: whether it bites depends on the checkpoint, not on us: measured on 4.52.3,
#: a checkpoint declaring 4.37.0 leaves our values alone and one declaring 4.50.0
#: replaces them — silently, while the row still records what we asked for.
_USE_MODEL_DEFAULTS_SINCE = (4, 50)


def supports_use_model_defaults(version: str) -> bool:
    """Whether this transformers takes `generate(use_model_defaults=...)`.

    Below 4.50 the keyword does not exist and the overwrite it suppresses does
    not happen either, so not passing it is correct there.
    """
    parts = []
    for chunk in version.split(".")[:2]:
        digits = "".join(character for character in chunk if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if len(parts) < 2:
        raise ProvenanceError("cannot read a transformers version from %r" % version)
    return tuple(parts) >= _USE_MODEL_DEFAULTS_SINCE


def generation_config_fields(sampling: "Sampling", pad_token_id: int,
                             stop_token_ids: Sequence[int]) -> Dict:
    """Every field of the `GenerationConfig` the engine builds — nothing implicit."""
    if not stop_token_ids:
        raise ConfigError("no stop token: every sample would run to max_new_tokens")
    fields = dict(sampling.generation_kwargs())
    fields["pad_token_id"] = pad_token_id
    fields["eos_token_id"] = list(stop_token_ids)
    return fields


@dataclass(frozen=True)
class Preset:
    """A named, complete run configuration: what to sample, and what to load.

    A preset pins everything measured to move a result, which is more than the
    sampling knobs. It deliberately does not pin the seed — vary it per run with
    `Sampling.with_seed` — or `samples_per_prompt`, which the caller passes;
    `recommended_samples_per_prompt` says what the operator standardised on.
    """

    name: str
    sampling: Sampling
    dtype: str
    attn_implementation: str
    recommended_samples_per_prompt: int
    rationale: str

    def check_engine(self, description: "EngineDescription") -> None:
        """Raise unless the loaded model matches the pins.

        Asking for bf16 + sdpa and silently getting something else is a wrong
        result, not a warning.
        """
        mismatches = []
        if _dtype_name(description.dtype) != _dtype_name(self.dtype):
            mismatches.append("dtype is %s, preset pins %s"
                              % (description.dtype, self.dtype))
        if description.attn_implementation != self.attn_implementation:
            mismatches.append("attn_implementation is %s, preset pins %s"
                              % (description.attn_implementation, self.attn_implementation))
        if mismatches:
            raise ConfigError("engine does not match preset %r: %s"
                              % (self.name, "; ".join(mismatches)))


#: The one measured configuration. "neutral" because no knob in it biases the
#: answer distribution: each sits at the value that leaves the model's own
#: distribution alone (`top_p` 1.0, `top_k` 0, `min_p` 0.0, no repetition penalty).
#:
#: It is semantically identical to the vLLM `SamplingParams` that produced the
#: authors' published baseline — their coef 0 path IS vLLM, so that baseline is a
#: vLLM number — and, unlike theirs, it is internally consistent across beta by
#: construction, because one engine serves every point. Forensics measured that
#: HF alone reproduces that baseline; no vLLM arm is needed. "Reproduces" is not
#: "matches exactly": see `run` on what a given sample size can resolve.
#:
#: `attn_implementation` is pinned because it is not cosmetic. Swapping only
#: sdpa -> eager moved the Overfishing mode from 50 to 55 (KS D=0.815), and a
#: forward-pass probe put the two kernels 3.4 logits apart with no padding
#: involved — enough to flip the first token's argmax. It is therefore recorded
#: on every row, exactly like a sampling knob. The pinned value is the one the
#: rest of this preset was measured under.
NEUTRAL = Preset(
    name="neutral",
    sampling=Sampling(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                      repetition_penalty=1.0, max_new_tokens=1000, min_new_tokens=1,
                      seed=0),
    dtype="bfloat16",
    attn_implementation="sdpa",
    recommended_samples_per_prompt=200,
    rationale=("semantically identical to the vLLM SamplingParams behind the authors' "
               "published baseline, and internally consistent across beta because one "
               "engine serves every point"),
)

#: Named presets, declared here and nowhere else. There is deliberately no
#: `as_published` preset: nobody has asked for one, and it would encode a
#: configuration we would then have to keep true.
PRESETS = {NEUTRAL.name: NEUTRAL}


def preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ConfigError("no preset %r; declared presets: %s" % (name, sorted(PRESETS)))


def _dtype_name(value: str) -> str:
    """`torch.bfloat16` and `bfloat16` name the same dtype."""
    return value.split(".")[-1]


@dataclass(frozen=True)
class EngineDescription:
    """What an engine must be able to say about itself, for the result rows."""

    engine: str
    engine_version: str
    torch_version: str
    model_id: str
    model_revision: str
    dtype: str
    attn_implementation: str
    stop_token_ids: Tuple[int, ...]

    def __post_init__(self):
        blank = [f.name for f in dataclasses.fields(self) if not getattr(self, f.name)]
        if blank:
            raise ProvenanceError("engine description is incomplete: %s" % blank)


@dataclass(frozen=True)
class Provenance:
    """Everything about a run that is not per-prompt."""

    engine: EngineDescription
    chat_template_sha256: str
    repo_commit: str
    repo_dirty: bool


def provenance(engine, tokenizer) -> Provenance:
    """Collect the run-level provenance, failing loudly on anything unknowable."""
    description = engine.describe()
    if not isinstance(description, EngineDescription):
        raise ProvenanceError("engine.describe() returned %s, expected EngineDescription"
                              % type(description).__name__)
    return Provenance(description, elicit.chat_template_fingerprint(tokenizer),
                      repo_commit(), repo_is_dirty())


@dataclass(frozen=True)
class Row:
    """One generation, scored, with the whole configuration that produced it.

    `game_id` is the unambiguous key: one game is one question scored one way.
    `upstream_question_id` is not — `altruism_v1/dictator` and
    `altruism_v3/dictator` both carry `altruism_0`, so pooling two question sets
    and grouping on it silently merges $10-stake and $100-stake games. It is kept
    under that name, rather than renamed away, because it is what joins these rows
    to the committed judge CSVs; `games.py` owns the value and is not this
    branch's to change.
    """

    game_id: str
    # upstream's own label ("altruism_0"): the join key into their committed judge
    # CSVs, and ambiguous by their design — v1 and v3 both use it. Group by game_id.
    upstream_question_id: str
    question_set: str
    family: str
    mode: str
    persona: str
    reading: str
    sample_index: int
    batch_index: int
    batch_size: int
    seed: int
    continuation: str
    answer: str
    value: Optional[float]
    tag: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    max_new_tokens: int
    min_new_tokens: int
    stop_token_ids: str
    engine: str
    engine_version: str
    torch_version: str
    model_id: str
    model_revision: str
    dtype: str
    attn_implementation: str
    chat_template_sha256: str
    question_sha256: str
    prompt_sha256: str
    repo_commit: str
    repo_dirty: bool


#: The CSV schema, in column order.
ROW_FIELDS = tuple(f.name for f in dataclasses.fields(Row))


class Task(NamedTuple):
    """One prompt to generate once."""

    game: Game
    prompt: elicit.RenderedPrompt
    sample_index: int


def plan(pairs: Iterable[Tuple[str, str]], tokenizer, samples_per_prompt: int,
         persona: Optional[elicit.Persona] = None) -> List[Task]:
    """Expand (game_id, mode) pairs into one task per sample, in run order.

    Each pair is rendered once and repeated, so identical prompts land in the same
    batch and padding stays tight.
    """
    if samples_per_prompt < 1:
        raise ConfigError("samples_per_prompt must be >= 1, got %d" % samples_per_prompt)
    tasks = []
    for game_id, mode in pairs:
        game = games.by_id(game_id)
        prompt = elicit.render(game, mode, tokenizer, persona)
        for sample_index in range(samples_per_prompt):
            tasks.append(Task(game, prompt, sample_index))
    return tasks


def run(pairs: Iterable[Tuple[str, str]], engine, sampling: Sampling, tokenizer,
        samples_per_prompt: int, *, reading: str, batch_size: int,
        persona: Optional[elicit.Persona] = None, on_rows: Optional[Callable] = None
        ) -> List[Row]:
    """Generate and score every (game_id, mode) pair, `samples_per_prompt` times.

    `engine` is the caller's; this function never chooses or substitutes one.
    Deterministic given `sampling.seed`, `batch_size` and the task order.

    Nothing that moves a number defaults, and that includes the three arguments
    below. A default would be recorded on the row as though someone had chosen it:

    * `reading` — `games.py` keeps both readings of a contradictory question and
      picks no winner. Defaulting here would pick one; on the v1 Dictator that is
      $4.50 against $1.69.
    * `batch_size` — batches are seeded `seed + batch_index`, so composition
      determines the draws.
    * `samples_per_prompt` — measured, n=50 resolves nothing below $13-16 per
      game: two runs of a byte-identical configuration differ by $6-11 on their
      own. The standard is n=200 (SE of a mean $1.44, typical run-to-run gap
      $1.63, 95% of gaps under $4.00 on the Dictator), carried as
      `preset(name).recommended_samples_per_prompt`.

    `on_rows` is called with each batch's rows as they are scored, before the next
    batch is generated; an exception from it stops the run. Pass `RowWriter.write`
    to keep a long run on disk — at n=200 a single OOM otherwise costs every
    generation since the start, on a device that can lose 20 GiB to another tenant
    mid-run.
    """
    if batch_size < 1:
        raise ConfigError("batch_size must be >= 1, got %d" % batch_size)
    tasks = plan(pairs, tokenizer, samples_per_prompt, persona)
    for game in {task.game.id: task.game for task in tasks}.values():
        # a reading the game does not declare must fail before any generation
        game.answer_space.reading(reading)
    record = provenance(engine, tokenizer)

    rows = []
    for batch_index, batch in enumerate(_batches(tasks, batch_size)):
        # Each batch is seeded `seed + batch_index`: one seed for the whole run
        # would hand every repeat of a prompt the same draw. Batch composition
        # therefore changes which draws a prompt gets — `batch_size` is part of
        # the configuration, and `batch_index` is recorded on every row.
        continuations = engine.generate([task.prompt.text for task in batch], sampling,
                                        sampling.seed + batch_index)
        continuations = list(continuations)
        if len(continuations) != len(batch):
            raise RuntimeError("engine returned %d continuations for %d prompts"
                               % (len(continuations), len(batch)))
        batch_rows = [_row(task, continuation, batch_index, batch_size, sampling,
                           record, reading)
                      for task, continuation in zip(batch, continuations)]
        if on_rows is not None:
            on_rows(batch_rows)
        rows.extend(batch_rows)
    return rows


class RowWriter:
    """A CSV that grows as the rows are produced, so a killed run keeps what ran.

    Header once, then append and flush per call — hand `write` to `run`'s
    `on_rows`. Use it as a context manager.

    An unresolved answer keeps its row and its tag; `value` is left empty. It is
    never a zero.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.written = 0
        self._handle = None
        self._writer = None

    def __enter__(self) -> "RowWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=ROW_FIELDS)
        self._writer.writeheader()
        self._handle.flush()
        return self

    def __exit__(self, *_exception):
        self._handle.close()
        self._handle = self._writer = None
        return False

    def write(self, rows: Iterable[Row]) -> int:
        """Append `rows`, flush, and return the running total."""
        if self._writer is None:
            raise RuntimeError("RowWriter is closed; use it as a context manager")
        rows = list(rows)
        for row in rows:
            if not isinstance(row, Row):
                raise TypeError("RowWriter takes Row objects, got %s" % type(row).__name__)
            record = dataclasses.asdict(row)
            if record["value"] is None:
                record["value"] = ""
            self._writer.writerow(record)
        # flushed per batch: an OOM kill must not take the rows already generated
        self._handle.flush()
        self.written += len(rows)
        return self.written


def write_rows(path, rows: Sequence[Row]) -> Path:
    """Write every row at once. For a long run prefer `RowWriter` — see `run`."""
    with RowWriter(path) as writer:
        writer.write(rows)
    return Path(path)


class HuggingFaceEngine:
    """`transformers` `generate()` — the one engine.

    HF rather than vLLM because steering needs forward hooks, which vLLM cannot
    give us; the same path then serves beta=0 and beta!=0.

    Constructing this imports torch and transformers. The tokenizer stays the
    caller's: left padding is borrowed for the length of a call and given back.
    """

    name = "huggingface"

    def __init__(self, model, tokenizer, model_id: str, model_revision: Optional[str] = None,
                 stop_token_ids: Optional[Sequence[int]] = None,
                 attn_implementation: Optional[str] = None):
        import torch
        import transformers

        self._torch = torch
        self._transformers = transformers
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._model_revision = resolve_revision(model, model_revision)
        self._attn_implementation = resolve_attn_implementation(model, attn_implementation)
        self._stop_token_ids = (tuple(int(token) for token in stop_token_ids)
                                if stop_token_ids is not None
                                else model_stop_tokens(model, tokenizer))
        self._pass_use_model_defaults = supports_use_model_defaults(transformers.__version__)
        if tokenizer.pad_token_id is None:
            raise ConfigError("tokenizer has no pad token and batched generation pads; "
                              "set one on the tokenizer before building the engine")

    @classmethod
    def load(cls, model_id: str, revision: Optional[str] = None, **from_pretrained):
        """Load weights and tokenizer. Needs a GPU and the model; never run by the tests.

        `from_pretrained` is passed through untouched so the caller, not this
        module, owns the dtype/device spelling their transformers version wants.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if revision is not None:
            from_pretrained["revision"] = revision
        model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, **({"revision": revision} if revision is not None else {}))
        return cls(model, tokenizer, model_id, revision)

    @property
    def tokenizer(self):
        """The tokenizer to hand `run` — the same one that rendered the prompts."""
        return self._tokenizer

    def describe(self) -> EngineDescription:
        return EngineDescription(
            engine=self.name,
            engine_version=self._transformers.__version__,
            torch_version=self._torch.__version__,
            model_id=self._model_id,
            model_revision=self._model_revision,
            dtype=str(self._model.dtype),
            attn_implementation=self._attn_implementation,
            stop_token_ids=self._stop_token_ids,
        )

    def generate(self, prompts: Sequence[str], sampling: Sampling, seed: int) -> List[str]:
        """Continue each prompt once. Returns only what the model added."""
        torch = self._torch
        torch.manual_seed(seed)
        config = self._transformers.GenerationConfig(
            **generation_config_fields(sampling, self._tokenizer.pad_token_id,
                                       self._stop_token_ids))
        call = {"generation_config": config}
        if self._pass_use_model_defaults:
            # without this the config above is only a suggestion: see
            # `supports_use_model_defaults`
            call["use_model_defaults"] = False
        with self._left_padding():
            # add_special_tokens=False: the chat template already wrote them.
            batch = self._tokenizer(list(prompts), return_tensors="pt", padding=True,
                                    add_special_tokens=False)
        batch = {key: value.to(self._model.device) for key, value in batch.items()}
        with torch.no_grad():
            output = self._model.generate(**batch, **call)
        prompt_len = batch["input_ids"].shape[1]
        return [self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                for row in output]

    @contextmanager
    def _left_padding(self):
        """Borrow the caller's tokenizer for one batch, then put it back.

        Decoder-only batched generation needs left padding; the tokenizer is not
        ours to leave that way.
        """
        previous = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            yield
        finally:
            self._tokenizer.padding_side = previous


def repo_commit() -> str:
    """The commit the code came from."""
    return _git("rev-parse", "HEAD")


def repo_is_dirty() -> bool:
    """Whether the tree differs from that commit, ignoring compiled python.

    This repo commits `__pycache__`, so merely running python dirties the tree;
    counting that would make the flag meaningless.
    """
    changed = [line[3:].strip() for line in _git("status", "--porcelain").splitlines()
               if line.strip()]
    return any("__pycache__" not in path and not path.endswith(".pyc")
               for path in changed)


def _row(task: Task, continuation: str, batch_index: int, batch_size: int,
         sampling: Sampling, record: Provenance, reading: str) -> Row:
    game = task.game
    # The model's assistant turn is what it wrote plus what the mode pre-filled;
    # scoring the continuation alone would drop the stub's "give to Agent 2".
    answer = task.prompt.assistant_prefill + continuation
    extraction = games.score(game, answer, reading)
    engine = record.engine
    return Row(
        game_id=game.id,
        upstream_question_id=game.question_id,
        question_set=game.question_set,
        family=game.family,
        mode=task.prompt.mode,
        persona=task.prompt.persona,
        reading=reading,
        sample_index=task.sample_index,
        batch_index=batch_index,
        batch_size=batch_size,
        seed=sampling.seed,
        continuation=continuation,
        answer=answer,
        value=extraction.value,
        tag=extraction.tag,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        repetition_penalty=sampling.repetition_penalty,
        max_new_tokens=sampling.max_new_tokens,
        min_new_tokens=sampling.min_new_tokens,
        stop_token_ids="|".join(str(token) for token in engine.stop_token_ids),
        engine=engine.engine,
        engine_version=engine.engine_version,
        torch_version=engine.torch_version,
        model_id=engine.model_id,
        model_revision=engine.model_revision,
        dtype=engine.dtype,
        attn_implementation=engine.attn_implementation,
        chat_template_sha256=record.chat_template_sha256,
        question_sha256=game.question_sha256,
        prompt_sha256=task.prompt.sha256,
        repo_commit=record.repo_commit,
        repo_dirty=record.repo_dirty,
    )


def _batches(tasks: Sequence[Task], size: int):
    for start in range(0, len(tasks), size):
        yield tasks[start:start + size]


_COMMIT_SHA_RX = re.compile(r"^[0-9a-f]{40}$")


def resolve_revision(model, requested: Optional[str] = None) -> str:
    """The commit the weights came from, as a sha — never a moving ref.

    `revision="main"` names a branch that points somewhere else next month, so
    recording it would defeat the field. The loaded model knows the sha it
    resolved to, so prefer that and accept a caller's value only if it is one.
    """
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    if _is_commit_sha(resolved):
        return resolved
    if _is_commit_sha(requested):
        return requested
    raise ProvenanceError(
        "cannot resolve the model revision to a commit sha (the loaded model reports "
        "%r, the caller passed %r); a branch or tag name is not a revision — pass "
        "model_revision=<40-hex sha>" % (resolved, requested))


def resolve_attn_implementation(model, requested: Optional[str] = None) -> str:
    """Which attention kernel the loaded model actually uses.

    Read from the model, so the row records what ran rather than what was asked
    for. Not cosmetic and not incidental — see `NEUTRAL`.
    """
    resolved = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if isinstance(resolved, str) and resolved:
        return resolved
    if isinstance(requested, str) and requested:
        return requested
    raise ProvenanceError(
        "cannot tell which attention implementation the model loaded with; pass "
        "attn_implementation= explicitly — the choice moves results (see NEUTRAL)")


def model_stop_tokens(model, tokenizer) -> Tuple[int, ...]:
    """The checkpoint's full stop set.

    Deliberately read from the model's generation_config: a stop token is not a
    sampling knob. `Qwen2.5-7B-Instruct` declares two (`<|im_end|>` and
    `<|endoftext|>`) and the tokenizer's `eos_token_id` is only the first, so
    passing that alone lets a sample that emits the other run to
    `max_new_tokens` — and `skip_special_tokens` then glues the overrun into the
    answer we score.
    """
    declared = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if declared is None:
        declared = tokenizer.eos_token_id
    if declared is None:
        raise ConfigError("neither the model nor the tokenizer declares a stop token")
    tokens = (declared,) if isinstance(declared, int) else tuple(declared)
    return tuple(int(token) for token in tokens)


def _is_commit_sha(value) -> bool:
    return isinstance(value, str) and bool(_COMMIT_SHA_RX.match(value))


def _git(*args: str) -> str:
    result = subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ProvenanceError("git %s failed in %s: %s"
                              % (" ".join(args), REPO_ROOT, result.stderr.strip()))
    return result.stdout.strip()

