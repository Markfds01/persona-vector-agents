"""Run (game, mode) pairs through one model and write scored, self-describing rows.

The engine and every sampling parameter are the caller's explicit choice. Nothing
here picks either from an incidental condition: upstream's `eval/eval_persona.py`
selects the inference engine with `if vector is not None`, so its steered runs go
through HF `generate()` and its unsteered runs through vLLM, with different
sampling configuration — its beta=0 point is not comparable with its own steered
points. One path, always.

Every row carries the whole configuration that produced it, because a result
whose sampling settings, model revision and code commit are unrecorded cannot be
reproduced or contradicted.

Importing this module needs neither torch nor transformers; the HF engine imports
them when it is constructed.
"""

import csv
import dataclasses
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from audit import elicit, games
from audit.games import Game
from audit.paths import REPO_ROOT


class ConfigError(ValueError):
    """A run was configured in a way that would produce an untrustworthy result."""


class ProvenanceError(RuntimeError):
    """Something a result row must record could not be determined."""


#: Every knob that moves a number. All required, none defaulted.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty",
                   "max_new_tokens", "seed")


@dataclass(frozen=True)
class Sampling:
    """The full sampling configuration of a run.

    Nothing defaults, at either layer. `Qwen2.5-7B-Instruct` ships
    `generation_config.json` with `temperature: 0.7, top_p: 0.8, top_k: 20,
    repetition_penalty: 1.05`; inheriting those silently moves a game's mean by
    $7-13, so `generation_kwargs` states all of them and the engine hands
    `generate()` a config built only from them.
    """

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int
    seed: int

    def __post_init__(self):
        _require_number("temperature", self.temperature, low=0.0)
        _require_number("top_p", self.top_p, low=0.0, high=1.0, exclusive_low=True)
        _require_int("top_k", self.top_k, low=0)
        _require_number("repetition_penalty", self.repetition_penalty, low=0.0,
                        exclusive_low=True)
        _require_int("max_new_tokens", self.max_new_tokens, low=1)
        _require_int("seed", self.seed, low=0)

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
        """Exactly what the engine passes to `generate()` — nothing else is inherited."""
        return {
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }


#: Named sampling presets. Empty on purpose: no setting is known to reproduce the
#: authors' published numbers yet. When one is, it is declared here — one place,
#: no call site.
PRESETS = {}


def preset(name: str) -> Sampling:
    try:
        return PRESETS[name]
    except KeyError:
        raise ConfigError("no sampling preset %r; declared presets: %s"
                          % (name, sorted(PRESETS)))


@dataclass(frozen=True)
class EngineDescription:
    """What an engine must be able to say about itself, for the result rows."""

    engine: str
    engine_version: str
    torch_version: str
    model_id: str
    model_revision: str
    dtype: str

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
    """One generation, scored, with the whole configuration that produced it."""

    game_id: str
    question_id: str
    question_set: str
    family: str
    mode: str
    persona: str
    reading: str
    sample_index: int
    batch_index: int
    seed: int
    continuation: str
    answer: str
    value: Optional[float]
    tag: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int
    engine: str
    engine_version: str
    torch_version: str
    model_id: str
    model_revision: str
    dtype: str
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
        samples_per_prompt: int, persona: Optional[elicit.Persona] = None,
        reading: str = "stated", batch_size: int = 8) -> List[Row]:
    """Generate and score every (game_id, mode) pair, `samples_per_prompt` times.

    `engine` is the caller's; this function never chooses or substitutes one.
    Deterministic given `sampling.seed`, `batch_size` and the task order.
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
        for task, continuation in zip(batch, continuations):
            rows.append(_row(task, continuation, batch_index, sampling, record, reading))
    return rows


def write_rows(path, rows: Sequence[Row]) -> Path:
    """Write the rows as CSV with `ROW_FIELDS` as its header.

    An unresolved answer keeps its row and its tag; `value` is left empty. It is
    never a zero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Row):
                raise TypeError("write_rows takes Row objects, got %s" % type(row).__name__)
            record = dataclasses.asdict(row)
            if record["value"] is None:
                record["value"] = ""
            writer.writerow(record)
    return path


class HuggingFaceEngine:
    """`transformers` `generate()` — the one engine.

    HF rather than vLLM because steering needs forward hooks, which vLLM cannot
    give us; the same path then serves beta=0 and beta!=0.

    Constructing this imports torch and transformers, and puts the tokenizer in
    left-padding mode, which decoder-only batched generation requires.
    """

    name = "huggingface"

    def __init__(self, model, tokenizer, model_id: str, model_revision: Optional[str] = None):
        import torch
        import transformers

        self._torch = torch
        self._transformers = transformers
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._model_revision = model_revision or _resolved_revision(model)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        )

    def generate(self, prompts: Sequence[str], sampling: Sampling, seed: int) -> List[str]:
        """Continue each prompt once. Returns only what the model added."""
        torch = self._torch
        torch.manual_seed(seed)
        # A config built from `sampling` alone, so the model's own
        # generation_config.json contributes nothing.
        config = self._transformers.GenerationConfig(
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            **sampling.generation_kwargs())
        # add_special_tokens=False: the chat template already wrote them.
        batch = self._tokenizer(list(prompts), return_tensors="pt", padding=True,
                                add_special_tokens=False)
        batch = {key: value.to(self._model.device) for key, value in batch.items()}
        with torch.no_grad():
            output = self._model.generate(**batch, generation_config=config)
        prompt_len = batch["input_ids"].shape[1]
        return [self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                for row in output]


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


def _row(task: Task, continuation: str, batch_index: int, sampling: Sampling,
         record: Provenance, reading: str) -> Row:
    game = task.game
    # The model's assistant turn is what it wrote plus what the mode pre-filled;
    # scoring the continuation alone would drop the stub's "give to Agent 2".
    answer = task.prompt.assistant_prefill + continuation
    extraction = games.score(game, answer, reading)
    engine = record.engine
    return Row(
        game_id=game.id,
        question_id=game.question_id,
        question_set=game.question_set,
        family=game.family,
        mode=task.prompt.mode,
        persona=task.prompt.persona,
        reading=reading,
        sample_index=task.sample_index,
        batch_index=batch_index,
        seed=sampling.seed,
        continuation=continuation,
        answer=answer,
        value=extraction.value,
        tag=extraction.tag,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        repetition_penalty=sampling.repetition_penalty,
        max_new_tokens=sampling.max_new_tokens,
        engine=engine.engine,
        engine_version=engine.engine_version,
        torch_version=engine.torch_version,
        model_id=engine.model_id,
        model_revision=engine.model_revision,
        dtype=engine.dtype,
        chat_template_sha256=record.chat_template_sha256,
        question_sha256=game.question_sha256,
        prompt_sha256=task.prompt.sha256,
        repo_commit=record.repo_commit,
        repo_dirty=record.repo_dirty,
    )


def _batches(tasks: Sequence[Task], size: int):
    for start in range(0, len(tasks), size):
        yield tasks[start:start + size]


def _resolved_revision(model) -> str:
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not revision:
        raise ProvenanceError(
            "cannot resolve the model revision from the loaded model; pass "
            "model_revision= explicitly — a result row without it is not reproducible")
    return revision


def _git(*args: str) -> str:
    result = subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ProvenanceError("git %s failed in %s: %s"
                              % (" ".join(args), REPO_ROOT, result.stderr.strip()))
    return result.stdout.strip()


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
