"""Apply the authors' activation steering to the audit's own generation path.

`activation_steer.ActivationSteerer` is the thing being reproduced. If we add the
vector anywhere other than where they add it, every steered number we produce is
incomparable with theirs and nothing downstream would notice, so this module
matches their arithmetic rather than reimplementing the idea:

* their `--layer L` hooks the layer list at index `L - 1`, whose output is
  `hidden_states[L]`; the vector is indexed `[L]`. Both are the same tensor, and
  `hooked_module` asserts it.
* the vector is cast to the model's parameter dtype **before** it is scaled, so
  the delta is `coeff * bfloat16(v)`, not `bfloat16(coeff * v)`. On this vector
  the two differ in the last bf16 mantissa bit; matching them is free.
* `positions="all"` adds at every position of every forward pass, prompt tokens
  and each decode step alike. `scripts/eval_steering.sh` uses that mode.
* a decoder layer may return a bare tensor or a tuple; only the first entry is
  touched, and the rest are carried through untouched.

Two deliberate differences, neither of which can move a number:

* the hook site is resolved by an explicit path with an assertion, not by
  `ActivationSteerer`'s first-match search over five architecture spellings. That
  search reaches `model.layers` only because Qwen2 has no `transformer.h`;
  `upstream_layer_list` reproduces the search so a test can prove the two agree
  instead of assuming it.
* the vector is detached and copied before use. `torch.as_tensor` returns the
  *same* object when dtype and device already match, so upstream can alias a
  caller's tensor and keep its `requires_grad`; a copy removes both hazards.

Importing this module needs no torch. Every entry point that does imports it.
"""

import csv
import dataclasses
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from audit import generate
from audit.generate import ConfigError, Row, ROW_FIELDS
from audit.paths import REPO_ROOT

#: The attribute paths `ActivationSteerer._locate_layer` searches, in its order.
#: First match wins there, so the order is part of the behaviour being matched.
UPSTREAM_LAYER_ATTRS = ("transformer.h", "encoder.layer", "model.layers",
                        "gpt_neox.layers", "block")

#: The position policies upstream implements. `eval_steering.sh` uses "all".
POSITIONS = ("all", "prompt", "response")

#: The provenance columns a steered row carries on top of `generate.ROW_FIELDS`.
#: These belong on `generate.Row` — see this module's entry in `audit/README.md`;
#: adding them there was out of scope for the task that wrote this file.
STEERING_FIELDS = ("steer_coeff", "steer_layer", "steer_module_index",
                   "steer_module_path", "steer_positions", "steer_vector",
                   "steer_vector_sha256", "steer_delta_dtype")


class SteeringError(ConfigError):
    """A steering setup that would not reproduce what it claims to reproduce."""


@dataclass(frozen=True)
class Steering:
    """One steering configuration: which vector, which layer, how hard, where.

    `layer` is upstream's 1-based `--layer`: it indexes the vector file directly
    and the layer list at `layer - 1`. Keeping their spelling means a run can be
    compared with their command line without arithmetic in between.
    """

    vector_path: str
    layer: int
    coeff: float
    positions: str = "all"

    def __post_init__(self):
        if not isinstance(self.layer, int) or isinstance(self.layer, bool):
            raise SteeringError("layer must be an integer, got %r" % (self.layer,))
        if self.layer < 1:
            raise SteeringError("layer is upstream's 1-based --layer and indexes the "
                                "layer list at layer-1, so it must be >= 1, got %d"
                                % self.layer)
        if isinstance(self.coeff, bool) or not isinstance(self.coeff, (int, float)):
            raise SteeringError("coeff must be a number, got %r" % (self.coeff,))
        if self.positions not in POSITIONS:
            raise SteeringError("positions must be one of %s, got %r"
                                % (list(POSITIONS), self.positions))
        if not Path(self.vector_path).is_file():
            raise SteeringError("no steering vector at %s" % self.vector_path)

    @property
    def module_index(self) -> int:
        """Index into the layer list — upstream's `layer_idx = layer - 1`."""
        return self.layer - 1

    @property
    def vector_sha256(self) -> str:
        return _file_fingerprint(str(Path(self.vector_path).resolve()))

    def load_vector(self):
        """Row `layer` of the vector file, as stored (float32 on CPU).

        Loaded with `weights_only=False` because that is how upstream loads it and
        the file is a plain tensor either way.
        """
        import torch

        stored = torch.load(self.vector_path, weights_only=False)
        if not torch.is_tensor(stored) or stored.ndim != 2:
            raise SteeringError(
                "%s holds %s; a persona vector file is a 2-D tensor indexed by layer"
                % (self.vector_path, type(stored).__name__ if not torch.is_tensor(stored)
                   else "a %d-D tensor" % stored.ndim))
        if self.layer >= stored.shape[0]:
            raise SteeringError("%s has %d layer rows (0-%d); layer %d is out of range"
                                % (self.vector_path, stored.shape[0],
                                   stored.shape[0] - 1, self.layer))
        return stored[self.layer]


def upstream_layer_list(model):
    """The layer list `ActivationSteerer._locate_layer` would find on this model.

    Their search takes the first attribute path that resolves to something
    subscriptable, so on a model exposing two of them the order decides. Kept here
    so the hook site can be *asserted* equal to theirs rather than assumed.
    """
    for path in UPSTREAM_LAYER_ATTRS:
        current = model
        for part in path.split("."):
            if not hasattr(current, part):
                break
            current = getattr(current, part)
        else:
            if hasattr(current, "__getitem__"):
                return path, current
    raise SteeringError("no layer list on this model under any of %s"
                        % list(UPSTREAM_LAYER_ATTRS))


def hooked_module(model, layer: int):
    """The module upstream hooks for `--layer <layer>`, with its attribute path.

    Returns `(path, module)`. Raises unless the explicit `model.layers` path and
    upstream's search land on the same object: a silently different hook site is
    the one failure that would invalidate every steered number without showing up
    anywhere else.
    """
    path, layers = upstream_layer_list(model)
    if not -len(layers) <= layer - 1 < len(layers):
        raise SteeringError("layer %d needs layer list index %d, but %s has %d layers"
                            % (layer, layer - 1, path, len(layers)))
    module = layers[layer - 1]
    explicit = getattr(getattr(model, "model", None), "layers", None)
    if explicit is not None and explicit[layer - 1] is not module:
        raise SteeringError(
            "upstream's search hooks %s[%d] but model.model.layers[%d] is a different "
            "module; the vector would be added at the wrong depth" % (path, layer - 1,
                                                                      layer - 1))
    return "%s[%d]" % (path, layer - 1), module


class ActivationSteering:
    """`ActivationSteerer` with `positions` semantics matched, as a context manager.

    Installs one forward hook on the layer `hooked_module` resolves and adds
    `coeff * vector` to that layer's output for as long as the block runs.
    """

    def __init__(self, model, steering: Steering, vector=None):
        import torch

        self.steering = steering
        self.module_path, self._module = hooked_module(model, steering.layer)
        self._handle = None

        parameter = next(model.parameters())
        source = steering.load_vector() if vector is None else vector
        if source.ndim != 1:
            raise SteeringError("steering vector must be 1-D, got %d-D" % source.ndim)
        hidden = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden is not None and source.numel() != hidden:
            raise SteeringError("steering vector has %d elements, model hidden_size is %d"
                                % (source.numel(), hidden))
        # cast first, scale second — upstream's order, and the two disagree in bf16
        self.vector = source.detach().clone().to(dtype=parameter.dtype,
                                                 device=parameter.device)
        self.delta = steering.coeff * self.vector
        self._is_tensor = torch.is_tensor

    @property
    def columns(self) -> Dict[str, object]:
        """What actually ran, for the result rows — resolved, not requested."""
        return {
            "steer_coeff": self.steering.coeff,
            "steer_layer": self.steering.layer,
            "steer_module_index": self.steering.module_index,
            "steer_module_path": self.module_path,
            "steer_positions": self.steering.positions,
            "steer_vector": _repo_relative(self.steering.vector_path),
            "steer_vector_sha256": self.steering.vector_sha256,
            "steer_delta_dtype": str(self.delta.dtype),
        }

    def _add(self, tensor):
        delta = self.delta.to(tensor.device)
        if self.steering.positions == "all":
            return tensor + delta
        if self.steering.positions == "prompt":
            # upstream reads a width-1 forward as "this is a decode step". It is
            # their tell, and it misfires on a one-token prompt.
            if tensor.shape[1] == 1:
                return tensor
            return tensor + delta
        updated = tensor.clone()
        updated[:, -1, :] += delta
        return updated

    def _hook(self, _module, _inputs, output):
        if self._is_tensor(output):
            return self._add(output)
        if isinstance(output, (tuple, list)):
            if not output or not self._is_tensor(output[0]):
                return output
            return (self._add(output[0]), *output[1:])
        return output

    def __enter__(self) -> "ActivationSteering":
        if self._handle is not None:
            raise SteeringError("this steering is already installed")
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exception):
        self.remove()
        return False

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class SteeredEngine:
    """A `generate` engine with the steering hook live for the whole batch.

    The hook is installed around `generate()` and removed after it, which is where
    upstream puts it: inside it, every prompt forward and every decode step of the
    batch sees the vector.

    Wrapping rather than subclassing keeps the unsteered path the one in
    `generate.py`: a steered run and a beta=0 run differ by this object and
    nothing else.
    """

    def __init__(self, engine, model, steering: Steering, vector=None):
        self._engine = engine
        self._model = model
        self.steering = steering
        self._vector = vector
        # built once so a bad layer, dtype or vector fails before any generation
        self._probe = ActivationSteering(model, steering, vector)

    @property
    def columns(self) -> Dict[str, object]:
        return self._probe.columns

    def describe(self) -> generate.EngineDescription:
        return self._engine.describe()

    def generate(self, prompts: Sequence[str], sampling, seed: int) -> List[str]:
        with ActivationSteering(self._model, self.steering, self._vector):
            return self._engine.generate(prompts, sampling, seed)


class SteeredRowWriter:
    """`generate.RowWriter` plus the steering columns, flushed per batch.

    A separate writer rather than a subclass because `RowWriter` fixes its
    fieldnames and its record shape in the two methods there are; overriding both
    is the whole class. These columns belong on `generate.Row` — see
    `audit/README.md`.
    """

    def __init__(self, path, columns: Dict[str, object]):
        missing = [name for name in STEERING_FIELDS if name not in columns]
        unknown = sorted(set(columns) - set(STEERING_FIELDS))
        if missing or unknown:
            raise SteeringError("steering columns are not usable: missing %s, unknown %s"
                                % (missing, unknown))
        self.path = Path(path)
        self.columns = dict(columns)
        self.written = 0
        self._handle = None
        self._writer = None

    def __enter__(self) -> "SteeredRowWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle,
                                      fieldnames=ROW_FIELDS + STEERING_FIELDS)
        self._writer.writeheader()
        self._handle.flush()
        return self

    def __exit__(self, *_exception):
        self._handle.close()
        self._handle = self._writer = None
        return False

    def write(self, rows: Iterable[Row]) -> int:
        """Append `rows` with the steering columns, flush, return the running total."""
        if self._writer is None:
            raise RuntimeError("SteeredRowWriter is closed; use it as a context manager")
        rows = list(rows)
        for row in rows:
            if not isinstance(row, Row):
                raise TypeError("SteeredRowWriter takes Row objects, got %s"
                                % type(row).__name__)
            record = dataclasses.asdict(row)
            if record["value"] is None:
                record["value"] = ""
            record.update(self.columns)
            self._writer.writerow(record)
        # flushed per batch: an OOM kill must not take the rows already generated
        self._handle.flush()
        self.written += len(rows)
        return self.written


def rows_path(directory, steering: Steering, game_id: str, mode: str) -> Path:
    """Where one coefficient's rows go. The coefficient is in the name and the file."""
    return Path(directory) / ("%s_%s_layer%d_%s_coef%s.csv"
                              % (game_id.replace("/", "-"), mode, steering.layer,
                                 steering.positions, _coefficient_label(steering.coeff)))


def sweep(pairs, engine, model, steerings: Sequence[Steering], sampling, tokenizer,
          samples_per_prompt: int, *, reading: str, batch_size: int, out_dir,
          persona=None, vector=None,
          on_coefficient=None) -> "Dict[float, Tuple[Path, List[Row]]]":
    """Run `generate.run` once per steering, one flushed CSV per coefficient.

    Every coefficient runs the same prompts at the same seed, so the coefficient is
    the only thing that differs between two points — which is the comparison the
    sweep exists to make. `samples_per_prompt`, `reading` and `batch_size` are the
    caller's, exactly as in `generate.run`; nothing here defaults a number.

    `on_coefficient(steering, path, rows)` is called as each coefficient finishes.
    """
    if not steerings:
        raise SteeringError("a sweep needs at least one steering")
    pairs = list(pairs)
    if not pairs:
        raise SteeringError("a sweep needs at least one (game_id, mode) pair")
    results = {}
    for steering in steerings:
        if steering.coeff in results:
            raise SteeringError("coefficient %r appears twice in the sweep"
                                % (steering.coeff,))
        steered = SteeredEngine(engine, model, steering, vector)
        game_id, mode = pairs[0]
        path = rows_path(out_dir, steering, game_id if len(pairs) == 1 else "pooled",
                         mode if len(pairs) == 1 else "pooled")
        with SteeredRowWriter(path, steered.columns) as writer:
            rows = generate.run(pairs, steered, sampling, tokenizer, samples_per_prompt,
                                reading=reading, batch_size=batch_size, persona=persona,
                                on_rows=writer.write)
        results[steering.coeff] = (path, rows)
        if on_coefficient is not None:
            on_coefficient(steering, path, rows)
    return results


def response_avg_projection(model, tokenizer, prompt: str, answer: str, vector,
                            layer: int) -> float:
    """`eval/cal_projection.py`'s `proj` metric for one (prompt, answer) pair.

    Their arithmetic, transcribed: mean the layer-`layer` hidden states over the
    answer tokens, then `(a*b).sum() / b.norm()` — a scalar projection, so its sign
    and scale both depend on getting the layer and the vector's orientation right.
    The answer span starts at `len(tokenizer.encode(prompt))`, which is their
    definition and is only a token boundary if the tokenizer does not merge across
    the seam; `projection_span_is_stable` reports when it does not.

    Runs one forward pass. Needs the model on a device that can hold the sequence.
    """
    import torch

    inputs = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_len >= inputs["input_ids"].shape[1]:
        raise SteeringError("the answer contributes no tokens: prompt is %d tokens and "
                            "prompt+answer is %d" % (prompt_len,
                                                     inputs["input_ids"].shape[1]))
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
    reference = vector.detach().cpu()
    return ((response_avg * reference).sum(dim=-1) / reference.norm(dim=-1)).item()


def projection_span_is_stable(tokenizer, prompt: str, answer: str) -> bool:
    """Whether the prompt's tokens really are a prefix of `prompt + answer`'s.

    When the tokenizer merges across the seam they are not, and the answer span
    `cal_projection.py` slices is off by the merged token. Reported, never fixed
    silently: their committed column was computed the unstable way.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    joint_ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    return joint_ids[:len(prompt_ids)] == prompt_ids


def _coefficient_label(coeff: float) -> str:
    """`-5.0` and `-5` name the same point; the file name says which run made it."""
    return ("%g" % coeff)


def _repo_relative(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


@lru_cache(maxsize=None)
def _file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
