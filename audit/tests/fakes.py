"""Stand-ins that let the elicitation and generation tests run with no model.

Not collected by pytest: it is a helper module, not a test module.
"""

import contextlib
import random
import types

from audit.generate import EngineDescription

#: What Qwen2.5-Instruct's template inserts when the conversation has no system turn.
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class QwenLikeTokenizer:
    """The system/user branch of Qwen2.5-Instruct's chat template, transcribed.

    Transcribed from the template in that model's `tokenizer_config.json`, so the
    byte assertions are the shape a real run produces. It is still a fake: it
    cannot notice that the upstream template changed. What catches that is
    `chat_template_sha256`, recorded on every result row.
    """

    chat_template = "{# fake: Qwen2.5-Instruct system/user subset #}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **_ignored):
        if tokenize:
            raise ValueError("this fake renders text only")
        messages = list(messages)
        parts = []
        if messages and messages[0]["role"] == "system":
            parts.append("<|im_start|>system\n%s<|im_end|>\n" % messages[0]["content"])
            messages = messages[1:]
        else:
            parts.append("<|im_start|>system\n%s<|im_end|>\n" % DEFAULT_SYSTEM)
        for message in messages:
            parts.append("<|im_start|>%s\n%s<|im_end|>\n"
                         % (message["role"], message["content"]))
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class TokenizingTokenizer(QwenLikeTokenizer):
    """A tokenizer that returns ids — the mistake `render` must refuse."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **_ignored):
        return [0, 1, 2]


class TemplatelessTokenizer(QwenLikeTokenizer):
    """A tokenizer whose template cannot be fingerprinted."""

    chat_template = None


FAKE_ENGINE_DESCRIPTION = EngineDescription(
    engine="fake",
    engine_version="0.0.0",
    torch_version="0.0.0",
    model_id="fake/model",
    model_revision="0" * 40,
    dtype="torch.float32",
    attn_implementation="sdpa",
    stop_token_ids=(151645, 151643),
)


class ScriptedEngine:
    """Replays canned continuations in order and records what it was asked."""

    def __init__(self, continuations, description=FAKE_ENGINE_DESCRIPTION):
        self._continuations = list(continuations)
        self._description = description
        self._served = 0
        #: [(prompts, seed)] — one entry per batch.
        self.calls = []

    def describe(self):
        return self._description

    def generate(self, prompts, sampling, seed):
        prompts = list(prompts)
        self.calls.append((prompts, seed))
        served = [self._continuations[(self._served + i) % len(self._continuations)]
                  for i in range(len(prompts))]
        self._served += len(prompts)
        return served


class SeededEngine:
    """A continuation drawn from the seed, so a run's determinism is observable."""

    def __init__(self, description=FAKE_ENGINE_DESCRIPTION):
        self._description = description

    def describe(self):
        return self._description

    def generate(self, prompts, sampling, seed):
        rng = random.Random(seed)
        return ["%d to Agent 2." % rng.randint(0, 100) for _ in prompts]


# --- enough of torch/transformers to pin what the HF engine calls --------------
# The engine path itself was exercised against a real tiny model during review;
# these fakes keep that contract asserted in a suite with neither installed.

class FakeGenerationConfig:
    """Records the fields the engine builds it from."""

    def __init__(self, **fields):
        self.fields = dict(fields)


def fake_transformers(version="4.52.3"):
    module = types.ModuleType("transformers")
    module.__version__ = version
    module.GenerationConfig = FakeGenerationConfig
    return module


def fake_torch(version="2.6.0"):
    module = types.ModuleType("torch")
    module.__version__ = version
    module.seeds = []
    module.manual_seed = module.seeds.append
    module.no_grad = contextlib.nullcontext
    return module


class FakeTensor:
    """A batch of id rows, with the two operations the engine performs on one."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def to(self, _device):
        return self

    def __iter__(self):
        return iter(self.rows)


class FakeHfTokenizer:
    """One id per character, padded on whichever side it is asked for."""

    chat_template = "{# fake: engine-path tokenizer #}"

    def __init__(self, pad_token_id=151643, eos_token_id=151645):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = "right"
        #: the padding side in force at each call — the engine must borrow "left"
        self.padding_sides_seen = []
        self.add_special_tokens_seen = []

    def __call__(self, prompts, return_tensors=None, padding=None, add_special_tokens=None):
        self.padding_sides_seen.append(self.padding_side)
        self.add_special_tokens_seen.append(add_special_tokens)
        width = max(len(prompt) for prompt in prompts)
        rows = []
        for prompt in prompts:
            ids = [ord(character) for character in prompt]
            pad = [self.pad_token_id] * (width - len(ids))
            rows.append(pad + ids if self.padding_side == "left" else ids + pad)
        mask = [[1] * width for _ in prompts]
        return {"input_ids": FakeTensor(rows), "attention_mask": FakeTensor(mask)}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token) for token in ids if token < 0x110000)


class FakeCausalModel:
    """Appends a fixed continuation to every row and records its generate kwargs."""

    #: the ids the fake always appends, decoding to "42"
    CONTINUATION = (ord("4"), ord("2"))

    def __init__(self, commit_hash="a" * 40, eos_token_id=(151645, 151643),
                 dtype="torch.bfloat16", attn_implementation="sdpa"):
        self.config = types.SimpleNamespace(_commit_hash=commit_hash,
                                            _attn_implementation=attn_implementation)
        self.generation_config = types.SimpleNamespace(eos_token_id=list(eos_token_id))
        self.device = "cpu"
        self.dtype = dtype
        #: [kwargs] — one entry per generate() call, minus the batch tensors
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append(kwargs)
        return FakeTensor([list(row) + list(self.CONTINUATION) for row in input_ids])
