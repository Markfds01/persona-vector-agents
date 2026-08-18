"""Stand-ins that let the elicitation and generation tests run with no model.

Not collected by pytest: it is a helper module, not a test module.
"""

import random

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
