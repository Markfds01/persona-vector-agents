"""HTTP client for the local llama-server (llama.cpp) endpoint, nothing else.

The judges for `results/general-question-transfer` are plain HTTP calls to the
llama.cpp `llama-server` running on PimientoRojo (its `/v1/chat/completions`
route), reached through `PIMIENTOROJO_BASE_URL`. This module is the only code that
talks to it. What it guarantees, each in the code rather than in a convention:

* **The endpoint comes from the environment and only from there.** The base URL is
  `PIMIENTOROJO_BASE_URL`, the key (optional) `PIMIENTOROJO_API_KEY`. There is no
  default URL, no other environment variable is read and there is no second
  server anywhere: an unset or unreachable endpoint is an error, never a fallback.
* **The served model is checked before anything is sent.** `JudgeClient.open()`
  calls `/v1/models`, records the raw response, and raises `ModelMismatch` unless
  the configured id is served. Every completion's own `model` field is checked too.
  If the endpoint is unreachable, `open()` raises and nothing is written.
* **Deterministic requests.** `temperature=0`, a fixed `seed`, an explicit
  `max_tokens`, one user message. The request payload is the cache key: its
  canonical JSON's sha256.
* **An append-only cache.** One JSON line per finished call, flushed and fsynced,
  never rewritten. A transport failure is not cached (a re-run may retry it); a
  call the server answered is, whatever its status.
* **Retries only on transport errors** - connection failure, timeout, HTTP 5xx -
  capped, each attempt logged and kept on the result. A 4xx, an unparseable answer
  or a low-mass score is a result, not a reason to ask again.
* **Every call ends in a status.** `ok`, `unparseable`, `refusal`, `low_mass`,
  `no_logprobs`, `few_logprobs`, `truncated`, `http_error`, `transport_error`.
  Nothing is dropped, imputed or re-asked for a better answer.

Three modes, fixed per judge before any judgment exists, never switched; the first
two are 0-100 scorers and the third is the pairwise forced choice (PREREG section 6):

* `logprob_0_100` - upstream's: one completion token, `top_logprobs`, the expected
  value over tokens that are integers 0..100, `low_mass` if their total probability
  is under 0.25 (`judge.py:_aggregate_0_100_score`). Only valid for a judge whose
  tokenizer makes every "0".."100" one token in that position - see
  `results/general-question-transfer/scripts/judging/tokenizer_check.py`.
* `text_0_100` - the judge writes text; exactly one `<score>NN</score>` with
  0 <= NN <= 100 is a score, anything else is `unparseable`. For a reasoning model
  (gpt-oss) only the FINAL channel is parsed: a server that separates reasoning
  returns it as `content`; a raw harmony transcript is cut at its last
  `<|channel|>final<|message|>`.
* `choice_1_2` - the pairwise judgment: the judge writes text; exactly one
  `<choice>1</choice>` or `<choice>2</choice>` in the final channel is a choice (the
  score is 1.0 or 2.0), anything else is `unparseable`. No logprobs.

The client never sees what a call is about. A request is a rendered prompt - the
question and the answer substituted into a verbatim template - and an opaque key
the caller chooses; arm, vector, k and game never reach this module. `judge_many`
sends requests in a seeded shuffled order.

Stdlib only (urllib in worker threads under asyncio), so it imports under every
interpreter this repository is tested with. It does not import `judge.py` or
`eval/` - both read credentials at import, and `eval/eval_persona.py` switches
engines between beta 0 and steered points.
"""

import asyncio
import dataclasses
import datetime as _dt
import hashlib
import http.client
import json
import logging
import math
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

log = logging.getLogger("lab.llama_server_client")

ENV_BASE_URL = "PIMIENTOROJO_BASE_URL"
ENV_API_KEY = "PIMIENTOROJO_API_KEY"

MODES = ("logprob_0_100", "text_0_100", "choice_1_2")
#: upstream's threshold: under this much probability on 0..100 there is no score
MIN_NUMBER_MASS = 0.25

STATUSES = ("ok", "unparseable", "refusal", "low_mass", "no_logprobs", "few_logprobs",
            "truncated", "http_error", "transport_error")

_SCORE_TAG = re.compile(r"<score>(\d{1,3})</score>")
_ANY_SCORE_TAG = re.compile(r"<score>", re.IGNORECASE)
_CHOICE_TAG = re.compile(r"<choice>([12])</choice>")
_ANY_CHOICE_TAG = re.compile(r"<choice>", re.IGNORECASE)
_INTEGER_TOKEN = re.compile(r"^[0-9]{1,3}$")
_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_END = re.compile(r"<\|(?:return|end|call)\|>")


class JudgeError(RuntimeError):
    """Base class: a judge run that must stop."""


class ConfigError(JudgeError):
    """The client was configured in a way that cannot produce a valid judgment."""


class EndpointUnavailable(JudgeError):
    """The endpoint could not be reached at all. The run stops; nothing is written."""


class ModelMismatch(JudgeError):
    """The endpoint does not serve the configured model id."""


class CacheCorrupt(JudgeError):
    """A cache line does not parse or does not match its key."""


@dataclasses.dataclass(frozen=True)
class JudgeConfig:
    """Everything that shapes a request, fixed before a run.

    `extra_payload` is merged into every request verbatim and is part of the cache
    key - e.g. `{"chat_template_kwargs": {"reasoning_effort": "low"}}` for gpt-oss on
    llama-server. It may not override a field this class owns.
    """

    model: str
    mode: str
    max_tokens: int
    seed: int
    concurrency: int = 4
    top_logprobs: int = 20
    timeout_s: float = 600.0
    max_transport_retries: int = 3
    retry_backoff_s: float = 2.0
    temperature: float = 0.0
    extra_payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ConfigError("mode %r; have %s" % (self.mode, MODES))
        if self.temperature != 0.0:
            raise ConfigError("judges run at temperature 0, got %r" % self.temperature)
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ConfigError("max_tokens must be an explicit positive int")
        if self.mode == "logprob_0_100" and self.max_tokens != 1:
            raise ConfigError("logprob_0_100 reads one token: max_tokens must be 1")
        if self.mode == "logprob_0_100" and self.top_logprobs < 20:
            raise ConfigError("logprob_0_100 needs top_logprobs >= 20")
        if not isinstance(self.seed, int):
            raise ConfigError("seed must be an explicit int")
        if self.concurrency < 1 or self.max_transport_retries < 0:
            raise ConfigError("concurrency >= 1 and max_transport_retries >= 0")
        owned = {"model", "messages", "temperature", "seed", "max_tokens", "logprobs",
                 "top_logprobs", "stream", "n", "cache_prompt"}
        clash = owned & set(self.extra_payload)
        if clash:
            raise ConfigError("extra_payload may not set %s" % sorted(clash))

    def payload(self, prompt: str) -> Dict[str, Any]:
        """The exact request body for one rendered prompt."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stream": False,
            "cache_prompt": False,
        }
        if self.mode == "logprob_0_100":
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        body.update(json.loads(json.dumps(dict(self.extra_payload))))
        return body


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def render(template: str, question: str, answer: str) -> str:
    """Substitute the question and answer, and nothing else, into a verbatim template.

    Upstream renders with `str.format(**kwargs)`; this is the same substitution,
    and it refuses a template naming any other field so nothing else can leak in.
    """
    fields = set(re.findall(r"(?<!\{)\{([^{}]*)\}(?!\})", template))
    if fields != {"question", "answer"}:
        raise ConfigError("template must have exactly {question} and {answer}, has %s"
                          % sorted(fields))
    return template.format(question=question, answer=answer)


@dataclasses.dataclass
class JudgeResult:
    key: str
    status: str
    score: Optional[float]
    payload_sha256: str
    served_model: Optional[str]
    raw_response: Optional[Dict[str, Any]]
    detail: Dict[str, Any]
    attempts: List[Dict[str, Any]]
    timestamp: str
    from_cache: bool = False

    def as_row(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --- scoring ----------------------------------------------------------------

def score_logprobs(top_logprobs: Sequence[Mapping[str, Any]]):
    """Upstream's 0..100 aggregation over one token's top logprobs.

    Returns (status, score, detail). A token counts if, stripped of whitespace, it
    is 1-3 ASCII digits with value <= 100. (Upstream used `int(token)`, which also
    accepts "+5", "4_2" and non-ASCII digits; none of those is a score token for a
    tokenizer that passes the single-token check, and accepting them would make a
    malformed token count as a number.) Entries are summed, so " 42" and "42" both
    contribute, as in upstream.
    """
    mass, weighted = 0.0, 0.0
    tokens = []
    for entry in top_logprobs:
        token = entry.get("token", "")
        prob = math.exp(float(entry.get("logprob", -math.inf)))
        tokens.append([token, prob])
        stripped = token.strip()
        if _INTEGER_TOKEN.match(stripped) and int(stripped) <= 100:
            mass += prob
            weighted += int(stripped) * prob
    detail = {"number_mass": mass, "top_logprobs": tokens}
    if mass < MIN_NUMBER_MASS:
        top = max(tokens, key=lambda t: t[1])[0] if tokens else ""
        status = "refusal" if top.strip().upper().startswith("REF") else "low_mass"
        return status, None, detail
    return "ok", weighted / mass, detail


def final_channel(content: str):
    """The final-channel text of a reply, or None if it has channels but no final."""
    if "<|channel|>" not in content:
        return content
    idx = content.rfind(_HARMONY_FINAL)
    if idx < 0:
        return None
    tail = content[idx + len(_HARMONY_FINAL):]
    end = _HARMONY_END.search(tail)
    return tail[:end.start()] if end else tail


def score_text(content: Optional[str]):
    """Exactly one `<score>NN</score>`, 0..100, in the final channel."""
    if content is None:
        return "unparseable", None, {"reason": "no content"}
    final = final_channel(content)
    if final is None:
        return "unparseable", None, {"reason": "channel markers without a final channel"}
    tags = _SCORE_TAG.findall(final)
    loose = len(_ANY_SCORE_TAG.findall(final))
    detail = {"final_text": final, "n_score_tags": loose}
    if len(tags) == 1 and loose == 1:
        value = int(tags[0])
        if value <= 100 and (tags[0] == "0" or not tags[0].startswith("0")):
            return "ok", float(value), detail
        detail["reason"] = "score out of range or zero-padded"
        return "unparseable", None, detail
    if loose == 0 and "REFUSAL" in final.upper():
        return "refusal", None, detail
    detail["reason"] = "expected exactly one <score>NN</score>, found %d tag(s)" % loose
    return "unparseable", None, detail


def score_choice(content: Optional[str]):
    """Exactly one `<choice>1</choice>` or `<choice>2</choice>` in the final channel."""
    if content is None:
        return "unparseable", None, {"reason": "no content"}
    final = final_channel(content)
    if final is None:
        return "unparseable", None, {"reason": "channel markers without a final channel"}
    tags = _CHOICE_TAG.findall(final)
    loose = len(_ANY_CHOICE_TAG.findall(final))
    detail = {"final_text": final, "n_choice_tags": loose}
    if len(tags) == 1 and loose == 1:
        return "ok", float(tags[0]), detail
    detail["reason"] = "expected exactly one <choice>1|2</choice>, found %d tag(s)" % loose
    return "unparseable", None, detail


def interpret(config: JudgeConfig, response: Mapping[str, Any]):
    """(status, score, detail) for one completion body."""
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "unparseable", None, {"reason": "no choices in response"}
    finish = choice.get("finish_reason")
    message = choice.get("message") or {}
    if config.mode == "logprob_0_100":
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content or not content[0].get("top_logprobs"):
            return "no_logprobs", None, {"finish_reason": finish}
        top = content[0]["top_logprobs"]
        if len(top) < config.top_logprobs:
            return "few_logprobs", None, {"n_top_logprobs": len(top),
                                          "requested": config.top_logprobs}
        status, score, detail = score_logprobs(top)
        detail["finish_reason"] = finish
        detail["sampled_token"] = content[0].get("token")
        return status, score, detail
    if finish == "length":
        return "truncated", None, {"finish_reason": finish,
                                   "content": message.get("content")}
    parse = score_choice if config.mode == "choice_1_2" else score_text
    status, score, detail = parse(message.get("content"))
    detail["finish_reason"] = finish
    return status, score, detail


# --- transport ----------------------------------------------------------------

class _Transport(Exception):
    """Retryable: the request may not have been served."""


class _HttpError(Exception):
    def __init__(self, code, body):
        super().__init__(code)
        self.code, self.body = code, body


def _base_url() -> str:
    base = os.environ.get(ENV_BASE_URL, "").strip()
    if not base:
        raise ConfigError("%s is not set; there is no default endpoint and no fallback"
                          % ENV_BASE_URL)
    return base.rstrip("/")


def _request(method: str, url: str, body: Optional[bytes], timeout: float):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = os.environ.get(ENV_API_KEY, "")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:4000]
        if exc.code >= 500:
            raise _Transport("HTTP %d: %s" % (exc.code, text[:200]))
        raise _HttpError(exc.code, text)
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
            http.client.HTTPException) as exc:
        raise _Transport("%s: %s" % (type(exc).__name__, exc))
    except json.JSONDecodeError as exc:
        raise _HttpError(200, "response is not JSON: %s" % exc)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


# --- cache --------------------------------------------------------------------

class Cache:
    """Append-only JSONL, keyed on the request payload's sha256. First write wins."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CacheCorrupt("%s:%d does not parse (%s); refusing to guess"
                                           % (self.path, number, exc))
                    key = entry.get("payload_sha256")
                    if key != payload_sha256(entry.get("payload")):
                        raise CacheCorrupt("%s:%d key does not match its payload"
                                           % (self.path, number))
                    self._entries.setdefault(key, entry)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(key)

    def __len__(self):
        return len(self._entries)

    def append(self, key: str, payload: Mapping[str, Any], response: Mapping[str, Any],
               served_model: Optional[str], timestamp: str) -> None:
        entry = {"payload_sha256": key, "payload": payload, "response": response,
                 "served_model": served_model, "timestamp": timestamp}
        with self._lock:
            if key in self._entries:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(entry) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._entries[key] = entry


# --- client -------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class JudgeRequest:
    """One rendered prompt and the caller's opaque key for it."""

    key: str
    prompt: str


class JudgeClient:
    """Use as `async with JudgeClient(config, cache_path) as client:`."""

    def __init__(self, config: JudgeConfig, cache_path, sleep=time.sleep):
        self.config = config
        self.cache = Cache(cache_path)
        self.base_url: Optional[str] = None
        self.models_response: Optional[Dict[str, Any]] = None
        self.server_props: Optional[Dict[str, Any]] = None
        self._sleep = sleep
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *exc):
        return False

    async def open(self) -> Dict[str, Any]:
        """Check the endpoint serves the configured model. Raises before any write."""
        self.base_url = _base_url()
        try:
            models = await asyncio.to_thread(
                _request, "GET", self.base_url + "/v1/models", None, self.config.timeout_s)
        except _Transport as exc:
            raise EndpointUnavailable("%s/v1/models unreachable: %s" % (ENV_BASE_URL, exc))
        except _HttpError as exc:
            raise EndpointUnavailable("/v1/models returned HTTP %s" % exc.code)
        ids = [m.get("id") for m in (models.get("data") or []) if isinstance(m, dict)]
        if self.config.model not in ids:
            raise ModelMismatch("configured model %r is not served; /v1/models lists %s"
                                % (self.config.model, ids))
        self.models_response = models
        # llama-server describes the loaded weights and slots at /props. Recorded when
        # present, recorded as unavailable when not.
        try:
            self.server_props = await asyncio.to_thread(
                _request, "GET", self.base_url + "/props", None, self.config.timeout_s)
        except (_Transport, _HttpError) as exc:
            self.server_props = {"unavailable": str(exc)[:300]}
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        return models

    def provenance(self) -> Dict[str, Any]:
        """Run-level record: what was asked of which server. No URL (it is env-only)."""
        return {
            "endpoint": "env:%s" % ENV_BASE_URL,
            "configured_model": self.config.model,
            "models_response": self.models_response,
            "server_props": self.server_props,
            "config": dataclasses.asdict(self.config),
        }

    async def judge(self, request: JudgeRequest, use_cache: bool = True) -> JudgeResult:
        """One call. `use_cache=False` re-sends (determinism check) and caches nothing."""
        if self._semaphore is None:
            raise JudgeError("call open() first: the model check precedes any request")
        payload = self.config.payload(request.prompt)
        key = payload_sha256(payload)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                status, score, detail = interpret(self.config, hit["response"])
                return JudgeResult(request.key, status, score, key, hit["served_model"],
                                   hit["response"], detail, [], hit["timestamp"], True)
        async with self._semaphore:
            return await asyncio.to_thread(self._send, request, payload, key, use_cache)

    def _send(self, request, payload, key, use_cache) -> JudgeResult:
        body = canonical_json(payload).encode("utf-8")
        attempts: List[Dict[str, Any]] = []
        url = self.base_url + "/v1/chat/completions"
        for attempt in range(self.config.max_transport_retries + 1):
            started = _now()
            try:
                response = _request("POST", url, body, self.config.timeout_s)
            except _Transport as exc:
                attempts.append({"attempt": attempt, "at": started, "error": str(exc)})
                log.warning("judge %s transport error (attempt %d/%d): %s", request.key,
                            attempt + 1, self.config.max_transport_retries + 1, exc)
                if attempt < self.config.max_transport_retries:
                    self._sleep(self.config.retry_backoff_s * (2 ** attempt))
                continue
            except _HttpError as exc:
                attempts.append({"attempt": attempt, "at": started,
                                 "error": "HTTP %s" % exc.code})
                return JudgeResult(request.key, "http_error", None, key, None, None,
                                   {"http_status": exc.code, "body": exc.body}, attempts,
                                   _now())
            served = response.get("model")
            if served is not None and served != self.config.model:
                raise ModelMismatch("completion served by %r, configured %r"
                                    % (served, self.config.model))
            attempts.append({"attempt": attempt, "at": started, "error": None})
            stamp = _now()
            if use_cache:
                self.cache.append(key, payload, response, served, stamp)
            status, score, detail = interpret(self.config, response)
            return JudgeResult(request.key, status, score, key, served, response, detail,
                               attempts, stamp)
        return JudgeResult(request.key, "transport_error", None, key, None, None,
                           {"reason": "gave up after %d attempts" % len(attempts)},
                           attempts, _now())

    async def judge_many(self, requests: Sequence[JudgeRequest], order_seed: int,
                         use_cache: bool = True) -> List[JudgeResult]:
        """Every request judged, sent in a seeded shuffled order, returned in input order."""
        keys = [r.key for r in requests]
        if len(set(keys)) != len(keys):
            raise ConfigError("request keys must be unique")
        order = list(range(len(requests)))
        random.Random(order_seed).shuffle(order)
        tasks = {i: asyncio.ensure_future(self.judge(requests[i], use_cache)) for i in order}
        await asyncio.gather(*tasks.values())
        return [tasks[i].result() for i in range(len(requests))]
