"""`lab.llama_server_client` against a fake llama-server on localhost.

No network beyond 127.0.0.1 and no model. The fake server is scripted per test:
each POST to /v1/chat/completions pops the next behaviour (a response body, an
HTTP status, a delay past the client's timeout, or a dropped connection), and the
server records every request it saw, so "the client did not ask again" and "the
client did not ask anyone else" are assertions on what actually arrived.
"""

import asyncio
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lab import llama_server_client as jc

MODEL = "judge-under-test"


class FakeServer:
    def __init__(self, models=(MODEL,), props=None):
        self.models = list(models)
        self.props = props
        self.script = []
        self.requests = []
        self.default = None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                server.requests.append(("GET", self.path, None))
                if self.path == "/v1/models":
                    self._send(200, {"object": "list",
                                     "data": [{"id": m, "object": "model"} for m in server.models]})
                elif self.path == "/props" and server.props is not None:
                    self._send(200, server.props)
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(("POST", self.path, body))
                action = server.script.pop(0) if server.script else server.default
                kind = action[0]
                if kind == "json":
                    self._send(200, action[1])
                elif kind == "status":
                    self._send(action[1], {"error": "scripted %d" % action[1]})
                elif kind == "sleep":
                    time.sleep(action[1])
                    self._send(200, completion_text("<score>1</score>"))
                elif kind == "drop":
                    self.close_connection = True
                    self.connection.close()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01},
                                       daemon=True)
        self.thread.start()

    def posts(self):
        return [r for r in self.requests if r[0] == "POST"]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def completion_logprobs(pairs, model=MODEL, sampled=None):
    top = [{"token": t, "logprob": math.log(p) if p > 0 else -1e9, "bytes": None}
           for t, p in pairs]
    return {"model": model, "choices": [{
        "index": 0, "finish_reason": "length",
        "message": {"role": "assistant", "content": sampled or pairs[0][0]},
        "logprobs": {"content": [{"token": sampled or pairs[0][0], "logprob": top[0]["logprob"],
                                  "top_logprobs": top}]}}]}


def completion_text(content, finish="stop", model=MODEL):
    return {"model": model, "choices": [{"index": 0, "finish_reason": finish,
                                         "message": {"role": "assistant", "content": content}}]}


def twenty(pairs):
    """Pad to 20 entries with negligible non-number tokens."""
    pad = [("tok%d" % i, 1e-9) for i in range(20 - len(pairs))]
    return list(pairs) + pad


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer(props={"model_path": "x.gguf", "total_slots": 2,
                            "build_info": "b11836-f5027fb21"})
    monkeypatch.setenv(jc.ENV_BASE_URL, srv.url)
    monkeypatch.delenv(jc.ENV_API_KEY, raising=False)
    yield srv
    srv.close()


def text_config(**kw):
    base = dict(model=MODEL, mode="text_0_100", max_tokens=64, seed=0, concurrency=2,
                timeout_s=5.0, max_transport_retries=2, retry_backoff_s=0.0)
    base.update(kw)
    return jc.JudgeConfig(**base)


def logprob_config(**kw):
    base = dict(model=MODEL, mode="logprob_0_100", max_tokens=1, seed=0, concurrency=2,
                timeout_s=5.0, max_transport_retries=2, retry_backoff_s=0.0)
    base.update(kw)
    return jc.JudgeConfig(**base)


def run(config, cache_path, prompts, use_cache=True, order_seed=0):
    async def go():
        async with jc.JudgeClient(config, cache_path, sleep=lambda s: None) as client:
            reqs = [jc.JudgeRequest("k%d" % i, p) for i, p in enumerate(prompts)]
            return client, await client.judge_many(reqs, order_seed=order_seed,
                                                    use_cache=use_cache)
    return asyncio.run(go())


# --- endpoint and model checks ------------------------------------------------

def test_model_id_mismatch_aborts_before_any_request_or_write(monkeypatch, tmp_path):
    srv = FakeServer(models=("some-other-model",))
    monkeypatch.setenv(jc.ENV_BASE_URL, srv.url)
    cache = tmp_path / "cache.jsonl"
    try:
        with pytest.raises(jc.ModelMismatch):
            run(text_config(), cache, ["p"])
        assert srv.posts() == []
        assert not cache.exists()
    finally:
        srv.close()


def test_unreachable_endpoint_raises_and_writes_nothing(monkeypatch, tmp_path):
    srv = FakeServer()
    url = srv.url
    srv.close()                       # nothing listens there any more
    monkeypatch.setenv(jc.ENV_BASE_URL, url)
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(jc.EndpointUnavailable):
        run(text_config(timeout_s=1.0), cache, ["p"])
    assert not cache.exists()


def test_no_fallback_provider_when_endpoint_unset(monkeypatch, tmp_path):
    decoy = FakeServer()
    try:
        monkeypatch.delenv(jc.ENV_BASE_URL, raising=False)
        monkeypatch.setenv("LLM_BASE_URL", decoy.url)
        monkeypatch.setenv("LLM_API_BASE", decoy.url)
        monkeypatch.setenv("LLM_API_KEY", "sk-decoy")
        with pytest.raises(jc.ConfigError):
            run(text_config(), tmp_path / "c.jsonl", ["p"])
        assert decoy.requests == []
    finally:
        decoy.close()


def test_no_fallback_when_endpoint_fails_mid_run(server, monkeypatch, tmp_path):
    decoy = FakeServer()
    try:
        monkeypatch.setenv("LLM_BASE_URL", decoy.url)
        server.default = ("status", 503)
        _, results = run(text_config(), tmp_path / "c.jsonl", ["p"])
        assert results[0].status == "transport_error"
        assert decoy.requests == []
    finally:
        decoy.close()


def test_completion_from_a_different_model_aborts(server, tmp_path):
    server.script = [("json", completion_text("<score>5</score>", model="impostor"))]
    with pytest.raises(jc.ModelMismatch):
        run(text_config(), tmp_path / "c.jsonl", ["p"])


def test_provenance_records_models_and_props_without_url(server, tmp_path):
    server.default = ("json", completion_text("<score>5</score>"))
    client, _ = run(text_config(), tmp_path / "c.jsonl", ["p"])
    prov = client.provenance()
    assert prov["models_response"]["data"][0]["id"] == MODEL
    assert prov["server_props"]["total_slots"] == 2
    assert server.url not in json.dumps(prov)


def test_props_missing_is_recorded_as_unavailable(monkeypatch, tmp_path):
    srv = FakeServer(props=None)
    monkeypatch.setenv(jc.ENV_BASE_URL, srv.url)
    try:
        srv.default = ("json", completion_text("<score>5</score>"))
        client, _ = run(text_config(), tmp_path / "c.jsonl", ["p"])
        assert "unavailable" in client.server_props
    finally:
        srv.close()


# --- request shape --------------------------------------------------------------

def test_payload_is_deterministic_and_explicit(server, tmp_path):
    server.default = ("json", completion_logprobs(twenty([("50", 1.0)])))
    run(logprob_config(), tmp_path / "c.jsonl", ["hello"])
    body = server.posts()[0][2]
    assert body["temperature"] == 0.0 and body["seed"] == 0 and body["max_tokens"] == 1
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["model"] == MODEL
    assert body["cache_prompt"] is False


def test_template_date_string_is_sent_and_keyed(server, tmp_path):
    cfg = text_config(extra_payload={"chat_template_kwargs": {"date_string": "23 Oct 2026"}})
    server.default = ("json", completion_text("<score>5</score>"))
    run(cfg, tmp_path / "c.jsonl", ["p"])
    assert server.posts()[0][2]["chat_template_kwargs"] == {"date_string": "23 Oct 2026"}
    other = text_config(extra_payload={"chat_template_kwargs": {"date_string": "24 Oct 2026"}})
    assert jc.payload_sha256(cfg.payload("p")) != jc.payload_sha256(other.payload("p"))


@pytest.mark.parametrize("extra", [{"cache_prompt": True}, {"cache_prompt": False}])
def test_cache_prompt_is_owned_by_the_config(extra):
    with pytest.raises(jc.ConfigError):
        text_config(extra_payload=extra)


def test_extra_payload_is_sent_and_keyed(server, tmp_path):
    cfg = text_config(extra_payload={"chat_template_kwargs": {"reasoning_effort": "low"}})
    server.default = ("json", completion_text("<score>5</score>"))
    _, results = run(cfg, tmp_path / "c.jsonl", ["p"])
    assert server.posts()[0][2]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert results[0].payload_sha256 == jc.payload_sha256(cfg.payload("p"))
    assert jc.payload_sha256(cfg.payload("p")) != jc.payload_sha256(text_config().payload("p"))


@pytest.mark.parametrize("kw", [dict(temperature=0.5), dict(max_tokens=0),
                                dict(mode="logprob_0_100", max_tokens=5),
                                dict(mode="logprob_0_100", max_tokens=1, top_logprobs=5),
                                dict(mode="nonsense"),
                                dict(extra_payload={"temperature": 1.0}),
                                dict(extra_payload={"seed": 3})])
def test_config_refuses_invalid_settings(kw):
    with pytest.raises(jc.ConfigError):
        text_config(**kw)


def test_render_substitutes_only_question_and_answer():
    t = "Q: {question}\nA: {answer}"
    assert jc.render(t, "q{x}", "a {y}") == "Q: q{x}\nA: a {y}"
    with pytest.raises(jc.ConfigError):
        jc.render("Q: {question} A: {answer} arm: {arm}", "q", "a")
    with pytest.raises(jc.ConfigError):
        jc.render("Q: {question}", "q", "a")


def test_upstream_templates_render():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    trait = json.loads((repo / "data_generation/trait_data_eval/altruistic.json").read_text())
    out = jc.render(trait["eval_prompt"], "QQ", "AA")
    assert "QQ" in out and "AA" in out


# --- scoring: logprob_0_100 -------------------------------------------------------

def test_logprob_expected_value(server, tmp_path):
    server.default = ("json", completion_logprobs(twenty([("80", 0.5), ("90", 0.3),
                                                          ("70", 0.2)])))
    _, (r,) = run(logprob_config(), tmp_path / "c.jsonl", ["p"])
    assert r.status == "ok"
    assert r.score == pytest.approx((80 * 0.5 + 90 * 0.3 + 70 * 0.2) / 1.0, abs=1e-6)
    assert r.detail["number_mass"] == pytest.approx(1.0, abs=1e-6)


def test_logprob_renormalises_over_number_mass_and_ignores_non_numbers(server, tmp_path):
    server.default = ("json", completion_logprobs(twenty([("40", 0.3), ("Sure", 0.6),
                                                          ("101", 0.05), ("60", 0.05)])))
    _, (r,) = run(logprob_config(), tmp_path / "c.jsonl", ["p"])
    assert r.status == "ok"
    assert r.score == pytest.approx((40 * 0.3 + 60 * 0.05) / 0.35, abs=1e-6)


def test_logprob_low_mass(server, tmp_path):
    server.default = ("json", completion_logprobs(twenty([("I", 0.8), ("50", 0.2)])))
    _, (r,) = run(logprob_config(), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("low_mass", None)


def test_logprob_refusal(server, tmp_path):
    server.default = ("json", completion_logprobs(twenty([("REF", 0.9), ("50", 0.05)])))
    _, (r,) = run(logprob_config(), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("refusal", None)


def test_logprob_missing_and_short(server, tmp_path):
    server.script = [("json", completion_text("50", finish="length")),
                     ("json", completion_logprobs([("50", 0.9), ("60", 0.1)]))]
    _, results = run(logprob_config(concurrency=1), tmp_path / "c.jsonl", ["a", "b"],
                     order_seed=1)
    assert sorted(r.status for r in results) == ["few_logprobs", "no_logprobs"]


@pytest.mark.parametrize("token,counted", [("42", True), (" 42", True), ("100", True),
                                           ("101", False), ("+5", False), ("4_2", False),
                                           ("٤٢", False), ("042", True), ("", False)])
def test_integer_token_rule(token, counted):
    status, score, detail = jc.score_logprobs([{"token": token, "logprob": 0.0}])
    assert (status == "ok") is counted


# --- scoring: text_0_100 ------------------------------------------------------------

@pytest.mark.parametrize("content,status,score", [
    ("Reasoning... <score>73</score>", "ok", 73.0),
    ("<score>0</score>", "ok", 0.0),
    ("<score>100</score>", "ok", 100.0),
    ("<score>101</score>", "unparseable", None),
    ("<score>07</score>", "unparseable", None),
    ("<score>5</score> or <score>6</score>", "unparseable", None),
    ("<score>5</score> <score>5</score>", "unparseable", None),
    ("<score> 5 </score>", "unparseable", None),
    ("73", "unparseable", None),
    ("", "unparseable", None),
    ("REFUSAL", "refusal", None),
])
def test_text_parse(content, status, score):
    got_status, got_score, _ = jc.score_text(content)
    assert (got_status, got_score) == (status, score)


def test_text_truncated(server, tmp_path):
    server.default = ("json", completion_text("thinking <score>5", finish="length"))
    _, (r,) = run(text_config(), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("truncated", None)


def test_harmony_final_channel_only():
    raw = ("<|channel|>analysis<|message|>maybe <score>10</score><|end|>"
           "<|start|>assistant<|channel|>final<|message|><score>64</score><|return|>")
    assert jc.score_text(raw)[:2] == ("ok", 64.0)
    no_final = "<|channel|>analysis<|message|>hmm <score>10</score><|end|>"
    assert jc.score_text(no_final)[0] == "unparseable"


def test_reasoning_field_is_not_parsed(server, tmp_path):
    body = completion_text("<score>64</score>")
    body["choices"][0]["message"]["reasoning_content"] = "draft <score>10</score>"
    server.default = ("json", body)
    _, (r,) = run(text_config(), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("ok", 64.0)


# --- cache ------------------------------------------------------------------------

def test_cache_hit_sends_nothing(server, tmp_path):
    server.default = ("json", completion_text("<score>5</score>"))
    cache = tmp_path / "c.jsonl"
    run(text_config(), cache, ["p"])
    assert len(server.posts()) == 1
    _, (r,) = run(text_config(), cache, ["p"])
    assert len(server.posts()) == 1
    assert r.from_cache and (r.status, r.score) == ("ok", 5.0)


def test_cache_is_append_only_and_keyed_on_full_payload(server, tmp_path):
    server.default = ("json", completion_text("<score>5</score>"))
    cache = tmp_path / "c.jsonl"
    run(text_config(), cache, ["p", "q"])
    first = cache.read_text()
    run(text_config(seed=1), cache, ["p"])            # different payload -> new key
    second = cache.read_text()
    assert second.startswith(first)
    assert len(second.splitlines()) == 3
    for line in second.splitlines():
        entry = json.loads(line)
        assert entry["payload_sha256"] == jc.payload_sha256(entry["payload"])


def test_unparseable_answers_are_cached_too(server, tmp_path):
    server.default = ("json", completion_text("no tag"))
    cache = tmp_path / "c.jsonl"
    run(text_config(), cache, ["p"])
    _, (r,) = run(text_config(), cache, ["p"])
    assert r.from_cache and r.status == "unparseable"
    assert len(server.posts()) == 1


def test_resend_bypasses_and_does_not_write_cache(server, tmp_path):
    server.default = ("json", completion_text("<score>5</score>"))
    cache = tmp_path / "c.jsonl"
    run(text_config(), cache, ["p"])
    before = cache.read_text()
    _, (r,) = run(text_config(), cache, ["p"], use_cache=False)
    assert not r.from_cache and len(server.posts()) == 2
    assert cache.read_text() == before


def test_corrupt_cache_refuses(tmp_path):
    cache = tmp_path / "c.jsonl"
    cache.write_text('{"payload_sha256": "abc", "payload": {"x": 1}}\n')
    with pytest.raises(jc.CacheCorrupt):
        jc.Cache(cache)
    cache.write_text('{"payload_sha256": "abc"\n')
    with pytest.raises(jc.CacheCorrupt):
        jc.Cache(cache)


# --- retries --------------------------------------------------------------------

def test_retries_on_5xx_then_succeeds(server, tmp_path):
    server.script = [("status", 503), ("status", 502),
                     ("json", completion_text("<score>9</score>"))]
    _, (r,) = run(text_config(), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("ok", 9.0)
    assert [a["error"] is None for a in r.attempts] == [False, False, True]
    assert len(server.posts()) == 3


def test_retries_are_capped_and_not_cached(server, tmp_path):
    server.default = ("status", 500)
    cache = tmp_path / "c.jsonl"
    _, (r,) = run(text_config(max_transport_retries=2), cache, ["p"])
    assert r.status == "transport_error" and r.score is None
    assert len(r.attempts) == 3 and len(server.posts()) == 3
    assert not cache.exists() or cache.read_text() == ""


def test_dropped_connection_is_transport(server, tmp_path):
    server.script = [("drop",), ("json", completion_text("<score>3</score>"))]
    _, (r,) = run(text_config(), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("ok", 3.0) and len(r.attempts) == 2


def test_timeout_is_transport(server, tmp_path):
    server.script = [("sleep", 1.5)]
    server.default = ("json", completion_text("<score>4</score>"))
    _, (r,) = run(text_config(timeout_s=0.5), tmp_path / "c.jsonl", ["p"])
    assert (r.status, r.score) == ("ok", 4.0) and len(r.attempts) == 2


@pytest.mark.parametrize("code", [400, 401, 404, 422])
def test_4xx_is_not_retried(server, tmp_path, code):
    server.default = ("status", code)
    _, (r,) = run(text_config(), tmp_path / "c.jsonl", ["p"])
    assert r.status == "http_error" and r.detail["http_status"] == code
    assert len(server.posts()) == 1


def test_unparseable_and_low_mass_are_not_retried(server, tmp_path):
    server.script = [("json", completion_text("no score here")),
                     ("json", completion_logprobs(twenty([("I", 1.0)])))]
    run(text_config(), tmp_path / "a.jsonl", ["p"])
    run(logprob_config(), tmp_path / "b.jsonl", ["p"])
    assert len(server.posts()) == 2


# --- order and blindness ----------------------------------------------------------

def test_judge_many_sends_shuffled_and_returns_in_order(server, tmp_path):
    server.default = ("json", completion_text("<score>1</score>"))
    prompts = ["prompt-%02d" % i for i in range(20)]
    _, results = run(text_config(concurrency=1), tmp_path / "c.jsonl", prompts, order_seed=7)
    sent = [b["messages"][0]["content"] for _, _, b in server.posts()]
    assert sorted(sent) == prompts and sent != prompts
    assert [r.key for r in results] == ["k%d" % i for i in range(20)]


def test_duplicate_keys_refused(server, tmp_path):
    async def go():
        async with jc.JudgeClient(text_config(), tmp_path / "c.jsonl") as client:
            await client.judge_many([jc.JudgeRequest("a", "x"), jc.JudgeRequest("a", "y")], 0)
    with pytest.raises(jc.ConfigError):
        asyncio.run(go())


def test_request_carries_nothing_but_the_prompt(server, tmp_path):
    server.default = ("json", completion_text("<score>1</score>"))
    run(text_config(), tmp_path / "c.jsonl", ["Q and A only"])
    body = server.posts()[0][2]
    assert set(body) == {"model", "messages", "temperature", "seed", "max_tokens", "stream",
                       "cache_prompt"}
    assert "k0" not in json.dumps(body)


def test_judge_before_open_refused(tmp_path):
    client = jc.JudgeClient(text_config(), tmp_path / "c.jsonl")
    with pytest.raises(jc.JudgeError):
        asyncio.run(client.judge(jc.JudgeRequest("a", "x")))
