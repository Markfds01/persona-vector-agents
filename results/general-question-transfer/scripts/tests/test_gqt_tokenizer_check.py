"""The tokenizer check's own logic, without downloading anything.

The real check reads GGUF headers over HTTP and runs llama-tokenize; here a GGUF
header is built in memory and the tokenizer is a fake, so what is tested is the
parser, the vocab-only writer and the single-token rule.
"""

import struct
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "judging"))

import tokenizer_check as tc  # noqa: E402


def _s(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key, value):
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_u32(key, value):
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str_array(key, values):
    return _s(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + \
        struct.pack("<Q", len(values)) + b"".join(_s(v) for v in values)


def fake_gguf(tensor_count=3, extra=b""):
    kvs = [_kv_string("general.name", "fake"),
           _kv_u32("general.alignment", 32),
           _kv_u32("split.count", 2),
           _kv_str_array("tokenizer.ggml.tokens", ["<s>", "4", "2", "42"]),
           _kv_string("tokenizer.ggml.pre", "llama-bpe")]
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count) + \
        struct.pack("<Q", len(kvs))
    return head + b"".join(kvs) + extra


def test_reads_kv_and_reports_end():
    data = fake_gguf(extra=b"TENSORINFO...")
    version, tensors, kv, end = tc.read_gguf_kv(data)
    assert (version, tensors) == (3, 3)
    assert kv["tokenizer.ggml.tokens"] == ["<s>", "4", "2", "42"]
    assert kv["tokenizer.ggml.pre"] == "llama-bpe"
    assert data[end:] == b"TENSORINFO..."


def test_truncated_header_asks_for_more():
    data = fake_gguf()
    with pytest.raises(tc.Truncated):
        tc.read_gguf_kv(data[:len(data) - 5])


def test_not_gguf():
    with pytest.raises(ValueError):
        tc.read_gguf_kv(b"NOPE" + b"\0" * 40)


def test_vocab_only_has_zero_tensors_same_vocab_no_split_keys():
    data = fake_gguf(extra=b"\xff" * 100)
    _, _, _, end = tc.read_gguf_kv(data)
    out = tc.vocab_only_gguf(data, end)
    version, tensors, kv, kv_end = tc.read_gguf_kv(out)
    assert tensors == 0
    assert kv["tokenizer.ggml.tokens"] == ["<s>", "4", "2", "42"]
    assert not any(k.startswith("split.") for k in kv)
    assert len(out) % 32 == 0
    assert set(out[kv_end:]) <= {0}


class FakeTokenizer:
    """Single token for 0..99, two for 100, and a space is always its own token."""

    def encode(self, text):
        ids = []
        for part in text.split("|"):
            if not part:
                continue
            if part.startswith(" "):
                ids.append(-1)
                part = part[1:]
            if part.isdigit() and int(part) < 100:
                ids.append(int(part))
            elif part == "100":
                ids += [10, 0]
            else:
                ids.append(9999)
        return ids


def test_single_token_report_variants():
    tok = FakeTokenizer()
    prefix = "ASSISTANT|"
    report = tc.single_token_report(tok.encode, tok.encode(prefix), prefix)
    assert report["bare"]["status"] == "fail"
    assert report["bare"]["n_single_token"] == 100
    assert list(report["bare"]["failures"]) == ["100"]
    assert report["leading_space"]["n_single_token"] == 0
    assert report["first_assistant"]["n_single_token"] == 100


def test_first_assistant_not_run_without_template():
    report = tc.single_token_report(FakeTokenizer().encode)
    assert report["first_assistant"]["status"] == "not run"


def test_prefix_merge_is_a_failure():
    """If the score merges with the end of the prefix, the prefix ids change."""
    def encode(text):
        return [ord(c) for c in text.replace("X7", "Z")]
    report = tc.single_token_report(encode, encode("X"), "X")
    assert "7" in report["first_assistant"]["failures"]
    assert "8" not in report["first_assistant"]["failures"]


def test_judges_pin_full_identifiers():
    for judge in tc.JUDGES.values():
        assert len(judge["gguf_revision"]) == 40 and len(judge["gguf_sha256"]) == 64
        assert len(judge["hf_revision"]) == 40
