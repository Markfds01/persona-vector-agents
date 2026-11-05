"""Is every score "0".."100" exactly one token for each candidate judge?

Upstream's `logprob_0_100` reads a score off ONE token's top logprobs. That is a
valid reading only if every integer 0-100 is a single token for the judge's
tokenizer, in the position the judge actually emits it. This script tests that
instead of assuming it, for three variants of each string:

* `bare`            - the string on its own ("42"),
* `leading_space`   - with a leading space (" 42"),
* `first_assistant` - as the judge's first assistant token: the chat prefix the
  judge's own template renders for one user turn, plus the string. The check is
  ids(prefix + s) == ids(prefix) + [one id], which also catches a merge across the
  prefix boundary.

Two tokenizers are tested per judge, because they are two different claims:

1. **the GGUF-embedded vocabulary**, tokenized by `llama-tokenize` from the SAME
   llama.cpp build the judge server runs. Only the GGUF header is downloaded (an
   HTTP range read of the exact file at a pinned revision, whose full-file sha256
   is compared against Hugging Face's LFS metadata), and a vocab-only GGUF - the
   header's key/value section verbatim, zero tensors - is written for the
   tokenizer to load.
2. **the Hugging Face tokenizer** of the base model, when it is public. A gated
   repository without access is recorded as "not run (gated)", never guessed.
   When both exist, the GGUF token list and merges are compared with the HF
   tokenizer's, so a reader can see whether the two claims are one.

What this script does NOT check, and records as not run: whether the served
endpoint returns `top_logprobs` >= 20. That needs the live server.

Stdlib only for the GGUF parsing; `transformers` (and its jinja2) for the HF side
and for rendering the chat templates.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

SCORE_STRINGS = tuple(str(i) for i in range(101))
VARIANTS = ("bare", "leading_space", "first_assistant")

#: One user turn, used only to render the chat prefix. Its content is irrelevant
#: to the check (the suffix is what is tested) but is recorded.
PROBE_MESSAGES = ({"role": "user", "content": "Rate this answer."},)

#: The judges the operator fixed on 2026-10-23 (decision 2), with the exact files
#: the judge server loads. `sha256` is the full-file digest the orchestrator
#: verified on the server; this script re-checks it against Hugging Face's LFS
#: metadata at the pinned revision before trusting the header it reads.
JUDGES = {
    "llama-3.3-70b-instruct-q8_0": {
        "gguf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "gguf_revision": "b6c5c9f176f3279204034e1d16d393105e95cb88",
        "gguf_file": "Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
        "gguf_sha256": "0c4ec5083aec2efa104269e8b63e2d9f3322e2fa50511184314093a7b4697878",
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct",
        "hf_revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "reasoning": False,
    },
    "gpt-oss-20b-mxfp4": {
        "gguf_repo": "ggml-org/gpt-oss-20b-GGUF",
        "gguf_revision": "15dcfc7e7c608c6161507211ed43c5b590801b4b",
        "gguf_file": "gpt-oss-20b-MXFP4.gguf",
        "gguf_sha256": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
        "hf_repo": "openai/gpt-oss-20b",
        "hf_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "reasoning": True,
    },
}

#: The other two candidates the task named. Recorded for completeness; they are
#: not the judges the operator chose, and their checks are what was possible.
OTHER_CANDIDATES = {
    "gpt-oss-120b": {"hf_repo": "openai/gpt-oss-120b",
                     "hf_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
                     "reasoning": True},
    "llama-4-scout-17b-16e-instruct": {
        "hf_repo": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "hf_revision": "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
        "reasoning": False},
}

# --- GGUF header ------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
#: GGUF value types -> struct format of a scalar (None: string / array).
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
            7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class Truncated(Exception):
    """The bytes end before the key/value section does; read more."""


class _Cursor:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise Truncated(self.pos + n)
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q")).decode("utf-8")

    def value(self, vtype):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            item_type = self.unpack("<I")
            count = self.unpack("<Q")
            return [self.value(item_type) for _ in range(count)]
        raise ValueError("unknown GGUF value type %d at offset %d" % (vtype, self.pos))


def read_gguf_kv(data):
    """Parse a GGUF header's key/value section from its leading bytes.

    Returns (version, tensor_count, kv dict, end offset of the kv section).
    Raises `Truncated` when `data` ends inside it.
    """
    cur = _Cursor(data)
    if cur.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = cur.unpack("<I")
    if version not in (2, 3):
        raise ValueError("unsupported GGUF version %d" % version)
    tensor_count = cur.unpack("<Q")
    kv_count = cur.unpack("<Q")
    kv = {}
    for _ in range(kv_count):
        key = cur.string()
        kv[key] = cur.value(cur.unpack("<I"))
    return version, tensor_count, kv, cur.pos


def vocab_only_gguf(data, kv_end):
    """The same header with zero tensors: the key/value bytes verbatim.

    Split-file keys are dropped (a vocab-only file is not part of a split), and the
    file is padded to the declared alignment so the empty data section starts on it.
    """
    version, _, kv, end = read_gguf_kv(data[:kv_end])
    alignment = int(kv.get("general.alignment", 32))
    # Re-walk the kv section to copy each pair's raw bytes, minus split keys.
    cur = _Cursor(data)
    cur.take(4 + 4 + 8 + 8)
    kept, n_kept = [], 0
    for _ in range(len(kv)):
        start = cur.pos
        key = cur.string()
        cur.value(cur.unpack("<I"))
        if not key.startswith("split."):
            kept.append(data[start:cur.pos])
            n_kept += 1
    body = b"".join(kept)
    out = GGUF_MAGIC + struct.pack("<I", version) + struct.pack("<Q", 0) \
        + struct.pack("<Q", n_kept) + body
    out += b"\x00" * ((-len(out)) % alignment)
    return out


def fetch_head(url, dest, start_bytes=8 << 20, max_bytes=512 << 20, token=None):
    """Range-read the start of `url` until its kv section parses; cache to `dest`."""
    nbytes = start_bytes
    while True:
        if dest.exists() and dest.stat().st_size >= nbytes:
            data = dest.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-%d" % (nbytes - 1)})
            if token:
                request.add_header("Authorization", "Bearer " + token)
            with urllib.request.urlopen(request, timeout=300) as response:
                data = response.read()
            dest.write_bytes(data)
        try:
            return data, read_gguf_kv(data)
        except Truncated:
            if nbytes >= max_bytes:
                raise
            nbytes *= 2


def hf_lfs_sha256(repo, revision, filename):
    """Hugging Face's recorded LFS sha256 of one file at one revision."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            return sibling.lfs.sha256 if sibling.lfs else None, sibling.size
    raise KeyError("%s@%s has no %s" % (repo, revision, filename))


# --- the single-token check ---------------------------------------------------

def single_token_report(encode, prefix_ids=None, prefix_text=None):
    """Per variant, which of "0".."100" are NOT exactly one token.

    `encode(text)` returns ids with no BOS added; special-token text is parsed.
    `prefix_text` / `prefix_ids` are the rendered chat prefix and its ids, for
    `first_assistant`; without them that variant is reported as not run.
    """
    report = {}
    for variant in VARIANTS:
        failures = {}
        if variant == "first_assistant" and prefix_text is None:
            report[variant] = {"status": "not run", "reason": "no chat prefix"}
            continue
        for s in SCORE_STRINGS:
            if variant == "bare":
                ids = encode(s)
                ok = len(ids) == 1
            elif variant == "leading_space":
                ids = encode(" " + s)
                ok = len(ids) == 1
            else:
                full = encode(prefix_text + s)
                ok = (len(full) == len(prefix_ids) + 1
                      and list(full[:len(prefix_ids)]) == list(prefix_ids))
                ids = full[len(prefix_ids):] if list(full[:len(prefix_ids)]) == list(prefix_ids) \
                    else ["<prefix not preserved>"]
            if not ok:
                failures[s] = list(ids)
        report[variant] = {
            "status": "pass" if not failures else "fail",
            "n_tested": len(SCORE_STRINGS),
            "n_single_token": len(SCORE_STRINGS) - len(failures),
            "failures": failures if len(failures) <= 12 else
            dict(list(failures.items())[:12]) | {"...": "%d more" % (len(failures) - 12)},
        }
    return report


class LlamaTokenize:
    """`llama-tokenize` from the judge server's own llama.cpp build."""

    def __init__(self, binary, model):
        self.binary = Path(binary)
        self.model = Path(model)
        self.env = dict(os.environ, LD_LIBRARY_PATH=str(self.binary.parent))

    def version(self):
        out = subprocess.run([str(self.binary), "--version"], env=self.env,
                             capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()

    def encode(self, text):
        out = subprocess.run(
            [str(self.binary), "-m", str(self.model), "-p", text, "--ids", "--no-bos",
             "--log-disable"],
            env=self.env, capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError("llama-tokenize failed: %s" % out.stderr[-2000:])
        line = out.stdout.strip().splitlines()[-1]
        return json.loads(line)


def render_prefix(chat_template, bos_token, reasoning):
    """The judge's chat prefix for one user turn, from its own template.

    For a reasoning model the assistant's first token is a channel marker, never a
    digit, so the scored position is after the FINAL channel header; that is what
    `text_0_100` parses and what a logprob read would have to target.
    """
    from transformers.utils.chat_template_utils import _compile_jinja_template
    template = _compile_jinja_template(chat_template)
    text = template.render(messages=list(PROBE_MESSAGES), add_generation_prompt=True,
                           bos_token=bos_token, eos_token="", tools=None,
                           date_string="26 Jul 2024")
    if reasoning:
        text += "<|channel|>final<|message|>"
    return text


def gguf_side(name, judge, work, llama_tokenize_bin):
    url = "https://huggingface.co/%s/resolve/%s/%s" % (
        judge["gguf_repo"], judge["gguf_revision"], judge["gguf_file"])
    record = {"repo": judge["gguf_repo"], "revision": judge["gguf_revision"],
              "file": judge["gguf_file"], "expected_sha256": judge["gguf_sha256"]}
    lfs_sha, size = hf_lfs_sha256(judge["gguf_repo"], judge["gguf_revision"], judge["gguf_file"])
    record["hf_lfs_sha256"] = lfs_sha
    record["file_size"] = size
    record["sha256_matches_expected"] = lfs_sha == judge["gguf_sha256"]
    if not record["sha256_matches_expected"]:
        record["status"] = "not run: the pinned revision's file is not the expected one"
        return record, None
    head_path = work / ("%s.head" % name)
    data, (version, tensor_count, kv, kv_end) = fetch_head(url, head_path)
    record["header_bytes_read"] = len(data)
    record["gguf_version"] = version
    record["tensor_count"] = tensor_count
    tokens = kv.get("tokenizer.ggml.tokens", [])
    merges = kv.get("tokenizer.ggml.merges", [])
    chat_template = kv.get("tokenizer.chat_template")
    record["tokenizer_model"] = kv.get("tokenizer.ggml.model")
    record["tokenizer_pre"] = kv.get("tokenizer.ggml.pre")
    record["n_tokens"] = len(tokens)
    record["n_merges"] = len(merges)
    record["general_name"] = kv.get("general.name")
    record["file_type"] = kv.get("general.file_type")
    record["chat_template_sha256"] = (hashlib.sha256(chat_template.encode()).hexdigest()
                                      if chat_template else None)
    bos_id = kv.get("tokenizer.ggml.bos_token_id")
    bos = tokens[bos_id] if bos_id is not None and bos_id < len(tokens) else ""
    vocab_path = work / ("%s.vocab-only.gguf" % name)
    vocab_path.write_bytes(vocab_only_gguf(data, kv_end))
    record["vocab_only_gguf_sha256"] = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    tok = LlamaTokenize(llama_tokenize_bin, vocab_path)
    record["llama_tokenize_version"] = tok.version()
    prefix = render_prefix(chat_template, bos, judge["reasoning"]) if chat_template else None
    if prefix is not None and bos and prefix.startswith(bos):
        prefix = prefix[len(bos):]   # --no-bos: compare like with like
    record["chat_prefix"] = prefix
    prefix_ids = tok.encode(prefix) if prefix is not None else None
    record["single_token"] = single_token_report(tok.encode, prefix_ids, prefix)
    record["status"] = "run"
    return record, {"tokens": tokens, "merges": merges}


def hf_side(repo, revision, reasoning, gguf_vocab=None):
    record = {"repo": repo, "revision": revision}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    except Exception as exc:  # gated, missing, offline: recorded, never guessed
        name = type(exc).__name__
        gated = "Gated" in name or "403" in str(exc) or "401" in str(exc)
        record["status"] = "not run (gated)" if gated else "not run"
        record["reason"] = "%s: %s" % (name, str(exc).splitlines()[0][:300])
        return record
    record["status"] = "run"
    record["transformers_tokenizer_class"] = type(tokenizer).__name__

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    template = getattr(tokenizer, "chat_template", None)
    prefix = render_prefix(template, tokenizer.bos_token or "", reasoning) if template else None
    if prefix is not None and tokenizer.bos_token and prefix.startswith(tokenizer.bos_token):
        prefix = prefix[len(tokenizer.bos_token):]
    record["chat_prefix"] = prefix
    record["chat_template_sha256"] = (hashlib.sha256(template.encode()).hexdigest()
                                      if template else None)
    record["single_token"] = single_token_report(encode, encode(prefix) if prefix else None,
                                                 prefix)
    if gguf_vocab is not None:
        record["vocab_identity_with_gguf"] = vocab_identity(tokenizer, gguf_vocab)
    return record


def vocab_identity(tokenizer, gguf_vocab):
    """Do the GGUF token list and merges equal the HF tokenizer.json's?"""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    model = backend["model"]
    hf_vocab = model.get("vocab", {})
    added = {t["id"]: t["content"] for t in backend.get("added_tokens", [])}
    hf_tokens = [None] * max(len(gguf_vocab["tokens"]),
                             1 + max(list(hf_vocab.values()) + list(added) or [0]))
    for token, idx in hf_vocab.items():
        hf_tokens[idx] = token
    for idx, token in added.items():
        hf_tokens[idx] = token
    mismatched = [i for i, t in enumerate(gguf_vocab["tokens"])
                  if i >= len(hf_tokens) or hf_tokens[i] != t]
    hf_merges = [" ".join(m) if isinstance(m, list) else m for m in model.get("merges", [])]
    # A GGUF pads its token list to the embedding size with "[PAD<id>]" entries the
    # HF tokenizer has no id for; those are not a disagreement about any real token.
    padding = [i for i in mismatched
               if (i >= len(hf_tokens) or hf_tokens[i] is None)
               and gguf_vocab["tokens"][i] == "[PAD%d]" % i]
    return {
        "n_gguf_tokens": len(gguf_vocab["tokens"]),
        "n_hf_tokens": sum(t is not None for t in hf_tokens),
        "n_token_mismatches": len(mismatched),
        "n_mismatches_that_are_gguf_padding_only": len(padding),
        "every_hf_token_identical": len(mismatched) == len(padding),
        "first_mismatches": [[i, gguf_vocab["tokens"][i], hf_tokens[i] if i < len(hf_tokens) else None]
                             for i in mismatched[:10]],
        "merges_identical": hf_merges == list(gguf_vocab["merges"]),
        "n_gguf_merges": len(gguf_vocab["merges"]),
        "n_hf_merges": len(hf_merges),
    }


def summarize(record):
    def verdict(single):
        if not single:
            return "not run"
        statuses = {v: single[v]["status"] for v in VARIANTS}
        return statuses
    return {"gguf": verdict(record.get("gguf", {}).get("single_token")),
            "hf": verdict(record.get("hf", {}).get("single_token"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llama-tokenize", required=True,
                    help="llama-tokenize from the judge server's llama.cpp build")
    ap.add_argument("--work", required=True, help="scratch directory for GGUF headers")
    ap.add_argument("--out", required=True, help="JSON report path")
    args = ap.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "results/general-question-transfer/scripts/judging/tokenizer_check.py",
        "llama_tokenize_binary_sha256": hashlib.sha256(
            Path(args.llama_tokenize).read_bytes()).hexdigest(),
        "endpoint_top_logprobs_ge_20": "not run (needs the live judge server; checkpoint-1 "
                                       "item makes no endpoint call)",
        "judges": {}, "other_candidates": {},
    }
    for name, judge in JUDGES.items():
        print("judge %s" % name, file=sys.stderr)
        gguf_record, gguf_vocab = gguf_side(name, judge, work, args.llama_tokenize)
        record = {"gguf": gguf_record,
                  "hf": hf_side(judge["hf_repo"], judge["hf_revision"], judge["reasoning"],
                                gguf_vocab)}
        record["summary"] = summarize(record)
        report["judges"][name] = record
    for name, cand in OTHER_CANDIDATES.items():
        print("candidate %s" % name, file=sys.stderr)
        record = {"gguf": {"status": "not run (no GGUF pinned for this candidate)"},
                  "hf": hf_side(cand["hf_repo"], cand["hf_revision"], cand["reasoning"])}
        record["summary"] = summarize(record)
        report["other_candidates"][name] = record
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["summary"] for k, v in
                      {**report["judges"], **report["other_candidates"]}.items()}, indent=1))


if __name__ == "__main__":
    main()
