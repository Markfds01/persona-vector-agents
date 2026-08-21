"""Scalar version of the layer-0 token decode: how much of the direction IS digits.

Needs the tokenizer and the safetensors embedding shard, so it runs under the
project venv. CPU only - the embedding matrix is read off disk and never moved
to a GPU, and no forward pass happens.
"""

import json
import math
import os
from pathlib import Path

import torch

import common

SEED = 20260820
SNAPSHOT = os.environ.get(
    "DM_SNAPSHOT",
    "/home/marco/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")


def response_token_ids(tok, per_family=200):
    """Distinct token ids the model actually emitted, sampled across all six games."""
    import csv
    ids = set()
    for family in common.FAMILIES:
        path = common.ACTS / family / "rows.csv"
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        step = max(1, len(rows) // per_family)
        for row in rows[::step]:
            ids.update(tok.encode(row["continuation"]))
    return ids


def digit_share(vectors, snapshot, top_random=10):
    """Fraction of a layer-0 direction's norm lying in the span of '0'-'9'.

    Layer 0 of `response_avg` is the mean input embedding of the response, so the
    direction lives in embedding space and this is a scalar version of the token
    decode: how much of the direction IS the digit tokens. Read it against the
    empirical control bands below, never against the spherical figure
    sqrt(10/3584) = 0.053 - embeddings are not isotropic and that figure is far
    too generous. It is reported for reference only.
    """
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    emb = load_file(str(Path(snapshot) / index["weight_map"][key]))[key].float()
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28")
    digit_ids = [tok.encode(str(d))[0] for d in range(10)]
    basis, _ = torch.linalg.qr(emb[digit_ids].double().T)

    # The theoretical baseline sqrt(10/3584)=0.053 understates the real one:
    # token embeddings are not isotropic, so ANY direction has more overlap with
    # ANY ten of them than chance in a spherical sense would give. 32 random
    # ten-token subspaces give the baseline this comparison actually needs.
    g = torch.Generator().manual_seed(SEED)
    controls = [torch.linalg.qr(
        emb[torch.randperm(emb.shape[0], generator=g)[:len(digit_ids)].tolist()]
        .double().T)[0] for _ in range(32)]

    # A SIGNED read of the same thing: does the direction point at "the answer is
    # a large digit" rather than "the answer is zero"? The subspace share above is
    # unsigned, and overfishing's digit polarity is inverted, so the sign matters.
    big = emb[[tok.encode(d)[0] for d in "4567"]].double().mean(dim=0)
    zero = emb[tok.encode("0")[0]].double()
    axis = (big - zero)
    axis = axis / axis.norm()

    # The control that actually matters: ten tokens the model really emitted in
    # these responses. A random vocabulary token is mostly rare junk, so a random
    # subspace over the whole vocabulary is not the right comparison for a
    # direction built out of response text.
    response_ids = sorted(response_token_ids(tok))
    pool = torch.tensor(response_ids)
    response_controls = [torch.linalg.qr(
        emb[pool[torch.randperm(len(pool), generator=g)[:len(digit_ids)]].tolist()]
        .double().T)[0] for _ in range(32)]

    # The span share is unsigned and says nothing about WHICH digit or which
    # pole. A vector whose only digit content is '0' on the self-interested side
    # is a different object from one where '4'-'7' mark the altruistic side, and
    # the aggregate cannot tell them apart - so report the per-digit cosines and
    # the share carried by '0' alone.
    zero_basis = (emb[[tok.encode("0")[0]]].double().T
                  / emb[tok.encode("0")[0]].double().norm())

    out = {"theoretical_spherical_baseline": math.sqrt(len(digit_ids) / emb.shape[1]),
           "digit_token_ids": digit_ids,
           "n_distinct_response_tokens": len(response_ids)}
    for name, path in vectors.items():
        v = torch.load(path, map_location="cpu").double()[0]
        shares = torch.tensor([float((c.T @ v).norm() / v.norm()) for c in controls])
        resp = torch.tensor([float((c.T @ v).norm() / v.norm())
                             for c in response_controls])
        out[name] = {
            "digit_span_share": float((basis.T @ v).norm() / v.norm()),
            "random_vocab_10_span_share_p97.5": float(shares.quantile(0.975)),
            "random_response_10_span_share_mean": float(resp.mean()),
            "random_response_10_span_share_p97.5": float(resp.quantile(0.975)),
            "cos_vs_big_digit_minus_zero": float(torch.dot(axis, v) / v.norm()),
            "zero_only_span_share": float((zero_basis.T @ v).norm() / v.norm()),
            "cos_per_digit": {str(d): float(
                torch.dot(emb[digit_ids[d]].double(), v)
                / (emb[digit_ids[d]].double().norm() * v.norm()))
                for d in range(10)},
        }
    return out




def main():
    vecs = {}
    for policy in ("strict", "relaxed"):
        for scheme in common.WEIGHTINGS:
            name = "%s_%s" % (scheme, policy)
            vecs[name] = common.OUT / "vectors" / (
                "decision_pooled_%s_response_avg_diff_%s.pt" % (scheme, policy))
        for f in common.FAMILIES:
            vecs["game_%s_%s" % (f, policy)] = common.OUT / "vectors" / (
                "decision_%s_response_avg_diff_cellbalanced_%s.pt" % (f, policy))
    vecs["archived_dictator"] = common.DICTATOR_VECTOR
    vecs["their_altruism"] = common.ALTRUISM_VECTOR
    out = digit_share(vecs, SNAPSHOT)
    (common.OUT / "digit_share.json").write_text(json.dumps(out, indent=2))
    for k, v in sorted(out.items()):
        if isinstance(v, dict):
            print("%-38s digits %.3f  resp10 p97.5 %.3f  vocab10 p97.5 %.3f  "
                  "cos(big-'0') %+.3f"
                  % (k, v["digit_span_share"],
                     v["random_response_10_span_share_p97.5"],
                     v["random_vocab_10_span_share_p97.5"],
                     v["cos_vs_big_digit_minus_zero"]))


if __name__ == "__main__":
    main()
