"""What is the layer-0 direction actually made of? CPU only, no GPU, no model forward.

Layer 0 of `response_avg` is the mean INPUT EMBEDDING of the response tokens, so a
layer-0 difference vector is literally a difference of average token embeddings.
That makes the layer-0 AUC interpretable in a way no other layer is: if a pole
pair is separable at layer 0, the separation is carried by which tokens the answer
contains, and the direction can be read back out in token space.

This reads the embedding matrix straight out of the safetensors shard - the
weights are never put on a GPU and the model is never run - and reports the tokens
whose embeddings have the largest positive and negative cosine with the direction.
A layer-0 direction whose extremes are digits and currency tokens IS the answer-
wording confound, named. One whose extremes are not is evidence the confound is
absent, though a null result in token space is weaker than a positive one.
"""

import argparse
import json
from pathlib import Path

import torch

SNAPSHOT = ("/home/marco/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
            "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
EMBED_KEY = "model.embed_tokens.weight"


def load_embeddings(snapshot):
    from safetensors.torch import load_file
    index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][EMBED_KEY]
    return load_file(str(Path(snapshot) / shard))[EMBED_KEY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", nargs="+", required=True,
                    help="(29, 3584) .pt files to decode at layer 0")
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28")

    embeddings = load_embeddings(args.snapshot).float()
    norms = embeddings.norm(dim=1).clamp_min(1e-12)
    print("embedding matrix %s" % (tuple(embeddings.shape),), flush=True)

    report = {}
    for path in args.vectors:
        vector = torch.load(path, map_location="cpu").float()
        if vector.shape[1] != embeddings.shape[1]:
            raise SystemExit("%s: width %d != embedding width %d"
                             % (path, vector.shape[1], embeddings.shape[1]))
        direction = vector[0]
        cos = (embeddings @ direction) / (norms * direction.norm().clamp_min(1e-12))
        top = torch.topk(cos, args.top)
        bottom = torch.topk(-cos, args.top)
        name = Path(path).name
        entry = {
            "altruistic_pole_tokens": [
                {"token": tokenizer.decode([int(i)]), "cos": float(c)}
                for c, i in zip(top.values, top.indices)],
            "self_interested_pole_tokens": [
                {"token": tokenizer.decode([int(i)]), "cos": float(-c)}
                for c, i in zip(bottom.values, bottom.indices)],
        }
        report[name] = entry
        print("\n=== %s (layer 0) ===" % name)
        print("  toward the ALTRUISTIC pole: %s"
              % ", ".join("%r%+.3f" % (t["token"], t["cos"])
                          for t in entry["altruistic_pole_tokens"]))
        print("  toward the SELF-INTERESTED pole: %s"
              % ", ".join("%r%+.3f" % (t["token"], t["cos"])
                          for t in entry["self_interested_pole_tokens"]))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
