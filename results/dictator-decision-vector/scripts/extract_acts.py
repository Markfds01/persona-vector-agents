"""Teacher-forced activation capture, matching `generate_vec.py:28-38` exactly.

For each generated row: re-render the prompt it was generated from, verify the
rendered bytes against the fingerprint the row carries, run ONE forward pass over
prompt+response with `output_hidden_states=True`, and pool three ways per layer:

  prompt_avg    mean over token positions [0, prompt_len)
  prompt_last   the single position prompt_len-1
  response_avg  mean over token positions [prompt_len, end)

Batch size is 1 and no padding is involved, which is what upstream does and also
the only way the position slices mean what they say.

Two things upstream does not check are checked here and reported rather than
silently absorbed:
  * prefix stability - the prompt's own tokens must be a prefix of prompt+response
    tokens. BPE can merge across the seam; a row where it does has the wrong
    prompt_len and is dropped, counted, and named.
  * an empty response, which would make response_avg a NaN.

dtype and attn_implementation are pinned and written into the meta file: a
persona vector is the output of a forward pass, and this model's sdpa and eager
kernels diverge by ~3.4 logits at bf16.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import elicit, games
from scratch import decision_grid

SHARD_ROWS = 250


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, help="CSV written by gen_grid.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.rows, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    decision_grid.register()

    # only rows the scorer resolved carry a label; the rest are counted, never guessed
    todo = [(i, r) for i, r in enumerate(rows) if r["value"] != ""]
    print("rows=%d scored=%d unscored=%d" % (len(rows), len(todo), len(rows) - len(todo)),
          flush=True)

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype,
        attn_implementation=args.attn, device_map={"": 0})
    model.eval()
    resolved_attn = model.config._attn_implementation
    if resolved_attn != args.attn:
        raise RuntimeError("asked for attn %r, model loaded %r" % (args.attn, resolved_attn))
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    meta = {
        "rows_csv": str(Path(args.rows).resolve()),
        "model_id": args.model,
        "model_revision": model.config._commit_hash,
        "dtype": str(model.dtype),
        "attn_implementation": resolved_attn,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
        "n_hidden_states": n_layers + 1,
        "chat_template_sha256": elicit.chat_template_fingerprint(tokenizer),
        "batch_size": 1,
        "padding": "none",
    }

    prompt_cache = {}

    def prompt_for(game_id, mode):
        key = (game_id, mode)
        if key not in prompt_cache:
            prompt_cache[key] = elicit.render(games.by_id(game_id), mode, tokenizer)
        return prompt_cache[key]

    shard = []
    shard_index = 0
    dropped = []
    started = time.time()

    def flush():
        nonlocal shard, shard_index
        if not shard:
            return
        payload = {
            "row_index": torch.tensor([s[0] for s in shard], dtype=torch.long),
            "prompt_len": torch.tensor([s[1] for s in shard], dtype=torch.long),
            "total_len": torch.tensor([s[2] for s in shard], dtype=torch.long),
            "prompt_avg": torch.stack([s[3] for s in shard]),
            "prompt_last": torch.stack([s[4] for s in shard]),
            "response_avg": torch.stack([s[5] for s in shard]),
        }
        torch.save(payload, out_dir / ("shard_%04d.pt" % shard_index))
        shard_index += 1
        shard = []

    for done, (row_index, row) in enumerate(todo):
        rendered = prompt_for(row["game_id"], row["mode"])
        if rendered.sha256 != row["prompt_sha256"]:
            raise RuntimeError("row %d: re-rendered prompt %s != recorded %s"
                               % (row_index, rendered.sha256, row["prompt_sha256"]))
        prompt = rendered.text
        answer = row["answer"]
        text = prompt + answer

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"][0].tolist()
        if ids[:prompt_len] != prompt_ids:
            dropped.append({"row_index": row_index, "reason": "prompt tokens are not a "
                            "prefix of prompt+answer tokens"})
            continue
        if len(ids) <= prompt_len:
            dropped.append({"row_index": row_index, "reason": "empty response"})
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        states = out.hidden_states
        if len(states) != n_layers + 1:
            raise RuntimeError("expected %d hidden states, got %d"
                               % (n_layers + 1, len(states)))
        p_avg = torch.stack([states[l][0, :prompt_len, :].mean(dim=0).float().cpu()
                             for l in range(n_layers + 1)])
        p_last = torch.stack([states[l][0, prompt_len - 1, :].float().cpu()
                              for l in range(n_layers + 1)])
        r_avg = torch.stack([states[l][0, prompt_len:, :].mean(dim=0).float().cpu()
                             for l in range(n_layers + 1)])
        for name, tensor in (("prompt_avg", p_avg), ("prompt_last", p_last),
                             ("response_avg", r_avg)):
            if not torch.isfinite(tensor).all():
                raise RuntimeError("row %d: non-finite %s" % (row_index, name))
        del out, states
        shard.append((row_index, prompt_len, len(ids), p_avg, p_last, r_avg))
        if len(shard) >= SHARD_ROWS:
            flush()
            print("%5d/%5d  %6.1fs" % (done + 1, len(todo), time.time() - started),
                  flush=True)
    flush()

    meta["n_shards"] = shard_index
    meta["dropped"] = dropped
    meta["n_scored_rows"] = len(todo)
    meta["n_captured"] = len(todo) - len(dropped)
    meta["seconds"] = time.time() - started
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print("captured %d of %d scored rows, dropped %d, %.1f min"
          % (meta["n_captured"], len(todo), len(dropped), meta["seconds"] / 60.0), flush=True)


if __name__ == "__main__":
    main()
