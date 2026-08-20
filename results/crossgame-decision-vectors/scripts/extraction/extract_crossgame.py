"""Teacher-forced activation capture for the cross-game grid, CPU-pooled per layer.

Identical in method to the Dictator run's `extract_acts.py` (which in turn matches
`generate_vec.py:28-38`): re-render the prompt each row was generated from, verify
the rendered bytes against the fingerprint the row carries, run ONE forward pass
over prompt+response with `output_hidden_states=True`, and pool per layer:

  prompt_avg    mean over token positions [0, prompt_len)
  prompt_last   the single position prompt_len-1
  response_avg  mean over token positions [prompt_len, end)

Only `response_avg` carries decision information for an outcome-conditioned
contrast - causal masking makes every prompt-side activation identical within a
prompt cell - but all three are captured so that fact stays checkable rather than
asserted.

Batch size is 1 and no padding is involved: that is what upstream does, and it is
the only way the position slices mean what they say.

Two failures are reported rather than absorbed:
  * prefix instability - the prompt's tokens must be a prefix of prompt+response
    tokens; BPE can merge across the seam, which would put the wrong tokens in
    the response slice. Such a row is dropped, counted and named.
  * an empty response, which would make response_avg a NaN.

dtype and attn_implementation are pinned and written into the meta file: this
model's sdpa and eager kernels diverge at bf16, so a vector built under one is
not comparable with a vector built under the other. Every run in this project is
sdpa/bf16.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import elicit, games
from scratch import crossgame_grid

SHARD_ROWS = 250


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="CSV written by gen_crossgame.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--load-attempts", type=int, default=60)
    ap.add_argument("--load-retry-seconds", type=int, default=30)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.rows, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    crossgame_grid.register()

    # only rows the scorer resolved carry a label; the rest are counted, never guessed
    todo = [(i, r) for i, r in enumerate(rows) if r["value"] != ""]
    print("rows=%d scored=%d unscored=%d" % (len(rows), len(todo), len(rows) - len(todo)),
          flush=True)

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    # The device is shared and the other tenant cycles models on it without warning;
    # a load that OOMs is transient, not a result. Retry the LOAD only.
    model = None
    for attempt in range(1, args.load_attempts + 1):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=args.revision, torch_dtype=dtype,
                attn_implementation=args.attn, device_map={"": args.device})
            break
        except torch.OutOfMemoryError as exc:
            free, total = torch.cuda.mem_get_info(args.device)
            print("load attempt %d/%d OOMed (%.1f GiB free of %.1f): %s"
                  % (attempt, args.load_attempts, free / 2**30, total / 2**30, exc),
                  flush=True)
            torch.cuda.empty_cache()
            if attempt == args.load_attempts:
                raise
            time.sleep(args.load_retry_seconds)
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
        "device": "cuda:%d" % args.device,
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
        if not shard:
            return 0
        payload = {
            "row_index": torch.tensor([s[0] for s in shard], dtype=torch.long),
            "prompt_len": torch.tensor([s[1] for s in shard], dtype=torch.long),
            "total_len": torch.tensor([s[2] for s in shard], dtype=torch.long),
            "prompt_avg": torch.stack([s[3] for s in shard]),
            "prompt_last": torch.stack([s[4] for s in shard]),
            "response_avg": torch.stack([s[5] for s in shard]),
        }
        torch.save(payload, out_dir / ("shard_%04d.pt" % shard_index))
        return 1

    for done, (row_index, row) in enumerate(todo):
        rendered = prompt_for(row["game_id"], row["mode"])
        if rendered.sha256 != row["prompt_sha256"]:
            raise RuntimeError("row %d: re-rendered prompt %s != recorded %s"
                               % (row_index, rendered.sha256, row["prompt_sha256"]))
        prompt = rendered.text
        text = prompt + row["answer"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"][0].tolist()
        if ids[:prompt_len] != prompt_ids:
            dropped.append({"row_index": row_index, "game_id": row["game_id"],
                            "reason": "prompt tokens are not a prefix of "
                                      "prompt+answer tokens"})
            continue
        if len(ids) <= prompt_len:
            dropped.append({"row_index": row_index, "game_id": row["game_id"],
                            "reason": "empty response"})
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
            shard_index += flush()
            shard = []
            print("%5d/%5d  %6.1fs" % (done + 1, len(todo), time.time() - started),
                  flush=True)
    shard_index += flush()

    meta["n_shards"] = shard_index
    meta["dropped"] = dropped
    meta["n_scored_rows"] = len(todo)
    meta["n_captured"] = len(todo) - len(dropped)
    meta["seconds"] = time.time() - started
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print("captured %d of %d scored rows, dropped %d, %.1f min"
          % (meta["n_captured"], len(todo), len(dropped), meta["seconds"] / 60.0),
          flush=True)


if __name__ == "__main__":
    main()
