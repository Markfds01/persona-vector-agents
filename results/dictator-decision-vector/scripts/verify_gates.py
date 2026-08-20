"""The three gates that must pass before a steering sweep is worth running.

1. beta=0 is a byte-exact no-op against NO HOOK AT ALL, together with the three
   controls that make that statement mean something: the unhooked run is itself
   reproducible at a fixed seed and batch, the hook actually fired, and a nonzero
   coefficient changes the ids.
2. The hook site is `model.model.layers[19]` for --layer 20, and that module's
   output is byte-equal to `hidden_states[20]` (and to neither [19] nor [21]).
3. The vector file loaded is the file we think it is: sha256 and norm, measured.

Nothing here is inferred. Every line printed is the result of a run.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from audit import elicit, games, generate, steer


def sha16(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def gen_ids(model, tokenizer, prompts, sampling, seed, stop_token_ids, max_new_tokens):
    """The engine's generate path, but returning ids instead of decoded text."""
    fields = generate.generation_config_fields(sampling, tokenizer.pad_token_id,
                                               stop_token_ids)
    fields["max_new_tokens"] = max_new_tokens
    config = __import__("transformers").GenerationConfig(**fields)
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(list(prompts), return_tensors="pt", padding=True,
                          add_special_tokens=False)
    finally:
        tokenizer.padding_side = previous
    batch = {k: v.to(model.device) for k, v in batch.items()}
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**batch, generation_config=config,
                             use_model_defaults=False)
    return out.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    preset = generate.preset("neutral")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    model.eval()

    engine = generate.HuggingFaceEngine(model, tokenizer, args.model, args.revision)
    preset.check_engine(engine.describe())
    description = engine.describe()
    stop_token_ids = description.stop_token_ids

    report = {"engine": description._asdict() if hasattr(description, "_asdict")
              else {"engine": description.engine,
                    "engine_version": description.engine_version,
                    "torch_version": description.torch_version,
                    "model_id": description.model_id,
                    "model_revision": description.model_revision,
                    "dtype": description.dtype,
                    "attn_implementation": description.attn_implementation,
                    "stop_token_ids": list(description.stop_token_ids)},
              "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
              "device_name": torch.cuda.get_device_name(0)}

    # ---- gate 3: the vector is the file we think it is ----------------------
    stored = torch.load(args.vector, weights_only=False)
    row = stored[args.layer]
    report["vector"] = {
        "path": str(Path(args.vector).resolve()),
        "sha256_16": sha16(args.vector),
        "shape": list(stored.shape),
        "dtype": str(stored.dtype),
        "layer": args.layer,
        "layer_norm_fp32": float(row.detach().float().norm()),
    }

    # ---- gate 2: the hook site ---------------------------------------------
    path, module = steer.hooked_module(model, args.layer)
    site = {"resolved_path": path,
            "is_model_model_layers_19": module is model.model.layers[args.layer - 1]}
    captured = {}

    def capture(_m, _i, output):
        captured["value"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = module.register_forward_hook(capture)
    probe = tokenizer("Agent 1 receives an endowment of $100.", return_tensors="pt",
                      add_special_tokens=False)
    probe = {k: v.to(model.device) for k, v in probe.items()}
    with torch.no_grad():
        out = model(**probe, output_hidden_states=True)
    handle.remove()
    got = captured["value"]
    site["n_hidden_states"] = len(out.hidden_states)
    site["equals_hidden_states_20"] = bool(torch.equal(got, out.hidden_states[args.layer]))
    site["equals_hidden_states_19"] = bool(torch.equal(got, out.hidden_states[args.layer - 1]))
    site["equals_hidden_states_21"] = bool(torch.equal(got, out.hidden_states[args.layer + 1]))
    report["hook_site"] = site
    del out, captured, got

    # ---- gate 1: beta=0 no-op, with its three controls ----------------------
    game = games.by_id("altruism_v3/dictator")
    prompt = elicit.render(game, "free", tokenizer).text
    prompts = [prompt] * 4
    sampling = preset.sampling.with_seed(0)
    seed = 0

    def run(steering=None):
        counter = {"calls": 0}
        if steering is None:
            ids = gen_ids(model, tokenizer, prompts, sampling, seed, stop_token_ids,
                          args.max_new_tokens)
            return ids, counter["calls"]
        active = steer.ActivationSteering(model, steering)
        original = active._hook

        def counting(module, inputs, output):
            counter["calls"] += 1
            return original(module, inputs, output)

        active._hook = counting
        with active:
            ids = gen_ids(model, tokenizer, prompts, sampling, seed, stop_token_ids,
                          args.max_new_tokens)
        return ids, counter["calls"]

    unhooked_a, _ = run(None)
    unhooked_b, _ = run(None)
    zero = steer.Steering(args.vector, args.layer, 0.0, positions="all", norm="unit")
    zero_ids, zero_calls = run(zero)
    nonzero = steer.Steering(args.vector, args.layer, 20.0, positions="all", norm="unit")
    nonzero_ids, nonzero_calls = run(nonzero)

    report["noop_gate"] = {
        "unhooked_reproducible": bool(torch.equal(unhooked_a, unhooked_b)),
        "beta0_equals_unhooked": bool(torch.equal(zero_ids, unhooked_a)),
        "beta0_hook_invocations": zero_calls,
        "nonzero_hook_invocations": nonzero_calls,
        "nonzero_changes_ids": bool(not torch.equal(nonzero_ids, unhooked_a)),
        "nonzero_beta_unit": 20.0,
        "shapes": {"unhooked": list(unhooked_a.shape), "beta0": list(zero_ids.shape),
                   "nonzero": list(nonzero_ids.shape)},
    }

    passed = (site["is_model_model_layers_19"] and site["equals_hidden_states_20"]
              and not site["equals_hidden_states_19"]
              and not site["equals_hidden_states_21"]
              and report["noop_gate"]["unhooked_reproducible"]
              and report["noop_gate"]["beta0_equals_unhooked"]
              and zero_calls > 0 and report["noop_gate"]["nonzero_changes_ids"])
    report["all_gates_passed"] = bool(passed)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("GATES FAILED")


if __name__ == "__main__":
    main()
