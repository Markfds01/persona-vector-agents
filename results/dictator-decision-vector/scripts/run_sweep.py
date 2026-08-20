"""The decision-vector steering sweep, matched to their vector's sweep by
INTERVENTION SIZE rather than by raw coefficient.

Their altruism vector and our decision vector have different lengths, so the same
raw beta is a different sized push. Every point here is placed at a unit beta of
k * ||their altruism vector at layer 20||, for k = 0, +-1 .. +-5, which is exactly
the delta magnitude their raw beta k produced. `norm="unit"` is used, so the
coefficient on the row IS the unit beta; the equivalent raw beta for our vector is
unit_beta / steer_vector_norm, and both are written to the manifest.

Two sweeps run: the real cell-balanced decision vector at all 11 points, and the
shuffled-label null vector at the two extremes and zero. The null is steered at the
SAME unit betas, so it is the same sized intervention along a randomly-labelled
direction.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from audit import generate, steer


def their_norm(path, layer):
    stored = torch.load(path, weights_only=False)
    return float(stored[layer].detach().float().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--null-vector", required=True)
    ap.add_argument("--their-vector", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_norm = their_norm(args.their_vector, args.layer)
    # sign-paired outward from zero
    ks = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    real_betas = [k * reference_norm for k in ks]
    null_ks = [0, 5, -5]
    null_betas = [k * reference_norm for k in null_ks]

    preset = generate.preset("neutral")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    model.eval()
    engine = generate.HuggingFaceEngine(model, tokenizer, args.model, args.revision)
    preset.check_engine(engine.describe())
    description = engine.describe()

    pairs = [("altruism_v3/dictator", "free")]
    sampling = preset.sampling.with_seed(args.seed)
    started = time.time()
    manifest = []

    def make_progress(tag):
        def on_coefficient(steering, path, rows):
            parsed = [r.value for r in rows if r.value is not None]
            mean = sum(parsed) / len(parsed) if parsed else float("nan")
            print("[%s] unit_beta=%+9.4f n=%d parsed=%d mean=%.2f  %s  %.1f min elapsed"
                  % (tag, steering.coeff, len(rows), len(parsed), mean, path.name,
                     (time.time() - started) / 60.0), flush=True)
        return on_coefficient

    def do_sweep(vector_path, betas, k_values, tag):
        # one directory per arm: the two arms sit at the SAME unit betas, so a
        # shared directory would have the null overwrite the real rows.
        arm_dir = out_dir / tag
        arm_dir.mkdir(parents=True, exist_ok=True)
        steerings = [steer.Steering(vector_path, args.layer, beta, positions="all",
                                    norm="unit") for beta in betas]
        results = steer.sweep(pairs, engine, model, steerings, sampling, tokenizer,
                              samples_per_prompt=args.samples, reading="stated",
                              batch_size=args.batch_size, out_dir=arm_dir,
                              on_coefficient=make_progress(tag))
        probe = steer.ActivationSteering(model, steerings[0])
        vector_norm = probe.vector_norm
        probe.remove()
        for k, beta in zip(k_values, betas):
            path, rows = results[beta]
            manifest.append({
                "arm": tag,
                "their_raw_beta": k,
                "unit_beta": beta,
                "our_raw_beta_equivalent": beta / vector_norm,
                "vector": str(Path(vector_path).resolve()),
                "vector_sha256_16": steerings[0].vector_sha256,
                "vector_layer_norm": vector_norm,
                "rows_csv": str(path.resolve()),
                "n_rows": len(rows),
            })
        return results

    do_sweep(args.vector, real_betas, ks, "decision")
    do_sweep(args.null_vector, null_betas, null_ks, "shuffled-null")

    provenance = {
        "their_vector": str(Path(args.their_vector).resolve()),
        "their_layer_norm": reference_norm,
        "layer": args.layer,
        "positions": "all",
        "norm_mode": "unit",
        "game": "altruism_v3/dictator",
        "mode": "free",
        "reading": "stated",
        "preset": "neutral",
        "samples_per_prompt": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "engine": description.engine,
        "engine_version": description.engine_version,
        "torch_version": description.torch_version,
        "model_id": description.model_id,
        "model_revision": description.model_revision,
        "dtype": description.dtype,
        "attn_implementation": description.attn_implementation,
        "stop_token_ids": list(description.stop_token_ids),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "repo_commit": generate.repo_commit(),
        "repo_dirty": generate.repo_is_dirty(),
        "minutes": (time.time() - started) / 60.0,
        "coefficients": manifest,
    }
    (out_dir.parent / "sweep_provenance.json").write_text(json.dumps(provenance, indent=2))
    with (out_dir.parent / "coefficients.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print("done in %.1f min" % provenance["minutes"], flush=True)


if __name__ == "__main__":
    main()
