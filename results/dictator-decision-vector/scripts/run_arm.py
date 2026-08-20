"""Run one steering arm: one vector, a list of k values, one output directory.

k is indexed in THEIR raw beta, and the unit beta actually applied is
k * ||their altruism vector at layer 20||, so every arm sits at the same
intervention magnitudes as every other. Same engine, seed, batching and preset as
the first sweep; only the vector and the coefficients differ.
"""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from audit import generate, steer

ap = argparse.ArgumentParser()
ap.add_argument("--vector", required=True)
ap.add_argument("--their-vector", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--ks", required=True, help="comma-separated k values")
ap.add_argument("--manifest", required=True)
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
ap.add_argument("--samples", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

reference_norm = float(torch.load(args.their_vector, weights_only=False)[args.layer]
                       .detach().float().norm())
ks = [int(k) for k in args.ks.split(",")]
betas = [k * reference_norm for k in ks]

preset = generate.preset("neutral")
tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
model = AutoModelForCausalLM.from_pretrained(
    args.model, revision=args.revision, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map={"": 0})
model.eval()
engine = generate.HuggingFaceEngine(model, tokenizer, args.model, args.revision)
preset.check_engine(engine.describe())

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
started = time.time()

def progress(steering, path, rows):
    parsed = [r.value for r in rows if r.value is not None]
    print("[%s] unit_beta=%+9.4f n=%d parsed=%d mean=%.2f  %s  %.1f min"
          % (args.arm, steering.coeff, len(rows), len(parsed),
             sum(parsed) / len(parsed) if parsed else float("nan"), path.name,
             (time.time() - started) / 60.0), flush=True)

steerings = [steer.Steering(args.vector, args.layer, b, positions="all", norm="unit")
             for b in betas]

# resolved BEFORE the sweep: the manifest row for a finished coefficient must be
# writable the instant that coefficient lands, not after the whole arm
probe = steer.ActivationSteering(model, steerings[0])
vector_norm = probe.vector_norm
probe.remove()
beta_to_k = {b: k for k, b in zip(ks, betas)}
manifest = Path(args.manifest)


def record(steering, path, rows):
    """Append one finished coefficient to the manifest immediately.

    The GPU here is shared with a tenant that loads and unloads models on its own
    schedule, so this process can be evicted mid-arm. Appending per coefficient
    means an eviction costs the coefficient still generating, not the record of
    the ones already on disk.
    """
    existing = list(csv.DictReader(open(manifest, encoding="utf-8")))
    fields = list(existing[0].keys())
    row = {"arm": args.arm, "their_raw_beta": beta_to_k[steering.coeff],
           "unit_beta": steering.coeff,
           "our_raw_beta_equivalent": steering.coeff / vector_norm,
           "vector": str(Path(args.vector).resolve()),
           "vector_sha256_16": steering.vector_sha256,
           "vector_layer_norm": vector_norm,
           "rows_csv": str(path.resolve()), "n_rows": len(rows)}
    with manifest.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(existing)
        w.writerow({f: row[f] for f in fields})
    print("   manifest updated for k=%+d" % row["their_raw_beta"], flush=True)


def on_coefficient(steering, path, rows):
    progress(steering, path, rows)
    record(steering, path, rows)


results = steer.sweep([("altruism_v3/dictator", "free")], engine, model, steerings,
                      preset.sampling.with_seed(args.seed), tokenizer,
                      samples_per_prompt=args.samples, reading="stated",
                      batch_size=args.batch_size, out_dir=out_dir,
                      on_coefficient=on_coefficient)
print("done in %.1f min" % ((time.time() - started) / 60.0), flush=True)
