"""Fill in the shuffled-label null arm at k = +-1 and +-3.

The three-point null (0, +-5) establishes that a randomly-labelled direction of
the same length moves the Dictator, but not the SHAPE of that movement: with one
point per side a dose-response and a single quirk are indistinguishable. These
four points make the null arm 7 points, matching the real arm's spacing on the
odd coefficients.

Same vector file, same layer, same positions, same unit-beta convention, same
seed, same batch size, same engine as the first run - only the coefficients differ.
"""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from audit import generate, steer

ap = argparse.ArgumentParser()
ap.add_argument("--null-vector", required=True)
ap.add_argument("--their-vector", required=True)
ap.add_argument("--out-dir", required=True)
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
ks = [1, -1, 3, -3]
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
    print("[null-extra] unit_beta=%+9.4f n=%d parsed=%d mean=%.2f  %s  %.1f min"
          % (steering.coeff, len(rows), len(parsed),
             sum(parsed) / len(parsed) if parsed else float("nan"), path.name,
             (time.time() - started) / 60.0), flush=True)

steerings = [steer.Steering(args.null_vector, args.layer, b, positions="all",
                            norm="unit") for b in betas]
results = steer.sweep([("altruism_v3/dictator", "free")], engine, model, steerings,
                      preset.sampling.with_seed(args.seed), tokenizer,
                      samples_per_prompt=args.samples, reading="stated",
                      batch_size=args.batch_size, out_dir=out_dir,
                      on_coefficient=progress)

probe = steer.ActivationSteering(model, steerings[0])
vector_norm = probe.vector_norm
probe.remove()
rows_out = []
for k, beta in zip(ks, betas):
    path, rows = results[beta]
    rows_out.append({"arm": "shuffled-null", "their_raw_beta": k, "unit_beta": beta,
                     "our_raw_beta_equivalent": beta / vector_norm,
                     "vector": str(Path(args.null_vector).resolve()),
                     "vector_sha256_16": steerings[0].vector_sha256,
                     "vector_layer_norm": vector_norm,
                     "rows_csv": str(path.resolve()), "n_rows": len(rows)})
manifest = Path(args.out_dir).parent.parent / "coefficients.csv"
existing = list(csv.DictReader(open(manifest, encoding="utf-8")))
with manifest.open("w", encoding="utf-8", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(existing[0].keys()))
    w.writeheader()
    w.writerows(existing)
    w.writerows([{k2: r[k2] for k2 in existing[0].keys()} for r in rows_out])
print("done in %.1f min" % ((time.time() - started) / 60.0), flush=True)
