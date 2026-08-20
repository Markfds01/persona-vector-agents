"""The MATCHED empirical cosine null for the cell-balanced decision vector.

1000 within-cell label permutations, cell-balanced rebuild each time, cosine to
the cell-balanced real vector at layer 20 - the same construction as the vector
under test and as build_null_vector.py's single draw. analyze.py's
shuffle_null_vs_ours is a different construction (global permutation, naive
pooled rebuild, scored against the naive vector) and is not a null for this.
"""
import argparse, csv, json, sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parents[1]      # results/dictator-decision-vector
sys.path.insert(0, str(R / "scripts"))
from build_null_vector import label, poles

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--acts", required=True,
                help="the acts_seed0 directory (2.1 GB, not committed)")
ap.add_argument("--rows", default=str(R / "extraction/grid_seed0.csv"))
ap.add_argument("--reference", default=str(R / "vectors/decision_response_avg_diff_cellbalanced.pt"))
ap.add_argument("--out", default=str(R / "analysis/matched_cosine_null.json"))
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--draws", type=int, default=1000)
ap.add_argument("--seed", type=int, default=20260819)
args = ap.parse_args()

LAYER, DRAWS, SEED = args.layer, args.draws, args.seed

rows = list(csv.DictReader(open(args.rows, encoding="utf-8")))
index, mats = [], []
for path in sorted(Path(args.acts).glob("shard_*.pt")):
    payload = torch.load(path, map_location="cpu")
    index.extend(payload["row_index"].tolist())
    mats.append(payload["response_avg"][:, LAYER, :].double())   # layer 20 only
    del payload
A = torch.cat(mats, dim=0)                                       # (n, 3584) float64
del mats
print("layer-%d activations: %s" % (LAYER, tuple(A.shape)), flush=True)

alt, self_, _m, _e = poles(label(rows, index))
by_cell = {}
for it in alt:
    by_cell.setdefault(it["cell"], ([], []))[0].append(it["position"])
for it in self_:
    by_cell.setdefault(it["cell"], ([], []))[1].append(it["position"])
usable = [(c, torch.tensor(a), torch.tensor(s)) for c, (a, s) in sorted(by_cell.items()) if a and s]
print("usable cells: %d" % len(usable), flush=True)

def balanced20(cells):
    return torch.stack([A.index_select(0, a).mean(0) - A.index_select(0, s).mean(0)
                        for _c, a, s in cells]).mean(0)

real = balanced20(usable)
ref = torch.load(args.reference, map_location="cpu").double()[LAYER]
rel = float((ref - real).norm() / ref.norm())
print("rebuild vs shipped artifact at layer %d, relative deviation: %.3e" % (LAYER, rel), flush=True)

rv_n = real / real.norm()
g = torch.Generator().manual_seed(SEED)
cs = []
for d in range(DRAWS):
    cells = []
    for cell, a, s in usable:
        pool = torch.cat([a, s])
        picked = pool[torch.randperm(len(pool), generator=g)]
        cells.append((cell, picked[:len(a)], picked[len(a):]))
    fake = balanced20(cells)
    cs.append(float(fake @ rv_n / fake.norm()))
    if (d + 1) % 250 == 0:
        print("  %d draws" % (d + 1), flush=True)

t = torch.tensor(cs, dtype=torch.float64)
obs = 0.24231978212677696
sd, mean = float(t.std(unbiased=True)), float(t.mean())
beat = int((t.abs() >= obs).sum())
out = {
    "what": "empirical cosine null for the cell-balanced decision vector, matched construction",
    "construction": "within-cell label permutation, cell-balanced rebuild, cosine to the "
                    "cell-balanced real vector at layer 20",
    "layer": LAYER, "draws": DRAWS, "seed": SEED,
    "rebuild_relative_deviation_vs_shipped_at_layer": rel,
    "observed_cos_shuffled_null_vs_real": obs,
    "null": {"n": DRAWS, "mean": mean, "sd": sd, "min": float(t.min()), "max": float(t.max()),
             "abs_max": float(t.abs().max()),
             "p2.5": float(t.quantile(0.025)), "p97.5": float(t.quantile(0.975))},
    "observed_z_vs_null_sd": obs / sd,
    "draws_reaching_observed_abs": beat,
    "empirical_p_two_sided": beat / DRAWS,
    "one_over_sd_squared": 1.0 / sd ** 2,
    "note": "analyze.py's shuffle_null_vs_ours (sd 0.34635) is a DIFFERENT construction - "
            "global permutation, naive pooled rebuild, scored against the naive vector - "
            "and is not the matched null for the cell-balanced vector that was steered.",
}
Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
