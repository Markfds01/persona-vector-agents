"""Three controls the headline numbers need to be readable.

1. Shuffled-label SEPARATION. The real direction separates held-out rows at
   d=2.3 / AUC 0.93. That only means something if the same split-half procedure
   on permuted labels lands at AUC 0.5 - otherwise the number is a property of
   the procedure, not of the labels.
2. A shuffle null per SHIPPED vector. The null's width depends on the target:
   it is the overlap between the target and the subspace the row cloud lives in.
   Comparing our cosine to `altruism` against a null computed for `altruism` is
   valid; reusing that null for `expected_altruism` is not.
3. Per-layer empirical p-values against their altruism vector, so "which layer
   aligns best" is not read off raw cosines that have different nulls per layer.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.analyze import (VARIANTS, auc, cosines, label, load, naive_vector,
                             poles, summarize)


def shuffled_separation(acts, alt, self_, seed, draws, layer):
    """Split-half fit-and-score with the pole labels permuted. Should be AUC 0.5."""
    g = torch.Generator().manual_seed(seed)
    positions = torch.tensor([i["position"] for i in alt + self_], dtype=torch.long)
    x = acts.index_select(0, positions).double()
    n, n_alt = x.shape[0], len(alt)
    out = []
    for _ in range(draws):
        perm = torch.randperm(n, generator=g)
        fake_alt, fake_self = perm[:n_alt], perm[n_alt:]
        half_a = fake_alt[: len(fake_alt) // 2]
        half_b = fake_alt[len(fake_alt) // 2:]
        half_c = fake_self[: len(fake_self) // 2]
        half_d = fake_self[len(fake_self) // 2:]
        direction = (x.index_select(0, half_a).mean(0)
                     - x.index_select(0, half_c).mean(0))[layer]
        unit = direction / direction.norm().clamp_min(1e-12)
        pa = x.index_select(0, half_b)[:, layer, :] @ unit
        ps = x.index_select(0, half_d)[:, layer, :] @ unit
        pooled = math.sqrt((pa.var().item() + ps.var().item()) / 2)
        out.append((auc(pa, ps), (pa.mean().item() - ps.mean().item()) / pooled))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors-dir", default="persona_vectors/Qwen2.5-7B-Instruct")
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--sep-shuffles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    rows, index, acts, _meta = load(args.rows, args.acts)
    labels = label(rows, index)
    alt, self_, _m, _e = poles(labels, False)
    L = args.layer
    report = {"layer": L, "shuffles": args.shuffles, "sep_shuffles": args.sep_shuffles,
              "seed": args.seed, "n_altruistic": len(alt), "n_self": len(self_)}

    a = acts["response_avg"]
    real = naive_vector(a, alt, self_)

    # --- 1. shuffled-label separation ---------------------------------------
    sep = shuffled_separation(a, alt, self_, args.seed, args.sep_shuffles, L)
    aucs = torch.tensor([s[0] for s in sep])
    ds = torch.tensor([s[1] for s in sep])
    report["shuffled_label_separation_response_avg"] = {
        "auc": summarize(aucs), "cohens_d": summarize(ds)}

    # --- 2 + 3. per-target nulls --------------------------------------------
    positions = torch.tensor([i["position"] for i in alt + self_], dtype=torch.long)
    x = a.index_select(0, positions).double()
    total = x.sum(dim=0)
    n, n_alt = x.shape[0], len(alt)
    g = torch.Generator().manual_seed(args.seed)
    fakes = []
    for _ in range(args.shuffles):
        pick = torch.randperm(n, generator=g)[:n_alt]
        sum_alt = x.index_select(0, pick).sum(dim=0)
        fakes.append(sum_alt / n_alt - (total - sum_alt) / (n - n_alt))
    fakes = torch.stack(fakes)                                   # (draws, L, H)

    targets = {}
    for path in sorted(Path(args.vectors_dir).glob("*_response_avg_diff.pt")):
        vec = torch.load(path, map_location="cpu").double()
        observed = cosines(real, vec)[L].item()
        null = torch.stack([cosines(f, vec)[L] for f in fakes])
        beat = (null.abs() >= abs(observed)).sum().item()
        targets[path.stem] = {
            "observed_cos": observed,
            "null": summarize(null),
            "draws_reaching_observed": beat,
            "empirical_p_two_sided": beat / float(args.shuffles),
        }
    report["per_target_null_at_layer"] = targets

    theirs = torch.load(Path(args.vectors_dir) / "altruism_response_avg_diff.pt",
                        map_location="cpu").double()
    per_layer = []
    observed_all = cosines(real, theirs)
    null_all = torch.stack([cosines(f, theirs) for f in fakes])   # (draws, L)
    for l in range(observed_all.shape[0]):
        beat = (null_all[:, l].abs() >= abs(observed_all[l].item())).sum().item()
        per_layer.append({"layer": l, "observed_cos": observed_all[l].item(),
                          "null_sd": null_all[:, l].std().item(),
                          "null_p97.5": null_all[:, l].quantile(0.975).item(),
                          "empirical_p_two_sided": beat / float(args.shuffles)})
    report["per_layer_vs_their_altruism"] = per_layer

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", out)
    s = report["shuffled_label_separation_response_avg"]
    print("shuffled-label separation: AUC mean %.4f sd %.4f max %.4f | d mean %.3f max %.3f"
          % (s["auc"]["mean"], s["auc"]["sd"], s["auc"]["max"],
             s["cohens_d"]["mean"], s["cohens_d"]["max"]))
    for name, t in sorted(targets.items(), key=lambda kv: -abs(kv[1]["observed_cos"])):
        print("%-45s cos=%+.4f  null sd %.4f p97.5 %.4f  p=%.3f"
              % (name, t["observed_cos"], t["null"]["sd"], t["null"]["p97.5"],
                 t["empirical_p_two_sided"]))


if __name__ == "__main__":
    main()
