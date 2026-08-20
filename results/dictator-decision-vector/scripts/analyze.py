"""Build the outcome-defined Dictator vector and measure it. CPU only.

Poles are read off what the model DID, never off a rating:
  self-interested  transferred exactly 0
  altruistic       transferred at least half the endowment
Labelled by FRACTION of the endowment, because the endowment varies across the
grid; everything strictly between is discarded.

`audit.parse` resolves an amount along one of several paths and names the path in
its `tag`. Two of those paths - `complement` and `keep` - do not read the amount
off the response, they subtract a read amount from the pot. That subtraction
turns "Agent 1 would send $0" into "gave the whole endowment", which is not a
small error: it lands exactly on the pole boundary, in the wrong pole. One such
row was found by hand in the 60-generation smoke run. So the poles are built from
the direct-read tags only, and the subtraction tags are reported separately and
re-run as a sensitivity check rather than silently included or silently dropped.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VARIANTS = ("prompt_avg", "prompt_last", "response_avg")

#: value read directly off the response
DIRECT_TAGS = ("a2_anchor", "a2_near", "verb_obj", "answer_is", "bare", "bare_int")
#: value derived as pot minus a read amount
SUBTRACTION_TAGS = ("complement", "keep")


def load(rows_csv, act_dir):
    with open(rows_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    shards = sorted(Path(act_dir).glob("shard_*.pt"))
    if not shards:
        raise SystemExit("no activation shards in %s" % act_dir)
    index, acts = [], {v: [] for v in VARIANTS}
    for path in shards:
        payload = torch.load(path, map_location="cpu")
        index.extend(payload["row_index"].tolist())
        for v in VARIANTS:
            acts[v].append(payload[v])
    acts = {v: torch.cat(tensors, dim=0) for v, tensors in acts.items()}
    if len(index) != acts["response_avg"].shape[0]:
        raise SystemExit("shard row count disagrees with activation count")
    meta = json.loads((Path(act_dir) / "meta.json").read_text())
    return rows, index, acts, meta


def label(rows, index):
    """Per captured activation: endowment, wording, fraction given, tag."""
    out = []
    for position, row_index in enumerate(index):
        row = rows[row_index]
        game_id = row["game_id"]
        wording = game_id.split("/")[1]
        endowment = int(game_id.split("/e")[1])
        value = float(row["value"])
        out.append({
            "position": position,
            "row_index": row_index,
            "wording": wording,
            "endowment": endowment,
            "cell": "%s/e%d" % (wording, endowment),
            "value": value,
            "fraction": value / endowment,
            "tag": row["tag"],
        })
    return out


def poles(labels, allow_subtraction):
    allowed = set(DIRECT_TAGS) | (set(SUBTRACTION_TAGS) if allow_subtraction else set())
    alt, self_, middle, excluded = [], [], [], []
    for item in labels:
        if item["tag"] not in allowed:
            excluded.append(item)
            continue
        if item["fraction"] == 0.0:
            self_.append(item)
        elif item["fraction"] >= 0.5:
            alt.append(item)
        else:
            middle.append(item)
    return alt, self_, middle, excluded


def mean_of(acts, items):
    idx = torch.tensor([i["position"] for i in items], dtype=torch.long)
    return acts.index_select(0, idx).double().mean(dim=0)


def naive_vector(acts, alt, self_):
    return mean_of(acts, alt) - mean_of(acts, self_)


def balanced_vector(acts, alt, self_):
    """Equal weight per prompt cell, so pole composition cannot fake a direction.

    A cell is one (wording, endowment). Only cells with rows in BOTH poles can
    contribute a within-cell difference; the rest are reported, not silently used.
    """
    by_cell = {}
    for item in alt:
        by_cell.setdefault(item["cell"], ([], []))[0].append(item)
    for item in self_:
        by_cell.setdefault(item["cell"], ([], []))[1].append(item)
    usable = [(cell, a, s) for cell, (a, s) in sorted(by_cell.items()) if a and s]
    if not usable:
        raise SystemExit("no cell has both poles")
    stack = torch.stack([mean_of(acts, a) - mean_of(acts, s) for _c, a, s in usable])
    return stack.mean(dim=0), [c for c, _a, _s in usable], len(by_cell)


def cosines(ours, theirs):
    """Per-layer cosine between two (L, H) tensors, NaN-safe on a zero row."""
    a = ours.double()
    b = theirs.double()
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1)
    out = torch.where(den > 0, num / den, torch.full_like(num, float("nan")))
    return out


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random altruistic projection > a random self one)."""
    values = torch.cat([pos, neg])
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=values.dtype)
    # average ranks over ties
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    n1, n2 = len(pos), len(neg)
    r1 = ranks[:n1].sum()
    return ((r1 - n1 * (n1 + 1) / 2) / (n1 * n2)).item()


def split_half_separation(acts, alt, self_, seed):
    """Fit the direction on half the rows, score the other half. Per layer.

    In-sample separation is circular - the direction is the thing that maximises
    it. Fitting and scoring on disjoint halves is what makes "layer L separates
    best" a claim about the model rather than about the fit.
    """
    g = torch.Generator().manual_seed(seed)

    def halve(items):
        order = torch.randperm(len(items), generator=g).tolist()
        cut = len(items) // 2
        return [items[i] for i in order[:cut]], [items[i] for i in order[cut:]]

    alt_a, alt_b = halve(alt)
    self_a, self_b = halve(self_)
    direction = naive_vector(acts, alt_a, self_a)          # (L, H)
    other_half = naive_vector(acts, alt_b, self_b)
    reliability = cosines(direction, other_half)
    idx_b_alt = torch.tensor([i["position"] for i in alt_b])
    idx_b_self = torch.tensor([i["position"] for i in self_b])
    xa = acts.index_select(0, idx_b_alt).double()          # (n, L, H)
    xs = acts.index_select(0, idx_b_self).double()
    unit = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    proj_a = (xa * unit.unsqueeze(0)).sum(dim=2)           # (n, L)
    proj_s = (xs * unit.unsqueeze(0)).sum(dim=2)
    n_layers = proj_a.shape[1]
    out = []
    for l in range(n_layers):
        a, s = proj_a[:, l], proj_s[:, l]
        pooled = math.sqrt((a.var(unbiased=True).item() + s.var(unbiased=True).item()) / 2)
        d = (a.mean().item() - s.mean().item()) / pooled if pooled > 0 else float("nan")
        out.append({"layer": l, "cohens_d": d, "auc": auc(a, s)})
    return out, len(alt_a), len(self_a), len(alt_b), len(self_b), reliability


def shuffle_null(acts, alt, self_, ours, theirs, draws, seed):
    """Rebuild the vector from the SAME activations with the labels permuted.

    Two nulls, because two different claims need one:
      vs_ours   - could a random split of this data produce this direction
      vs_theirs - could a random split of this data reach this cosine to theirs
    """
    positions = torch.tensor([i["position"] for i in alt + self_], dtype=torch.long)
    n_alt = len(alt)
    x = acts.index_select(0, positions).double()           # (n, L, H)
    total = x.sum(dim=0)
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    vs_ours, vs_theirs = [], []
    for _ in range(draws):
        perm = torch.randperm(n, generator=g)
        pick = perm[:n_alt]
        sum_alt = x.index_select(0, pick).sum(dim=0)
        fake = sum_alt / n_alt - (total - sum_alt) / (n - n_alt)
        vs_ours.append(cosines(fake, ours))
        vs_theirs.append(cosines(fake, theirs))
    return torch.stack(vs_ours), torch.stack(vs_theirs)


def summarize(values):
    v = values[torch.isfinite(values)]
    if v.numel() == 0:
        return {"n": 0}
    return {
        "n": int(v.numel()),
        "mean": v.mean().item(),
        "sd": v.std(unbiased=True).item(),
        "min": v.min().item(),
        "max": v.max().item(),
        "abs_max": v.abs().max().item(),
        "p2.5": v.quantile(0.025).item(),
        "p97.5": v.quantile(0.975).item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors-dir", default="persona_vectors/Qwen2.5-7B-Instruct")
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, index, acts, meta = load(args.rows, args.acts)
    labels = label(rows, index)

    report = {"meta": meta, "seed": args.seed, "shuffles": args.shuffles,
              "headline_layer": args.layer, "n_generated": len(rows),
              "n_captured": len(labels)}

    # ---- what the scorer did with the whole run -----------------------------
    tag_counts = {}
    for row in rows:
        tag_counts[row["tag"]] = tag_counts.get(row["tag"], 0) + 1
    report["tag_counts_all_rows"] = tag_counts
    report["n_unresolved"] = sum(v for k, v in tag_counts.items()
                                 if k in ("empty", "refusal", "unparsed"))

    for policy, allow in (("primary_direct_tags_only", False),
                          ("sensitivity_with_subtraction_tags", True)):
        alt, self_, middle, excluded = poles(labels, allow)
        entry = {"n_altruistic": len(alt), "n_self_interested": len(self_),
                 "n_middle_discarded": len(middle), "n_tag_excluded": len(excluded)}
        by_cell = {}
        for name, group in (("altruistic", alt), ("self_interested", self_),
                            ("middle", middle)):
            for item in group:
                by_cell.setdefault(item["cell"], {}).setdefault(name, 0)
                by_cell[item["cell"]][name] += 1
        entry["by_cell"] = by_cell
        by_endowment = {}
        for name, group in (("altruistic", alt), ("self_interested", self_),
                            ("middle", middle)):
            for item in group:
                key = str(item["endowment"])
                by_endowment.setdefault(key, {}).setdefault(name, 0)
                by_endowment[key][name] += 1
        entry["by_endowment"] = by_endowment
        by_wording = {}
        for name, group in (("altruistic", alt), ("self_interested", self_),
                            ("middle", middle)):
            for item in group:
                by_wording.setdefault(item["wording"], {}).setdefault(name, 0)
                by_wording[item["wording"]][name] += 1
        entry["by_wording"] = by_wording
        report[policy] = entry

    alt, self_, middle, excluded = poles(labels, False)
    alt_s, self_s, _m, _e = poles(labels, True)

    d = acts["response_avg"].shape[2]
    report["random_null_sd"] = 1.0 / math.sqrt(d)
    report["hidden_size"] = d

    theirs = {}
    for variant in VARIANTS:
        path = Path(args.vectors_dir) / ("altruism_%s_diff.pt" % variant)
        theirs[variant] = torch.load(path, map_location="cpu").double()

    per_variant = {}
    for variant in VARIANTS:
        a = acts[variant]
        ours = naive_vector(a, alt, self_)
        bal, usable_cells, total_cells = balanced_vector(a, alt, self_)
        ours_sens = naive_vector(a, alt_s, self_s)
        torch.save(ours.float(), out_dir / ("decision_%s_diff.pt" % variant))
        torch.save(bal.float(), out_dir / ("decision_%s_diff_cellbalanced.pt" % variant))

        cos_theirs = cosines(ours, theirs[variant])
        entry = {
            "norms": ours.norm(dim=1).tolist(),
            "norms_cellbalanced": bal.norm(dim=1).tolist(),
            "their_norms": theirs[variant].norm(dim=1).tolist(),
            "cos_vs_theirs_by_layer": cos_theirs.tolist(),
            "cos_ours_vs_cellbalanced_by_layer": cosines(ours, bal).tolist(),
            "cos_sensitivity_vs_primary_by_layer": cosines(ours_sens, ours).tolist(),
            "cos_sensitivity_vs_theirs_by_layer": cosines(ours_sens, theirs[variant]).tolist(),
            "cos_cellbalanced_vs_theirs_by_layer": cosines(bal, theirs[variant]).tolist(),
            "usable_cells": usable_cells,
            "n_cells_seen": total_cells,
        }
        sep, na, ns, nb_a, nb_s, reliability = split_half_separation(
            a, alt, self_, args.seed)
        entry["split_half"] = {"fit_altruistic": na, "fit_self": ns,
                               "score_altruistic": nb_a, "score_self": nb_s,
                               "by_layer": sep,
                               # how much of the direction survives being rebuilt
                               # from a disjoint half - the ceiling any cosine
                               # against another vector could reach
                               "reliability_cos_by_layer": reliability.tolist()}
        null_ours, null_theirs = shuffle_null(a, alt, self_, ours, theirs[variant],
                                              args.shuffles, args.seed)
        entry["shuffle_null_vs_ours"] = {
            "layer_%d" % args.layer: summarize(null_ours[:, args.layer]),
            "per_layer_mean": null_ours.mean(dim=0).tolist(),
            "per_layer_abs_max": null_ours.abs().max(dim=0).values.tolist(),
            "per_layer_p97.5": null_ours.quantile(0.975, dim=0).tolist(),
        }
        entry["shuffle_null_vs_theirs"] = {
            "layer_%d" % args.layer: summarize(null_theirs[:, args.layer]),
            "per_layer_mean": null_theirs.mean(dim=0).tolist(),
            "per_layer_abs_max": null_theirs.abs().max(dim=0).values.tolist(),
            "per_layer_p97.5": null_theirs.quantile(0.975, dim=0).tolist(),
        }
        real = cos_theirs[args.layer].item()
        beat = (null_theirs[:, args.layer].abs() >= abs(real)).sum().item()
        entry["shuffle_draws_reaching_real_cos_vs_theirs_at_layer"] = beat
        per_variant[variant] = entry
    report["variants"] = per_variant

    # ---- our response_avg against every shipped Qwen trait vector ------------
    ours_ra = naive_vector(acts["response_avg"], alt, self_)
    others = {}
    for path in sorted(Path(args.vectors_dir).glob("*_response_avg_diff.pt")):
        vec = torch.load(path, map_location="cpu").double()
        others[path.stem] = cosines(ours_ra, vec)[args.layer].item()
    report["cos_vs_all_shipped_response_avg_at_layer"] = others

    # ---- our three variants against each other ------------------------------
    cross = {}
    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            cross["%s__%s" % (a, b)] = cosines(naive_vector(acts[a], alt, self_),
                                               naive_vector(acts[b], alt, self_)
                                               )[args.layer].item()
    report["cos_between_our_variants_at_layer"] = cross

    with (out_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote %s" % (out_dir / "analysis.json"))
    L = args.layer
    for variant in VARIANTS:
        e = per_variant[variant]
        print("%-13s layer %d: cos_vs_theirs=%+.4f  norm=%.4f (theirs %.4f)  "
              "shuffle|cos| mean %.4f max %.4f  splithalf d=%.3f auc=%.3f rel=%.3f"
              % (variant, L, e["cos_vs_theirs_by_layer"][L], e["norms"][L],
                 e["their_norms"][L],
                 e["shuffle_null_vs_theirs"]["layer_%d" % L]["mean"],
                 e["shuffle_null_vs_theirs"]["layer_%d" % L]["abs_max"],
                 e["split_half"]["by_layer"][L]["cohens_d"],
                 e["split_half"]["by_layer"][L]["auc"],
                 e["split_half"]["reliability_cos_by_layer"][L]))


if __name__ == "__main__":
    main()
