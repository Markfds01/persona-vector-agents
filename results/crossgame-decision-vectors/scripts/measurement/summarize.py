"""Print the report's tables from analysis.json. Reading only - computes nothing new."""

import json
import sys
from pathlib import Path

A = json.loads(Path(sys.argv[1]).read_text())
LAYER = A["headline_layer"]
SHORT = {"dictator": "dict", "trust": "trust", "ultimatum": "ultim", "apology": "apol",
         "overfishing": "fish", "prisoners_dilemma": "pd"}


def fam_short(f):
    return SHORT.get(f, f[:5])


print("hidden=%d layers+emb=%d headline_layer=%d seed=%d shuffles=%d pair_shuffles=%d"
      % (A["hidden_size"], A["n_layers_plus_embedding"], LAYER, A["seed"],
         A["shuffles"], A["pair_shuffles"]))
print("theoretical random cosine sd = %.4f (quoted only to be dismissed)"
      % A["theoretical_random_cosine_sd"])
print("cos(archived dictator vector, their shipped altruism) @%d = %+.4f"
      % (LAYER, A["cos_dictator_vs_their_altruism_by_layer"][LAYER]))

print("\n### parse coverage over every generated row")
for fam, tags in sorted(A["tag_counts_all_generated_rows"].items()):
    total = sum(tags.values())
    unres = sum(v for k, v in tags.items() if k in ("empty", "refusal", "unparsed"))
    print("  %-18s n=%-5d unresolved=%-4d (%.1f%%)  %s"
          % (fam, total, unres, 100.0 * unres / total,
             " ".join("%s=%d" % kv for kv in sorted(tags.items(), key=lambda kv: -kv[1]))))

for policy, R in A["by_policy"].items():
    print("\n\n" + "=" * 78)
    print("POLE POLICY: %s" % policy.upper())
    print("=" * 78)

    print("\n### per-game poles, cells, norms, and each game's OWN out-of-sample AUC")
    print("  %-18s %6s %6s %7s %6s %8s %8s %8s %6s %8s %8s"
          % ("game", "alt", "self", "middle", "cells", "L0 AUC", "L20 AUC", "best AUC",
             "@L", "|v|L20", "|v|bal"))
    for fam, e in R["per_game"].items():
        if not e["usable"]:
            print("  %-18s  --- no vector: %s" % (fam, e["reason"]))
            continue
        sh = e["split_half"]
        bl = sh["by_layer"] if sh else None
        best = max(bl, key=lambda r: r["auc"]) if bl else None
        print("  %-18s %6d %6d %7d %6s %8s %8s %8s %6s %8.2f %8s"
              % (fam, e["n_altruistic"], e["n_self_interested"],
                 e["n_middle_discarded"],
                 "%d/%d" % (len(e["usable_cells"]), e["n_cells_seen"]),
                 "%.3f" % bl[0]["auc"] if bl else "n/a",
                 "%.3f" % bl[LAYER]["auc"] if bl else "n/a",
                 "%.3f" % best["auc"] if best else "n/a",
                 str(best["layer"]) if best else "-",
                 e["norms_unbalanced"][LAYER],
                 "%.2f" % e["norms_balanced"][LAYER] if e["norms_balanced"] else "n/a"))

    P = R["pooled"]
    if not P.get("usable"):
        print("\n  POOLED: not usable")
        continue
    sh = P["split_half"]; bl = sh["by_layer"]
    best = max(bl, key=lambda r: r["auc"])
    print("\n### pooled")
    print("  alt=%d self=%d middle=%d tag-excluded=%d  usable cells %d/%d"
          % (P["n_altruistic"], P["n_self_interested"], P["n_middle_discarded"],
             P["n_tag_excluded"], P["usable_cells"], P["n_cells_seen"]))
    print("  usable cells by game: %s" % P["usable_cells_by_family"])
    print("  pole mix by game:     %s" % {k: (v["altruistic"], v["self_interested"])
                                          for k, v in P["pole_mix_by_family"].items()})
    print("  LAYER 0 AUC  = %.4f   <-- the token-surface test" % bl[0]["auc"])
    print("  LAYER %d AUC = %.4f" % (LAYER, bl[LAYER]["auc"]))
    print("  best layer %d AUC = %.4f" % (best["layer"], best["auc"]))
    print("  norms @%d: unbalanced %.2f  cell-balanced %.2f"
          % (LAYER, P["norms_unbalanced"][LAYER], P["norms_balanced"][LAYER]))
    print("  AUC by layer: %s"
          % " ".join("%d:%.3f" % (r["layer"], r["auc"]) for r in bl))

    print("\n### leave-one-game-out: fit on the others, score the held-out game")
    print("  %-18s %6s %6s %9s %9s %9s %5s" % ("held out", "alt", "self", "L0 AUC",
                                               "L20 AUC", "best AUC", "@L"))
    for fam, e in R["leave_one_game_out"].items():
        if not e.get("usable"):
            print("  %-18s  --- not usable" % fam)
            continue
        bl2 = e["by_layer"]; b2 = max(bl2, key=lambda r: r["auc"])
        print("  %-18s %6d %6d %9.3f %9.3f %9.3f %5d"
              % (fam, e["n_held_altruistic"], e["n_held_self"], bl2[0]["auc"],
                 bl2[LAYER]["auc"], b2["auc"], b2["layer"]))

    for key, title in (("agreement_matrix_cellbalanced",
                        "agreement matrix, cell-balanced, layer %d" % LAYER),
                       ("agreement_matrix_layer0_cellbalanced",
                        "agreement matrix, cell-balanced, LAYER 0")):
        M = R[key]
        names = M["families"]
        print("\n### %s  (label-shuffled null in brackets: mean +/- sd, p97.5)" % title)
        print("      " + "".join("%9s" % fam_short(n) for n in names))
        for i, a in enumerate(names):
            row = "  %-4s" % fam_short(a)
            for j, b in enumerate(names):
                if i == j:
                    row += "%9s" % "1.000"
                else:
                    k = (a + "|" + b) if (a + "|" + b) in M["pairs"] else (b + "|" + a)
                    row += "%9.3f" % M["pairs"][k]["cosine"]
            print(row)
        print("  label-null per pair:")
        for k, v in M["pairs"].items():
            n = v["label_null"]
            print("    %-40s cos=%+.3f   null mean %+.3f sd %.3f  p2.5 %+.3f p97.5 %+.3f"
                  % (k, v["cosine"], n["mean"], n["sd"], n["p2.5"], n["p97.5"]))

    V = R["vs_existing"]
    print("\n### the pooled vector against the vectors that already exist (layer %d)" % LAYER)
    nd = V["label_null_vs_dictator_layer_%d" % LAYER]
    nt = V["label_null_vs_their_altruism_layer_%d" % LAYER]
    ns = V["label_null_vs_pooled_itself_layer_%d" % LAYER]
    print("  cos(pooled cell-balanced, archived Dictator-only) = %+.4f"
          % V["cos_pooled_balanced_vs_dictator"][LAYER])
    print("      label-null: mean %+.3f sd %.3f  p2.5 %+.3f p97.5 %+.3f"
          % (nd["mean"], nd["sd"], nd["p2.5"], nd["p97.5"]))
    print("  cos(pooled cell-balanced, their shipped altruism) = %+.4f"
          % V["cos_pooled_balanced_vs_their_altruism"][LAYER])
    print("      label-null: mean %+.3f sd %.3f  p2.5 %+.3f p97.5 %+.3f"
          % (nt["mean"], nt["sd"], nt["p2.5"], nt["p97.5"]))
    print("  label-null of pooled against ITSELF: mean %+.3f sd %.3f p97.5 %+.3f"
          % (ns["mean"], ns["sd"], ns["p97.5"]))
    print("  per game:")
    for fam, e in V["per_game"].items():
        print("    %-18s vs dictator %+.3f   vs their altruism %+.3f   vs pooled %+.3f"
              % (fam, e["cos_vs_dictator_vector"][LAYER],
                 e["cos_vs_their_altruism"][LAYER], e["cos_vs_pooled"][LAYER]))

    print("\n### prompt-side variants (expected void: causal masking)")
    for variant, e in R["prompt_side_check"].items():
        sh2 = e["split_half"]
        print("  %-12s L0 AUC %s  L%d AUC %s  |v|@%d %.2f"
              % (variant,
                 "%.3f" % sh2["by_layer"][0]["auc"] if sh2 else "n/a", LAYER,
                 "%.3f" % sh2["by_layer"][LAYER]["auc"] if sh2 else "n/a",
                 LAYER, e["norms"][LAYER]))
