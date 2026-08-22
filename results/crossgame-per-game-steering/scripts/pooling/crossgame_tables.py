"""The six games on one axis: the tables the report quotes, and the CSVs behind them.

Nothing is computed here that `analyze_game.py` did not already compute — this
stage only lays the six games side by side on the shared unit-beta ladder and
prints them. Keeping it separate is what lets the per-game numbers be checked one
game at a time without rendering anything.

What can honestly be put in one table is the POLE SHARE: same orientation and
same definition in every game. The own-measure column is printed beside it with
its unit, and never summed, averaged or compared across rows.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prompting"))
sys.path.insert(0, str(HERE.parents[3]))

import eval_games  # noqa: E402

SHORT = {"dictator": "dictator", "trust": "trust", "ultimatum": "ultimatum",
         "apology": "apology", "overfishing": "overfishing",
         "prisoners_dilemma": "prisoners_dil"}


def point_rows(analysis):
    """One flat row per (game, arm, k) — the table every other view is cut from."""
    policy = analysis["policy"]
    out = []
    for family, game in analysis["games"].items():
        for arm, block in game["arms"].items():
            for k, point in block["points"].items():
                pole = point["poles"][policy]
                measure = point["measure"]
                move = block["move_from_zero"].get(k, {})
                out.append({
                    "game": family,
                    "arm": arm,
                    "k": int(k),
                    "unit_beta": point["unit_beta"],
                    "raw_beta_equivalent": point["raw_beta_equivalent"],
                    "vector_layer_norm": point["vector_layer_norm"],
                    "n_rows": point["n_rows"],
                    "n_parsed": point["n_parsed"],
                    "parse_coverage": point["parse_coverage"],
                    "measure_units": game["measure_units"],
                    "mean": measure.get("mean"),
                    "sd": measure.get("sd"),
                    "ci_low": measure.get("ci_low"),
                    "ci_high": measure.get("ci_high"),
                    "p_altruistic": pole["altruistic"]["p"],
                    "p_altruistic_lo": pole["altruistic"]["lo"],
                    "p_altruistic_hi": pole["altruistic"]["hi"],
                    "p_self_interested": pole["self_interested"]["p"],
                    "p_middle": pole["middle"]["p"],
                    "mean_move_from_zero": move.get("mean", {}).get("diff"),
                    "mean_move_p": move.get("mean", {}).get("p"),
                    "p_altruistic_move_from_zero": move.get("altruistic", {}).get("diff"),
                    "p_altruistic_move_excludes_zero":
                        move.get("altruistic", {}).get("excludes_zero"),
                    "share_trigram_repeated_5x":
                        point["degeneracy"].get("share_trigram_repeated_5x"),
                })
    out.sort(key=lambda row: (row["game"], row["arm"], row["k"]))
    return out


def print_per_game(analysis):
    policy = analysis["policy"]
    print("### per game: the vector steered, its norm, and what the steering did")
    print("  %-14s %8s %7s  %-24s %-24s %s" % (
        "game", "|v|@20", "n/point", "P(altruistic) at k=-5,0,+5",
        "own measure at k=-5,0,+5", "units"))
    for family, game in analysis["games"].items():
        points = game["arms"]["decision"]["points"]
        norm = _first(points)["vector_layer_norm"]
        n = _first(points)["n_rows"]

        def share(k):
            entry = points.get(str(k))
            return None if entry is None else entry["poles"][policy]["altruistic"]["p"]

        def mean(k):
            entry = points.get(str(k))
            return None if entry is None else entry["measure"].get("mean")

        print("  %-14s %8.4f %7d  %-24s %-24s %s" % (
            SHORT[family], norm, n,
            " ".join(_share(share(k)) for k in (-5, 0, 5)),
            " ".join(_mean(mean(k)) for k in (-5, 0, 5)),
            game["measure_units"]))
    print()


def print_ladder(analysis, arm="decision"):
    policy = analysis["policy"]
    ks = sorted({int(k) for game in analysis["games"].values()
                 for k in game["arms"].get(arm, {}).get("points", {})})
    print("### P(altruistic pole), %s policy, %s arm — the six games on one unit-beta axis"
          % (policy, arm))
    header = "  %-14s" % "game" + "".join("%9s" % ("k=%+d" % k) for k in ks)
    print(header)
    for family, game in analysis["games"].items():
        points = game["arms"].get(arm, {}).get("points", {})
        cells = []
        for k in ks:
            entry = points.get(str(k))
            cells.append("%9s" % (_share(entry["poles"][policy]["altruistic"]["p"])
                                  if entry else "-"))
        print("  %-14s%s" % (SHORT[family], "".join(cells)))
    print()


def print_monotonicity(analysis):
    policy = analysis["policy"]
    print("### monotone in beta? (on P(altruistic), %s policy)" % policy)
    print("  %-14s %8s %10s %12s %s" % ("game", "rho", "direction", "strict", "reversals"))
    for family, game in analysis["games"].items():
        mono = game["monotonicity"][policy]
        alt = mono.get("altruistic", {})
        reversals = mono.get("significant_reversals_on_altruistic_share", [])
        print("  %-14s %8s %10s %12s %s" % (
            SHORT[family],
            "n/a" if alt.get("spearman_rho_vs_k") is None
            else "%+.3f" % alt["spearman_rho_vs_k"],
            alt.get("direction") or "-",
            alt.get("strictly_monotone"),
            "none" if not reversals
            else " ".join("%+d->%+d %.3f" % (r["from_k"], r["to_k"], r["diff"])
                          for r in reversals)))
    print()


def print_null(analysis):
    policy = analysis["policy"]
    print("### the real arm against its own shuffled-label null, at matched unit beta")
    print("  %-14s %5s %26s %26s" % ("game", "k", "P(alt) decision - null",
                                     "mean decision - null"))
    for family, game in analysis["games"].items():
        contrast = game.get("decision_vs_null", {})
        for k in sorted(contrast, key=int):
            entry = contrast[k]
            share = entry["altruistic_decision_minus_null"]
            mean = entry["mean_decision_minus_null"]
            print("  %-14s %5s %26s %26s" % (
                SHORT[family], k,
                "n/a" if share.get("diff") is None
                else "%+.3f [%+.3f,%+.3f]%s" % (share["diff"], share["lo"], share["hi"],
                                                "" if share["excludes_zero"] else " ns"),
                "n/a" if mean.get("diff") is None
                else "%+7.2f p=%s" % (mean["diff"], _p(mean.get("p")))))
    print()


def _p(value):
    """`None` where the test has none to give — two identical arms at beta=0."""
    return "n/a" if value is None else "%.1e" % value


def _first(points):
    return points[sorted(points, key=int)[0]]


def _share(value):
    return "-" if value is None else "%.3f" % value


def _mean(value):
    return "-" if value is None else "%.2f" % value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, help="analyze_game.py's JSON")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis).read_text())
    rows = point_rows(analysis)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("pole policy: %s   games: %d   points: %d\n"
          % (analysis["policy"], len(analysis["games"]), len(rows)))
    print_per_game(analysis)
    print_ladder(analysis, "decision")
    print_ladder(analysis, "shuffled-null")
    print_monotonicity(analysis)
    print_null(analysis)


if __name__ == "__main__":
    main()
