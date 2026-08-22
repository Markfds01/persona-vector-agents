"""Turn one game's row CSVs into the numbers its section of the report states.

Two measures per point, and the distinction between them is the whole reason this
file is not a mean-plotter:

* the game's OWN measure — dollars given, fish caught, P(Cooperate). It is what a
  reader wants to see, and it is NOT comparable across games: the units differ,
  and in Overfishing a HIGHER number is the less altruistic choice.
* the POLE SHARES — P(altruistic) and P(self-interested), under the same pole
  definitions the vectors were built from. Same orientation in every game, so
  these are what the cross-game table compares and what a monotonicity claim is
  made on.

Both pole policies are scored, from the same values, because reclassifying costs
nothing and the policy moved two of the six games' vectors a long way (Ultimatum
37 degrees, Overfishing 49). Strict is primary throughout, matching the vectors.

Unparsed answers keep their row and their tag and are never scored as zero; every
share is over parsed rows and carries its own n.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "prompting"))
sys.path.insert(0, str(HERE.parents[3]))

import eval_games  # noqa: E402
import stats  # noqa: E402

import poles  # noqa: E402

UNRESOLVED_TAGS = poles.UNRESOLVED_TAGS

#: The two arms every game runs, and which ladder each one covers.
ARMS = ("decision", "shuffled-null")


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_point(path, family, policy_names=poles.POLICIES):
    """Everything measured at one beta point of one arm of one game."""
    rows = read_rows(path)
    values, tags = [], {}
    for row in rows:
        tags[row["tag"]] = tags.get(row["tag"], 0) + 1
        if row["value"] != "":
            values.append(float(row["value"]))
    entry = {
        "rows_csv": path.name,
        "n_rows": len(rows),
        "n_parsed": len(values),
        "parse_coverage": len(values) / len(rows) if rows else None,
        "n_unresolved": sum(count for tag, count in tags.items()
                            if tag in UNRESOLVED_TAGS),
        "tags": dict(sorted(tags.items())),
        "measure": stats.describe(values),
        "degeneracy": stats.degeneracy([row["continuation"] for row in rows]),
        "poles": {},
    }
    for policy in policy_names:
        classified = [eval_games.classify(family, value, policy) for value in values]
        n = len(classified)
        entry["poles"][policy] = {
            "n_parsed": n,
            "altruistic": stats.wilson(classified.count(poles.ALT), n),
            "self_interested": stats.wilson(classified.count(poles.SELF), n),
            "middle": stats.wilson(classified.count(poles.MIDDLE), n),
        }
    first = rows[0] if rows else {}
    entry["steering"] = {column: first.get(column) for column in (
        "steer_coeff", "steer_layer", "steer_norm_mode", "steer_positions",
        "steer_vector", "steer_vector_sha256", "steer_vector_norm",
        "steer_module_path", "steer_delta_dtype")}
    entry["run"] = {column: first.get(column) for column in (
        "seed", "batch_size", "model_revision", "dtype", "attn_implementation",
        "prompt_sha256", "question_sha256")}
    entry["_values"] = values
    entry["_poles"] = {policy: [eval_games.classify(family, v, policy) for v in values]
                       for policy in policy_names}
    return entry


def pole_counts(point, policy, which):
    bucket = point["poles"][policy][which]
    return bucket["k"], bucket["n"]


def move_from_zero(points, policy):
    """Every point against this arm's own beta=0, on the mean and on both poles."""
    zero = points.get(0)
    if zero is None:
        return {}
    out = {}
    for k in sorted(points):
        point = points[k]
        entry = {"mean": stats.welch(point["measure"], zero["measure"])}
        for which in ("altruistic", "self_interested"):
            k1, n1 = pole_counts(point, policy, which)
            k0, n0 = pole_counts(zero, policy, which)
            entry[which] = stats.newcombe(k1, n1, k0, n0)
        out[k] = entry
    return out


def monotonicity(points, policy):
    """Is the effect monotone in beta, and where does it demonstrably reverse?

    Reported three ways, because a bare yes/no on 11 noisy points is not an
    answer. The rank correlation says which direction the ladder runs; the strict
    walk says whether the observed shares never step backwards; the violation list
    says which backward steps are larger than their own 95% interval — the only
    ones a reader should treat as real.
    """
    ks = sorted(points)
    if len(ks) < 3:
        return {"n_points": len(ks)}
    result = {"n_points": len(ks), "ks": ks}
    for which, series in (("altruistic",
                           [points[k]["poles"][policy]["altruistic"]["p"] for k in ks]),
                          ("measure", [points[k]["measure"].get("mean") for k in ks])):
        if any(value is None for value in series):
            result[which] = {"available": False}
            continue
        rho = stats.spearman([float(k) for k in ks], series)
        non_decreasing = all(b >= a for a, b in zip(series, series[1:]))
        non_increasing = all(b <= a for a, b in zip(series, series[1:]))
        result[which] = {
            "available": True,
            "series": series,
            "spearman_rho_vs_k": rho,
            "direction": (None if rho is None else ("increasing" if rho > 0 else
                                                    "decreasing" if rho < 0 else None)),
            "strictly_monotone": non_decreasing or non_increasing,
        }
    # Only the pole share gets significance-tested reversals: it is a share, so it
    # has an exact interval, and it is the series the report makes claims on.
    series = [points[k]["poles"][policy]["altruistic"]["p"] for k in ks]
    rho = result["altruistic"].get("spearman_rho_vs_k") if result["altruistic"].get(
        "available") else None
    expected = 1 if (rho or 0.0) >= 0 else -1
    violations = []
    for left, right in zip(ks, ks[1:]):
        a_k, a_n = pole_counts(points[left], policy, "altruistic")
        b_k, b_n = pole_counts(points[right], policy, "altruistic")
        step = stats.newcombe(b_k, b_n, a_k, a_n)
        if step["diff"] is None:
            continue
        against = (step["diff"] * expected) < 0
        if against and step["excludes_zero"]:
            violations.append({"from_k": left, "to_k": right, "diff": step["diff"],
                               "lo": step["lo"], "hi": step["hi"]})
    result["significant_reversals_on_altruistic_share"] = violations
    result["monotone_up_to_noise"] = not violations
    result["altruistic_series"] = series
    return result


def decision_vs_null(decision, null, policy):
    """The real arm against its matched null, at every beta both arms ran."""
    out = {}
    for k in sorted(set(decision) & set(null)):
        entry = {"mean_decision_minus_null": stats.welch(decision[k]["measure"],
                                                         null[k]["measure"])}
        for which in ("altruistic", "self_interested"):
            k1, n1 = pole_counts(decision[k], policy, which)
            k2, n2 = pole_counts(null[k], policy, which)
            entry[which + "_decision_minus_null"] = stats.newcombe(k1, n1, k2, n2)
        out[k] = entry
    return out


def analyze(family, rows_root, coefficients, policy):
    game = eval_games.by_family(family)
    arms = {}
    for arm in ARMS:
        points = {}
        for entry in coefficients:
            if entry["family"] != family or entry["arm"] != arm:
                continue
            path = Path(rows_root) / entry["rows_csv"]
            point = summarize_point(path, family)
            point["k"] = entry["k"]
            point["unit_beta"] = entry["unit_beta"]
            point["raw_beta_equivalent"] = entry["raw_beta_equivalent"]
            point["vector"] = entry["vector"]
            point["vector_layer_norm"] = entry["vector_layer_norm"]
            points[entry["k"]] = point
        if points:
            arms[arm] = points
    if "decision" not in arms:
        raise SystemExit("%s: no decision arm in the manifest" % family)

    result = {
        "family": family,
        "game_id": game.id,
        "measure_units": eval_games.MEASURE_UNITS[family],
        "altruistic_is_high_on_own_measure": eval_games.ALTRUISTIC_IS_HIGH[family],
        "pole_scale": game.pole_scale,
        "primary_policy": policy,
        "arms": {},
    }
    for arm, points in arms.items():
        result["arms"][arm] = {
            "points": {str(k): _public(point) for k, point in sorted(points.items())},
            "move_from_zero": {str(k): v for k, v
                               in move_from_zero(points, policy).items()},
        }
    result["monotonicity"] = {p: monotonicity(arms["decision"], p)
                              for p in poles.POLICIES}
    if "shuffled-null" in arms:
        result["beta0_identical_across_arms"] = _beta0_identical(rows_root, coefficients,
                                                                 family)
        result["decision_vs_null"] = {
            str(k): v for k, v in decision_vs_null(arms["decision"],
                                                   arms["shuffled-null"], policy).items()}
        result["null_move_from_zero"] = {
            str(k): v for k, v in move_from_zero(arms["shuffled-null"], policy).items()}
    return result


def _beta0_identical(rows_root, coefficients, family):
    """At beta=0 the delta is exactly zero for both vectors, so the two arms must
    have generated the same answers. Anything else means the arms are not sharing
    an origin and every "move from zero" in the report is measured against the
    wrong baseline. Returns None when one arm did not run beta=0."""
    paths = {}
    for entry in coefficients:
        if entry["family"] == family and entry["k"] == 0:
            paths[entry["arm"]] = Path(rows_root) / entry["rows_csv"]
    if set(paths) != set(ARMS):
        return None
    left = read_rows(paths["decision"])
    right = read_rows(paths["shuffled-null"])
    return (len(left) == len(right)
            and all(a["answer"] == b["answer"] for a, b in zip(left, right)))


def _public(point):
    return {key: value for key, value in point.items() if not key.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-root", required=True,
                    help="the rows/ directory the sweep wrote")
    ap.add_argument("--coefficients", required=True,
                    help="coefficients.csv the sweep wrote")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="strict", choices=list(poles.POLICIES))
    ap.add_argument("--games", default=",".join(eval_games.FAMILIES))
    args = ap.parse_args()

    with open(args.coefficients, encoding="utf-8", newline="") as handle:
        coefficients = [
            {"family": row["family"], "arm": row["arm"], "k": int(row["k"]),
             "unit_beta": float(row["unit_beta"]),
             "raw_beta_equivalent": float(row["raw_beta_equivalent"]),
             "vector": row["vector"],
             "vector_layer_norm": float(row["vector_layer_norm"]),
             "rows_csv": row["rows_csv"]}
            for row in csv.DictReader(handle)]

    families = [f.strip() for f in args.games.split(",") if f.strip()]
    present = {entry["family"] for entry in coefficients}
    missing = [f for f in families if f not in present]
    if missing:
        raise SystemExit("no rows for %s in %s" % (missing, args.coefficients))

    out = {"policy": args.policy, "games": {}}
    for family in families:
        out["games"][family] = analyze(family, args.rows_root, coefficients,
                                       args.policy)
        headline = out["games"][family]["monotonicity"][args.policy]
        print("%-20s points=%d  monotone_up_to_noise=%s  rho=%s"
              % (family, headline.get("n_points"),
                 headline.get("monotone_up_to_noise"),
                 _fmt(headline.get("altruistic", {}).get("spearman_rho_vs_k"))),
              flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


def _fmt(value):
    return "n/a" if value is None else "%+.3f" % value


if __name__ == "__main__":
    main()
