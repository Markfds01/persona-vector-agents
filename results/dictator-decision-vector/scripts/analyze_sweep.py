"""Turn the three arms of rows into the numbers the report states.

Three arms are read: our decision vector at 11 matched intervention sizes, their
altruism vector at the same 11 (the fleet's existing sweep, n=200, same game, same
pipeline, same seed), and the shuffled-label null at three of them.

Everything is computed from the rows. Unparsed answers are a reported category and
are never scored as zero. No LLM judge is involved anywhere, so "coherence" here is
a set of measured surface degeneracy statistics, named as such and not a rating.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

# ---------------------------------------------------------------- statistics


def _betacf(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                          + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t."""
    x = df / (df + t * t)
    half = 0.5 * betainc(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def t_ppf975(df):
    """The 0.975 quantile of Student's t, by bisection on the survival function."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > 0.025:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def describe(values):
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "se": None, "ci_low": None,
                "ci_high": None, "median": mean}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    crit = t_ppf975(n - 1)
    ordered = sorted(values)
    median = (ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "ci_low": mean - crit * se, "ci_high": mean + crit * se, "median": median}


def welch(a, b):
    """Difference of means a - b with a Welch 95% CI and two-sided p."""
    if a["n"] < 2 or b["n"] < 2:
        return {"diff": None}
    diff = a["mean"] - b["mean"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    if se == 0.0:
        return {"diff": diff, "se": 0.0, "df": None, "ci_low": diff, "ci_high": diff,
                "p": None}
    num = (a["se"] ** 2 + b["se"] ** 2) ** 2
    den = a["se"] ** 4 / (a["n"] - 1) + b["se"] ** 4 / (b["n"] - 1)
    df = num / den
    crit = t_ppf975(df)
    t = diff / se
    return {"diff": diff, "se": se, "df": df, "ci_low": diff - crit * se,
            "ci_high": diff + crit * se, "t": t, "p": 2.0 * t_sf(abs(t), df)}


def ols(xs, ys):
    """Slope and its 95% CI, over individual observations."""
    n = len(xs)
    if n < 3:
        return {"slope": None}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"slope": None}
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx)
    crit = t_ppf975(n - 2)
    return {"slope": slope, "se": se, "ci_low": slope - crit * se,
            "ci_high": slope + crit * se, "n": n,
            "p": 2.0 * t_sf(abs(slope / se), n - 2) if se > 0 else None}


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


# ------------------------------------------------------------- degeneracy


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NON_LATIN = re.compile(r"[Ͱ-᳿⺀-￯]")


def degeneracy(texts):
    """Surface statistics that go bad when steering breaks the text.

    NOT a coherence rating: no judge is involved. These are the four things that
    actually degrade under a large activation edit and can be counted for free -
    length, self-repetition, script drift, and the share the scorer could not read
    (counted by the caller from tags).
    """
    if not texts:
        return {"n": 0}
    chars, words, tri_ratio, repeats, nonlatin = [], [], [], 0, 0
    for text in texts:
        chars.append(len(text))
        tokens = _WORD.findall(text.lower())
        words.append(len(tokens))
        if len(tokens) >= 5:
            grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
            ratio = len(set(grams)) / len(grams)
            tri_ratio.append(ratio)
            counts = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            if max(counts.values()) >= 5:
                repeats += 1
        if _NON_LATIN.search(text):
            nonlatin += 1
    return {
        "n": len(texts),
        "mean_chars": sum(chars) / len(chars),
        "mean_words": sum(words) / len(words),
        "mean_distinct_trigram_ratio": (sum(tri_ratio) / len(tri_ratio)) if tri_ratio else None,
        "share_trigram_repeated_5x": repeats / len(texts),
        "share_with_non_latin_script": nonlatin / len(texts),
    }


# ------------------------------------------------------------------- loading


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_point(path, pot=100.0):
    rows = read_rows(path)
    values, tags = [], {}
    for row in rows:
        tags[row["tag"]] = tags.get(row["tag"], 0) + 1
        if row["value"] != "":
            values.append(float(row["value"]))
    stats = describe(values)
    unresolved = sum(count for tag, count in tags.items()
                     if tag in ("empty", "refusal", "unparsed"))
    modes = {
        "share_zero": sum(1 for v in values if v == 0.0) / len(values) if values else None,
        "share_half": sum(1 for v in values if v == pot / 2) / len(values) if values else None,
        "share_full": sum(1 for v in values if v == pot) / len(values) if values else None,
    }
    entry = {"rows_csv": str(path), "n_rows": len(rows), "n_parsed": len(values),
             "parse_coverage": len(values) / len(rows) if rows else None,
             "n_unresolved": unresolved, "tags": dict(sorted(tags.items())),
             "modes": modes, "degeneracy": degeneracy([r["continuation"] for r in rows])}
    entry.update(stats)
    entry["values"] = values
    first = rows[0] if rows else {}
    for column in ("steer_coeff", "steer_norm_mode", "steer_vector",
                   "steer_vector_sha256", "steer_vector_norm", "steer_layer",
                   "steer_module_path", "steer_positions", "attn_implementation",
                   "dtype", "model_revision", "seed", "batch_size"):
        if column in first:
            entry[column] = first[column]
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-dir", required=True)
    ap.add_argument("--null-dir", required=True)
    ap.add_argument("--theirs-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    by_arm = {}
    for entry in manifest:
        by_arm.setdefault(entry["arm"], []).append(entry)

    result = {"arms": {}, "comparisons": {}}

    for arm, entries in by_arm.items():
        points = {}
        for entry in entries:
            k = int(entry["their_raw_beta"])
            point = summarize_point(Path(entry["rows_csv"]))
            point["their_raw_beta"] = k
            point["unit_beta"] = float(entry["unit_beta"])
            point["our_raw_beta_equivalent"] = float(entry["our_raw_beta_equivalent"])
            point["vector_sha256_16"] = entry["vector_sha256_16"]
            point["vector_layer_norm"] = float(entry["vector_layer_norm"])
            points[k] = point
        result["arms"][arm] = points

    theirs = {}
    for k in range(-5, 6):
        path = Path(args.theirs_dir) / ("altruism_v3-dictator_free_layer20_all_coef%d.csv" % k)
        if not path.is_file():
            raise SystemExit("missing their rows at %s" % path)
        point = summarize_point(path)
        point["their_raw_beta"] = k
        point["unit_beta"] = None       # filled below from the measured norm
        theirs[k] = point
    result["arms"]["theirs"] = theirs

    decision = result["arms"]["decision"]
    for k, point in theirs.items():
        match = decision.get(k)
        point["unit_beta"] = match["unit_beta"] if match else None

    # ---- matched-size comparison, per coefficient --------------------------
    matched = {}
    for k in sorted(decision):
        matched[k] = {
            "their_raw_beta": k,
            "unit_beta": decision[k]["unit_beta"],
            "our_raw_beta_equivalent": decision[k]["our_raw_beta_equivalent"],
            "decision": {key: decision[k].get(key)
                         for key in ("n", "mean", "sd", "se", "ci_low", "ci_high",
                                     "median", "n_parsed", "parse_coverage")},
            "theirs": {key: theirs[k].get(key)
                       for key in ("n", "mean", "sd", "se", "ci_low", "ci_high",
                                   "median", "n_parsed", "parse_coverage")},
            "difference_decision_minus_theirs": welch(decision[k], theirs[k]),
        }
    result["comparisons"]["matched"] = matched

    null = result["arms"].get("shuffled-null", {})
    null_cmp = {}
    for k in sorted(null):
        null_cmp[k] = {
            "their_raw_beta": k,
            "unit_beta": null[k]["unit_beta"],
            "null": {key: null[k].get(key) for key in ("n", "mean", "sd", "se",
                                                       "ci_low", "ci_high", "n_parsed",
                                                       "parse_coverage")},
            "null_minus_decision": welch(null[k], decision[k]) if k in decision else None,
            "null_minus_theirs": welch(null[k], theirs[k]) if k in theirs else None,
        }
        if k in decision:
            null_cmp[k]["decision_move_from_zero"] = welch(decision[k], decision[0])
            null_cmp[k]["null_move_from_zero"] = welch(null[k], null[0])
            null_cmp[k]["theirs_move_from_zero"] = welch(theirs[k], theirs[0])
    result["comparisons"]["null"] = null_cmp

    # ---- slopes -------------------------------------------------------------
    def arm_slopes(points, name):
        out = {}
        for label, keys in (("full", sorted(points)),
                            ("positive_arm", [k for k in sorted(points) if k >= 0]),
                            ("negative_arm", [k for k in sorted(points) if k <= 0])):
            xs, ys = [], []
            for k in keys:
                for v in points[k]["values"]:
                    xs.append(float(k))
                    ys.append(v)
            out[label] = ols(xs, ys)
            out[label + "_n_points"] = len(keys)
        # per-unit-beta slope: the per-k slope divided by the unit beta of one k step
        per_k = None
        for k in sorted(points):
            if k != 0 and points[k].get("unit_beta"):
                per_k = abs(points[k]["unit_beta"] / k)
                break
        if per_k:
            for label in ("full", "positive_arm", "negative_arm"):
                s = out[label]
                if s.get("slope") is not None:
                    s["slope_per_unit_beta"] = s["slope"] / per_k
        out["means_by_k"] = {k: points[k]["mean"] for k in sorted(points)}
        out["argmin_negative_arm"] = min([k for k in sorted(points) if k <= 0],
                                         key=lambda k: points[k]["mean"])
        neg = [k for k in sorted(points) if k <= 0]
        # monotone decreasing as beta goes down?  walk from 0 to the most negative
        walk = sorted(neg, reverse=True)
        out["negative_arm_monotone_decreasing"] = all(
            points[walk[i + 1]]["mean"] <= points[walk[i]]["mean"]
            for i in range(len(walk) - 1))
        out["name"] = name
        return out

    result["slopes"] = {name: arm_slopes(points, name)
                        for name, points in result["arms"].items() if len(points) > 2}

    # ---- curve correlation --------------------------------------------------
    ks = sorted(decision)
    result["curve_correlation_decision_vs_theirs"] = pearson(
        [decision[k]["mean"] for k in ks], [theirs[k]["mean"] for k in ks])

    # ---- the beta=0 cross-arm identity check --------------------------------
    if 0 in decision and 0 in null:
        a = read_rows(decision[0]["rows_csv"])
        b = read_rows(null[0]["rows_csv"])
        same = (len(a) == len(b)
                and all(x["answer"] == y["answer"] for x, y in zip(a, b)))
        result["beta0_identical_across_arms"] = bool(same)

    for arm in result["arms"].values():
        for point in arm.values():
            point.pop("values", None)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["comparisons"]["matched"], indent=2)[:4000])


if __name__ == "__main__":
    main()
