"""The statistics this package reports, in the standard library and nothing else.

Small on purpose. Every function here answers one question the report asks, and
each is the estimator that stays valid where this data actually lives: shares
near 0 and 1 (three of the six games park their poles on one value), and groups
whose variances differ by an order of magnitude across the beta ladder.

* means -> Student's t interval, and Welch for a difference of two of them. The
  ladder's extremes have an SD several times the baseline's, so a pooled-variance
  test would be wrong in the direction that matters.
* shares -> Wilson, and Newcombe's hybrid-score interval for a difference. A Wald
  interval on P(gives exactly $0) = 0.985 is nonsense; both of these are not.

No p-value here is corrected for multiplicity, and the report says so where it
quotes one.
"""

import math
import re

Z975 = 1.959963984540054


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
                          + b * math.log1p(-x) + a * math.log(x)) \
        * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t on `df` degrees of freedom."""
    x = df / (df + t * t)
    half = 0.5 * betainc(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def t_ppf975(df):
    """The 0.975 quantile of Student's t, by bisection on the survival function."""
    low, high = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if t_sf(mid, df) > 0.025:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def describe(values):
    """n, mean, sd, se, a 95% t interval and the median. `n < 2` gets no interval."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "se": None, "ci_low": None,
                "ci_high": None, "median": mean}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    ordered = sorted(values)
    median = (ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))
    crit = t_ppf975(n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci_low": mean - crit * se,
            "ci_high": mean + crit * se, "median": median}


def welch(a, b):
    """`a["mean"] - b["mean"]` with a Welch 95% interval and a two-sided p."""
    if a.get("n", 0) < 2 or b.get("n", 0) < 2:
        return {"diff": None}
    diff = a["mean"] - b["mean"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    if se == 0.0:
        return {"diff": diff, "se": 0.0, "df": None, "ci_low": diff, "ci_high": diff,
                "p": None}
    numerator = (a["se"] ** 2 + b["se"] ** 2) ** 2
    denominator = a["se"] ** 4 / (a["n"] - 1) + b["se"] ** 4 / (b["n"] - 1)
    df = numerator / denominator
    crit = t_ppf975(df)
    t = diff / se
    return {"diff": diff, "se": se, "df": df, "ci_low": diff - crit * se,
            "ci_high": diff + crit * se, "t": t, "p": 2.0 * t_sf(abs(t), df)}


def wilson(successes, n, z=Z975):
    """A share and its Wilson 95% interval. Valid at 0 and 1, where Wald is not."""
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "k": successes, "n": 0}
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return {"p": p, "lo": centre - half, "hi": centre + half, "k": successes, "n": n}


def newcombe(k1, n1, k2, n2):
    """`p1 - p2` with Newcombe's hybrid-score interval, built from two Wilsons."""
    if n1 == 0 or n2 == 0:
        return {"diff": None}
    p1, p2 = k1 / n1, k2 / n2
    a = wilson(k1, n1)
    b = wilson(k2, n2)
    low = (p1 - p2) - math.sqrt((p1 - a["lo"]) ** 2 + (b["hi"] - p2) ** 2)
    high = (p1 - p2) + math.sqrt((a["hi"] - p1) ** 2 + (p2 - b["lo"]) ** 2)
    return {"diff": p1 - p2, "lo": low, "hi": high,
            "excludes_zero": not (low <= 0.0 <= high)}


def spearman(xs, ys):
    """Rank correlation, with midranks for ties. None when either side is constant."""
    n = len(xs)
    if n < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((v - mx) ** 2 for v in rx)
    syy = sum((v - my) ** 2 for v in ry)
    if sxx == 0.0 or syy == 0.0:
        return None
    sxy = sum((u - mx) * (v - my) for u, v in zip(rx, ry))
    return sxy / math.sqrt(sxx * syy)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        midrank = 0.5 * (position + end) + 1.0
        for i in range(position, end + 1):
            ranks[order[i]] = midrank
        position = end + 1
    return ranks


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NON_LATIN = re.compile(r"[Ͱ-᳿⺀-￯]")


def degeneracy(texts):
    """Surface statistics that go bad when a large edit breaks the text.

    NOT a coherence rating — no judge is involved. Length, self-repetition and
    script drift are what actually degrade under a large activation edit and can
    be counted for free; the share the scorer could not read is counted from tags
    by the caller.
    """
    if not texts:
        return {"n": 0}
    chars, words, trigram_ratios, repeats, non_latin = [], [], [], 0, 0
    for text in texts:
        chars.append(len(text))
        tokens = _WORD.findall(text.lower())
        words.append(len(tokens))
        if len(tokens) >= 5:
            grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
            trigram_ratios.append(len(set(grams)) / len(grams))
            counts = {}
            for gram in grams:
                counts[gram] = counts.get(gram, 0) + 1
            if max(counts.values()) >= 5:
                repeats += 1
        if _NON_LATIN.search(text):
            non_latin += 1
    return {
        "n": len(texts),
        "mean_chars": sum(chars) / len(chars),
        "mean_words": sum(words) / len(words),
        "mean_distinct_trigram_ratio": (sum(trigram_ratios) / len(trigram_ratios)
                                        if trigram_ratios else None),
        "share_trigram_repeated_5x": repeats / len(texts),
        "share_with_non_latin_script": non_latin / len(texts),
    }
