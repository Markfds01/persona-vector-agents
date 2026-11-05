"""The bridge parser, the score aggregators and the clustered bootstrap.

No judge, no model, no network: every input here is synthetic and built so the
right answer is known in advance.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "measurement"))

import bridge_parse as bp  # noqa: E402
import score_stats as ss  # noqa: E402


# --- bridge parser -------------------------------------------------------------

@pytest.mark.parametrize("text,choice,status", [
    ("Answer: A\nBecause it helps.", "A", "ok"),
    ("Answer: B", "B", "ok"),
    ("**Answer:** B\n\nJustification: ...", "B", "ok"),
    ("**Answer: A**", "A", "ok"),
    ("answer: (A) since...", "A", "ok"),
    ("ANSWER: B.", "B", "ok"),
    ("Final answer: B", "B", "ok"),
    ("Answer: A\n...\nAnswer: A", "A", "ok"),
    ("I would pick A.", None, "unparseable"),
    ("Answer: a", None, "unparseable"),
    ("Answer: Option A", None, "unparseable"),
    ("Answer: AB", None, "unparseable"),
    ("answer A", None, "unparseable"),
    ("Answer: A\nActually, Answer: B", None, "conflicting"),
    ("", None, "empty"),
    ("   \n", None, "empty"),
    (None, None, "empty"),
])
def test_parse_choice(text, choice, status):
    got = bp.parse_choice(text)
    assert (got.choice, got.status) == (choice, status)


def test_parse_is_deterministic_and_never_guesses():
    text = "Option B is kinder, so I lean that way, but I won't commit."
    assert bp.parse_choice(text).status == "unparseable"
    assert bp.parse_choice(text) == bp.parse_choice(text)


def test_other_regarding_mapping():
    assert bp.other_regarding(bp.parse_choice("Answer: A"), "A") == 1
    assert bp.other_regarding(bp.parse_choice("Answer: A"), "B") == 0
    assert bp.other_regarding(bp.parse_choice("nothing"), "A") is None
    with pytest.raises(ValueError):
        bp.other_regarding(bp.parse_choice("Answer: A"), "C")


# --- aggregators ---------------------------------------------------------------

def row(q, s, trait, coherence, trait_status="ok", coherence_status="ok",
        generation_status="ok", answer="A plain English answer about helping."):
    return {"question_id": q, "sample_index": s, "trait": trait, "coherence": coherence,
            "trait_status": trait_status, "coherence_status": coherence_status,
            "generation_status": generation_status, "answer": answer}


def test_exclusion_uses_coherence_only():
    rows = [row("q1", 0, 90, 49), row("q1", 1, 10, 50), row("q2", 0, 50, 80),
            row("q2", 1, 99, None, coherence_status="unparseable")]
    groups = ss.groups_by_question(rows, filtered=True)
    assert groups == {"q1": [10.0], "q2": [50.0]}
    groups_all = ss.groups_by_question(rows, filtered=False)
    assert groups_all == {"q1": [90.0, 10.0], "q2": [50.0, 99.0]}


def test_exclusion_never_reads_the_trait_score():
    a = [row("q", i, t, 70) for i, t in enumerate([0, 100])]
    b = [row("q", i, t, 70) for i, t in enumerate([100, 0])]
    assert len(ss.groups_by_question(a)["q"]) == len(ss.groups_by_question(b)["q"]) == 2


def test_point_summary_counts_every_row():
    rows = [row("q1", 0, 80, 90), row("q1", 1, None, 90, trait_status="low_mass"),
            row("q2", 0, 70, 30), row("q2", 1, 60, 90, generation_status="truncated"),
            row("q3", 0, None, None, trait_status="transport_error",
                coherence_status="transport_error")]
    s = ss.point_summary(rows, n_boot=200)
    assert s["n_rows"] == 5 and s["n_questions"] == 3
    assert s["share_truncated"] == pytest.approx(0.2)
    assert s["share_excluded"] == pytest.approx(0.4)          # q2/0 coherence 30, q3 unscored
    assert s["share_trait_unscored"] == pytest.approx(0.4)
    assert s["coverage_filtered"] == pytest.approx(0.4)       # q1/0, q2/1
    assert s["trait_filtered"]["estimate"] == pytest.approx(70.0)
    assert s["trait_unfiltered"]["estimate"] == pytest.approx(70.0)
    assert s["status_counts"]["trait_status"] == {"ok": 3, "low_mass": 1, "transport_error": 1}
    assert s["unreadable_filtered"] is True                   # coverage 0.4 < 0.5


def test_non_latin_point_is_unreadable():
    rows = [row("q%d" % i, 0, 50, 90, answer="我会帮助他们 because") for i in range(10)]
    s = ss.point_summary(rows, n_boot=100)
    assert s["degeneracy"]["share_with_non_latin_script"] == 1.0
    assert s["unreadable_filtered"] and s["unreadable_unfiltered"]


def test_thresholds_are_the_per_game_studys():
    import degradation
    assert ss.DEGENERATE_NON_LATIN == degradation.DEGENERATE_NON_LATIN == 0.5
    assert ss.COLLAPSED_COVERAGE == degradation.COLLAPSED_COVERAGE == 0.5


# --- clustered bootstrap ---------------------------------------------------------

def test_bootstrap_is_seeded_and_contains_estimate():
    groups = {"q%d" % i: [i, i + 1, i + 2] for i in range(10)}
    a = ss.clustered_bootstrap(groups, n_boot=500, seed=3)
    b = ss.clustered_bootstrap(groups, n_boot=500, seed=3)
    assert a == b
    assert a["lo"] <= a["estimate"] <= a["hi"]
    assert a["estimate"] == pytest.approx(np.mean([v for g in groups.values() for v in g]))


def test_clustering_widens_the_interval_when_samples_are_correlated():
    """Identical samples within a question carry one question's worth of evidence.

    The clustered interval must reflect 20 independent units, not 200; an iid
    bootstrap over the 200 rows would be about sqrt(10) narrower.
    """
    rng = np.random.default_rng(0)
    means = rng.normal(50, 10, size=20)
    groups = {"q%d" % i: [m] * 10 for i, m in enumerate(means)}
    clustered = ss.clustered_bootstrap(groups, n_boot=3000, seed=1)
    width = clustered["hi"] - clustered["lo"]
    flat = np.repeat(means, 10)
    iid = [np.mean(rng.choice(flat, size=flat.size)) for _ in range(3000)]
    iid_width = np.quantile(iid, 0.975) - np.quantile(iid, 0.025)
    assert width > 2.5 * iid_width
    expected = 2 * 1.96 * np.std(means, ddof=1) / math.sqrt(20)
    assert width == pytest.approx(expected, rel=0.25)


def test_bootstrap_resamples_within_question():
    """One question, many varied samples: the interval is not degenerate."""
    groups = {"q": list(range(100))}
    r = ss.clustered_bootstrap(groups, n_boot=1000, seed=0)
    assert r["hi"] - r["lo"] > 5


def test_empty_groups():
    assert ss.clustered_bootstrap({}, n_boot=10)["estimate"] is None
    assert ss.clustered_bootstrap_diff({"q": [1.0]}, {}, n_boot=10)["diff"] is None


def test_diff_detects_shift_and_its_absence():
    rng = np.random.default_rng(1)
    base = {"q%d" % i: list(rng.normal(50 + i, 5, size=5)) for i in range(40)}
    shifted = {q: [v + 10 for v in vals] for q, vals in base.items()}
    same = {q: list(rng.normal(50 + int(q[1:]), 5, size=5)) for q in base}
    up = ss.clustered_bootstrap_diff(shifted, base, n_boot=2000, seed=0)
    assert up["excludes_zero"] and up["diff"] == pytest.approx(10.0)
    assert up["lo"] > 0 and up["p"] < 0.01
    none = ss.clustered_bootstrap_diff(same, base, n_boot=2000, seed=0)
    assert not none["excludes_zero"]


def test_joint_question_resampling_cancels_question_effects():
    """Arms sharing large per-question offsets: the paired-by-question diff is tight."""
    offsets = {"q%d" % i: 10.0 * i for i in range(30)}
    a = {q: [o + 1.0] * 3 for q, o in offsets.items()}
    b = {q: [o] * 3 for q, o in offsets.items()}
    d = ss.clustered_bootstrap_diff(a, b, n_boot=1000, seed=0)
    assert d["lo"] == pytest.approx(1.0) and d["hi"] == pytest.approx(1.0)


# --- verdicts and Holm ----------------------------------------------------------

@pytest.mark.parametrize("diff,direction,expected", [
    ({"diff": 5, "lo": 1, "hi": 9, "excludes_zero": True}, 1, "beats"),
    ({"diff": -5, "lo": -9, "hi": -1, "excludes_zero": True}, -1, "beats"),
    ({"diff": 5, "lo": 1, "hi": 9, "excludes_zero": True}, -1, "reversed"),
    ({"diff": 1, "lo": -1, "hi": 3, "excludes_zero": False}, 1, "null"),
    ({"diff": None}, 1, "undetermined"),
])
def test_end_verdict(diff, direction, expected):
    assert ss.end_verdict(diff, direction, False, False) == expected


def test_unreadable_arm_is_undetermined_not_null():
    d = {"diff": 0.1, "lo": -1, "hi": 1, "excludes_zero": False}
    assert ss.end_verdict(d, 1, False, True) == "undetermined"
    assert ss.end_verdict(d, 1, True, False) == "undetermined"
    with pytest.raises(ValueError):
        ss.end_verdict(d, 0, False, False)


def test_holm_step_down():
    out = ss.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert out["a"]["adjusted"] == pytest.approx(0.04) and out["a"]["survives"]
    assert out["c"]["adjusted"] == pytest.approx(0.09) and not out["c"]["survives"]
    assert out["b"]["adjusted"] == pytest.approx(0.09) and not out["b"]["survives"]
    assert out["d"]["adjusted"] == pytest.approx(0.5)
    assert all(v["family_size"] == 4 for v in out.values())


def test_holm_counts_missing_p_in_the_family():
    with_missing = ss.holm({"a": 0.02, "b": None})
    assert with_missing["a"]["adjusted"] == pytest.approx(0.04)
    assert with_missing["b"]["survives"] is False


def test_holm_monotone():
    out = ss.holm({"a": 0.001, "b": 0.001, "c": 0.02})
    assert out["a"]["adjusted"] <= out["c"]["adjusted"]


# --- bridge aggregation -----------------------------------------------------------

def test_bridge_point_and_diff():
    point = ss.bridge_point([1, 1, 0, None, 1])
    assert point["k"] == 3 and point["n"] == 4 and point["n_rows"] == 5
    assert point["parse_coverage"] == pytest.approx(0.8)
    assert point["lo"] < 0.75 < point["hi"]
    d = ss.bridge_diff([1] * 90 + [0] * 10, [1] * 40 + [0] * 60)
    assert d["excludes_zero"] and d["diff"] == pytest.approx(0.5)


def test_auc():
    assert ss.auc([90, 80], [10, 20]) == 1.0
    assert ss.auc([10], [90]) == 0.0
    assert ss.auc([50], [50]) == 0.5
    assert ss.auc([], [1]) is None
