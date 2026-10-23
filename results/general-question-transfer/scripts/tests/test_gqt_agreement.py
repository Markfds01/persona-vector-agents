"""Agreement statistics on synthetic labels whose answer is known."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "measurement"))

import agreement as ag  # noqa: E402


def test_bins():
    assert [ag.to_bin(x) for x in (0, 19.9, 20, 39, 40, 60, 79.9, 80, 100)] == \
        [0, 0, 1, 1, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError):
        ag.to_bin(101)


def test_perfect_agreement():
    a = [5, 25, 45, 65, 85, 10, 90]
    s = ag.pair_stats(a, list(a))
    assert s["spearman"] == pytest.approx(1.0)
    assert s["qw_kappa_binned"] == pytest.approx(1.0)
    assert s["krippendorff_alpha_interval_binned"] == pytest.approx(1.0)


def test_kappa_known_value():
    # 2x2 with weights (n_cat=2): quadratic == linear == unweighted kappa.
    a = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    b = [0, 1, 1, 1, 0, 1, 0, 0, 1, 1]
    # po = 0.8; pa0=.4, pb0=.4; pe = .4*.4 + .6*.6 = .52; kappa = (.8-.52)/.48
    assert ag.quadratic_weighted_kappa(a, b, n_cat=2) == pytest.approx((0.8 - 0.52) / 0.48)


def test_kappa_penalises_far_disagreement_more():
    near = ag.quadratic_weighted_kappa([0, 1, 2, 3, 4] * 4, [1, 1, 2, 3, 4] * 4)
    far = ag.quadratic_weighted_kappa([0, 1, 2, 3, 4] * 4, [4, 1, 2, 3, 4] * 4)
    assert near > far


def test_krippendorff_textbook_nominal():
    # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", the
    # 4-coder / 12-unit example with missing values: nominal alpha = 0.743.
    data = [  # coders A, B, C, D over units 1..12
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    units = [[data[c][u] for c in range(4)] for u in range(12)]
    assert ag.krippendorff_alpha(units, "nominal") == pytest.approx(0.743, abs=1e-3)


def test_krippendorff_interval_textbook():
    # Same data, interval metric: alpha = 0.849 (Krippendorff 2011).
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    units = [[data[c][u] for c in range(4)] for u in range(12)]
    assert ag.krippendorff_alpha(units, "interval") == pytest.approx(0.849, abs=1e-3)


def test_noise_lowers_agreement_and_ci_covers_point():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 100, size=200)
    close = np.clip(truth + rng.normal(0, 5, size=200), 0, 100)
    noisy = np.clip(truth + rng.normal(0, 40, size=200), 0, 100)
    good = ag.bootstrap_pair(list(truth), list(close), n_boot=200, seed=0)
    bad = ag.bootstrap_pair(list(truth), list(noisy), n_boot=200, seed=0)
    assert good["spearman"] > 0.95 > bad["spearman"]
    assert good["qw_kappa_binned"] > bad["qw_kappa_binned"]
    for res in (good, bad):
        for k in ("spearman", "qw_kappa_binned", "krippendorff_alpha_interval_binned"):
            lo, hi = res[k + "_ci"]
            assert lo <= res[k] <= hi


def test_missing_ratings_are_counted_not_imputed():
    s = ag.pair_stats([10, None, 50, 90], [12, 40, None, 88])
    assert s["n_items"] == 4 and s["n_paired"] == 2


def test_cli_reads_grader_export(tmp_path):
    a = tmp_path / "human.csv"
    b = tmp_path / "judge.csv"
    a.write_text("item_id,trait,coherence\nx1,10,90\nx2,50,80\nx3,90,70\nx4,,60\n")
    b.write_text("item_id,trait,coherence\nx1,15,90\nx2,45,80\nx3,95,75\nx5,50,50\n")
    out = tmp_path / "agree.json"
    ag.main(["--a", str(a), "--b", str(b), "--score", "trait", "--out", str(out),
             "--n-boot", "50"])
    import json
    result = json.loads(out.read_text())
    assert result["n_paired"] == 3 and result["only_in_a"] == 1 and result["only_in_b"] == 1
