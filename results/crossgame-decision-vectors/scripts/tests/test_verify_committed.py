"""The reproduction gate, exercised with no activations, no weights and no GPU.

`verify_committed.py` decides whether the vectors committed beside it reproduce,
and until this file existed nothing tested it. `reproduction_gate` is a pure
function over the report dict the run assembles and `committed_comparison` needs
only two tensors, so both are checkable here: the pass, both failure modes, the
refusal to pass having compared nothing, and the NaN behaviour.

The tolerance this revision widened is checked against DATA rather than prose.
The committed `.pt` files are float32 and the recomputation is float64, so the
last two tests exhibit a float64 vector that stores to exactly the committed bits
and still moves the norm by 3e-8, and then read the drifts the committed
verification reports actually recorded.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
DIR = SCRIPTS.parent
VECTORS = DIR / "vectors"
ANALYSIS = DIR / "analysis"

#: the gate's own defaults, which are what the committed reports were produced under
MIN_COSINE = 1 - 1e-9
MAX_NORM_DRIFT = 1e-6


def _load_module():
    """Import the script by path: it is a directory's tool, not an installed module."""
    spec = importlib.util.spec_from_file_location("verify_committed",
                                                  SCRIPTS / "verify_committed.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_committed", module)
    spec.loader.exec_module(module)
    return module


verify = _load_module()


def report(per_game=None, pooled=None):
    return {"A_per_game_vs_committed": dict(per_game or {}),
            "A2_pooled_vs_committed": dict(pooled or {})}


def entry(cos=1.0, drift=0.0, layer=20):
    """One vector's comparison record, as `committed_comparison` writes it."""
    return {"cos_layer20": cos, "min_cos_over_layers": cos,
            "norm_layer20": 1.0, "norm_layer20_committed": 1.0,
            "norm_ratio_layer20": 1.0 + drift,
            "worst_norm_drift": drift, "worst_norm_drift_layer": layer,
            "norm_at_worst_drift": 1.0 + drift,
            "norm_at_worst_drift_committed": 1.0}


def gate(out, min_cosine=MIN_COSINE, max_norm_drift=MAX_NORM_DRIFT):
    return verify.reproduction_gate(out, min_cosine, max_norm_drift)


# --- the gate's verdicts ---------------------------------------------------------

def test_a_reproducing_corpus_passes():
    result = gate(report(per_game={"dictator": entry(cos=1.0 - 1e-15, drift=4e-9)},
                         pooled={"cell_balanced": entry()}))
    assert result["passed"] and result["failures"] == []
    assert result["n_vectors_checked"] == 2
    assert result["worst_norm_drift_over_layers"] == 4e-9


def test_a_scale_error_fails_on_the_norm_no_cosine_can_see():
    """A vector twice as long reads cosine 1.000000; the norm is the only witness."""
    result = gate(report(per_game={"dictator": entry(cos=1.0, drift=1.0)}))
    assert not result["passed"]
    assert "layer-20 norm" in result["failures"][0]


def test_a_scale_error_away_from_layer_20_fails_too():
    """Gating layer 20 alone let a doubled layer 5 through at 1.000000 on both
    numbers, while sections 1, 3 and 5 are read at layer 0."""
    result = gate(report(per_game={"dictator": entry(cos=1.0, drift=1.0, layer=5)}))
    assert not result["passed"]
    assert "layer-5 norm" in result["failures"][0]


def test_a_turned_vector_fails_on_the_cosine():
    result = gate(report(pooled={"cell_balanced": entry(cos=0.3)}))
    assert not result["passed"]
    assert "worst layer cosine" in result["failures"][0]


def test_a_gate_over_nothing_does_not_pass():
    """Until this guard existed, an empty section reported `passed` proving nothing."""
    result = gate(report())
    assert not result["passed"]
    assert result["n_vectors_checked"] == 0
    assert "proves nothing" in result["failures"][0]


def test_a_vector_that_could_not_be_compared_is_a_failure():
    result = gate(report(per_game={"dictator": {"error": "not committed: x.pt"}}))
    assert not result["passed"]
    assert "not committed" in result["failures"][0]


def test_a_nan_never_passes_either_check():
    """`not (worst >= floor)` and `not (drift <= max)` are written that way on
    purpose: a NaN would satisfy the reversed comparisons."""
    nan = float("nan")
    assert not gate(report(per_game={"a": entry(cos=nan)}))["passed"]
    assert not gate(report(per_game={"a": entry(drift=nan)}))["passed"]


def test_the_margins_are_reported_not_only_the_thresholds():
    result = gate(report(per_game={"a": entry(cos=1.0 - 2e-15, drift=1e-9),
                                   "b": entry(cos=1.0 - 5e-15, drift=3e-9)}))
    assert result["worst_min_cos_over_layers"] == 1.0 - 5e-15
    assert result["worst_norm_drift_over_layers"] == 3e-9


# --- the comparison the gate reads ----------------------------------------------

COMMITTED = VECTORS / "decision_dictator_response_avg_diff_cellbalanced_strict.pt"


def committed_vector():
    if not COMMITTED.is_file():
        pytest.skip("%s is not in this checkout" % COMMITTED.name)
    return torch.load(COMMITTED, map_location="cpu")


def test_a_vector_compared_against_itself_reproduces_exactly():
    """Which also pins what `--min-cosine`'s help calls float64 reassociation noise."""
    committed = committed_vector()
    got = verify.committed_comparison(committed.double(), committed)
    assert 1.0 - got["min_cos_over_layers"] < 1e-14
    assert got["min_cos_over_layers"] >= MIN_COSINE
    assert got["worst_norm_drift"] == 0.0


def test_the_comparison_finds_a_scale_error_on_any_layer():
    committed = committed_vector()
    for layer in (0, 5, 20, 28):
        recomputed = committed.double().clone()
        recomputed[layer] *= 2.0
        got = verify.committed_comparison(recomputed, committed)
        assert got["worst_norm_drift_layer"] == layer
        assert got["worst_norm_drift"] == pytest.approx(1.0)


def test_a_real_comparison_reaches_the_real_gate():
    """The two halves against each other, so the record the gate reads is the
    record the comparison writes — not a fixture that agrees with neither."""
    committed = committed_vector()
    clean = verify.committed_comparison(committed.double(), committed)
    assert gate(report(per_game={"dictator": clean}))["passed"]

    doubled = committed.double().clone()
    doubled[0] *= 2.0
    result = gate(report(per_game={"dictator":
                                   verify.committed_comparison(doubled, committed)}))
    assert not result["passed"]
    assert "layer-0 norm" in result["failures"][0]


# --- the tolerance, against data rather than prose -------------------------------

def test_float32_storage_alone_moves_the_norm_past_the_old_floor():
    """Two float64 vectors that save to the SAME committed file, 3e-8 apart in norm.

    The committed `.pt` is float32; the recomputation `committed_comparison`
    compares it with is float64. Scaling by 1 + 2**-25 is an eighth of a float32
    ulp, so it rounds back to exactly the bytes on disk — and it moves the norm by
    2**-25 = 3.0e-8. A `--max-norm-drift` of 1e-9 is therefore below the floor
    storage itself sets, and 1e-6 is above it while still failing a scale error of
    one part in 1e5.
    """
    committed = committed_vector()
    assert committed.dtype is torch.float32
    exact = committed[20].double()
    stored_the_same = exact * (1.0 + 2.0 ** -25)
    assert torch.equal(stored_the_same.float(), committed[20])

    drift = abs(1.0 - (stored_the_same.norm() / exact.norm()).item())
    assert drift == pytest.approx(2.0 ** -25, rel=1e-6)
    assert 1e-9 < drift < 1e-6


@pytest.mark.parametrize("name", ["verification.json", "verification_relaxed.json"])
def test_the_recorded_drifts_clear_1e_6_and_would_not_have_cleared_1e_9(name):
    """The committed reports, over a corpus proven bit-identical to the archive.

    This is the whole argument for widening the tolerance, read off the artifacts
    rather than asserted: every observed drift is inside 1e-6, and at least one is
    outside 1e-9, so the old default failed vectors that do reproduce.
    """
    path = ANALYSIS / name
    if not path.is_file():
        pytest.skip("%s is not in this checkout" % name)
    out = json.loads(path.read_text(encoding="utf-8"))
    entries = [e for section in ("A_per_game_vs_committed", "A2_pooled_vs_committed")
               for e in out[section].values() if "error" not in e]
    assert len(entries) == 15
    drifts = [e["worst_norm_drift"] for e in entries]
    assert max(drifts) < MAX_NORM_DRIFT
    assert max(drifts) > 1e-9

    strict = gate({"A_per_game_vs_committed": out["A_per_game_vs_committed"],
                   "A2_pooled_vs_committed": out["A2_pooled_vs_committed"]},
                  max_norm_drift=1e-9)
    assert not strict["passed"]
    assert gate({"A_per_game_vs_committed": out["A_per_game_vs_committed"],
                 "A2_pooled_vs_committed": out["A2_pooled_vs_committed"]})["passed"]
