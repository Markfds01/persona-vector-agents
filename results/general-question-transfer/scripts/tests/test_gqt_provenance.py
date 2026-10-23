"""Every arm vector is what PREREG.md says it is, and the orthogonal nulls rebuild.

`torch` is imported outright, as elsewhere in this repository: the contract here is
about tensors, and a module-scope skip would turn a torch-less run green.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent
REPO = PACKAGE.parents[1]
sys.path.insert(0, str(SCRIPTS / "provenance"))

import arm_vectors  # noqa: E402
import cost_estimate  # noqa: E402

REPORT = json.loads((PACKAGE / "provenance" / "arm_vectors.json").read_text())


def test_report_lists_exactly_the_specs_arms():
    roles = [a["role"] for a in REPORT["arms"]]
    assert roles.count("decision") == 6
    assert roles.count("decision_relaxed") == 2
    assert roles.count("decision_dictator_only") == 1
    assert roles.count("trait") == 1
    assert roles.count("shuffled_null") == 6
    assert roles.count("orthogonal_null") == 6
    assert len(roles) == 22


@pytest.mark.parametrize("arm", REPORT["arms"], ids=lambda a: a["arm"])
def test_every_arm_file_matches_its_sha256_and_loads_weights_only(arm):
    data = (REPO / arm["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == arm["sha256"]
    vec = torch.load(REPO / arm["path"], map_location="cpu", weights_only=True)
    assert tuple(vec.shape) == (29, 3584) and vec.dtype == torch.float32


def test_every_sha256_in_the_prereg_table_matches():
    prereg = (PACKAGE / "PREREG.md").read_text()
    for arm in REPORT["arms"]:
        assert "| %s |" % arm["arm"] in prereg
        assert "`%s` | `%s`" % (arm["path"], arm["sha256"]) in prereg


def test_orthogonal_nulls_rebuild_bit_identically():
    """Values bit-identical on any torch; file bytes only on the torch that wrote them."""
    built = arm_vectors.build_orthogonal_nulls(write=False)
    for arm in REPORT["arms"]:
        if arm["role"] == "orthogonal_null":
            committed = torch.load(REPO / arm["path"], map_location="cpu", weights_only=True)
            assert torch.equal(built[arm["game"]]["tensor"], committed)
            if torch.__version__ == REPORT["torch_version"]:
                assert built[arm["game"]]["sha256"] == arm["sha256"]
            assert abs(built[arm["game"]]["cos_after_save_vs_decision"]) < 1e-8
            assert built[arm["game"]]["worst_layer_abs_cos_after_save"] < 1e-8


def test_orthogonalise_removes_exactly_the_projection():
    torch.manual_seed(0)
    real = torch.randn(3, 16)
    null = torch.randn(3, 16)
    out = arm_vectors.orthogonalise(null, real).double()
    for layer in range(3):
        assert abs(float((out[layer] * real[layer].double()).sum())) < 1e-5
    # the component orthogonal to real is unchanged
    unit = real.double() / real.double().norm(dim=1, keepdim=True)
    resid = null.double() - (null.double() * unit).sum(1, keepdim=True) * unit
    assert torch.allclose(out, resid, atol=1e-6)


def test_cost_design_arithmetic():
    d = cost_estimate.design()
    assert d["points_per_question_set"] == 22 * 6 + 2
    assert d["generations"] == 134 * 200 * 3
    assert d["judged_rows"] == d["generations"] - 600
    calls = cost_estimate.judge_calls(d)
    assert calls["primary"]["pairwise"] == 2 * 22 * 6 * 200 * 2
    committed = json.loads((PACKAGE / "analysis" / "cost.json").read_text())
    assert committed["design"] == d and committed["judge_calls"] == calls
