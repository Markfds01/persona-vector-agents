"""The vector manifest's two ways of being quietly wrong.

It can replace a committed artifact with a different file of the same name, and
it can report an angle that is not the angle. Both produce a manifest that looks
right, so both are pinned here. The rest of the module is bookkeeping over the
null-build reports and is exercised end to end by the committed MANIFEST.json,
which the last test reads back.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
PACKAGE = SCRIPTS.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


manifest = _load("build_vector_manifest",
                 SCRIPTS / "measurement" / "build_vector_manifest.py")


def _vector(rows=29, hidden=8, fill=1.0):
    tensor = torch.zeros(rows, hidden)
    tensor[manifest.LAYER, 0] = fill
    return tensor


# --- the angle ----------------------------------------------------------------

def test_a_vector_is_zero_degrees_from_itself():
    v = _vector()
    cosine, degrees = manifest.angle_between(v, v, manifest.LAYER)
    assert cosine == pytest.approx(1.0)
    assert degrees == pytest.approx(0.0, abs=1e-6)


def test_a_rescaled_vector_is_still_zero_degrees_away():
    cosine, degrees = manifest.angle_between(_vector(fill=1.0), _vector(fill=7.5),
                                             manifest.LAYER)
    assert degrees == pytest.approx(0.0, abs=1e-6)


def test_an_opposed_vector_is_a_hundred_and_eighty_degrees_away():
    _cos, degrees = manifest.angle_between(_vector(fill=1.0), _vector(fill=-1.0),
                                           manifest.LAYER)
    assert degrees == pytest.approx(180.0, abs=1e-6)


def test_a_right_angle_reads_as_ninety_degrees():
    a, b = _vector(), torch.zeros(29, 8)
    b[manifest.LAYER, 1] = 1.0
    _cos, degrees = manifest.angle_between(a, b, manifest.LAYER)
    assert degrees == pytest.approx(90.0, abs=1e-6)


def test_the_angle_is_measured_at_the_named_layer_and_not_over_the_whole_tensor():
    a, b = _vector(), _vector()
    b[0, 1] = 1000.0  # a different layer entirely
    _cos, degrees = manifest.angle_between(a, b, manifest.LAYER)
    assert degrees == pytest.approx(0.0, abs=1e-6)


# --- placing the real vectors -------------------------------------------------

def _write(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def test_a_missing_source_vector_stops_the_run(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        manifest.place_real_vectors(tmp_path / "source", tmp_path / "vectors")
    assert "no vector at" in str(excinfo.value)


def _full_source(tmp_path, fill=1.0):
    source = tmp_path / "source"
    for family in manifest.FAMILIES:
        for policy in manifest.POLICIES:
            _write(source / manifest.decision_name(family, policy), _vector(fill=fill))
    return source


def test_every_real_vector_is_copied_in(tmp_path):
    source = _full_source(tmp_path)
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    placed = manifest.place_real_vectors(source, vectors)
    assert len(placed) == len(manifest.FAMILIES) * len(manifest.POLICIES)
    for name in placed:
        assert (vectors / name).read_bytes() == (source / name).read_bytes()


def test_copying_twice_changes_nothing_and_reports_nothing_new(tmp_path):
    source = _full_source(tmp_path)
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    manifest.place_real_vectors(source, vectors)
    assert manifest.place_real_vectors(source, vectors) == []


def test_a_present_file_that_is_a_different_vector_is_never_overwritten(tmp_path):
    source = _full_source(tmp_path)
    vectors = tmp_path / "vectors"
    name = manifest.decision_name(manifest.FAMILIES[0], "strict")
    _write(vectors / name, _vector(fill=99.0))
    before = (vectors / name).read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        manifest.place_real_vectors(source, vectors)
    assert "refusing to replace" in str(excinfo.value)
    assert (vectors / name).read_bytes() == before


# --- the reports this manifest copies its bookkeeping out of ------------------

def _report(package, name, entries):
    path = package / "provenance" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def _entry(family, policy, seed=1, layer=20):
    return {"family": family, "policy": policy, "seed": seed, "layer": layer,
            "n_altruistic": 1, "n_self_interested": 1, "n_cells_usable": 1,
            "n_cells_seen": 1, "rows_csv": "x.csv", "rows_csv_sha256": "ab",
            "committed_vector": "v.pt", "committed_vector_sha256_16": "cd",
            "rebuild_max_relative_layer_deviation": 0.0}


def test_the_relaxed_report_holding_strict_entries_is_refused(tmp_path):
    """This module never rebuilds a vector: an entry read under the wrong name
    puts that policy's row counts and provenance on every entry silently."""
    _report(tmp_path, "null_vectors.json", [_entry("dictator", "strict")])
    _report(tmp_path, "null_vectors_relaxed.json", [_entry("dictator", "strict")])
    with pytest.raises(SystemExit) as excinfo:
        manifest.load_reports(tmp_path, 1, 20)
    assert "relaxed report by name" in str(excinfo.value)


def test_a_null_built_at_another_layer_is_refused(tmp_path):
    """`build_nulls.py` takes --layer; an angle measured at one and described as
    another is wrong in a way no number in the file shows."""
    _report(tmp_path, "null_vectors.json", [_entry("dictator", "strict", layer=16)])
    _report(tmp_path, "null_vectors_relaxed.json",
            [_entry("dictator", "relaxed", layer=16)])
    with pytest.raises(SystemExit) as excinfo:
        manifest.load_reports(tmp_path, 1, 20)
    assert "layer 16" in str(excinfo.value)


def test_a_matching_pair_of_reports_loads(tmp_path):
    _report(tmp_path, "null_vectors.json", [_entry("dictator", "strict")])
    _report(tmp_path, "null_vectors_relaxed.json", [_entry("dictator", "relaxed")])
    reports = manifest.load_reports(tmp_path, 1, 20)
    assert sorted(reports) == [("dictator", "relaxed"), ("dictator", "strict")]


# --- the committed manifest ---------------------------------------------------

@pytest.fixture(scope="module")
def committed():
    path = PACKAGE / "vectors" / "MANIFEST.json"
    if not path.is_file():
        pytest.skip("no committed vector manifest")
    return json.loads(path.read_text())


def test_the_manifest_holds_all_four_vectors_for_every_game(committed):
    seen = {(e["family"], e["policy"], e["role"]) for e in committed["vectors"]}
    expected = {(f, p, r) for f in manifest.FAMILIES for p in manifest.POLICIES
                for r in ("decision", "shuffled-null")}
    assert seen == expected


def test_every_vector_named_in_the_manifest_is_present_with_that_sha(committed):
    for entry in committed["vectors"]:
        path = PACKAGE / "vectors" / entry["file"]
        assert path.is_file(), entry["file"]
        assert manifest.sha256(path) == entry["sha256"], entry["file"]


def test_a_strict_decision_vector_is_zero_degrees_from_itself(committed):
    for entry in committed["vectors"]:
        if entry["role"] == "decision" and entry["policy"] == "strict":
            assert entry["angle_to_strict_decision_deg"] == pytest.approx(0.0, abs=1e-6)


def test_the_prisoners_dilemma_is_the_only_identical_relaxed_pair(committed):
    identical = sorted({e["family"] for e in committed["vectors"]
                        if e["policy"] == "relaxed"
                        and e["identical_to_strict_counterpart"]})
    assert identical == ["prisoners_dilemma"]


def test_no_strict_entry_claims_to_be_identical_to_a_strict_counterpart(committed):
    """A strict entry's counterpart is itself, and a strict NULL carrying the flag
    beside its 67-degree angle reads as the control being the treatment."""
    for entry in committed["vectors"]:
        if entry["policy"] == "strict":
            assert "identical_to_strict_counterpart" not in entry, entry["file"]
            assert "max_abs_diff_to_strict_counterpart_all_layers" not in entry
    assert not any(key.startswith("identical_to_strict")
                   and key != "identical_to_strict_counterpart"
                   for entry in committed["vectors"] for key in entry)


def test_a_null_is_fit_on_exactly_the_rows_its_real_vector_was(committed):
    by_key = {(e["family"], e["policy"], e["role"]): e for e in committed["vectors"]}
    for family in manifest.FAMILIES:
        for policy in manifest.POLICIES:
            real = by_key[(family, policy, "decision")]
            null = by_key[(family, policy, "shuffled-null")]
            assert null["n_rows_fit"] == real["n_rows_fit"]
            assert null["built_from_rows_csv_sha256"] == real["built_from_rows_csv_sha256"]


def test_every_null_cleared_the_rebuild_gate(committed):
    deviations = [e["rebuild_max_relative_layer_deviation"]
                  for e in committed["vectors"] if e["role"] == "shuffled-null"]
    assert len(deviations) == len(manifest.FAMILIES) * len(manifest.POLICIES)
    assert max(deviations) < 1e-6
