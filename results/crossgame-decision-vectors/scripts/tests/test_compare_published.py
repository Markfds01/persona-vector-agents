"""The rebuild comparator, on trees small enough to read.

Everything this file pins is a way the comparator could report agreement while
having compared nothing: a NaN leaf that satisfies no comparison, a subtree whose
two lists differ in length, a vector directory that is empty on one side. Each of
those shipped at least once.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_published",
                                                  SCRIPTS / "compare_published.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_published", module)
    spec.loader.exec_module(module)
    return module


compare = _load_module()

NAN = float("nan")


def diff(tmp_path, old, new):
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps(new), encoding="utf-8")
    return compare.diff_json(tmp_path / "old.json", tmp_path / "new.json", 25)


def test_a_leaf_that_matches_is_compared_and_counted(tmp_path):
    got = diff(tmp_path, {"auc": 0.9}, {"auc": 0.9})
    assert (got["n_numeric_leaves"], got["n_numeric_leaves_identical"]) == (1, 1)
    assert got["n_non_finite_leaves"] == 0


def test_a_leaf_that_moved_is_the_headline(tmp_path):
    got = diff(tmp_path, {"auc": 0.9}, {"auc": 0.4})
    assert got["n_numeric_leaves_that_moved"] == 1
    assert got["max_abs_delta"] == pytest.approx(0.5)
    assert got["max_abs_delta_path"] == "auc"


@pytest.mark.parametrize("old,new", [(NAN, 0.5), (0.5, NAN)])
def test_a_nan_that_appeared_or_vanished_is_a_difference(tmp_path, old, new):
    """`nan > 0` is False, so this leaf used to be counted as IDENTICAL and left
    out of `max_abs_delta`: the file printed as "nothing moved"."""
    got = diff(tmp_path, {"reliability": old}, {"reliability": new})
    assert got["n_structural_differences"] == 1
    assert got["n_numeric_leaves_identical"] == 0
    assert got["n_numeric_leaves"] == 0


def test_a_nan_on_both_sides_is_unverifiable_not_identical(tmp_path):
    """Zero within-cell variance at layer 0 makes these legitimately NaN. Nothing
    is wrong with them — but nothing was checked about them either."""
    got = diff(tmp_path, {"reliability": NAN}, {"reliability": NAN})
    assert got["n_non_finite_leaves"] == 1
    assert got["n_numeric_leaves"] == 0
    assert got["n_numeric_leaves_identical"] == 0
    assert got["non_finite_leaves"][0]["path"] == "reliability"


def test_an_infinity_that_changed_sign_is_a_difference(tmp_path):
    got = diff(tmp_path, {"x": float("inf")}, {"x": float("-inf")})
    assert got["n_structural_differences"] == 1


def test_lists_of_different_lengths_are_still_compared_over_their_overlap(tmp_path):
    """Returning here left the whole subtree with zero numeric leaves while the
    file still reported as unmoved."""
    got = diff(tmp_path, {"top": [1.0, 2.0]}, {"top": [1.0, 9.0, 3.0]})
    assert got["n_numeric_leaves"] == 2
    assert got["n_numeric_leaves_that_moved"] == 1
    assert got["n_structural_differences"] == 1


def test_a_key_only_one_side_has_is_structural(tmp_path):
    got = diff(tmp_path, {"a": 1.0}, {"a": 1.0, "b": 2.0})
    assert got["structural_differences"] == [{"path": "b", "only_in": "new"}]


def test_where_a_run_happened_is_provenance_not_a_delta(tmp_path):
    got = diff(tmp_path, {"acts_dir": "/a", "seconds": 1.0, "auc": 0.9},
               {"acts_dir": "/b", "seconds": 9.0, "auc": 0.9})
    assert got["n_numeric_leaves"] == 1 and got["n_numeric_leaves_that_moved"] == 0
    assert sorted(p["path"] for p in got["provenance_differences"]) == ["acts_dir",
                                                                        "seconds"]


# --- the vector side -------------------------------------------------------------

def write_vectors(directory, names, tensor=None):
    directory.mkdir(parents=True, exist_ok=True)
    tensor = torch.arange(29 * 8, dtype=torch.float32).view(29, 8) if tensor is None \
        else tensor
    for name in names:
        torch.save(tensor, directory / name)
    return directory


def run_main(monkeypatch, old_root, new_root, out):
    monkeypatch.setattr(sys, "argv", ["compare_published", "--old", str(old_root),
                                      "--new", str(new_root), "--out", str(out)])
    compare.main()
    return json.loads(Path(out).read_text(encoding="utf-8"))


def test_an_empty_new_vector_directory_is_refused(tmp_path, monkeypatch):
    """A wrong --new used to give 0 vectors compared, a null worst cosine, exit 0."""
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    write_vectors(old_root / "vectors", ["decision_x.pt"])
    (new_root / "pooled" / "vectors").mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, old_root, new_root, tmp_path / "out.json")
    assert "--new" in str(excinfo.value)


def test_an_empty_old_vector_directory_is_refused(tmp_path, monkeypatch):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    (old_root / "vectors").mkdir(parents=True)
    write_vectors(new_root / "pooled" / "vectors", ["decision_x.pt"])
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, old_root, new_root, tmp_path / "out.json")
    assert "--old" in str(excinfo.value)


def test_a_vector_only_the_rebuild_has_is_reported(tmp_path, monkeypatch):
    """Walking only the OLD tree hid every vector the rebuild added."""
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    write_vectors(old_root / "vectors", ["a.pt"])
    write_vectors(new_root / "pooled" / "vectors", ["a.pt", "b.pt"])
    report = run_main(monkeypatch, old_root, new_root, tmp_path / "out.json")
    vectors = [m for m in report["missing"] if m["path"].startswith("vectors/")]
    assert vectors == [{"path": "vectors/b.pt", "only_in": "new"}]
    assert report["vectors"]["a.pt"]["bit_identical"] is True
    assert report["summary"]["worst_vector_cosine"] == pytest.approx(1.0)
