"""`run_steering.sh` under both policies, without running anything it drives.

The script is the only documented way to reproduce either arm, and both arms are
committed side by side. A row's filename carries its game, arm and coefficient
but NOT its pole policy, so every output path has to be the policy's own or the
second round lands on the first one's artifacts. That is a data-loss bug rather
than a cosmetic one, and it is what these tests are for.

Nothing here needs torch, a GPU or the real package: `PY` is pointed at a
recorder that writes down how it was called, and the script is run against a
throwaay copy of its own directory layout.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_steering.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None,
                                reason="run_steering.sh is a bash script")

#: Every flag in the script that names a file, written or read, so a path cannot
#: be checked by eye and missed. The read ones matter too: the analysis stage has
#: to consume THIS policy's rows and the tables stage THIS policy's analysis.
PATH_FLAGS = ("--out-dir", "--report", "--out", "--out-csv", "--provenance",
              "--coefficients", "--rows-root", "--analysis")

#: vectors/ is shared by both policies ON PURPOSE: build_nulls.py puts the policy
#: in every null's own filename, so the two rounds cannot collide inside it. The
#: rows do not carry theirs, which is the whole reason for this file.
SHARED_BY_DESIGN = {"vectors"}


def _tree(tmp_path):
    """A throwaway checkout holding nothing but the script, at its real depth."""
    package = tmp_path / "results" / "crossgame-per-game-steering"
    scripts = package / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)

    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        "import sys\n"
        "with open(%r, 'a', encoding='utf-8') as handle:\n"
        "    handle.write('\\t'.join(sys.argv[1:]) + '\\n')\n"
        % str(tmp_path / "calls.tsv"))
    return package, scripts / SCRIPT.name, recorder


def _run(script, recorder, policy, tmp_path, **extra):
    environment = dict(os.environ,
                       PY="%s %s" % (sys.executable, recorder),
                       SAMPLES="100", ACTS=str(tmp_path / "acts"), POLICY=policy,
                       **extra)
    # PY is used unquoted nowhere, so the interpreter has to be one word
    environment["PY"] = str(tmp_path / "py")
    Path(environment["PY"]).write_text(
        '#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, recorder))
    os.chmod(environment["PY"], 0o755)
    return subprocess.run(["bash", str(script)], env=environment,
                          capture_output=True, text=True)


def _outputs(tmp_path, package):
    """`(stage, flag) -> path relative to the package`, over every invocation."""
    found = {}
    for line in (tmp_path / "calls.tsv").read_text().splitlines():
        argv = line.split("\t")
        stage = next(Path(part).name for part in argv if part.endswith(".py"))
        for flag in PATH_FLAGS:
            if flag in argv:
                value = Path(argv[argv.index(flag) + 1])
                found[(stage, flag)] = str(value.relative_to(package))
    return found


def _files(package):
    return {path.relative_to(package)
            for path in package.rglob("*") if path.is_file()}


def _written(package):
    """Everything the script produced: the copy of itself does not count."""
    return _files(package) - {Path("scripts") / SCRIPT.name}


# --- each policy lands where the committed round of that policy lives ---------

def test_the_strict_run_writes_the_paths_the_committed_strict_round_holds(tmp_path):
    package, script, recorder = _tree(tmp_path)
    assert _run(script, recorder, "strict", tmp_path).returncode == 0
    assert _outputs(tmp_path, package) == {
        ("build_nulls.py", "--out-dir"): "vectors",
        ("build_nulls.py", "--report"): "provenance/null_vectors_strict.json",
        ("run_sweep.py", "--out-dir"): "rows",
        ("run_sweep.py", "--provenance"): "provenance/sweep_provenance_strict.json",
        ("run_sweep.py", "--coefficients"): "provenance/coefficients_strict.csv",
        ("analyze_game.py", "--rows-root"): "rows",
        ("analyze_game.py", "--coefficients"): "provenance/coefficients_strict.csv",
        ("analyze_game.py", "--out"): "analysis/steering.json",
        ("crossgame_tables.py", "--analysis"): "analysis/steering.json",
        ("crossgame_tables.py", "--out-csv"): "analysis/points.csv",
    }


def test_the_relaxed_run_writes_the_paths_the_committed_relaxed_round_holds(tmp_path):
    package, script, recorder = _tree(tmp_path)
    assert _run(script, recorder, "relaxed", tmp_path).returncode == 0
    # provenance/ is flat under both policies: the policy is in every filename,
    # and build_vector_manifest.load_reports reads the report at exactly this one
    assert _outputs(tmp_path, package) == {
        ("build_nulls.py", "--out-dir"): "vectors",
        ("build_nulls.py", "--report"): "provenance/null_vectors_relaxed.json",
        ("run_sweep.py", "--out-dir"): "rows_relaxed",
        ("run_sweep.py", "--provenance"): "provenance/sweep_provenance_relaxed.json",
        ("run_sweep.py", "--coefficients"): "provenance/coefficients_relaxed.csv",
        ("analyze_game.py", "--rows-root"): "rows_relaxed",
        ("analyze_game.py", "--coefficients"): "provenance/coefficients_relaxed.csv",
        ("analyze_game.py", "--out"): "analysis/steering_relaxed.json",
        ("crossgame_tables.py", "--analysis"): "analysis/steering_relaxed.json",
        ("crossgame_tables.py", "--out-csv"): "analysis/points_relaxed.csv",
    }


# --- the bug itself ----------------------------------------------------------

def test_the_two_policies_share_no_path_but_the_vectors_directory(tmp_path):
    strict_package, strict_script, strict_recorder = _tree(tmp_path / "a")
    relaxed_package, relaxed_script, relaxed_recorder = _tree(tmp_path / "b")
    _run(strict_script, strict_recorder, "strict", tmp_path / "a")
    _run(relaxed_script, relaxed_recorder, "relaxed", tmp_path / "b")
    strict = set(_outputs(tmp_path / "a", strict_package).values())
    relaxed = set(_outputs(tmp_path / "b", relaxed_package).values())
    assert strict and relaxed
    assert strict & relaxed == SHARED_BY_DESIGN


def test_the_relaxed_run_leaves_every_file_the_strict_run_wrote_alone(tmp_path):
    """The documented `POLICY=relaxed` invocation, run over a finished strict round."""
    package, script, recorder = _tree(tmp_path)
    assert _run(script, recorder, "strict", tmp_path).returncode == 0
    before = {}
    for name in _written(package):
        (package / name).write_text("the committed strict round: %s" % name)
        before[name] = (package / name).read_text()
    assert before, "the strict run wrote nothing to protect"

    (tmp_path / "calls.tsv").unlink()
    assert _run(script, recorder, "relaxed", tmp_path).returncode == 0
    for name, content in before.items():
        assert (package / name).exists(), "%s was removed by the relaxed run" % name
        assert (package / name).read_text() == content, \
            "%s was overwritten by the relaxed run" % name


def test_no_environment_variable_can_move_a_policys_output_paths(tmp_path):
    """POLICY owns every output path, and README section 8 says so unconditionally.

    The log directory used to be settable, and pointing it at the package's own
    provenance/ on a relaxed run put the sweep manifest, the coefficients CSV and
    both logs back on top of the committed strict round's. Every provenance path
    is now derived from POLICY alone, so the environment aims at the strict names
    here and must not reach them.
    """
    package, script, recorder = _tree(tmp_path)
    assert _run(script, recorder, "relaxed", tmp_path,
                LOGS=str(package / "provenance"),
                PROVENANCE=str(package / "provenance"),
                SWEEP_PROVENANCE=str(package / "provenance"
                                     / "sweep_provenance_strict.json"),
                COEFFICIENTS=str(package / "provenance"
                                 / "coefficients_strict.csv")).returncode == 0
    assert _outputs(tmp_path, package)[("run_sweep.py", "--provenance")] == \
        "provenance/sweep_provenance_relaxed.json"
    assert _outputs(tmp_path, package)[("analyze_game.py", "--coefficients")] == \
        "provenance/coefficients_relaxed.csv"


# --- an unknown policy stops rather than inventing a third layout -------------

def test_an_unknown_policy_is_refused_before_anything_is_written(tmp_path):
    package, script, recorder = _tree(tmp_path)
    result = _run(script, recorder, "loose", tmp_path)
    assert result.returncode != 0
    assert "strict or relaxed" in result.stderr
    assert not (tmp_path / "calls.tsv").exists()
    assert _written(package) == set()
