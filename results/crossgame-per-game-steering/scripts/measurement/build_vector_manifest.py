"""Place every vector this package steered with beside its nulls, and describe them.

The steering rows name their vector by path and SHA, but the artifacts themselves
lived in two places: the real vectors in the vector study that fit them, the nulls
here. This collects all four kinds into one directory so the set a reader has to
check is the set they can open, and writes a manifest that says, per vector, how
long it is, how far it points from the strict vector of the same game, what it was
built from, and how many rows it was fit on.

A real vector is COPIED, never rebuilt: the bytes here must be the bytes the sweep
loaded, so the copy is refused if a file of that name is already present with
different content. Nulls are already written here by `build_nulls.py` and are only
described.

The one case worth naming: the Prisoner's Dilemma's strict and relaxed vectors are
the same tensor, and so are its two nulls. Its answer space is two points with no
middle, so widening the poles has nothing to widen. The manifest marks both pairs
`identical_to` their strict counterpart, which is what licenses the relaxed sweep
reusing its strict rows instead of regenerating them.

CPU only. It reads the two null-build reports for the row counts rather than
recounting them, because those reports are what the rebuild gate actually passed.
"""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
VECTOR_STUDY = HERE.parents[2] / "crossgame-decision-vectors"

FAMILIES = ("dictator", "trust", "ultimatum", "apology", "overfishing",
            "prisoners_dilemma")
POLICIES = ("strict", "relaxed")
LAYER = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def decision_name(family: str, policy: str) -> str:
    return "decision_%s_response_avg_diff_cellbalanced_%s.pt" % (family, policy)


def null_name(family: str, policy: str, seed: int) -> str:
    return ("null_%s_response_avg_diff_cellbalanced_%s_seed%d.pt"
            % (family, policy, seed))


def angle_between(a, b, layer: int):
    """`(cosine, degrees)` at one layer, in double so a 1.0 is a real 1.0."""
    x, y = a[layer].double(), b[layer].double()
    cosine = float((x * y).sum() / (x.norm() * y.norm()))
    return cosine, math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def place_real_vectors(source_dir: Path, vectors_dir: Path) -> list:
    """Copy each real vector in, refusing to replace a file that differs."""
    placed = []
    for family in FAMILIES:
        for policy in POLICIES:
            source = source_dir / decision_name(family, policy)
            if not source.is_file():
                raise SystemExit("no vector at %s" % source)
            target = vectors_dir / source.name
            if target.is_file():
                if sha256(target) != sha256(source):
                    raise SystemExit(
                        "%s already exists here and is NOT the vector at %s; "
                        "refusing to replace it" % (target, source))
            else:
                shutil.copy2(source, target)
                placed.append(target.name)
    return placed


def load_reports(package: Path, seed: int) -> dict:
    """`(family, policy) -> the null build report entry`, keyed for lookup."""
    reports = {}
    for policy, name in (("strict", "null_vectors.json"),
                         ("relaxed", "null_vectors_relaxed.json")):
        path = package / "provenance" / name
        if not path.is_file():
            raise SystemExit("no null-build report at %s; run build_nulls.py "
                             "--policy %s first" % (path, policy))
        for entry in json.loads(path.read_text()):
            if entry["seed"] != seed:
                raise SystemExit("%s carries seed %d, not %d"
                                 % (path, entry["seed"], seed))
            reports[(entry["family"], entry["policy"])] = entry
    return reports


def describe(path: Path, family: str, policy: str, role: str, strict_decision,
             report: dict, seed) -> dict:
    """One manifest entry. `strict_decision` is the game's strict real vector."""
    tensor = torch.load(path, weights_only=False)
    cosine, degrees = angle_between(tensor, strict_decision, LAYER)
    entry = {
        "file": path.name,
        "role": role,
        "family": family,
        "policy": policy,
        "layer": LAYER,
        "sha256": sha256(path),
        "norm_at_layer": float(tensor[LAYER].double().norm()),
        "cos_to_strict_decision_at_layer": cosine,
        "angle_to_strict_decision_deg": degrees,
        # both poles of the cell-balanced fit; a null permutes these labels
        # within each cell and so is fit on exactly the same rows
        "n_rows_fit": report["n_altruistic"] + report["n_self_interested"],
        "n_altruistic": report["n_altruistic"],
        "n_self_interested": report["n_self_interested"],
        "n_cells_usable": report["n_cells_usable"],
        "n_cells_seen": report["n_cells_seen"],
        "built_from_rows_csv": report["rows_csv"],
        "built_from_rows_csv_sha256": report["rows_csv_sha256"],
    }
    if role == "shuffled-null":
        entry["seed"] = seed
        entry["null_of"] = report["committed_vector"]
        entry["null_of_sha256_16"] = report["committed_vector_sha256_16"]
        entry["rebuild_max_relative_layer_deviation"] = report[
            "rebuild_max_relative_layer_deviation"]
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(VECTOR_STUDY / "vectors"),
                    help="the vector study's vectors/, where the real ones were fit")
    ap.add_argument("--vectors", default=str(PACKAGE / "vectors"))
    ap.add_argument("--package", default=str(PACKAGE))
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out", default=str(PACKAGE / "vectors" / "MANIFEST.json"))
    args = ap.parse_args()

    vectors_dir = Path(args.vectors)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    placed = place_real_vectors(Path(args.source), vectors_dir)
    reports = load_reports(Path(args.package), args.seed)

    entries = []
    for family in FAMILIES:
        strict_decision = torch.load(vectors_dir / decision_name(family, "strict"),
                                     weights_only=False)
        for policy in POLICIES:
            report = reports[(family, policy)]
            entries.append(describe(vectors_dir / decision_name(family, policy),
                                    family, policy, "decision", strict_decision,
                                    report, None))
            entries.append(describe(vectors_dir / null_name(family, policy, args.seed),
                                    family, policy, "shuffled-null", strict_decision,
                                    report, args.seed))

    # a relaxed artifact that IS its strict counterpart is the licence to reuse
    # that game's strict rows; it has to be stated, not inferred from an angle
    by_key = {(e["family"], e["policy"], e["role"]): e for e in entries}
    for family in FAMILIES:
        for role in ("decision", "shuffled-null"):
            strict = by_key[(family, "strict", role)]
            relaxed = by_key[(family, "relaxed", role)]
            a = torch.load(vectors_dir / strict["file"], weights_only=False)
            b = torch.load(vectors_dir / relaxed["file"], weights_only=False)
            same = bool(torch.equal(a, b))
            relaxed["identical_to_strict"] = same
            relaxed["max_abs_diff_to_strict_all_layers"] = float(
                (a.double() - b.double()).abs().max())
            strict["identical_to_strict"] = True
            strict["max_abs_diff_to_strict_all_layers"] = 0.0

    payload = {
        "layer": LAYER,
        "null_seed": args.seed,
        "source_of_real_vectors": repo_relative(Path(args.source)),
        "note": ("Every vector this package steered with, and every matched "
                 "shuffled-label null, under both pole policies. Angles are to the "
                 "SAME GAME's strict decision vector at layer %d. A real vector is "
                 "a byte copy of the vector study's artifact; the sweep loaded that "
                 "file, not a rebuild." % LAYER),
        "vectors": entries,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    if placed:
        print("copied %d real vectors in" % len(placed))
    for entry in entries:
        print("%-12s %-8s %-7s |v|=%7.4f  angle_to_strict=%6.2f deg  n_fit=%5d%s"
              % (entry["family"], entry["policy"], entry["role"][:7],
                 entry["norm_at_layer"], entry["angle_to_strict_decision_deg"],
                 entry["n_rows_fit"],
                 "  IDENTICAL to strict" if (entry["policy"] == "relaxed"
                                             and entry["identical_to_strict"]) else ""))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
