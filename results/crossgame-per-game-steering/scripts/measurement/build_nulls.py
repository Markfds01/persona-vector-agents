"""One shuffled-label null vector per game, matched to that game's real vector.

The steered arm asks whether adding a game's own decision direction changes what
the model does. On its own that is not an answer: any edit of that size at layer
20 does something. The null is the control — the SAME construction, on the SAME
activations, over the SAME cells, with only the pole labels permuted WITHIN each
cell. Steered at the same unit beta it is the same sized push along a direction
that is a random relabelling of the same data.

Within-cell permutation, sizes preserved, is the matched null for a cell-balanced
vector: it holds the prompt composition and every per-cell pole count fixed and
destroys only the association between a row's activation and the decision it
recorded. A global permutation would also reshuffle composition across cells,
which cell balancing exists to cancel.

The real vector is rebuilt here first and checked against the committed file. If
this script cannot reproduce the artifact, its null is not a null OF that
artifact and nothing is written. The rebuild runs through
`analyze_crossgame.balanced_vector` — the function that produced the committed
vectors — rather than a transcription of it, so the two cannot drift.

CPU only; it reads the activation shards `lab.extract` wrote and does linear
algebra. Those shards are ~7 GB and are not committed, so `--acts` is required.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_VECTORS = HERE.parents[2] / "crossgame-decision-vectors" / "scripts"
sys.path.insert(0, str(_VECTORS / "measurement"))
sys.path.insert(0, str(_VECTORS / "prompting"))
sys.path.insert(0, str(HERE.parents[3]))

import analyze_crossgame as A  # noqa: E402
import poles  # noqa: E402

#: The extraction corpus and the vectors built from it both live in the study that
#: produced them; this package adds a null beside each, never a second copy.
VECTOR_STUDY = HERE.parents[2] / "crossgame-decision-vectors"
EXTRACTION = VECTOR_STUDY / "extraction"
COMMITTED_VECTORS = VECTOR_STUDY / "vectors"

FAMILIES = ("dictator", "trust", "ultimatum", "apology", "overfishing",
            "prisoners_dilemma")

REPO_ROOT = HERE.parents[3]

#: A rebuild that differs from the committed vector by more than this, relative to
#: the layer's own norm, is a different vector and stops the run.
REBUILD_TOLERANCE = 1e-6


def sha16(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def committed_vector_path(family: str, policy: str) -> Path:
    return COMMITTED_VECTORS / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                                % (family, policy))


def permute_within_cells(alt, self_, generator):
    """Reassign the two poles inside each cell at random, keeping both counts.

    Returns new `(alt, self_)` label lists. Rows in a cell that holds only one
    pole are carried through unchanged — `balanced_vector` never uses them, and
    moving them would change which cells are usable.
    """
    by_cell = {}
    for item in alt:
        by_cell.setdefault(item["cell"], ([], []))[0].append(item)
    for item in self_:
        by_cell.setdefault(item["cell"], ([], []))[1].append(item)
    new_alt, new_self = [], []
    for cell in sorted(by_cell):
        cell_alt, cell_self = by_cell[cell]
        pool = cell_alt + cell_self
        if not cell_alt or not cell_self:
            new_alt.extend(cell_alt)
            new_self.extend(cell_self)
            continue
        order = torch.randperm(len(pool), generator=generator).tolist()
        shuffled = [pool[i] for i in order]
        new_alt.extend(shuffled[:len(cell_alt)])
        new_self.extend(shuffled[len(cell_alt):])
    return new_alt, new_self


def cosine_at(a, b, layer: int) -> float:
    x, y = a[layer].double(), b[layer].double()
    return float((x * y).sum() / (x.norm() * y.norm()))


def rebuild_deviation(committed, rebuilt) -> float:
    """Worst per-layer distance between two (L, H) vectors, relative to that
    layer's own norm. Per layer and not just at layer 20: a cosine is
    scale-invariant and a whole-tensor norm hides one layer moving."""
    return float(((committed - rebuilt).norm(dim=1)
                  / committed.norm(dim=1).clamp_min(1e-12)).max())


def check_rebuild(family: str, acts_dir, committed, rebuilt) -> float:
    """The gate: refuse to write a null unless the committed vector rebuilds here.

    If this activation corpus does not reproduce the artifact, a null built from
    it is not a null OF that artifact and nothing may be written. Returns the
    deviation it accepted, so the report records the number the gate passed
    rather than one recomputed beside it.
    """
    if committed.shape != rebuilt.shape:
        raise SystemExit("%s: committed vector is %s, rebuild is %s"
                         % (family, tuple(committed.shape), tuple(rebuilt.shape)))
    deviation = rebuild_deviation(committed, rebuilt)
    if deviation > REBUILD_TOLERANCE:
        raise SystemExit(
            "%s: rebuilding the committed vector from %s gives a max relative "
            "per-layer deviation of %.3g; a null built here would not be a null of "
            "the committed vector" % (family, acts_dir, deviation))
    return deviation


def build_one(family: str, acts_root: Path, out_dir: Path, seed: int, policy: str,
              layer: int) -> dict:
    rows_csv = EXTRACTION / ("%s.csv" % family)
    acts_dir = acts_root / family
    labels, acts, metas, _tags = A.load_all(
        {"families": [{"family": family, "rows_csv": str(rows_csv),
                       "acts_dir": str(acts_dir)}]})
    meta = metas[family]
    if meta["rows_csv_sha256"] != _file_sha256(rows_csv):
        raise SystemExit(
            "%s: the activations under %s were captured from a different rows CSV "
            "(%s) than the committed one; they are not this vector's activations"
            % (family, acts_dir, meta["rows_csv_sha256"]))

    response_avg = acts["response_avg"]
    alt, self_, middle, excluded = A.split_poles(labels, False, policy)
    real, usable_cells, n_cells = A.balanced_vector(response_avg, alt, self_)
    if real is None:
        raise SystemExit("%s: no cell holds both poles under %s; no vector to null"
                         % (family, policy))

    committed = committed_vector_path(family, policy)
    reference = torch.load(committed, weights_only=False).double()
    deviation = check_rebuild(family, acts_dir, reference, real)

    generator = torch.Generator().manual_seed(seed)
    null_alt, null_self = permute_within_cells(alt, self_, generator)
    null, null_cells, _n = A.balanced_vector(response_avg, null_alt, null_self)
    if sorted(null_cells) != sorted(usable_cells):
        raise SystemExit("%s: the permutation changed which cells are usable "
                         "(%d -> %d); it must not" % (family, len(usable_cells),
                                                      len(null_cells)))

    out_path = (out_dir / ("null_%s_response_avg_diff_cellbalanced_%s_seed%d.pt"
                           % (family, policy, seed))).resolve()
    torch.save(null.float(), out_path)
    return {
        "family": family,
        "policy": policy,
        "seed": seed,
        "layer": layer,
        "rows_csv": _repo_relative(rows_csv),
        "rows_csv_sha256": meta["rows_csv_sha256"],
        # the shards are ~7 GB, are not committed and sit outside the checkout, so
        # this is the corpus's own directory and never an absolute path off the
        # machine that ran it - a committed artifact in a public repo. The pin a
        # reader can actually check is rows_csv_sha256 above.
        "acts_dir": "%s/%s" % (Path(acts_root).resolve().name or ".", family),
        "n_altruistic": len(alt),
        "n_self_interested": len(self_),
        "n_middle_discarded": len(middle),
        "n_tag_excluded": len(excluded),
        "n_cells_seen": n_cells,
        "n_cells_usable": len(usable_cells),
        "rebuild_max_relative_layer_deviation": deviation,
        "committed_vector": _repo_relative(committed),
        "committed_vector_sha256_16": sha16(committed),
        "real_norm_at_layer": float(real[layer].norm()),
        "null_norm_at_layer": float(null[layer].norm()),
        "cos_null_vs_real_at_layer": cosine_at(null, real, layer),
        "null_vector": _repo_relative(out_path),
        "null_vector_sha256_16": sha16(out_path),
    }


def _repo_relative(path) -> str:
    """A path inside the checkout, named relative to it; anything else, in full."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True,
                    help="activation root written by lab.extract: <family>/shard_*.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--policy", default="strict", choices=list(poles.POLICIES))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise SystemExit("unknown families: %s" % unknown)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for family in families:
        entry = build_one(family, Path(args.acts), out_dir, args.seed, args.policy,
                          args.layer)
        report.append(entry)
        print("%-20s |real|=%7.4f |null|=%7.4f cos=%+.4f cells=%d/%d rebuild_dev=%.1e"
              % (family, entry["real_norm_at_layer"], entry["null_norm_at_layer"],
                 entry["cos_null_vs_real_at_layer"], entry["n_cells_usable"],
                 entry["n_cells_seen"], entry["rebuild_max_relative_layer_deviation"]),
              flush=True)
    Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
