"""Every vector this study will steer with: path, sha256, layer-20 norm, angles.

Run once at checkpoint 1, before any generation, so PREREG.md can name each arm by
the bytes it will load. Every file is loaded with `weights_only=True` - the repo's
vectors are bare (29, 3584) float32 tensors and need nothing more.

It also BUILDS the one arm that does not exist yet: the per-game orthogonal null,
which is each game's committed shuffled-label null with that game's strict decision
direction projected out, per layer, exactly as
`results/dictator-decision-vector/scripts/build_orthogonal_null.py` built the
Dictator's Null B (float64 Gram-Schmidt, saved float32, re-checked after the
float32 round trip because that file is what gets steered). The build is a
deterministic function of two committed files; a second run must reproduce the
same sha256, and `--check` asserts that instead of writing.

    python scripts/provenance/arm_vectors.py            # build + report
    python scripts/provenance/arm_vectors.py --check    # rebuild in memory, compare
"""

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[2]
REPO = HERE.parents[4]
LAYER = 20
REFERENCE_NORM = 10.5083
NULL_SEED = 20260821
GAMES = ("dictator", "trust", "ultimatum", "apology", "overfishing", "prisoners_dilemma")
PER_GAME = "results/crossgame-per-game-steering/vectors"
OUT_DIR = PACKAGE / "vectors"
REPORT = PACKAGE / "provenance" / "arm_vectors.json"


def decision_path(game, policy="strict"):
    return "%s/decision_%s_response_avg_diff_cellbalanced_%s.pt" % (PER_GAME, game, policy)


def shuffled_path(game):
    return "%s/null_%s_response_avg_diff_cellbalanced_strict_seed%d.pt" % (PER_GAME, game, NULL_SEED)


def orthogonal_path(game):
    return ("results/general-question-transfer/vectors/"
            "orthnull_%s_response_avg_diff_cellbalanced_strict_seed%d.pt" % (game, NULL_SEED))


#: The spec's arm list, in order. (arm id, role, game or None, repo-relative path,
#: the decision vector it is a null of / is compared with, or None)
ARMS = (
    [("decision/%s" % g, "decision", g, decision_path(g), None) for g in GAMES]
    + [("relaxed/%s" % g, "decision_relaxed", g, decision_path(g, "relaxed"), decision_path(g))
       for g in ("ultimatum", "overfishing")]
    + [("dictator_only", "decision_dictator_only", "dictator",
        "results/dictator-decision-vector/vectors/decision_response_avg_diff_cellbalanced.pt",
        decision_path("dictator"))]
    + [("trait/altruism", "trait", None,
        "persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt", None)]
    + [("shuffled_null/%s" % g, "shuffled_null", g, shuffled_path(g), decision_path(g))
       for g in GAMES]
    + [("orthogonal_null/%s" % g, "orthogonal_null", g, orthogonal_path(g), decision_path(g))
       for g in GAMES]
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return torch.load(REPO / path, map_location="cpu", weights_only=True)


def orthogonalise(null, real):
    """Null with the real direction projected out, per layer (Dictator Null B)."""
    null, real = null.double(), real.double()
    unit_real = real / real.norm(dim=1, keepdim=True).clamp_min(1e-12)
    projection = (null * unit_real).sum(dim=1, keepdim=True) * unit_real
    return (null - projection).float()


def serialise(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def cos_at(a, b, layer=LAYER):
    a, b = a[layer].double(), b[layer].double()
    return float((a * b).sum() / (a.norm() * b.norm()))


def build_orthogonal_nulls(write):
    built = {}
    for game in GAMES:
        null = load(shuffled_path(game))
        real = load(decision_path(game))
        residual = orthogonalise(null, real)
        data = serialise(residual)
        target = REPO / orthogonal_path(game)
        if write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        built[game] = {
            "sha256": sha256_bytes(data),
            "tensor": saved,
            "cos_shuffled_vs_decision": cos_at(null, real),
            "cos_after_save_vs_decision": cos_at(saved, real),
            "worst_layer_abs_cos_after_save": max(
                abs(cos_at(saved, real, layer)) for layer in range(saved.shape[0])
                if float(real[layer].norm()) > 0 and float(saved[layer].norm()) > 0),
            "source_null_sha256": sha256_bytes((REPO / shuffled_path(game)).read_bytes()),
            "source_decision_sha256": sha256_bytes((REPO / decision_path(game)).read_bytes()),
        }
    return built


def report(built):
    rows = []
    for arm, role, game, path, compare in ARMS:
        data = (REPO / path).read_bytes()
        vec = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        entry = {
            "arm": arm, "role": role, "game": game, "path": path,
            "sha256": sha256_bytes(data), "shape": list(vec.shape),
            "dtype": str(vec.dtype).replace("torch.", ""),
            "norm_at_layer_20": float(vec[LAYER].double().norm()),
            "loaded_with": "torch.load(weights_only=True)",
        }
        if compare is not None:
            other = load(compare)
            c = cos_at(vec, other)
            entry["compared_with"] = compare
            entry["cos_at_layer_20"] = c
            entry["angle_deg_at_layer_20"] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        if role == "orthogonal_null":
            entry["build"] = {k: v for k, v in built[game].items() if k != "tensor"}
        rows.append(entry)
    return {
        "layer": LAYER,
        "reference_norm": REFERENCE_NORM,
        "unit_beta": "k * %s with norm='unit' (every vector rescaled to unit norm at the "
                     "steered layer before scaling)" % REFERENCE_NORM,
        "torch_version": torch.__version__,
        "orthogonal_null_construction": (
            "v_orth = v_null - (v_null . v_hat) v_hat per layer, v_hat = decision / |decision|, "
            "float64, saved float32 via torch.save; as "
            "results/dictator-decision-vector/scripts/build_orthogonal_null.py"),
        "arms": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare with the committed files and report")
    args = ap.parse_args(argv)
    if args.check:
        built = build_orthogonal_nulls(write=False)
        committed = json.loads(REPORT.read_text())
        bad = []
        # torch.save's bytes depend on the torch version that writes them, so the
        # file sha256 can only be reproduced on the torch that built it; the tensor
        # values must reproduce bit-for-bit on any.
        same_torch = torch.__version__ == committed["torch_version"]
        for arm in committed["arms"]:
            if arm["role"] == "orthogonal_null":
                rebuilt = built[arm["game"]]
                if not torch.equal(rebuilt["tensor"], load(arm["path"])):
                    bad.append(arm["arm"] + " (values)")
                if same_torch and rebuilt["sha256"] != arm["sha256"]:
                    bad.append(arm["arm"] + " (file bytes)")
            if sha256_bytes((REPO / arm["path"]).read_bytes()) != arm["sha256"]:
                bad.append(arm["path"])
        if bad:
            raise SystemExit("MISMATCH: %s" % bad)
        print("ok: %d arms match their sha256; orthogonal nulls rebuild bit-identically%s"
              % (len(committed["arms"]), "" if same_torch else
                 " (values only: torch %s, built on %s)" % (torch.__version__,
                                                            committed["torch_version"])))
        return
    built = build_orthogonal_nulls(write=True)
    out = report(built)
    REPORT.write_text(json.dumps(out, indent=1) + "\n")
    for row in out["arms"]:
        print("%-28s %s  |v|@20=%.4f %s" % (row["arm"], row["sha256"][:16], row["norm_at_layer_20"],
                                            "cos=%+.4f" % row["cos_at_layer_20"]
                                            if "cos_at_layer_20" in row else ""))


if __name__ == "__main__":
    sys.exit(main())
