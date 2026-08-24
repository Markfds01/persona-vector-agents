"""Steer each of the six games with its OWN decision vector, on one shared axis.

Every point is placed by INTERVENTION SIZE, never by raw coefficient. The six
per-game vectors have six different layer-20 norms — 2.27 (overfishing) to 8.47
(prisoner's dilemma), a 3.7x span — so a shared raw beta would be six differently
sized pushes and the games would not be comparable, which is the entire point of
this run. Instead every game is stepped along the SAME ladder of unit betas,

    unit_beta = k * REFERENCE_NORM,   k = 0, +-1 .. +-5

with `norm="unit"`, so the coefficient on the row IS the length of the activation
edit added at layer 20 and one k is the same sized edit in every game.

The reference is `||altruism_response_avg_diff.pt[20]||` = 10.5083 — the trait
vector's layer-20 norm, which is the axis the Dictator-only sweep already used.
Two things follow from reusing it rather than inventing a new constant: the six
games are comparable to each other (any shared constant does that), and they are
also comparable to the Dictator-only sweep and to the raw beta k of the authors'
own sweep, for free. `--reference-vector` can change it; the manifest records the
resolved norm, the unit beta and the equivalent raw beta per game either way.

Two arms per game, in separate directories because they sit at the same unit
betas and a shared directory would have one overwrite the other:

* `decision` — that game's committed cell-balanced vector, 11 points;
* `shuffled-null` — that game's matched within-cell label permutation, at the two
  extremes and zero, the same sized push along a randomly relabelled direction.

Resumable at the granularity of one beta point. Each point is an independent
`generate.run` whose batches are seeded `seed + batch_index` from zero, so a
point regenerated on its own is bit-identical to the same point inside an
uninterrupted run — which is what makes skipping a finished CSV safe. A SHORT CSV
is never appended to: its remaining rows would land in differently composed
batches, so it is regenerated whole. A FULL one is trusted only once its rows name
the vector this run steers with: the file name carries the coefficient but neither
the vector nor the pole policy, so nothing else would tell the two arms apart.

The card is shared. The model load waits for real headroom rather than racing
another tenant for it, and retries a load that OOMs anyway; nothing else is
retried, and no process but this one is ever touched.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

import eval_games  # noqa: E402

from audit import generate, steer  # noqa: E402

REPO_ROOT = HERE.parents[3]

#: The vector study whose per-game vectors this sweep steers with.
VECTOR_STUDY = HERE.parents[2] / "crossgame-decision-vectors"

#: The trait vector whose layer-20 norm defines the unit-beta axis, so this sweep
#: shares an axis with the Dictator-only sweep and with the authors' raw beta.
DEFAULT_REFERENCE = (REPO_ROOT / "persona_vectors" / "Qwen2.5-7B-Instruct"
                     / "altruism_response_avg_diff.pt")

#: Sign-paired outward from zero, so an interrupted sweep still holds a symmetric
#: ladder rather than one whole arm and nothing of the other.
REAL_KS = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5)
NULL_KS = (0, 5, -5)

#: Weights are ~15.2 GiB; the rest is the working set at batch 20. Below this the
#: load is not attempted at all — the recorded failure mode on this card is a
#: tenant taking the room mid-load, not a batch that was too big.
DEFAULT_MIN_FREE_MIB = 16600


def reference_norm(path, layer: int) -> float:
    """The layer's norm of a stored (L, H) vector, measured in float32 as shipped.

    Loaded under torch's own `weights_only` default: every vector this reads is a
    bare tensor, so opting out would only give up the unpickling defense.
    """
    import torch

    stored = torch.load(path)
    return float(stored[layer].detach().float().norm())


def vector_paths(family: str, policy: str, vectors_dir: Path, nulls_dir: Path,
                 null_seed: int):
    real = vectors_dir / ("decision_%s_response_avg_diff_cellbalanced_%s.pt"
                          % (family, policy))
    null = nulls_dir / ("null_%s_response_avg_diff_cellbalanced_%s_seed%d.pt"
                        % (family, policy, null_seed))
    for path in (real, null):
        if not path.is_file():
            raise SystemExit("no vector at %s" % path)
    return real, null


def completed_rows(path: Path, expected: int, vector_sha256: str) -> bool:
    """True when `path` already holds THIS vector's whole point.

    `expected` readable data rows, every one of them generated with
    `vector_sha256`. A partial CSV is not resumable — see this module's docstring
    — so anything short is reported as absent and regenerated.

    The vector check is the one the file name cannot make. `steer.rows_path`
    encodes the game, the mode, the layer, the positions, the norm and the
    coefficient, but NOT the vector and not the pole policy, so
    `--policy relaxed --resume` over a strict rows directory would otherwise find
    every point complete, regenerate nothing, and write a manifest recording the
    relaxed vector, its sha, its norm and its raw-beta equivalent against strict
    rows. A rows directory holding another vector's answers is the wrong
    directory rather than a stale point, so this refuses instead of silently
    rebuilding one point of a set whose others would still be the wrong arm.
    """
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            found = [row.get("steer_vector_sha256") for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return False
    if len(found) != expected:
        return False
    wrong = sorted({sha for sha in found if sha != vector_sha256})
    if wrong:
        raise SystemExit(
            "%s holds %d rows steered with %s, not the %s this run steers with. "
            "The row path carries the coefficient but not the vector, so resuming "
            "here would report this point complete and record a vector it was not "
            "generated with. Point --out-dir at this policy's own rows directory, "
            "or re-run it without --resume."
            % (path, len(found), ", ".join(repr(sha) for sha in wrong),
               vector_sha256))
    return True


def wait_for_headroom(device: int, min_free_mib: int, timeout_seconds: int,
                      poll_seconds: int) -> int:
    """Block until the card reports `min_free_mib` free. Returns the MiB seen.

    The tenant on this card loads and unloads on its own schedule, so the useful
    protection is refusing to start rather than shrinking the batch: the peak
    above the weights is ~1.4 GiB against a multi-GiB swing from next door.
    """
    import torch

    deadline = time.time() + timeout_seconds
    while True:
        free, total = torch.cuda.mem_get_info(device)
        free_mib = free // (1 << 20)
        if free_mib >= min_free_mib:
            return free_mib
        if time.time() >= deadline:
            raise SystemExit(
                "cuda:%d has %d MiB free of %d, below the %d MiB this run needs, and "
                "it did not clear within %d s. Another tenant holds the card; nothing "
                "here will take it from them."
                % (device, free_mib, total // (1 << 20), min_free_mib,
                   timeout_seconds))
        print("waiting for headroom on cuda:%d: %d MiB free, need %d"
              % (device, free_mib, min_free_mib), flush=True)
        time.sleep(poll_seconds)


def load_model(args, preset, tokenizer):
    """Load the model on the pinned device, waiting for room and retrying an OOM.

    The model is built here rather than by `HuggingFaceEngine.load` because the
    steering hook needs the model object itself, and because the engine must use
    the same tokenizer that rendered the prompts.
    """
    import torch
    from transformers import AutoModelForCausalLM

    for attempt in range(1, args.load_attempts + 1):
        free_mib = wait_for_headroom(args.device, args.min_free_mib,
                                     args.headroom_timeout, args.headroom_poll)
        print("cuda:%d has %d MiB free; loading (attempt %d/%d)"
              % (args.device, free_mib, attempt, args.load_attempts), flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=args.revision, torch_dtype=torch.bfloat16,
                attn_implementation=preset.attn_implementation,
                device_map={"": args.device})
            break
        except torch.OutOfMemoryError as exc:
            print("load attempt %d/%d OOMed: %s"
                  % (attempt, args.load_attempts, exc), flush=True)
            torch.cuda.empty_cache()
            if attempt == args.load_attempts:
                raise
    model.eval()
    engine = generate.HuggingFaceEngine(model, tokenizer, args.model, args.revision)
    preset.check_engine(engine.describe())
    return engine, model


def run_arm(family, game_id, vector_path, ks, betas, arm, engine, model, tokenizer,
            sampling, args, out_dir):
    """One arm of one game: its points, skipping the ones already on disk."""
    arm_dir = out_dir / family / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    steerings = [steer.Steering(str(vector_path), args.layer, beta, positions="all",
                                norm="unit") for beta in betas]
    paths = {s.coeff: steer.rows_path(arm_dir, s, game_id, "free") for s in steerings}
    # every steering here is the same vector file; hashing it once is enough
    vector_sha256 = steerings[0].vector_sha256

    pending = [s for s in steerings
               if not (args.resume and completed_rows(paths[s.coeff], args.samples,
                                                      vector_sha256))]
    skipped = len(steerings) - len(pending)
    if skipped:
        print("[%s/%s] %d of %d points already complete" % (family, arm, skipped,
                                                            len(steerings)), flush=True)

    started = time.time()

    def on_coefficient(steering, path, rows):
        parsed = [r.value for r in rows if r.value is not None]
        mean = sum(parsed) / len(parsed) if parsed else float("nan")
        print("[%s/%s] unit_beta=%+9.4f n=%d parsed=%d mean=%.3f  %s  %.1f min"
              % (family, arm, steering.coeff, len(rows), len(parsed), mean, path.name,
                 (time.time() - started) / 60.0), flush=True)

    if pending:
        steer.sweep([(game_id, "free")], engine, model, pending, sampling, tokenizer,
                    samples_per_prompt=args.samples, reading="stated",
                    batch_size=args.batch_size, out_dir=arm_dir,
                    on_coefficient=on_coefficient)

    probe = steer.ActivationSteering(model, steerings[0])
    vector_norm = probe.vector_norm
    probe.remove()
    entries = []
    for k, steering in zip(ks, steerings):
        path = paths[steering.coeff]
        if not completed_rows(path, args.samples, vector_sha256):
            raise SystemExit("%s: %s is missing or short after the sweep" % (family, path))
        entries.append({
            "family": family,
            "game_id": game_id,
            "arm": arm,
            "k": k,
            "unit_beta": steering.coeff,
            "raw_beta_equivalent": steering.coeff / vector_norm,
            "vector": str(Path(vector_path).resolve().relative_to(REPO_ROOT)),
            "vector_sha256_16": vector_sha256,
            "vector_layer_norm": vector_norm,
            "rows_csv": str(path.resolve().relative_to(out_dir.resolve())),
            "n_rows": args.samples,
        })
    return entries


def _selected_ks(spec: str, whole, flag: str):
    """A subset of a declared ladder, in the ladder's own order. Empty means all."""
    if not spec.strip():
        return list(whole)
    wanted = [int(part) for part in spec.split(",") if part.strip()]
    unknown = sorted(set(wanted) - set(whole))
    if unknown:
        raise SystemExit("%s: %s are not on the ladder %s" % (flag, unknown, list(whole)))
    if not wanted:
        # an arm with no points is not an empty arm: run_arm probes the card with
        # steerings[0], and it would fail there after the other arm had generated
        raise SystemExit("%s: %r names no rung; leave it unset to run the whole "
                         "ladder %s" % (flag, spec, list(whole)))
    return [k for k in whole if k in wanted]


def write_coefficients(path: Path, record):
    """The flat manifest of every point landed so far. Returns the rows written.

    Rewritten per game and read by every stage downstream, so it goes through a
    staging file like the provenance JSON beside it: a sweep killed mid-write
    would otherwise leave a truncated manifest where a whole earlier one stood.
    """
    rows = [entry for game in record["games"].values()
            for entry in game["coefficients"]]
    if not rows:
        return rows
    staging = path.with_suffix(path.suffix + ".tmp")
    with staging.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staging, path)
    return rows


def write_json_atomically(path: Path, payload):
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(payload, indent=2))
    os.replace(staging, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="the rows/ root of the package")
    ap.add_argument("--provenance", required=True, help="JSON this run accumulates into")
    ap.add_argument("--coefficients", required=True,
                    help="flat CSV of every point landed, rewritten per game")
    ap.add_argument("--games", default=",".join(eval_games.FAMILIES))
    ap.add_argument("--policy", default="strict",
                    help="which pole policy's vectors to steer with")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--null-seed", type=int, default=20260821,
                    help="the seed the null vectors were built under")
    ap.add_argument("--vectors-dir", default=str(VECTOR_STUDY / "vectors"))
    ap.add_argument("--nulls-dir", default=str(HERE.parents[1] / "vectors"))
    ap.add_argument("--reference-vector", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    ap.add_argument("--headroom-timeout", type=int, default=3600)
    ap.add_argument("--headroom-poll", type=int, default=30)
    ap.add_argument("--load-attempts", type=int, default=60)
    ap.add_argument("--resume", action="store_true",
                    help="skip a beta point whose CSV already holds all its rows")
    ap.add_argument("--ks", default="",
                    help="comma-separated subset of the ladder; default the whole of it")
    ap.add_argument("--null-ks", default="",
                    help="comma-separated subset of the null ladder")
    args = ap.parse_args()

    real_ks = _selected_ks(args.ks, REAL_KS, "--ks")
    null_ks = _selected_ks(args.null_ks, NULL_KS, "--null-ks")

    families = [f.strip() for f in args.games.split(",") if f.strip()]
    unknown = sorted(set(families) - set(eval_games.FAMILIES))
    if unknown:
        raise SystemExit("unknown games: %s" % unknown)

    import torch
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(args.provenance).resolve()
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients_path = Path(args.coefficients).resolve()
    coefficients_path.parent.mkdir(parents=True, exist_ok=True)

    unit = reference_norm(args.reference_vector, args.layer)
    real_betas = [k * unit for k in real_ks]
    null_betas = [k * unit for k in null_ks]
    print("reference norm at layer %d = %.10f; unit betas %s"
          % (args.layer, unit, ["%+.4f" % b for b in real_betas]), flush=True)

    preset = generate.preset("neutral")
    sampling = preset.sampling.with_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    engine, model = load_model(args, preset, tokenizer)
    description = engine.describe()
    print("engine: %s" % (description,), flush=True)

    record = (json.loads(provenance_path.read_text())
              if provenance_path.exists() else {"games": {}})
    record["axis"] = {
        "reference_vector": str(Path(args.reference_vector).resolve().relative_to(REPO_ROOT)),
        "reference_layer_norm": unit,
        "layer": args.layer,
        "positions": "all",
        "norm_mode": "unit",
        "ks": list(real_ks),
        "null_ks": list(null_ks),
        "pole_policy_of_vectors": args.policy,
        "mode": "free",
        "reading": "stated",
        "preset": preset.name,
        "samples_per_prompt": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "null_vector_seed": args.null_seed,
        "engine": description.engine,
        "engine_version": description.engine_version,
        "torch_version": description.torch_version,
        "model_id": description.model_id,
        "model_revision": description.model_revision,
        "dtype": description.dtype,
        "attn_implementation": description.attn_implementation,
        "stop_token_ids": list(description.stop_token_ids),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": "cuda:%d" % args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "repo_commit": generate.repo_commit(),
        "repo_dirty": generate.repo_is_dirty(),
    }

    for family in families:
        game = eval_games.by_family(family)
        real_vector, null_vector = vector_paths(
            family, args.policy, Path(args.vectors_dir), Path(args.nulls_dir),
            args.null_seed)
        started = time.time()
        entries = run_arm(family, game.id, real_vector, real_ks, real_betas, "decision",
                          engine, model, tokenizer, sampling, args, out_dir)
        entries += run_arm(family, game.id, null_vector, null_ks, null_betas,
                           "shuffled-null", engine, model, tokenizer, sampling, args,
                           out_dir)
        # landed per game: an interruption costs the game in flight, not the set
        record["games"][family] = {
            "game_id": game.id,
            "pole_scale": game.pole_scale,
            "minutes": (time.time() - started) / 60.0,
            "coefficients": entries,
        }
        write_json_atomically(provenance_path, record)
        # rewritten per game, not once at the end: a run that dies on game five
        # must still leave an analysable manifest for the four that landed
        rows = write_coefficients(coefficients_path, record)
        print("=== %s done in %.1f min (%d points landed) ==="
              % (family, record["games"][family]["minutes"], len(rows)), flush=True)

    print("sweep complete: %d points over %d games"
          % (sum(len(game["coefficients"]) for game in record["games"].values()),
             len(record["games"])), flush=True)


if __name__ == "__main__":
    main()
