"""The generation stage: steered (and unhooked) answers to the frozen general questions.

Writes immutable rows only; judging is a separate stage that reads them
(`scripts/judging/run_judging.py`). One point = (arm, question set, k) = 40 questions x
`--samples` answers, one JSONL file under `rows/<arm>/`. Everything that moves a number
is PREREG section 4's and is not a flag: neutral preset, explicit `GenerationConfig`,
`use_model_defaults=False`, layer 20, `positions="all"`, `norm="unit"`,
`unit_beta = k x 10.5083`, `max_new_tokens` 1000, batch size 20, each batch seeded
`seed + batch_index`, each point seeded from `--seed`.

The pilot (PREREG section 14: one game arm, the trait vector and that game's two nulls at
k in {-3, 0, +3}) is `--arms decision/dictator,trait/altruism,shuffled_null/dictator,
orthogonal_null/dictator`, `--ks=-3,3`, both sets. Besides those, every run of a set writes
the shared k = 0 baseline (hook installed at coefficient 0; the trait vector is the file it
hooks), and the held-out set also writes the byte-identity controls: `unhooked` (no hook)
and `unhooked_repeat` (the same again, so the unhooked run is shown reproducible).

Resume (`--resume`) is per point and checked against the vector's sha256 on every row.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
REPO = PACKAGE.parents[1]
sys.path.insert(0, str(REPO))

from audit import generate, steer  # noqa: E402
from lab import general_questions as gq  # noqa: E402

LAYER = 20
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
BASELINE_VECTOR_ARM = "trait/altruism"


def arm_table():
    table = json.loads((PACKAGE / "provenance" / "arm_vectors.json").read_text())
    assert table["layer"] == LAYER and abs(table["reference_norm"] - gq.REFERENCE_NORM) < 1e-12
    return {a["arm"]: a for a in table["arms"]}


def build_arm(name, table):
    """An `Arm` and the sha256 the arm table pins for its vector."""
    if name in ("baseline", "unhooked", "unhooked_repeat"):
        if name != "baseline":
            return gq.Arm(name, "control_unhooked", None), None
        entry = table[BASELINE_VECTOR_ARM]
        return gq.Arm(name, "baseline", str(REPO / entry["path"])), entry["sha256"]
    if name not in table:
        raise SystemExit("unknown arm %r; have %s" % (name, sorted(table)))
    entry = table[name]
    return gq.Arm(name, entry["role"], str(REPO / entry["path"])), entry["sha256"]


def wait_for_headroom(device, min_free_mib, timeout_seconds, poll_seconds):
    import torch
    deadline = time.time() + timeout_seconds
    while True:
        free, total = torch.cuda.mem_get_info(device)
        if free // (1 << 20) >= min_free_mib:
            return free // (1 << 20)
        if time.time() >= deadline:
            raise SystemExit("cuda:%d has %d MiB free, below the %d MiB this run needs"
                             % (device, free // (1 << 20), min_free_mib))
        print("waiting for headroom on cuda:%d" % device, flush=True)
        time.sleep(poll_seconds)


def load_engine(args, preset):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    free = wait_for_headroom(args.device, args.min_free_mib, args.headroom_timeout, 30)
    print("cuda:%d has %d MiB free; loading" % (args.device, free), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        attn_implementation=preset.attn_implementation)
    model.to("cuda:%d" % args.device)
    model.eval()
    engine = gq.IdRecordingEngine(model, tokenizer, MODEL, REVISION)
    preset.check_engine(engine.describe())
    return engine, model, tokenizer


def write_json(path, payload):
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(staging, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--provenance-dir", required=True)
    ap.add_argument("--sets", default="heldout,bridge")
    ap.add_argument("--arms", required=True, help="comma-separated arm names from arm_vectors.json")
    ap.add_argument("--ks", required=True, help="comma-separated k, e.g. -3,3 (k = 0 is the baseline)")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-controls", action="store_true", help="skip baseline and unhooked points")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min-free-mib", type=int, default=18000)
    ap.add_argument("--headroom-timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    ks = [float(k) for k in args.ks.split(",") if k.strip()]
    if 0 in ks:
        raise SystemExit("k = 0 is the shared baseline, not an arm point; do not list it in --ks")
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    table = arm_table()
    arms = [build_arm(n, table) for n in arm_names]

    rows_dir, prov_dir = Path(args.rows_dir), Path(args.provenance_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    preset = generate.preset("neutral")
    sampling = preset.sampling.with_seed(args.seed)
    assert sampling.max_new_tokens == 1000 and sampling.temperature == 1.0
    engine, model, tokenizer = load_engine(args, preset)
    record = generate.provenance(engine, tokenizer)
    stop_ids = list(record.engine.stop_token_ids)
    print("engine: %s" % (record.engine,), flush=True)

    import torch
    trait_norm = float(torch.load(REPO / table[BASELINE_VECTOR_ARM]["path"],
                                  weights_only=True)[LAYER].float().norm())
    assert abs(trait_norm - gq.REFERENCE_NORM) < 5e-5, trait_norm

    for question_set in sets:
        questions, question_sha = gq.load_questions(PACKAGE, question_set)
        points_path = prov_dir / ("generation_points_%s.json" % question_set)
        state = json.loads(points_path.read_text()) if points_path.exists() else {"points": {}}
        state["run"] = {
            "model": MODEL, "model_revision": record.engine.model_revision,
            "dtype": record.engine.dtype, "attn_implementation": record.engine.attn_implementation,
            "engine_version": record.engine.engine_version, "torch_version": record.engine.torch_version,
            "stop_token_ids": stop_ids, "preset": preset.name, "samples": args.samples,
            "batch_size": args.batch_size, "seed": args.seed, "layer": LAYER, "positions": "all",
            "norm": "unit", "reference_norm": gq.REFERENCE_NORM, "use_model_defaults": False,
            "max_new_tokens": sampling.max_new_tokens, "question_set": question_set,
            "question_file_sha256": question_sha, "chat_template_sha256": record.chat_template_sha256,
            "repo_commit": record.repo_commit, "repo_dirty": record.repo_dirty,
            "device": "cuda:%d" % args.device, "device_name": torch.cuda.get_device_name(args.device),
            "trait_vector_layer_norm": trait_norm,
        }
        plan = []
        if not args.no_controls:
            plan.append(build_arm("baseline", table) + (0.0,))
            if question_set == "heldout":
                plan.append(build_arm("unhooked", table) + (0.0,))
                plan.append(build_arm("unhooked_repeat", table) + (0.0,))
        plan += [(arm, sha, k) for arm, sha in arms for k in ks]
        expected = len(questions) * args.samples
        for arm, sha, k in plan:
            path = rows_dir / (gq.point_slug(arm.name, question_set, k) + ".jsonl")
            key = gq.point_slug(arm.name, question_set, k)
            if args.resume and gq.point_complete(path, expected, sha, arm=arm.name,
                                                 max_new_tokens=sampling.max_new_tokens,
                                                 question_file_sha256=question_sha):
                print("skip (complete): %s" % key, flush=True)
                continue
            if arm.vector_path is not None:
                actual = gq.sha256_file(arm.vector_path)
                assert actual == sha, (arm.name, actual, sha)
                steering = steer.Steering(arm.vector_path, LAYER, k * arm.unit_beta_scale,
                                          positions="all", norm="unit")
                steered = gq.CountingSteeredEngine(engine, model, steering)
            else:
                steered = gq.UnhookedEngine(engine)
            started = time.time()
            summary = gq.generate_point(
                steered, tokenizer, questions, arm=arm, k=k, samples=args.samples,
                sampling=sampling, batch_size=args.batch_size, stop_ids=stop_ids,
                out_path=path, question_file_sha256=question_sha, record=record,
                vector_sha256=sha, log=lambda m: print(m, flush=True))
            summary["minutes"] = (time.time() - started) / 60.0
            if arm.vector_path is not None:
                summary["vector_norm_layer20"] = steered._steering.vector_norm
                summary["steer_module_path"] = steered.columns["steer_module_path"]
            state["points"][key] = summary
            write_json(points_path, state)
        write_json(points_path, state)
    print("generation complete", flush=True)


if __name__ == "__main__":
    main()
