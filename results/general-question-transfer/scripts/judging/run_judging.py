"""The judging stage: reads generation rows, writes judgment rows, touches nothing else.

    PIMIENTOROJO_BASE_URL=<the judge's endpoint> python run_judging.py --judge primary \\
        --rows-dir rows --out-dir judgments [--kinds trait,coherence,pairwise]
    PIMIENTOROJO_BASE_URL=<the judge's endpoint> python run_judging.py --judge primary \\
        --phase resend ...

`--judge primary` (Llama-3.3-70B Q8_0) and `--judge secondary` (gpt-oss-20b) differ in
config, not in code; the endpoint of the judge is the only thing in the environment
(`PIMIENTOROJO_BASE_URL`) and is never written anywhere. Before anything is sent the client
checks `/v1/models` for the configured served id and aborts on a mismatch; if the endpoint
cannot be reached the run stops with an error and writes NOTHING (no directory, no file).

Output, under `<out-dir>/<judge>/`: `trait.jsonl`, `coherence.jsonl`, `pairwise.jsonl` (one
row per call, whatever its status), `run_<id>.json` (the server's `/v1/models` and `/props`
as returned, the configs, the template hashes), `determinism_resend.jsonl` (phase resend)
and `_cache/` (the client's append-only request cache, git-ignored). Calls are sent in a
seeded shuffled order (PREREG 13); transport errors are retried a capped number of times by
the client and logged on the row; a re-run skips every call that already has a row other
than `transport_error`.
"""

import argparse
import asyncio
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gqt_judging as gj  # noqa: E402
from lab import llama_server_client as jc  # noqa: E402

CHUNK = 50


def append_rows(path, rows):
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


async def open_clients(configs, cache_dir, need_pairwise):
    clients = {}
    names = ["score"] + (["pairwise"] if need_pairwise else [])
    for name in names:
        client = jc.JudgeClient(configs[name], Path(cache_dir) / ("%s.jsonl" % name))
        await client.open()          # raises EndpointUnavailable / ModelMismatch before any write
        clients[name] = client
    return clients


async def judge_phase(args, plan, configs, out, run_id):
    kinds = [k for k in args.kinds.split(",") if k]
    plan = [t for t in plan if t.kind in kinds]
    existing = {k: gj.read_jsonl(out / ("%s.jsonl" % k)) for k in kinds} if out.exists() else {}
    done = set()
    for rows in existing.values():
        done |= gj.done_keys(rows)
    pending = gj.shuffled([t for t in plan if t.key not in done])
    print("%s: %d calls planned for kinds %s, %d done, %d to send"
          % (args.judge, len(plan), kinds, len(plan) - len(pending), len(pending)), flush=True)
    clients = await open_clients(configs, out / "_cache", any(t.kind == "pairwise" for t in pending))
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_id": run_id, "judge": gj.JUDGE_DIRS[args.judge], "phase": "judge", "kinds": kinds,
        "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "endpoint": clients["score"].provenance()["endpoint"],
        "configured_model": gj.SERVED_IDS[args.judge],
        "models_response": clients["score"].models_response,
        "server_props": clients["score"].server_props,
        "configs": {name: c.provenance()["config"] for name, c in clients.items()},
        "template_sha256": {k: gj.sha256_text(v) for k, v in gj.load_templates().items()},
        "format_instruction_sha256": gj.sha256_text(gj.TEXT_FORMAT_INSTRUCTION)
        if args.judge == "secondary" else None,
        "order_seed": gj.ORDER_SEED, "python": platform.python_version(),
        "n_planned": len(plan), "n_to_send": len(pending),
    }
    (out / ("run_%s.json" % run_id)).write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    started = time.time()
    sent = 0
    for start in range(0, len(pending), CHUNK):
        chunk = pending[start:start + CHUNK]
        for name in ("score", "pairwise"):
            group = [t for t in chunk if t.config_name == name]
            if not group:
                continue
            requests = [jc.JudgeRequest(t.key, t.prompt) for t in group]
            results = await clients[name].judge_many(requests, order_seed=gj.ORDER_SEED + start)
            by_kind = {}
            for task, result in zip(group, results):
                by_kind.setdefault(task.kind, []).append(
                    gj.judgment_row(args.judge, task, result, configs[name], run_id))
            for kind, rows in by_kind.items():
                append_rows(out / ("%s.jsonl" % kind), rows)
        sent += len(chunk)
        elapsed = time.time() - started
        print("  %d/%d sent, %.2f s/call" % (sent, len(pending), elapsed / sent), flush=True)
    provenance["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    provenance["seconds"] = time.time() - started
    provenance["n_sent"] = sent
    (out / ("run_%s.json" % run_id)).write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")


async def resend_phase(args, plan, configs, out, run_id):
    """Re-send 5% of the distinct request payloads cold (`use_cache=False`) and compare."""
    rows = {}
    for kind in gj.KINDS:
        rows.update(gj.latest_by_key(gj.read_jsonl(out / ("%s.jsonl" % kind))))
    missing = [t.key for t in plan if t.key not in rows or rows[t.key]["status"] == "transport_error"]
    if missing:
        raise SystemExit("%d planned calls have no finished row (e.g. %s): the re-send follows the "
                         "main pass, it does not replace it" % (len(missing), missing[0]))
    by_sha = {}
    for task in plan:
        by_sha.setdefault(rows[task.key]["payload_sha256"], task)
    chosen = gj.resend_selection(by_sha)
    clients = await open_clients(configs, out / "_cache", any(by_sha[s].kind == "pairwise" for s in chosen))
    target = out / "determinism_resend.jsonl"
    already = {r["payload_sha256"] for r in gj.read_jsonl(target)}
    print("%s: re-sending %d of %d distinct payloads (%d already done)"
          % (args.judge, len(chosen), len(by_sha), len(already & set(chosen))), flush=True)
    for sha in chosen:
        if sha in already:
            continue
        task = by_sha[sha]
        original = rows[task.key]
        config = configs[task.config_name]
        result = await clients[task.config_name].judge(jc.JudgeRequest(task.key, task.prompt),
                                                      use_cache=False)
        if result.payload_sha256 != sha:
            raise SystemExit("rebuilt payload differs from the recorded one for %s" % task.key)
        before = gj.comparable(original["raw_response"], config.mode)
        after = gj.comparable(result.raw_response, config.mode)
        delta = (None if original["score"] is None or result.score is None
                 else result.score - original["score"])
        append_rows(target, [{
            "payload_sha256": sha, "key": task.key, "kind": task.kind, "judge": gj.JUDGE_DIRS[args.judge],
            "status_original": original["status"], "status_resend": result.status,
            "score_original": original["score"], "score_resend": result.score,
            "score_delta": delta, "exact_match": before == after and before is not None,
            "same_status": original["status"] == result.status,
            "original_timestamp": original["timestamp"], "resend_timestamp": result.timestamp,
            "original": before if before != after else None, "resend": after if before != after else None,
            "run_id": run_id,
        }])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge", required=True, choices=("primary", "secondary"))
    ap.add_argument("--phase", default="judge", choices=("judge", "resend"))
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="the judgments/ root")
    ap.add_argument("--kinds", default="trait,coherence,pairwise",
                    help="add 'fairness' for the exploratory fairness rubric (docs/fairness_exploratory_prereg.md)")
    args = ap.parse_args(argv)

    rows = gj.load_generation_rows(args.rows_dir)
    if not rows:
        raise SystemExit("no generation rows under %s" % args.rows_dir)
    templates = gj.load_templates()
    configs = gj.judge_configs(args.judge)
    plan = gj.build_plan(args.judge, rows, templates,
                         with_fairness="fairness" in args.kinds.split(","))
    out = Path(args.out_dir) / gj.JUDGE_DIRS[args.judge]
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner = judge_phase if args.phase == "judge" else resend_phase
    try:
        asyncio.run(runner(args, plan, configs, out, run_id))
    except (jc.EndpointUnavailable, jc.ModelMismatch, jc.ConfigError) as exc:
        raise SystemExit("judging stopped, nothing written: %s: %s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
