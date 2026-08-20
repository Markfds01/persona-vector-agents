"""Generate cross-game grid cells at the `neutral` preset and write scored rows.

One model load, one engine, one sampling configuration. Rows are flushed per
batch so a GPU eviction keeps everything generated so far - the device is shared
and another tenant cycles models on it without warning.

`--families` and `--wordings` restrict the run to part of the grid, which is how
the calibration pass checks that a stake ladder actually populates both poles
before the full grid is paid for. Everything that moves a number is explicit.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import generate
from scratch import crossgame_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument("--samples", type=int, required=True, help="samples per prompt")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--families", default="", help="comma-separated; default all")
    parser.add_argument("--wordings", default="", help="comma-separated; default all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load-attempts", type=int, default=60)
    parser.add_argument("--load-retry-seconds", type=int, default=30)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    args = parser.parse_args()

    import torch

    preset = generate.preset("neutral")
    sampling = preset.sampling.with_seed(args.seed)

    crossgame_grid.register()
    selected = list(crossgame_grid.GRID)
    if args.families:
        want = set(args.families.split(","))
        unknown = want - set(crossgame_grid.FAMILIES)
        if unknown:
            raise SystemExit("unknown families: %s" % sorted(unknown))
        selected = [g for g in selected if g.family in want]
    if args.wordings:
        want = set(args.wordings.split(","))
        unknown = want - set(crossgame_grid.WORDING_NAMES)
        if unknown:
            raise SystemExit("unknown wordings: %s" % sorted(unknown))
        selected = [g for g in selected if g.wording in want]
    if not selected:
        raise SystemExit("selection is empty")
    pairs = [(game.id, "free") for game in selected]
    print("cells=%d samples=%d total=%d" % (len(pairs), args.samples,
                                            len(pairs) * args.samples), flush=True)

    # The device is shared and the other tenant cycles models on it without
    # warning; a load that OOMs is transient, not a result. Retry the LOAD only -
    # never a partially generated run - and report every attempt.
    engine = None
    for attempt in range(1, args.load_attempts + 1):
        try:
            engine = generate.HuggingFaceEngine.load(
                args.model,
                revision=args.revision,
                torch_dtype=torch.bfloat16,
                attn_implementation=preset.attn_implementation,
                device_map={"": args.device},
            )
            break
        except torch.OutOfMemoryError as exc:
            free, total = torch.cuda.mem_get_info(args.device)
            print("load attempt %d/%d OOMed (%.1f GiB free of %.1f): %s"
                  % (attempt, args.load_attempts, free / 2**30, total / 2**30, exc),
                  flush=True)
            torch.cuda.empty_cache()
            if attempt == args.load_attempts:
                raise
            time.sleep(args.load_retry_seconds)
    free, total = torch.cuda.mem_get_info(args.device)
    print("loaded; %.1f GiB free of %.1f on cuda:%d" % (free / 2**30, total / 2**30,
                                                        args.device), flush=True)
    preset.check_engine(engine.describe())
    print("engine:", engine.describe(), flush=True)
    print("device: cuda:%d" % args.device, flush=True)

    started = time.time()
    with generate.RowWriter(args.out) as writer:
        def on_rows(rows):
            writer.write(rows)
            elapsed = time.time() - started
            print("%6d rows  %7.1fs  %5.1f rows/min"
                  % (writer.written, elapsed, writer.written / (elapsed / 60.0)),
                  flush=True)

        rows = generate.run(pairs, engine, sampling, engine.tokenizer,
                            samples_per_prompt=args.samples,
                            reading="stated", batch_size=args.batch_size,
                            on_rows=on_rows)
    print("wrote %d rows to %s in %.1f min" % (len(rows), args.out,
                                               (time.time() - started) / 60.0), flush=True)


if __name__ == "__main__":
    main()
