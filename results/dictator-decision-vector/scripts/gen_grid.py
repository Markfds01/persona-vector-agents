"""Generate the Dictator grid at the `neutral` preset and write scored rows.

One model load, one engine, one sampling configuration. Rows are flushed per
batch so a GPU eviction keeps everything generated so far - the device is shared
and another tenant cycles models on it without warning.

Nothing about the run is defaulted here that `audit.generate.run` refuses to
default: samples_per_prompt, batch_size and reading are all explicit.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import generate
from scratch import decision_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument("--samples", type=int, required=True, help="samples per prompt")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    args = parser.parse_args()

    import torch

    preset = generate.preset("neutral")
    sampling = preset.sampling.with_seed(args.seed)

    game_ids = decision_grid.register()
    pairs = [(game_id, "free") for game_id in game_ids]

    engine = generate.HuggingFaceEngine.load(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=preset.attn_implementation,
        device_map={"": 0},
    )
    preset.check_engine(engine.describe())
    print("engine:", engine.describe(), flush=True)

    started = time.time()
    with generate.RowWriter(args.out) as writer:
        def on_rows(rows):
            writer.write(rows)
            print("%5d rows  %6.1fs" % (writer.written, time.time() - started), flush=True)

        rows = generate.run(pairs, engine, sampling, engine.tokenizer,
                            samples_per_prompt=args.samples,
                            reading="stated", batch_size=args.batch_size,
                            on_rows=on_rows)
    print("wrote %d rows to %s in %.1f min" % (len(rows), args.out,
                                               (time.time() - started) / 60.0), flush=True)


if __name__ == "__main__":
    main()
