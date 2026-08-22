# Generation run notes — August 2026

**A temporary record of one past run, not part of the pipeline.** Nothing here is
a contract and nothing re-runs from it. It describes the machine and the schedule
stage 2 (generation) happened on, and it can be deleted once that stops being
worth remembering. The reproducible pipeline is `scripts/run_extraction.sh` and
`scripts/run_analysis.sh`; stage 2 was not re-run for this revision and its output
is committed under `extraction/`.

It exists because the five shell wrappers that launched stage 2 — `run_all.sh`,
`main_run.sh`, `resume_run.sh`, `finish.sh`, `finish_pd.sh` — are gone rather than
kept beside the pipeline code. They hardcoded a worktree that no longer exists, a
`.venv` interpreter inside it, an output directory outside the repository, and an
`extract_crossgame.py` that was never committed; that step is now `lab/extract.py`,
driven by `run_extraction.sh`. The logs they wrote are still here: `main_run.log`,
`resume_run.log`, `finish.log`, `finish_pd.log`.

## How the run was launched

One game at a time, three separate processes per game:

1. **generate** — `gen_crossgame.py --out <rows>.csv --samples N --batch-size 32
   --seed 0 --families <one game> --device 0`
2. **capture** — `extract_crossgame.py --rows <rows>.csv --out-dir <acts>/<game>
   --device 0`, then the rows CSV copied beside the shards as `rows.csv`
   (superseded by `python -m lab.extract`)
3. **land** — `land_family.py --family <game> --rows <acts>/<game>/rows.csv
   --acts <acts>/<game> --out <out>`

A failure stopped the run. Nothing was retried in a loop: a game that failed was a
result to report, and the games already landed stayed landed.

Fixed for every game: `SEED=0`, `BATCH=32`, `DEVICE=0` (`cuda:0`). The wrappers
read `SAMPLES`, `BATCH`, `SEED`, `DEVICE`, `FAMILIES` and `OUT` (the activation
root, outside the repository) from the environment; nothing else was set.

Family order and samples per cell — the reasoning for the counts is `METHOD.md` §3:

| pass | families | samples per cell |
|---|---|---|
| 1 | overfishing | 48 |
| 2 | dictator, apology, trust, ultimatum | 24 |
| 3 | prisoners_dilemma | 8 |

Ordered so that an interruption cost the least important game: the game needing the
most samples first, the PD census last.

## What interrupted it, and what that forced

The card was shared with another tenant that loaded and unloaded models on its own
schedule, swinging between roughly 21.6 and 27.4 GiB against a 15.2 GiB model
during this run (and from 21 to 43 GiB and back inside a minute during the later PD
probe). That is the *situation*; the code's own reasons for landing per game and
for retrying a model load are stated in the code without reference to it.

What the resume wrapper added on top of the plain runner:

* **A free-memory precheck** before each stage — wait until the card reports at
  least 16600 MiB free (weights ~15.2 GiB plus a working set), give up after an
  hour.
* **A batch-size ladder**, 32 → 16 → 8 → 4, walking down until one size survived
  instead of waiting for a tenant that might never leave. Batch size changes WHICH
  draws happen — `audit.generate.run` seeds each batch as `seed + batch_index` —
  but not the distribution drawn from, and every row records its own `batch_size`.
  In the event every landed game generated at batch 32.
* **Retry at the granularity of a whole game, never a batch.** Resuming a
  half-finished game from its CSV would have generated the rest under a different
  batch composition than its seed implies. Restarting a game cost ~25 minutes.

Batch size was never the cause: the peak above the weights is ~1.4 GiB, against a
~6 GiB swing from the other tenant. The protection that mattered was refusing to
start until the card had room.

Two games were produced by their second attempt: `trust` (a CUDA OOM inside a
forward pass, then a second OOM on its first retry) and `prisoners_dilemma` (the
generation completed and reported success but its CSV could not be read —
`_csv.Error: line contains NUL` — and it was regenerated with the same parameters).
Each game is internally seed-consistent; the set was not produced by one
uninterrupted invocation. `README.md` §13 carries that as a property of the
committed data.
