# Relaxed steering run notes — 22 August 2026

**A temporary record of one run, not part of the pipeline.** Nothing here is a
contract and nothing re-runs from it. The reproducible pipeline is
`scripts/run_steering.sh` and the scripts under `scripts/`; the run specifics
that would have gone into a wrapper are here instead.

## What was run, and what was not

Five games, not six. `prisoners_dilemma` was **not regenerated**: its relaxed
decision vector and its relaxed shuffled-label null are element-wise identical to
the strict ones (max absolute difference exactly 0.0 across all 29 layers — its
answer space is two points with no middle, so widening a pole has nothing to
widen), and every row it produced would have been a bit-identical regeneration of
`rows/prisoners_dilemma/`. `vectors/MANIFEST.json` marks both pairs
`identical_to_strict`. That saved 14 points, 1,400 generations and about 45
minutes. **Its strict numbers are its relaxed numbers; they are not an
independent run and README § 11 says so.**

    python -u scripts/prompting/run_sweep.py \
        --out-dir <pkg>/rows_relaxed \
        --provenance <pkg>/provenance/relaxed/sweep_provenance.json \
        --games ultimatum,overfishing,dictator,trust,apology --policy relaxed \
        --samples 100 --batch-size 20 --seed 0 --null-seed 20260821 \
        --device 1 --headroom-timeout 7200 --resume

wrapped in a scratch retry loop that would have re-invoked it with `--resume`
after a crash. **It was never needed: the sweep completed on the first attempt.**

`rows_relaxed/` rather than `rows/` because a row's filename carries its game, arm
and coefficient but **not its pole policy**, so a shared output root would have
had the second run overwrite the first. `--provenance` points inside
`provenance/relaxed/` for the same reason: `run_sweep.py` writes
`coefficients.csv` beside whatever provenance file it is given.

**Those paths were passed by hand here, and `run_steering.sh` did not derive
them.** At the time every output path in that script was policy-independent, so
following its own documented `POLICY=relaxed` instruction would have overwritten
the committed strict round instead of producing this one. That is fixed: the
script now derives rows, analysis, tables, logs and the null-build report from
`POLICY`, `scripts/tests/test_run_steering.py` pins both layouts and pins that a
relaxed run leaves every file a strict run wrote untouched, and `README.md` § 8
gives the invocation that reproduces this round.

Fixed and identical to round 1: `SEED=0`, `BATCH=20`, `SAMPLES=100`, reference
norm 10.5083084106, the same 11-rung real ladder and 3-rung null ladder, null
seed 20260821. Only the vectors changed.

## Timings

2026-08-22, 14:25:05Z to 16:39:53Z, 2 h 15 min for 70 points / 7,000 generations.

| game | minutes | round 1 (strict) |
|---|---|---|
| ultimatum | 17.5 | 32.7 |
| overfishing | 37.8 | 56.0 |
| dictator | 20.0 | 30.0 |
| trust | 40.4 | 63.6 |
| apology | 19.0 | 29.7 |

Every game is faster than its round-1 counterpart, and none of that is a design
change: round 1 ran on a card it shared with another tenant, and this run had
`cuda:1` to itself for its whole duration. The cost estimate reported before the
run was ~3 h 32 min, from round 1's own per-game times; the realised 2 h 15 min
is under it for that reason.

## The machine, and the other tenant

Two NVIDIA A40s, shared with another tenant running `ollama` model servers on
their own schedule. **`cuda:1` was pinned** — it reported 45,217 MiB free at
14:25 and the model loaded on the first attempt. `cuda:0`, which round 1 used,
was taken by the other tenant during this run (down to ~1.2 GiB free at 14:47);
that is exactly why the device is pinned and why the headroom gate exists, and it
cost this run nothing.

**No process belonging to the other tenant was signalled at any point.** No OOM,
no retry, no resume.

## Checked rather than assumed

* **`k = 0` is a shared no-op across the two ROUNDS, not just across the two
  arms.** All 10 of this run's `k = 0` CSVs (5 games x 2 arms) hold answers that
  are character-for-character identical to round 1's, on a different physical GPU
  two hours later. The only columns that differ are the four that describe which
  artifact was loaded: `steer_vector`, `steer_vector_norm`, `steer_vector_sha256`
  and `repo_commit`.
* **`P(altruistic)` is identical under both pole policies**, at all 154 points of
  both rounds — but that is a **property of the classifier, not a measurement**.
  All three pole functions return the altruistic pole on a policy-independent
  condition and relaxed widens only the SELF pole, so it could not have come out
  otherwise. It is worth stating because it is what lets the strict and relaxed
  halves be compared on exactly the same measure, with only the vector differing;
  it is not worth counting as a check that could have failed.
* Every relaxed null was written only after `build_nulls.py` rebuilt the committed
  real vector it is a null of from the archived activations. Max relative
  per-layer deviation across the six relaxed builds: **2.9e-08**, against a 1e-06
  refusal threshold (`provenance/null_vectors_relaxed.json`).

## Anything that went wrong

Nothing. One attempt, no OOM, no resume, no waiting in the headroom gate.
