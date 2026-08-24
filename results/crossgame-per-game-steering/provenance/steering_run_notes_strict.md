# Steering run notes — 22 August 2026

**A temporary record of one run, not part of the pipeline.** Nothing here is a
contract and nothing re-runs from it. It describes the machine and the schedule
the sweep happened on, and it can be deleted once that stops being worth
remembering. The reproducible pipeline is `scripts/run_steering.sh` and the
scripts under `scripts/`; the run specifics that would have gone into a wrapper
are here instead, because a wrapper hardcoding one machine's interpreter and one
worktree's paths is exactly what this repository stopped committing.

## How it was launched

One process for all six games, holding the weights throughout:

    python -u scripts/prompting/run_sweep.py \
        --out-dir <pkg>/rows --provenance <pkg>/provenance/sweep_provenance.json \
        --samples 100 --batch-size 20 --seed 0 --device 0 \
        --headroom-timeout 7200 --resume

wrapped in a scratch retry loop that would have re-invoked it with `--resume`
after a crash. **It was never needed: the sweep completed on the first attempt.**

One process rather than one per game because each load is a fresh wait for
headroom on a contended card, and the per-beta resume already gives a finer
recovery granularity than per-game restart would.

Fixed: `SEED=0`, `BATCH=20`, `DEVICE=0` (`cuda:0`), `SAMPLES=100`, strict pole
policy, null-vector seed 20260821.

## Timings

2026-08-22, 08:01:15Z to 12:18:10Z, 4 h 17 min for 84 points / 8,400 generations.

| game | minutes | | game | minutes |
|---|---|---|---|---|
| dictator | 30.0 | | apology | 29.7 |
| trust | 63.6 | | overfishing | 56.0 |
| ultimatum | 32.7 | | prisoners_dilemma | 44.7 |

Trust and Overfishing are the slow ones because their `free`-mode answers are the
longest; the Prisoner's Dilemma's cost is concentrated in its two most positive
rungs, where the model produces long Chinese reasoning (README §3).

The cost was estimated and reported before the run rather than discovered during
it: 16,800 generations at the repo standard n=200 (~7 h) against 8,400 at n=100
(~3.5 h), from a measured 4.75–6.25 min per 200-sample point. The realised 4 h 17
min at n=100 is above that estimate, because the estimate was calibrated on the
Dictator and two of the six games are roughly twice as slow as it.

## The machine, and the other tenant

One NVIDIA A40 (44.4 GiB usable) out of two on the host, shared with another
tenant running `ollama` model servers on both cards on their own schedule. The
second card was held continuously at ~38.7 GiB for the whole session and was never
available; the first swung between roughly 1.2 and 32 GiB free against a 15.2 GiB
model.

**The sweep itself never had to wait.** It found 23,584 MiB free at 08:01:15Z and
loaded on the first attempt (`sweep.log`), and no OOM, no retry and no resume
occurred in the following four hours. The earlier sanity pass *did* sit in the
headroom gate through two dips before it could load, which is why the gate is
there; the sweep was simply lucky in its start time. That is the *situation*, and
the code's own reasons for gating, for retrying only the load, and for resuming
per beta point are stated in the code without reference to it.

No process belonging to the other tenant was signalled at any point.

## Anything that went wrong

Nothing during the sweep itself. Before it, a sanity pass over two games
(dictator and prisoners_dilemma, k = 0 and ±1, n = 20) was run and discarded; it
confirmed the plumbing, the output schema and the throughput, and its output is
not committed.

Two facts about the environment, recorded because the next run will want them:
the interpreter has to be one with `torch` 2.6.0+cu124 and `transformers` 4.52.3
(the versions every committed number in this repository was produced under), and
the activation shards `build_nulls.py` needs are ~7 GB and live outside the
checkout.
