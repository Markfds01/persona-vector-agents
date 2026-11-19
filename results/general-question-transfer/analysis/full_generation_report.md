# Full generation run — report

Written by `scripts/validation/full_generation_check.py` from `rows_full/` and `provenance/generation_full/` (numbers also in `provenance/generation_full/integrity.json`). Generation only: nothing has been judged. Blinding: counts only, no answer text.

## Checks

| check | result |
|---|---|
| all_points_present | True |
| all_points_whole_and_consistent | True |
| no_unplanned_files | True |
| hook_fired_exactly_as_expected | True |
| k0_identity_verdict | pass |
| every_row_names_the_full_run_interpreter | True |
| pass | True |

k = 0 identity: hooked baseline vs unhooked identical rows (120, 120); unhooked vs repeat (120, 120); k != 0 points identical to k = 0 (max over points) (0, 120); verdict **pass**.

Descriptive (not a check): the pilot's k = 0 baseline (torch 2.7.1 / transformers 4.54.0) against the full run's (torch 2.6.0 / transformers 4.52.3), identical output ids [identical, rows]: {'heldout': [120, 120], 'bridge': [120, 120]}. Full-run rows whose (row id, output ids) also occur in the pilot: 2400.

## Counts

Points: 111 planned, 111 present. Rows: 13320 (heldout 6120, in_distribution 1800, bridge 5400).
The study's 13,200 rows are these minus the `unhooked_repeat` control (120 rows).

| set | arm | k | rows | truncated | refusal (regex) | mean tokens | max tokens | minutes | device |
|---|---|---|---|---|---|---|---|---|---|
| bridge | baseline | +0 | 120 | 0 | 0 | 62 | 126 | 1.0 | cuda:1 |
| heldout | baseline | +0 | 120 | 0 | 1 | 380 | 661 | 6.1 | cuda:0 |
| in_distribution | baseline | +0 | 120 | 0 | 0 | 344 | 654 | 6.3 | cuda:1 |
| bridge | decision/apology | +3 | 120 | 0 | 0 | 56 | 108 | 0.8 | cuda:1 |
| heldout | decision/apology | +3 | 120 | 0 | 0 | 343 | 899 | 6.2 | cuda:0 |
| in_distribution | decision/apology | +3 | 120 | 0 | 0 | 313 | 544 | 4.6 | cuda:1 |
| bridge | decision/apology | -3 | 120 | 0 | 0 | 73 | 125 | 1.0 | cuda:1 |
| heldout | decision/apology | -3 | 120 | 0 | 2 | 428 | 687 | 6.5 | cuda:0 |
| in_distribution | decision/apology | -3 | 120 | 0 | 1 | 374 | 709 | 5.8 | cuda:1 |
| bridge | decision/dictator | +3 | 120 | 0 | 0 | 54 | 102 | 0.9 | cuda:1 |
| heldout | decision/dictator | +3 | 120 | 0 | 0 | 322 | 662 | 5.1 | cuda:0 |
| in_distribution | decision/dictator | +3 | 120 | 0 | 1 | 319 | 600 | 4.8 | cuda:1 |
| bridge | decision/dictator | -3 | 120 | 0 | 0 | 73 | 132 | 1.0 | cuda:1 |
| heldout | decision/dictator | -3 | 120 | 0 | 2 | 412 | 694 | 6.7 | cuda:0 |
| in_distribution | decision/dictator | -3 | 120 | 0 | 0 | 376 | 707 | 6.4 | cuda:1 |
| bridge | decision/overfishing | +3 | 120 | 0 | 0 | 53 | 99 | 0.8 | cuda:1 |
| heldout | decision/overfishing | +3 | 120 | 0 | 0 | 364 | 736 | 5.9 | cuda:0 |
| in_distribution | decision/overfishing | +3 | 120 | 0 | 0 | 353 | 651 | 5.4 | cuda:1 |
| bridge | decision/overfishing | -3 | 120 | 0 | 0 | 74 | 164 | 1.2 | cuda:1 |
| heldout | decision/overfishing | -3 | 120 | 0 | 0 | 402 | 658 | 6.3 | cuda:0 |
| in_distribution | decision/overfishing | -3 | 120 | 0 | 1 | 375 | 668 | 6.1 | cuda:1 |
| bridge | decision/prisoners_dilemma | +3 | 120 | 0 | 0 | 54 | 96 | 0.8 | cuda:1 |
| heldout | decision/prisoners_dilemma | +3 | 120 | 0 | 0 | 293 | 575 | 4.8 | cuda:0 |
| in_distribution | decision/prisoners_dilemma | +3 | 120 | 0 | 0 | 283 | 532 | 4.3 | cuda:1 |
| bridge | decision/prisoners_dilemma | -3 | 120 | 0 | 0 | 73 | 147 | 1.0 | cuda:1 |
| heldout | decision/prisoners_dilemma | -3 | 120 | 0 | 1 | 453 | 924 | 7.3 | cuda:0 |
| in_distribution | decision/prisoners_dilemma | -3 | 120 | 0 | 2 | 432 | 771 | 7.1 | cuda:1 |
| bridge | decision/trust | +3 | 120 | 0 | 0 | 53 | 115 | 0.8 | cuda:1 |
| heldout | decision/trust | +3 | 120 | 0 | 0 | 325 | 524 | 4.7 | cuda:0 |
| in_distribution | decision/trust | +3 | 120 | 0 | 0 | 307 | 575 | 3.8 | cuda:1 |
| bridge | decision/trust | -3 | 120 | 0 | 0 | 73 | 119 | 1.0 | cuda:1 |
| heldout | decision/trust | -3 | 120 | 0 | 0 | 407 | 653 | 6.4 | cuda:0 |
| in_distribution | decision/trust | -3 | 120 | 0 | 2 | 369 | 763 | 6.1 | cuda:1 |
| bridge | decision/ultimatum | +3 | 120 | 0 | 0 | 60 | 108 | 0.9 | cuda:1 |
| heldout | decision/ultimatum | +3 | 120 | 0 | 0 | 369 | 616 | 5.7 | cuda:0 |
| in_distribution | decision/ultimatum | +3 | 120 | 0 | 0 | 350 | 560 | 5.2 | cuda:1 |
| bridge | decision/ultimatum | -3 | 120 | 0 | 0 | 69 | 120 | 1.0 | cuda:1 |
| heldout | decision/ultimatum | -3 | 120 | 0 | 0 | 413 | 735 | 6.7 | cuda:0 |
| in_distribution | decision/ultimatum | -3 | 120 | 0 | 1 | 378 | 675 | 6.4 | cuda:1 |
| bridge | dictator_only | +3 | 120 | 0 | 0 | 53 | 110 | 0.8 | cuda:1 |
| heldout | dictator_only | +3 | 120 | 0 | 0 | 333 | 637 | 5.2 | cuda:0 |
| bridge | dictator_only | -3 | 120 | 0 | 0 | 73 | 130 | 1.1 | cuda:1 |
| heldout | dictator_only | -3 | 120 | 0 | 1 | 417 | 680 | 6.7 | cuda:0 |
| bridge | orthogonal_null/apology | +3 | 120 | 0 | 0 | 71 | 138 | 1.0 | cuda:1 |
| heldout | orthogonal_null/apology | +3 | 120 | 0 | 0 | 387 | 636 | 5.5 | cuda:1 |
| bridge | orthogonal_null/apology | -3 | 120 | 0 | 0 | 61 | 99 | 0.9 | cuda:1 |
| heldout | orthogonal_null/apology | -3 | 120 | 0 | 0 | 367 | 651 | 6.1 | cuda:1 |
| bridge | orthogonal_null/dictator | +3 | 120 | 0 | 0 | 74 | 176 | 1.1 | cuda:1 |
| heldout | orthogonal_null/dictator | +3 | 120 | 0 | 0 | 356 | 624 | 5.2 | cuda:1 |
| bridge | orthogonal_null/dictator | -3 | 120 | 0 | 0 | 59 | 124 | 0.9 | cuda:1 |
| heldout | orthogonal_null/dictator | -3 | 120 | 0 | 0 | 439 | 753 | 7.1 | cuda:1 |
| bridge | orthogonal_null/overfishing | +3 | 120 | 0 | 0 | 61 | 112 | 0.9 | cuda:1 |
| heldout | orthogonal_null/overfishing | +3 | 120 | 0 | 0 | 362 | 605 | 5.3 | cuda:1 |
| bridge | orthogonal_null/overfishing | -3 | 120 | 0 | 0 | 64 | 127 | 0.9 | cuda:1 |
| heldout | orthogonal_null/overfishing | -3 | 120 | 0 | 0 | 400 | 644 | 6.1 | cuda:1 |
| bridge | orthogonal_null/prisoners_dilemma | +3 | 120 | 0 | 0 | 67 | 123 | 1.0 | cuda:1 |
| heldout | orthogonal_null/prisoners_dilemma | +3 | 120 | 0 | 1 | 376 | 651 | 5.6 | cuda:1 |
| bridge | orthogonal_null/prisoners_dilemma | -3 | 120 | 0 | 0 | 61 | 117 | 0.9 | cuda:1 |
| heldout | orthogonal_null/prisoners_dilemma | -3 | 120 | 0 | 0 | 388 | 643 | 6.0 | cuda:1 |
| bridge | orthogonal_null/trust | +3 | 120 | 0 | 0 | 61 | 113 | 1.0 | cuda:1 |
| heldout | orthogonal_null/trust | +3 | 120 | 0 | 0 | 340 | 548 | 5.1 | cuda:1 |
| bridge | orthogonal_null/trust | -3 | 120 | 0 | 0 | 62 | 118 | 0.9 | cuda:1 |
| heldout | orthogonal_null/trust | -3 | 120 | 0 | 0 | 412 | 827 | 7.3 | cuda:1 |
| bridge | orthogonal_null/ultimatum | +3 | 120 | 0 | 0 | 70 | 124 | 1.0 | cuda:1 |
| heldout | orthogonal_null/ultimatum | +3 | 120 | 0 | 0 | 423 | 685 | 6.5 | cuda:1 |
| bridge | orthogonal_null/ultimatum | -3 | 120 | 0 | 0 | 58 | 135 | 0.9 | cuda:1 |
| heldout | orthogonal_null/ultimatum | -3 | 120 | 0 | 0 | 347 | 535 | 5.1 | cuda:1 |
| bridge | relaxed/overfishing | +3 | 120 | 0 | 0 | 62 | 130 | 1.0 | cuda:1 |
| heldout | relaxed/overfishing | +3 | 120 | 0 | 0 | 348 | 520 | 4.8 | cuda:0 |
| bridge | relaxed/overfishing | -3 | 120 | 0 | 0 | 62 | 101 | 0.9 | cuda:1 |
| heldout | relaxed/overfishing | -3 | 120 | 0 | 3 | 402 | 659 | 6.2 | cuda:0 |
| bridge | relaxed/ultimatum | +3 | 120 | 0 | 0 | 58 | 119 | 0.9 | cuda:1 |
| heldout | relaxed/ultimatum | +3 | 120 | 0 | 0 | 351 | 580 | 5.1 | cuda:0 |
| bridge | relaxed/ultimatum | -3 | 120 | 0 | 0 | 67 | 110 | 0.9 | cuda:1 |
| heldout | relaxed/ultimatum | -3 | 120 | 0 | 0 | 381 | 658 | 6.3 | cuda:0 |
| bridge | shuffled_null/apology | +3 | 120 | 0 | 0 | 66 | 180 | 1.1 | cuda:1 |
| heldout | shuffled_null/apology | +3 | 120 | 0 | 0 | 360 | 671 | 5.7 | cuda:0 |
| bridge | shuffled_null/apology | -3 | 120 | 0 | 0 | 68 | 112 | 0.9 | cuda:1 |
| heldout | shuffled_null/apology | -3 | 120 | 0 | 0 | 386 | 726 | 6.8 | cuda:0 |
| bridge | shuffled_null/dictator | +3 | 120 | 0 | 0 | 67 | 131 | 1.0 | cuda:1 |
| heldout | shuffled_null/dictator | +3 | 120 | 0 | 0 | 336 | 566 | 4.8 | cuda:0 |
| bridge | shuffled_null/dictator | -3 | 120 | 0 | 0 | 65 | 117 | 1.0 | cuda:1 |
| heldout | shuffled_null/dictator | -3 | 120 | 0 | 0 | 441 | 725 | 7.3 | cuda:0 |
| bridge | shuffled_null/overfishing | +3 | 120 | 0 | 0 | 61 | 96 | 0.8 | cuda:1 |
| heldout | shuffled_null/overfishing | +3 | 120 | 0 | 0 | 365 | 598 | 5.4 | cuda:0 |
| bridge | shuffled_null/overfishing | -3 | 120 | 0 | 0 | 61 | 134 | 1.0 | cuda:1 |
| heldout | shuffled_null/overfishing | -3 | 120 | 0 | 1 | 402 | 673 | 6.2 | cuda:0 |
| bridge | shuffled_null/prisoners_dilemma | +3 | 120 | 0 | 0 | 66 | 122 | 1.0 | cuda:1 |
| heldout | shuffled_null/prisoners_dilemma | +3 | 120 | 0 | 0 | 372 | 603 | 5.3 | cuda:0 |
| bridge | shuffled_null/prisoners_dilemma | -3 | 120 | 0 | 0 | 64 | 130 | 1.0 | cuda:1 |
| heldout | shuffled_null/prisoners_dilemma | -3 | 120 | 0 | 1 | 412 | 803 | 7.0 | cuda:0 |
| bridge | shuffled_null/trust | +3 | 120 | 0 | 0 | 62 | 168 | 1.0 | cuda:1 |
| heldout | shuffled_null/trust | +3 | 120 | 0 | 0 | 353 | 555 | 5.4 | cuda:0 |
| bridge | shuffled_null/trust | -3 | 120 | 0 | 0 | 61 | 98 | 0.8 | cuda:1 |
| heldout | shuffled_null/trust | -3 | 120 | 0 | 0 | 406 | 769 | 6.5 | cuda:0 |
| bridge | shuffled_null/ultimatum | +3 | 120 | 0 | 0 | 69 | 139 | 0.9 | cuda:1 |
| heldout | shuffled_null/ultimatum | +3 | 120 | 0 | 1 | 423 | 685 | 6.9 | cuda:0 |
| bridge | shuffled_null/ultimatum | -3 | 120 | 0 | 0 | 61 | 148 | 1.1 | cuda:1 |
| heldout | shuffled_null/ultimatum | -3 | 120 | 0 | 0 | 357 | 563 | 5.3 | cuda:0 |
| heldout | trait/altruism | +1 | 120 | 0 | 1 | 370 | 550 | 4.3 | cuda:0 |
| bridge | trait/altruism | +3 | 120 | 0 | 0 | 64 | 116 | 1.0 | cuda:1 |
| heldout | trait/altruism | +3 | 120 | 0 | 0 | 350 | 643 | 2.6 | cuda:1 |
| in_distribution | trait/altruism | +3 | 120 | 0 | 1 | 319 | 547 | 5.1 | cuda:1 |
| heldout | trait/altruism | +5 | 120 | 0 | 0 | 323 | 520 | 2.2 | cuda:0 |
| heldout | trait/altruism | -1 | 120 | 0 | 0 | 394 | 650 | 6.3 | cuda:0 |
| bridge | trait/altruism | -3 | 120 | 0 | 0 | 66 | 126 | 1.0 | cuda:1 |
| heldout | trait/altruism | -3 | 120 | 0 | 0 | 411 | 718 | 5.8 | cuda:1 |
| in_distribution | trait/altruism | -3 | 120 | 0 | 0 | 401 | 701 | 6.4 | cuda:1 |
| heldout | trait/altruism | -5 | 120 | 0 | 0 | 457 | 755 | 7.4 | cuda:0 |
| heldout | unhooked | +0 | 120 | 0 | 1 | 380 | 661 | 6.1 | cuda:0 |
| heldout | unhooked_repeat | +0 | 120 | 0 | 1 | 380 | 661 | 6.1 | cuda:0 |

## Truncation and refusal by set

| set | rows | truncated | share | refusal (regex) | share |
|---|---|---|---|---|---|
| heldout | 6120 | 0 | 0.00% | 17 | 0.28% |
| in_distribution | 1800 | 0 | 0.00% | 9 | 0.50% |
| bridge | 5400 | 0 | 0.00% | 0 | 0.00% |

Bridge parse statuses over all bridge rows (`scripts/measurement/bridge_parse.py`; the choices themselves are not reported here): {'conflicting': 11, 'ok': 5389}.

The refusal column is the per-game study's regex (`audit.parse._REFUSAL_RX`), a heuristic built for game answers; the judges' REFUSAL status is the measure the analysis uses.

## Run time

Sum of per-point generation minutes over all processes: 424 (7.1 GPU-process hours); the wall-clock window is in the part logs (`provenance/generation_full/generation_part*.log`).

Wall clock, first part start to last part end: 2026-11-03T11:12:30Z to 2026-11-03T13:04:35Z.

## How the run went (2026-11-03)

* Interpreter: torch 2.6.0+cu124 / transformers 4.52.3 (`provenance/interpreter_full_run.md`); every row
  names it (checked above).
* Processes: parts 1-4 (`scripts/run_full_generation.sh`) started 11:12:30 UTC, two per A40. When part 4
  finished (11:37:48), parts 1 and 2 were stopped and their remaining held-out points relaunched as parts
  5-7 (DEVIATIONS 2026-11-03, "Full-run processes rebalanced"); the two points in progress were
  regenerated whole. Part 3's driver printed a shell syntax error after its last point (12:54:29): the
  driver file had been edited (parts 5-7 added) while that bash was still reading it. Every point of
  part 3 had already been written and checked; only its closing `PART 3 COMPLETE` line is missing.
* The full run reproduces the pilot byte for byte where they overlap: all 2,400 pilot rows (same arm,
  set, k, question and sample) have identical output ids in the full run, although the pilot ran under
  torch 2.7.1+cu126 / transformers 4.54.0. So, for this model and these points, the interpreter change
  did not change a single generated token.
* Nothing was judged; the counts above contain no score.

