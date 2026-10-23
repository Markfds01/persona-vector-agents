# Checkpoint 2 pilot report: general-question transfer

**Status: pilot. Every number below is a plumbing number, not a result.** The PREREG
(`PREREG.md`, frozen at `provenance/freeze.md`) is followed as written; what it left open is in
`DEVIATIONS.md` (pilot scope, served ids, interpreter, pairwise parameters, judge order).

**Pilot scope.** The Dictator strict decision vector, the trait vector and the Dictator's shuffled
and orthogonal nulls at k in {-3, 0, +3}; held-out and bridge sets; 3 samples per question (120 rows
per point). Generation: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c70...`, bf16, sdpa, layer 20,
positions all, norm unit, `unit_beta = k x 10.5083`, neutral preset, `use_model_defaults=False`,
`max_new_tokens` 1000, batch 20 seeded `seed + batch_index`, torch 2.7.1+cu126 / transformers 4.54.0
(not the interpreter the README names; DEVIATIONS). Judges: primary Llama-3.3-70B-Instruct Q8_0
(`logprob_0_100`), secondary gpt-oss-20b MXFP4 (`text_0_100`, effort low), both through the local
llama-server on PimientoRojo.

## 1. Generation: rows, statuses, truncation

| set | point | rows | generation status | truncated | mean new tokens | hook fires / expected | minutes |
|---|---|---|---|---|---|---|---|
| bridge | baseline k+0 | 120 | ok 120 | 0.0% | 62 | 621 / 621 | 0.5 |
| heldout | baseline k+0 | 120 | ok 120 | 0.0% | 380 | 3503 / 3503 | 4.8 |
| bridge | decision/dictator k+3 | 120 | ok 120 | 0.0% | 54 | 547 / 547 | 0.4 |
| heldout | decision/dictator k+3 | 120 | ok 120 | 0.0% | 322 | 3021 / 3021 | 4.2 |
| bridge | decision/dictator k-3 | 120 | ok 120 | 0.0% | 73 | 649 / 649 | 0.5 |
| heldout | decision/dictator k-3 | 120 | ok 120 | 0.0% | 412 | 3792 / 3792 | 5.3 |
| bridge | orthogonal_null/dictator k+3 | 120 | ok 120 | 0.0% | 74 | 711 / 711 | 0.5 |
| heldout | orthogonal_null/dictator k+3 | 120 | ok 120 | 0.0% | 356 | 3088 / 3088 | 2.5 |
| bridge | orthogonal_null/dictator k-3 | 120 | ok 120 | 0.0% | 59 | 566 / 566 | 0.4 |
| heldout | orthogonal_null/dictator k-3 | 120 | ok 120 | 0.0% | 439 | 3984 / 3984 | 3.4 |
| bridge | shuffled_null/dictator k+3 | 120 | ok 120 | 0.0% | 67 | 634 / 634 | 0.5 |
| heldout | shuffled_null/dictator k+3 | 120 | ok 120 | 0.0% | 336 | 2859 / 2859 | 2.3 |
| bridge | shuffled_null/dictator k-3 | 120 | ok 120 | 0.0% | 65 | 619 / 619 | 0.5 |
| heldout | shuffled_null/dictator k-3 | 120 | ok 120 | 0.0% | 441 | 4050 / 4050 | 4.3 |
| bridge | trait/altruism k+3 | 120 | ok 120 | 0.0% | 64 | 618 / 618 | 0.5 |
| heldout | trait/altruism k+3 | 120 | ok 120 | 0.0% | 350 | 3206 / 3206 | 4.4 |
| bridge | trait/altruism k-3 | 120 | ok 120 | 0.0% | 66 | 660 / 660 | 0.5 |
| heldout | trait/altruism k-3 | 120 | ok 120 | 0.0% | 411 | 3839 / 3839 | 5.5 |
| heldout | unhooked k+0 | 120 | ok 120 | 0.0% | 380 | unhooked | 4.8 |
| heldout | unhooked_repeat k+0 | 120 | ok 120 | 0.0% | 380 | unhooked | 4.9 |

## 2. Hook controls and the k = 0 byte-identity check

Verdict: **PASS**. The check is void unless all three positive controls hold (`.dm-knowledge`).

| check | result |
|---|---|
| k = 0 with the hook installed vs unhooked: rows with identical generated ids | 120 / 120 |
| (a) the unhooked run reproduces itself (unhooked vs unhooked_repeat), identical rows | 120 / 120 |
| (b) hook fire count equals its exact expectation (sum over batches of the longest row) and is > 0 | True |
| (c) every k != 0 point changes ids vs k = 0 (identical rows per point) | decision/dictator k+3: 0/120; decision/dictator k-3: 0/120; orthogonal_null/dictator k+3: 0/120; orthogonal_null/dictator k-3: 0/120; shuffled_null/dictator k+3: 0/120; shuffled_null/dictator k-3: 0/120; trait/altruism k+3: 0/120; trait/altruism k-3: 0/120 |

## 3. Judging: calls, statuses, throughput achieved

| judge | kind | calls | status |
|---|---|---|---|
| gpt-oss-20b-mxfp4 | coherence | 270 | ok 270 |
| gpt-oss-20b-mxfp4 | pairwise | 480 | ok 480 |
| gpt-oss-20b-mxfp4 | trait | 630 | ok 630 |
| llama-3.3-70b-instruct-q8_0 | coherence | 2160 | ok 2160 |
| llama-3.3-70b-instruct-q8_0 | pairwise | 1920 | ok 1920 |
| llama-3.3-70b-instruct-q8_0 | trait | 1440 | ok 1440 |

Seconds per call as achieved (calls actually sent, `cache_prompt: false`, `--parallel 1`; the PREREG planned 3.82 s for the primary and 1.00 s for the secondary, pairwise unmeasured):

| judge | calls sent | s/call overall | inter-call gap by kind |
|---|---|---|---|
| gpt-oss-20b-mxfp4 | 1380 | 1.05 | coherence 0.92 s (median 0.92); pairwise 1.35 s (median 1.32); trait 0.87 s (median 0.80) |
| llama-3.3-70b-instruct-q8_0 | 5520 | 4.98 | coherence 3.26 s (median 2.58); pairwise 8.05 s (median 8.12); trait 3.48 s (median 3.72) |

## 4. Per-point primary-judge scores on the held-out set (plumbing, not results)

Coherence exclusion is by the primary judge's coherence < 50 (or not scored); the trait mean is filtered with a question-clustered 95% interval (10,000 draws), unfiltered beside it.

| point | rows | trait status | coherence status | excluded | coherence unscored | trait, filtered [95% CI] | trait, unfiltered | coherence mean |
|---|---|---|---|---|---|---|---|---|
| baseline k+0 | 120 | ok 120 | ok 120 | 0.0% | 0.0% | 84.4 [78.7, 88.7] | 84.4 | 97.9 |
| decision/dictator k+3 | 120 | ok 120 | ok 120 | 1.7% | 0.0% | 85.6 [83.2, 87.6] | 85.6 | 92.9 |
| decision/dictator k-3 | 120 | ok 120 | ok 120 | 3.3% | 0.0% | 77.5 [69.7, 84.6] | 76.4 | 93.5 |
| orthogonal_null/dictator k+3 | 120 | ok 120 | ok 120 | 0.0% | 0.0% | 83.8 [80.6, 86.7] | 83.8 | 95.5 |
| orthogonal_null/dictator k-3 | 120 | ok 120 | ok 120 | 0.0% | 0.0% | 84.8 [79.7, 88.8] | 84.8 | 95.7 |
| shuffled_null/dictator k+3 | 120 | ok 120 | ok 120 | 0.0% | 0.0% | 84.6 [81.5, 87.1] | 84.6 | 94.9 |
| shuffled_null/dictator k-3 | 120 | ok 120 | ok 120 | 0.0% | 0.0% | 85.7 [81.5, 89.0] | 85.7 | 95.8 |
| trait/altruism k+3 | 120 | ok 120 | ok 120 | 5.0% | 0.0% | 94.4 [91.4, 96.4] | 93.7 | 88.4 |
| trait/altruism k-3 | 120 | ok 120 | ok 120 | 0.0% | 0.0% | 63.6 [54.8, 71.5] | 63.6 | 94.6 |

## 5. Determinism re-send (5% of each judge's distinct requests, seed 20261026, cold)

| judge | re-sent | exact match | same status | score changed | |delta| |
|---|---|---|---|---|---|
| gpt-oss-20b-mxfp4 | 69 | 61 / 69 | 69 / 69 | 8 of 69 | mean 0.957, max 10.000 |
| llama-3.3-70b-instruct-q8_0 | 276 | 193 / 276 | 276 / 276 | 83 of 276 | mean 0.035, max 1.657 |

* gpt-oss-20b-mxfp4 mismatch `orthogonal_null/dictator|heldout|k+3|heldout_00|s2|trait`: 20.0 -> 30.0
* gpt-oss-20b-mxfp4 mismatch `baseline|bridge|k+0|bridge_01|s2|trait`: 80.0 -> 90.0
* gpt-oss-20b-mxfp4 mismatch `trait/altruism|bridge|k-3|bridge_10|s2|trait`: 90.0 -> 80.0
* gpt-oss-20b-mxfp4 mismatch `shuffled_null/dictator|heldout|k-3|heldout_08|s0|pairwise|baseline_first`: 2.0 -> 1.0
* gpt-oss-20b-mxfp4 mismatch `shuffled_null/dictator|heldout|k-3|heldout_21|s2|coherence`: 90.0 -> 100.0
* llama-3.3-70b-instruct-q8_0 mismatch `baseline|heldout|k+0|heldout_15|s1|trait`: 90.00892408569563 -> 90.00702826715873
* llama-3.3-70b-instruct-q8_0 mismatch `orthogonal_null/dictator|heldout|k-3|heldout_09|s2|coherence`: 99.99868137344548 -> 99.99887821521554
* llama-3.3-70b-instruct-q8_0 mismatch `baseline|bridge|k+0|bridge_23|s2|trait`: 99.99999994575003 -> 99.99999995523594
* llama-3.3-70b-instruct-q8_0 mismatch `trait/altruism|bridge|k+3|bridge_03|s0|trait`: 99.99999997462886 -> 99.99999997177038
* llama-3.3-70b-instruct-q8_0 mismatch `shuffled_null/dictator|heldout|k-3|heldout_34|s1|trait`: 71.23075955313568 -> 70.94540527836708

## 6. Bridge set: parsed choice, and each judge's trait score against it

| point | rows | parse coverage | P(other-regarding) | Wilson 95% |
|---|---|---|---|---|
| baseline k+0 | 120 | 100.0% | 0.892 | [0.823, 0.936] |
| decision/dictator k+3 | 120 | 100.0% | 0.892 | [0.823, 0.936] |
| decision/dictator k-3 | 120 | 100.0% | 0.850 | [0.775, 0.903] |
| orthogonal_null/dictator k+3 | 120 | 100.0% | 0.883 | [0.814, 0.929] |
| orthogonal_null/dictator k-3 | 120 | 100.0% | 0.867 | [0.794, 0.916] |
| shuffled_null/dictator k+3 | 120 | 100.0% | 0.883 | [0.814, 0.929] |
| shuffled_null/dictator k-3 | 120 | 100.0% | 0.850 | [0.775, 0.903] |
| trait/altruism k+3 | 120 | 100.0% | 0.833 | [0.757, 0.889] |
| trait/altruism k-3 | 120 | 100.0% | 0.892 | [0.823, 0.936] |

AUC = P(trait score of an other-regarding choice > that of a self-interested choice), on the bridge AUC subset (k = 0 baseline and the trait vector at +-3), question-clustered bootstrap:

| judge | rows used / candidates | AUC | 95% CI | by k |
|---|---|---|---|---|
| gpt-oss-20b-mxfp4 | 360 / 360 | 0.957 | [0.902, 1.000] | k+0 0.951 (n=120), k+3 0.962 (n=120), k-3 0.978 (n=120) |
| llama-3.3-70b-instruct-q8_0 | 360 / 360 | 0.999 | [0.996, 1.000] | k+0 1.000 (n=120), k+3 1.000 (n=120), k-3 0.997 (n=120) |

## 7. Judge-judge agreement on the overlap (primary vs secondary; bootstrap over items, 2,000 draws)

| score | items | Spearman [95% CI] | quadratic-weighted kappa, binned | Krippendorff alpha, binned |
|---|---|---|---|---|
| trait_heldout | 270 (270 paired) | 0.776 [0.714, 0.826] | 0.438 [0.316, 0.551] | 0.384 [0.230, 0.520] |
| coherence_heldout | 270 (270 paired) | 0.560 [0.472, 0.635] | 0.058 [-0.035, 0.206] | 0.054 [-0.043, 0.207] |
| trait_bridge_auc_subset | 360 (360 paired) | 0.693 [0.627, 0.749] | 0.866 [0.788, 0.928] | 0.866 [0.788, 0.928] |

Pairwise choice agreement (same call, both judges ok): 384 of 480 (80.0%).

## 8. Pairwise (secondary metric)

| judge | calls | status | order consistency | position bias (P(first) - 0.5) | P(steered chosen) by point |
|---|---|---|---|---|---|
| gpt-oss-20b-mxfp4 | 480 | ok 480 | 57.1% | 0.206 | decision/dictator k+3 0.33; decision/dictator k-3 0.48; orthogonal_null/dictator k+3 0.43; orthogonal_null/dictator k-3 0.58; shuffled_null/dictator k+3 0.40; shuffled_null/dictator k-3 0.50; trait/altruism k+3 0.80; trait/altruism k-3 0.08 |
| llama-3.3-70b-instruct-q8_0 | 1920 | ok 1920 | 71.2% | 0.141 | decision/dictator k+3 0.42; decision/dictator k-3 0.36; orthogonal_null/dictator k+3 0.42; orthogonal_null/dictator k-3 0.45; shuffled_null/dictator k+3 0.40; shuffled_null/dictator k-3 0.44; trait/altruism k+3 0.93; trait/altruism k-3 0.04 |

## 9. Human calibration sheet

164 items, seed 20261025, drawn from the held-out pilot rows (`human/calibration_sheet.csv`, page `human/grader.html`). Strata:

| stratum | rows available | quota | drawn |
|---|---|---|---|
| decision k-3 | 120 | 18 | 18 |
| decision k+3 | 120 | 18 | 18 |
| relaxed_dictator_only k-3 | 0 | 18 | 0 |
| relaxed_dictator_only k+3 | 0 | 18 | 0 |
| trait k-3 | 120 | 18 | 18 |
| trait k+3 | 120 | 18 | 18 |
| shuffled_null k-3 | 120 | 18 | 18 |
| shuffled_null k+3 | 120 | 18 | 18 |
| orthogonal_null k-3 | 120 | 18 | 18 |
| orthogonal_null k+3 | 120 | 18 | 18 |
| baseline k+0 | 120 | 20 | 20 |

**Human-human, human-judge agreement: not run.** It needs the two labellers' exports (`human_labels_<reviewer>_<date>.csv`, one per labeller); `scripts/measurement/agreement.py` computes it once they exist.

## 10. What the tables show, and questions for the operator

* **Headroom.** The primary judge's mean trait score at the unsteered baseline is 84.4 (held-out, 0-100), so the positive end of the ladder has 15.6 points of headroom; at the trait vector's k = +3 it is 94.4. A ceiling compresses decision-minus-null contrasts at the positive end.
* **Coherence exclusion** is non-zero at: decision/dictator k+3 1.7%; decision/dictator k-3 3.3%; trait/altruism k+3 5.0%; no point reaches the 50% usable-coverage collapse that makes a point `undetermined`.
* **Judge-judge.** Held-out trait Spearman 0.776, coherence Spearman 0.560 (secondary scores coherence differently from the primary: binned kappa 0.058). Exclusion is always by the primary judge, so the coherence disagreement does not enter the filter.
* **Determinism.** gpt-oss repeated exactly on 61 of 69 re-sent requests (not bitwise reproducible, as measured on the bench); the primary's own count is in section 5.

### Questions for the operator

1. **D12, the "poor agreement" rule (PREREG 9.4).** Not pre-registered; yours to fix before any full
   judging, and it goes to `DEVIATIONS.md` with its date. What exists so far is judge-side only (sections
   6 and 7); the human numbers need your labels and your colleague's. The draft's candidate thresholds
   (human-judge Spearman lower bound < 0.5 on trait; bridge AUC lower bound <= 0.5) were proposals, not
   adopted. Decide the statistic, the threshold and what happens below it (change judge, change mode,
   stop).
2. **The human sheet has 164 items, not about 200.** The pilot has no relaxed and no Dictator-only arm, so
   those two strata (2 x 18) are empty and were not topped up (DEVIATIONS). Accept 164, or ask for the
   two missing strata to be generated and judged (cheap on the generation side, about 36 more rows).
3. **Your colleague's identity** (name or initials) for `provenance/` (D10), and how the colleague gets the
   page: send **only** `human/grader.html` (or `human/calibration_sheet.csv` plus the page). Do **not**
   send `provenance/human_sample_key.json` or `human/judge_reveal_*.json`: the key names each item's arm and
   k, and the reveal files hold the judges' scores. Audit mode loads a reveal file only after an item is
   rated and freezes that rating.
4. **Served ids.** The servers answer to their model paths, not the `--alias` ids J10 expected. Keep the
   path ids (DEVIATIONS), or relaunch with `--alias` before the full run.
5. **Interpreter.** The pilot generated under torch 2.7.1 / transformers 4.54.0. If you want the full run
   under the earlier pins (2.6.0+cu124 / 4.52.3), supply that interpreter; the k = 0 rows are then
   comparable with the per-game study's, which this pilot's are not guaranteed to be.
6. **Pairwise parameters** (`max_tokens` 32 on the primary, 2048 on the secondary; no format instruction
   appended; template as frozen): approve or change (DEVIATIONS).
7. **Judging time.** Planned 37.3 h (44.3 h with long pairwise calls) for the full run; at the speeds this
   pilot achieved:

| judge | planned calls (PREREG 10) | pilot s/call overall | projected hours at the pilot's speeds |
|---|---|---|---|
| llama-3.3-70b-instruct-q8_0 | 33264 | 4.98 | 45.4 h |
| gpt-oss-20b-mxfp4 | 7182 | 1.05 | 2.1 h |

   The primary's 10560 pairwise calls (held-out, +-3, both orders) take 23.6 h of that at 8.05 s/call (a pairwise prompt holds two answers; single-answer scoring runs 3.3-3.5 s), so they are about half the primary's time. **Question: trim the pairwise pass** (e.g. one order only, or the primary on a 25% subset like the secondary), or keep it as pre-registered? Dropping it entirely is the PREREG's `div8_alt`.

8. **Anything flagged in sections 1-8** (truncation, coherence exclusion, a status other than `ok`, a
   determinism mismatch, an AUC interval reaching 0.5) is read off the tables above; nothing has been
   dropped, imputed or re-asked, and nothing beyond the pilot has run.

## 11. What was and was not run

Run: pilot generation (2,400 rows), both judges' pilot judging, the determinism re-send, the bridge AUC,
judge-judge agreement, pairwise, the human sheet and page export. **Not run:** human-human and
human-judge agreement (no labels exist), the in-distribution set, any full-study generation or judging,
the optional test-retest. Full judging does not start until you approve.
