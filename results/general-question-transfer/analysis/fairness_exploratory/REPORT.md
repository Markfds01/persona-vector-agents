# EXPLORATORY: fairness vs altruism on the checkpoint-2 pilot (report)

**Status: EXPLORATORY, post-hoc.** Pre-registered in `docs/fairness_exploratory_prereg.md` (commit ca333f7, unchanged). Not part of the confirmatory plan (`results/general-question-transfer/PREREG.md`). Logged in `results/general-question-transfer/DEVIATIONS.md`. Nothing here is a confirmatory result, and the pilot numbers are plumbing, not results.

Blinding: aggregates only. No individual general-question answer is quoted; the anchor responses (hand-written, not study outputs) are the only text shown, and only in the anchor section. Per-row scores are in `scores_by_row.csv` (row id, arm, k, question, sample, both scores, no answer text).

## 0. Headline

* **The validity check failed.** The fairness judge does not order the anchors as pre-registered: it scores all three *generous* anchors at 0 (the same as the selfish ones), so generous ≈ neutral does not hold. Every fairness result below is labelled as failing its validity check (prereg § 4).
* **P2 and P3 are supported; P1 is supported only against the trait vector.** Ours at +3 does not raise altruism (no change against baseline). The fairness gap against the trait vector comes mainly from the trait vector *lowering* fairness (58.8 vs 85.2 at baseline). Ours at +3 does **not** differ from baseline on fairness (+1.8, CI [−2.0, 6.2]) or from the two nulls. So "ours improves fairness" is not shown; "the trait vector hurts fairness" is what the data show.
* **Discriminant validity holds:** r = 0.22 between the fairness and altruism judges over the baseline and null rows (below the 0.8 threshold).

## 1. Anchor check (prereg § 4), primary judge `llama-3.3-70b-instruct-q8_0`

12 anchors, 3 questions × 4 kinds, one call each per rubric. Scores 0–100, logprob_0_100. Source: `anchor_check.json`.

| anchor kind | fairness (3 anchors) | altruism (3 anchors) |
|---|---|---|
| fair | 90.0, 90.0, 90.1 (mean 90.0) | 73.5, 28.9, 89.8 (mean 64.1) |
| generous | 0.0, 0.0, 0.0 (mean 0.0) | 100.0, 100.0, 100.0 (mean 100.0) |
| neutral | 55.7, 24.0, 80.2 (mean 53.3) | 49.3, 46.2, 77.4 (mean 57.6) |
| selfish | 0.0, 0.0, 0.0 (mean 0.0) | 0.0, 0.0, 0.0 (mean 0.0) |

Pre-registered fairness rule: mean(fair) > max(mean(generous), mean(neutral)): 90.0 > 53.3 **passes**. min(mean(generous), mean(neutral)) > mean(selfish): 0.0 > 0.0 **fails**. |mean(generous) − mean(neutral)| < 10: 53.3 **fails**. **Overall: FAIL** (`fairness_check_pass: false`).

Reading: the fairness judge separates *fair* from the rest. It scores every generous and every selfish anchor at the floor, so it does not rank generosity above selfishness. Generosity is penalised as hard as selfishness, consistent with the rubric's clause "giving everything away while ignoring others' or one's own legitimate claims is not fair", but it means the fairness scale has no resolution at the bottom. The altruism judge orders the anchors as expected (generous 100 > fair 64 > neutral 58 > selfish 0), so the two judges separate on what each is meant to measure.

## 2. Scores by point (fairness and altruism side by side)

Primary judge (llama-3.3-70b-instruct-q8_0). "analysed" = rows with both scores ok and coherence ≥ 50 (prereg § 5). Means are over analysed rows. Refusals: 0 at every point. All 1,080 fairness calls returned status ok.

| point | rows | analysed | fairness mean | altruism mean |
|---|---|---|---|---|
| baseline (k=0) | 120 | 120 | 85.17 | 84.42 |
| decision −3 | 120 | 116 | 83.42 | 77.54 |
| decision +3 | 120 | 118 | 86.92 | 85.58 |
| trait −3 | 120 | 120 | 81.83 | 63.58 |
| trait +3 | 120 | 114 | 58.78 | 94.42 |
| shuffled null −3 | 120 | 120 | 86.13 | 85.66 |
| shuffled null +3 | 120 | 120 | 85.86 | 84.58 |
| orthogonal null −3 | 120 | 120 | 86.14 | 84.77 |
| orthogonal null +3 | 120 | 120 | 85.94 | 83.85 |

Secondary judge (gpt-oss-20b-mxfp4), descriptive only, 30 rows per point, all analysed:

| point | fairness mean | altruism mean |
|---|---|---|
| baseline | 67.33 | 67.50 |
| decision −3 | 64.67 | 60.33 |
| decision +3 | 72.17 | 66.00 |
| trait −3 | 70.33 | 37.00 |
| trait +3 | 44.67 | 79.33 |
| shuffled null −3 | 67.67 | 69.00 |
| shuffled null +3 | 66.50 | 72.50 |
| orthogonal null −3 | 69.17 | 61.33 |
| orthogonal null +3 | 67.00 | 71.17 |

Rows: the brief's "7 points × 120 = 840" does not match the pilot. There are 9 judged held-out points, so 1,080 primary rows (and the 270-row secondary subset). All were judged, as the brief asks. The count difference is recorded in DEVIATIONS.md.

## 3. Pre-registered contrasts (primary judge)

Paired by question; 95% question-clustered percentile bootstrap (10,000 draws, seed 20261028), 40 questions. Columns: estimate, 95% CI, rows used, verdict. Verdict rule: "supported" if the CI excludes zero in the predicted direction; "contrary" if it excludes zero the other way; "not resolved" if it includes zero; "descriptive" if no direction was predicted.

| contrast | estimate | 95% CI | rows | verdict |
|---|---|---|---|---|
| a1: F(D+3) − F(base) | 1.75 | [−1.95, 6.21] | 238 | descriptive |
| **a2: F(D+3) − F(T+3)** [P1] | **28.35** | **[19.03, 38.17]** | 232 | **supported** |
| a2 at −3: F(D−3) − F(T−3) | 1.28 | [−4.75, 7.64] | 236 | descriptive |
| F(D+3) − F(S+3) | 1.06 | [−1.41, 3.18] | 238 | descriptive |
| F(D+3) − F(O+3) | 0.98 | [−1.00, 2.80] | 238 | descriptive |
| F(D−3) − F(S−3) | −3.02 | [−7.91, 1.43] | 236 | descriptive |
| F(D−3) − F(O−3) | −3.03 | [−8.37, 1.92] | 236 | descriptive |
| A(D+3) − A(base) | 1.17 | [−2.56, 6.11] | 238 | descriptive |
| **A(D+3) − A(T+3)** [P2] | **−8.36** | **[−11.57, −4.24]** | 232 | **supported** |
| A(D−3) − A(T−3) | 14.40 | [6.84, 22.12] | 236 | descriptive |
| **b1: interaction at +3 vs T** [P3] | **36.70** | **[26.41, 47.14]** | 232 | **supported** |
| c1: interaction at −3 vs T | −13.13 | [−20.89, −5.77] | 236 | descriptive |
| c2: interaction at +3 vs S | 0.05 | [−2.66, 2.70] | 238 | descriptive |
| c3: interaction at +3 vs O | −0.76 | [−3.72, 2.16] | 238 | descriptive |
| c4: interaction at −3 vs S | 4.66 | [−1.89, 11.40] | 236 | descriptive |
| c5: interaction at −3 vs O | 3.76 | [−3.96, 11.13] | 236 | descriptive |

Interaction terms: b1 = [F(D+3) − F(T+3)] − [A(D+3) − A(T+3)]; the c-terms follow the same pattern.

**Sensitivity (no coherence filter, 240 rows per contrast):** the verdicts are unchanged. a2 = 30.2 [20.9, 40.0]; P2 = −8.12 [−11.37, −4.50]; b1 = 38.3 [28.1, 49.1]. Full table in `results.json` (`contrasts_unfiltered`).

**Secondary judge (descriptive, enters no verdict; 26 questions, 60 rows per contrast):** the same signs as the primary for a2 (26.9 [18.1, 35.6]), P2 (−11.4 [−19.0, −4.0]) and b1 (38.4 [24.8, 51.8]). The fairness-versus-null contrasts at +3 are positive for both null arms (c2 12.6 [2.5, 23.4]; c3 8.0 [−1.9, 18.7]), which the primary judge does not show.

## 4. Discriminant validity (prereg § 8)

Pearson r between the fairness and altruism judge scores over the baseline and null rows (primary judge, coherence-filtered): **r = 0.22, 95% CI [−0.06, 0.50]**; Spearman = 0.31; 600 rows, 40 questions. Unfiltered: r = 0.22, CI [−0.06, 0.50]. **r is below 0.8**, so the fairness judge is not simply restating the altruism judge on these rows. The check passes.

## 5. Verdict per prediction

* **P1 (ours raises fairness more than the trait does; a2 > 0): SUPPORTED against the trait vector, with a caveat.** a2 = +28.4 [19.0, 38.2]. But the gap is driven by the trait vector *lowering* fairness (58.8 vs baseline 85.2, −26 points). Ours at +3 is not distinguishable from baseline (a1 = +1.8 [−2.0, 6.2], not resolved) or from either null (not resolved). The fairness judge's validity check failed (§ 1), and its generosity penalty may explain the trait result. So the data support "the trait vector reduces fairness as this judge scores it", not "ours improves fairness".
* **P2 (the trait vector raises altruism more than ours does; A(D+3) − A(T+3) < 0): SUPPORTED.** −8.4 [−11.6, −4.2]. The trait vector gives the expected altruism increase (94.4 vs 85.6 for ours). Ours at +3 does not raise altruism over baseline (+1.2 [−2.6, 6.1]).
* **P3 (the 2×2 interaction is positive; b1 > 0): SUPPORTED on the primary judge.** b1 = +36.7 [26.4, 47.1]. The interaction is large, but it comes from the trait-vector fairness drop, not from ours raising fairness. It is not significant against the null arms at +3 (c2 and c3 include zero). It is also not established at −3 (c1 is negative, meaning the ordering reverses at −3).

**Operator's wording ("training on game theory improves fairness and not altruism"):** the "not altruism" half is consistent with the data: ours at +3 does not change the altruism score. The "improves fairness" half is **not shown**: ours does not differ from baseline or from the nulls on fairness. The regex shares in the prereg (ours "fair" 65% vs baseline 22%) were not reproduced by the judge at the level of the score, and that is a finding for this rubric, not a confirmation.

## 6. Limitations

* The anchor check failed (§ 1). The fairness scale is floored at 0 for generosity and for selfishness, and it penalises generosity heavily. The trait-vector fairness drop may therefore measure "not generous" as much as "unfair". This needs a rubric revision and a new validity check before any fairness claim is made. Changing the rubric now would be a new analysis, and it is not done here.
* One judge (llama-3.3-70b) was the primary. The secondary (gpt-oss-20b) is on a 30-row subset per point and 26 questions, so its numbers are noisy.
* One arm pair, 40 held-out questions, 3 samples. The pilot is not a test of transfer.
* No multiplicity correction. Several contrasts are reported, and none of the verdicts is confirmatory.
* The fairness judgments came from three judge attempts (run files `run_20261026T150725Z`, `run_20261026T160049Z`; the first was killed at 700 of 1080 calls at about 4.45 s/call). Four primary calls were served from cache from the earlier attempt. `run_judging.py` keys each call, so the stored rows contain one entry per key (1,080 unique keys).

## 7. Cost

Self-hosted judges, no dollar cost. Primary: 1,080 logprob calls at about 4.45 s per call (about 80 min of judge time across the attempts) plus 12 anchor calls. Secondary: 270 calls at about 1.1 s (about 5 min). Total judge time is about 1.5 h. Below the 3 h budget.

## 8. Files

* `anchors.json` — the 12 hand-written anchors (committed in ca333f7).
* `anchor_check.py`, `anchor_check.json` — the validity check.
* `run_fairness_judging.sh`, `judging_fairness.log` — the judging run.
* `analyze_fairness.py` — the analysis (writes `results.json`, `scores_by_row.csv`, `tables.md`).
* Judgments: `results/general-question-transfer/judgments/*/fairness.jsonl`.
