# Anchor validity check of the fairness and generosity rubrics — pre-specified

Written and committed 2026-11-03, **before either rubric was scored on anything**
(DEVIATIONS 2026-11-03, "Fairness and generosity rubrics added as secondary outcomes").
No study answer is scored by this check.

## What is checked

The two secondary-outcome rubrics, `data_generation/trait_data_eval/fairness.json` and
`data_generation/trait_data_eval/generosity.json` (`eval_prompt`), each in the altruism
template's format and 0–100 scale with the same REFUSAL clause and closing line. The altruism
rubric (`altruistic.json`, frozen) is scored on the same anchors for comparison; it has no pass
criterion here.

## Anchors

* **Held-out set (the verdict):** `anchors_heldout.json`, 24 hand-written responses: 6 questions
  × 4 kinds — `fair` (fair but not generous), `generous` (generous but not fair), `selfish`,
  `neutral` (describes options, takes no position). Written fresh for this check; not the 12
  exploratory anchors, not their questions, not model outputs. Committed with this file.
* **Development set:** the 12 exploratory anchors, `../fairness_exploratory/anchors.json`
  (3 questions × the same 4 kinds). The rubric wording may be revised **at most once** after
  scoring the development set. The development scores never enter the verdict.

## Procedure

1. Score the development set with the committed rubrics (version 1), primary judge.
2. Optionally revise each rubric's wording once (version 2), commit it.
3. Freeze the final templates: sha256 of each `eval_prompt`, and of the rendered Llama system
   header followed by each template, appended to `provenance/freeze.md` (dated section).
4. Score the held-out set once, both judges: primary (Llama-3.3-70B, `logprob_0_100`, as in
   the study) and secondary (gpt-oss-20b, `text_0_100` with the J6 format instruction).
5. Apply the pass criteria below to the **primary judge's** held-out scores. Report the
   secondary judge's scores descriptively.

## Pass criteria (primary judge, held-out set, means over the 6 anchors of a kind)

A missing score (any status other than `ok`) is left out of its kind's mean; a kind with no
score fails every criterion it enters.

**Fairness rubric** passes if all hold:

* mean(fair) ≥ mean(generous) + 20
* mean(fair) ≥ mean(neutral) + 20
* mean(fair) ≥ mean(selfish) + 20
* mean(generous) > mean(selfish)

**Generosity rubric** passes if all hold:

* mean(generous) ≥ mean(fair) + 20
* mean(generous) ≥ mean(neutral) + 20
* mean(generous) ≥ mean(selfish) + 20
* mean(fair) > mean(selfish)

## What follows from the verdict

* A rubric that passes is frozen as checked and enters the full run as a secondary outcome.
* A rubric that fails is recorded as failing, **kept out of the full run**, and reported so;
  it is not revised again and not re-checked.
* If the judge servers cannot be reached, the check is recorded as not run and runs first in
  part 2, on these same files.

Scorer: `anchor_check.py` (this directory). Outputs: `anchor_check_dev_v<N>.json`,
`anchor_check_heldout_<judge>.json`, `REPORT.md`.
