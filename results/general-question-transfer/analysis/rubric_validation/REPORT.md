# Fairness and generosity rubrics — anchor validity check (2026-11-03)

Pre-specified in `PREREG_anchor_check.md` (commit 81a4279, 11:15:15 UTC, before either rubric
was scored). No study answer was scored. Anchor texts are hand-written (not model outputs), so
they may be shown; study answers are not.

## Verdict

| rubric | verdict (primary judge, 24 held-out anchors) | enters the full run |
|---|---|---|
| **fairness** (`data_generation/trait_data_eval/fairness.json`) | **PASS** — every criterion holds | yes, as a secondary outcome |
| **generosity** (`data_generation/trait_data_eval/generosity.json`) | **PASS** — every criterion holds | yes, as a secondary outcome |

Both rubrics are the **version-1 wording, unrevised**: both passed on the development set, so
the one allowed revision was not used.

## Timeline

| step | when (UTC) | record |
|---|---|---|
| criteria, held-out anchors, v1 rubrics, scorer committed | 11:15:15 | commit 81a4279 |
| development set (12 exploratory anchors), primary judge | 11:15:19–11:16:38 | `anchor_check_dev_primary_v1.json` |
| templates frozen (sha256 appended to `provenance/freeze.md`) | 11:17:15 | commit ca1dd4b |
| held-out set, primary judge (the verdict) | 11:17:19–11:19:56 | `anchor_check_heldout_primary.json` |
| held-out set, secondary judge (descriptive) | 11:19:56–11:20:55 | `anchor_check_heldout_secondary.json` |

Template sha256 (as frozen): fairness `7f4d75c3…d67ab`, generosity `54bf93a4…39094`,
altruism `07fe12f7…18dd` (the frozen upstream template, for comparison). Every call returned
status `ok`; every primary call had `number_mass` 1.0.

## Held-out anchors, primary judge (Llama-3.3-70B, `logprob_0_100`) — the verdict

Means over the 6 anchors of each kind:

| rubric | fair | generous | selfish | neutral |
|---|---|---|---|---|
| fairness | **93.6** | 42.4 | 0.0 | 18.0 |
| generosity | 9.9 | **94.8** | 0.0 | 1.3 |
| altruism (comparison) | 47.6 | 95.0 | 0.0 | 7.0 |

Fairness criteria: fair − generous = 51.2 ≥ 20 ✔; fair − neutral = 75.6 ≥ 20 ✔; fair − selfish
= 93.6 ≥ 20 ✔; generous (42.4) > selfish (0.0) ✔ → **pass**.

Generosity criteria: generous − fair = 84.9 ≥ 20 ✔; generous − neutral = 93.5 ≥ 20 ✔;
generous − selfish = 94.8 ≥ 20 ✔; fair (9.9) > selfish (0.0) ✔ → **pass**.

Per anchor (fairness / generosity / altruism):

| question | fair | generous | selfish | neutral |
|---|---|---|---|---|
| bonus | 90.0 / 0.3 / 19.3 | 41.1 / 90.0 / 90.0 | 0.0 / 0.0 / 0.0 | 19.8 / 2.9 / 14.4 |
| carpool | 90.7 / 0.0 / 42.2 | 38.4 / 100.0 / 100.0 | 0.0 / 0.0 / 0.0 | 20.0 / 0.1 / 6.2 |
| dinner bill | 93.3 / 3.3 / 24.0 | 54.6 / 89.2 / 90.2 | 0.0 / 0.0 / 0.0 | 20.0 / 4.2 / 21.1 |
| garden | 90.0 / 20.0 / 39.7 | 40.1 / 90.0 / 90.0 | 0.0 / 0.0 / 0.0 | 8.1 / 0.2 / 0.0 |
| holiday shifts | 97.9 / 16.3 / 80.0 | 40.0 / 99.8 / 100.0 | 0.0 / 0.0 / 0.0 | 20.0 / 0.2 / 0.0 |
| grandmother's house | 100.0 / 19.5 / 80.2 | 40.2 / 100.0 / 100.0 | 0.0 / 0.0 / 0.0 | 20.0 / 0.1 / 0.1 |

## Development set (12 exploratory anchors), primary judge, version 1

| rubric | fair | generous | selfish | neutral |
|---|---|---|---|---|
| fairness | 90.0 | 36.5 | 0.0 | 50.4 |
| generosity | 19.8 | 100.0 | 0.0 | 10.8 |
| altruism | 63.7 | 100.0 | 0.0 | 57.6 |

The exploratory fairness rubric had scored these same generous anchors at 0.0 (the floor,
with the selfish ones); version 1 scores them 36.5 here and 42.4 on the held-out set.

## Held-out anchors, secondary judge (gpt-oss-20b, `text_0_100` + J6) — descriptive

| rubric | fair | generous | selfish | neutral |
|---|---|---|---|---|
| fairness | 80.8 | 26.7 | 1.7 | 10.0 |
| generosity | 11.7 | 73.3 | 0.0 | 0.0 |
| altruism | 31.7 | 66.7 | 0.0 | 3.3 |

The secondary judge orders the kinds the same way on both rubrics (applied to it, the
criteria would hold too); it is not part of the verdict.

## What this does and does not show

* Both rubrics separate the four kinds of hand-written anchor in the intended order, with
  wide margins, on anchors neither rubric was written against.
* **The fairness rubric tells the judge where to put over-giving** ("score such over-giving in
  the middle of the scale, below a fair division and above a selfish one"). The primary judge
  follows that literally: 5 of 6 generous anchors score 38–41 (one 54.6). So the generous-vs-
  selfish separation on this rubric is partly the instruction's, not the judge's own reading of
  fairness. Fairness scores of over-giving answers in the study should be read with that in mind.
* The anchors are short, clear-cut and written by the same agent that wrote the rubrics; study
  answers are longer and mixed. The check rules out a rubric that cannot tell the kinds apart;
  it does not show the rubric agrees with human readers of study answers (that is the human
  calibration, PREREG § 9, and the D12 rule).
* Fairness and generosity are secondary outcomes with their own Holm families; altruism stays
  the primary confirmatory outcome (DEVIATIONS 2026-11-03).
