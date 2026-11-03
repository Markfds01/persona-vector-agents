# Deviations from PREREG.md

`PREREG.md` is frozen at checkpoint-1 approval and never edited after data exists.
Every departure from it — planned or forced, however small — is recorded here, in
the order it happened, BEFORE the affected step runs where that is possible. A
deviation that is discovered after the fact says so.

Format, one entry per deviation:

```
## YYYY-MM-DD — <short title>

* PREREG section: § <n> (<the sentence or table row it changes>)
* What changes: <exactly what is done instead>
* Why: <the reason, with evidence: a file, a log line, a measured number>
* Decided by: <operator | operator at checkpoint N | forced by <what>>
* Data existing at the time: <none | generation rows of <...> | judgments of <...>>
* Effect on results: <which numbers / verdicts it can move, or "none">
```

## 2026-10-23 — Pilot scope (checkpoint 2)

* PREREG section: § 14 ("The pilot ... one game arm + the trait vector + nulls at k in {-3, 0, +3}, small n") and § 9.1 (human sample strata)
* What changes: nothing is changed; § 14 leaves the game, the sets and n open, so the choice is recorded here. The game arm is the **Dictator strict decision vector** with the trait vector and the **Dictator's shuffled and orthogonal nulls**, at k in {-3, 0, +3} (the one shared k = 0 baseline hooks the trait vector at coefficient 0). Question sets: **held-out and bridge**, at the PREREG's **3 samples per question** (120 rows per point). The in-distribution set is not run (no decision-minus-null contrast needs it at the pilot). Controls: the held-out set also gets the `unhooked` and `unhooked_repeat` points. Total 2,400 generations (11 held-out points, 9 bridge points). Consequence for § 9.1: the pilot has no relaxed and no Dictator-only arm, so those two strata (18 rows each) are empty and are **not topped up**; the human sheet holds 164 items (8 x 18 + 20), not about 200.
* Why: the spec's pilot definition; the cheapest design that exercises every arm role the judges and the labellers will see, every stage and every control.
* Decided by: forced by the open § 14 text; the task brief names the Dictator strict vector and the held-out + bridge sets as the default.
* Data existing at the time: none (recorded before the pilot generation).
* Effect on results: none on the study; pilot numbers are plumbing, not results.

## 2026-10-23 — Served model ids are the model paths, not the PREREG J10 aliases

* PREREG section: § 11 J10 (served ids set with `--alias` to `llama-3.3-70b-instruct-q8_0` and `gpt-oss-20b-mxfp4`; the client aborts if `/v1/models` does not list the configured id)
* What changes: the pilot's configured ids are the ids the two servers actually report — `models/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf` and `models/gpt-oss-20b-MXFP4.gguf` — pinned in `scripts/judging/gqt_judging.py` (`SERVED_IDS`). The abort-on-mismatch rule is unchanged and tested. The judgment directories are still named by the J10 aliases.
* Why: the servers behind the tunnel were launched without `--alias` (`/v1/models` returned the paths, 2026-10-23); the check this run can make is against what they serve. The ids name the same GGUF files PREREG § 11 pins (the 70B's 00001-of-00002 shard, Q8_0; gpt-oss MXFP4); `/props` is recorded in every run file.
* Decided by: forced by the servers' launch; the operator may relaunch with `--alias` before the full run, in which case `SERVED_IDS` is the one line to change.
* Data existing at the time: none.
* Effect on results: none.

## 2026-10-23 — Generation interpreter differs from the one the README names

* PREREG section: § 3 / § 4 (steering and generation as in the per-game study); README "How to re-run" names torch 2.6.0+cu124 / transformers 4.52.3
* What changes: the pilot generated with **torch 2.7.1+cu126 and transformers 4.54.0** (the only interpreter on this machine with torch, transformers and pytest; `accelerate` is absent, so the model is loaded with `from_pretrained` and moved with `.to("cuda:N")`, not `device_map`). Both versions are recorded on every row. Transformers 4.54.0 is past 4.50, so `use_model_defaults=False` is passed and recorded.
* Why: no interpreter with the earlier pins is installed; building one was not part of the pilot.
* Decided by: forced by the machine; the operator may supply the earlier interpreter for the full run.
* Data existing at the time: none.
* Effect on results: the k = 0 point is not guaranteed bit-identical to the per-game study's k = 0 rows (different torch/transformers kernels); the identity checks inside this study (hooked k = 0 against unhooked, unhooked against its repeat) are not affected, because every point of the study runs under one interpreter.

## 2026-10-23 — Pairwise judging parameters and client mode the PREREG left open

* PREREG section: § 6 (pairwise template and parser "added to lab/llama_server_client.py before checkpoint 2"), § 11 J7 (`max_tokens`: Llama 1 / gpt-oss 2048 / Llama text 32)
* What changes: `lab/llama_server_client.py` gains a third mode `choice_1_2` (exactly one `<choice>1</choice>` or `<choice>2</choice>` in the final channel, else `unparseable`; no logprobs; tests in `lab/tests/test_llama_server_client.py`); the two 0-100 modes are untouched. Pairwise requests use `max_tokens` **32** on the primary and **2048** on the secondary (the reasoning budget J7 already sets for gpt-oss), temperature 0, seed 0, `cache_prompt: false`, the same `chat_template_kwargs` as that judge's other calls. The pairwise template is sent as the whole user message with no format instruction appended (it already fixes the reply format). The pairwise call's score is 1.0 or 2.0 (the response position chosen); the row stores which position the steered answer held. The parser reads "exactly one `<choice>1|2</choice>` in the final channel" as exactly one such tag, with other text around it allowed (as the 0-100 text parser allows text around its one `<score>` tag).
* Why: these values are not in the PREREG; they follow J7's reasoning and the template's own reply format.
* Decided by: implementation choice, flagged for the operator at checkpoint 2.
* Data existing at the time: none.
* Effect on results: pairwise only (secondary metric).

## 2026-10-23 — Judge order and the secondary judge's one-sitting rule

* PREREG section: § 11 D7/J13 ("run the whole secondary pass in one sitting")
* What changes: the pilot judging runs the secondary judge's whole pass first, then its determinism re-send, then the primary judge's trait/coherence, pairwise and re-send, so the two servers (one GPU) never compete and the secondary pass sits under one date. Nothing about what is judged changes. The send order is the seeded global shuffle (20261027) cut into chunks of 50 calls, each chunk re-shuffled by `judge_many` with `order_seed = 20261027 + chunk offset` and sent score calls first, pairwise calls second; a resume therefore changes the order of the calls it still has to send, not what they are.
* Why: D7 and the shared GPU.
* Decided by: implementation choice within D7.
* Data existing at the time: generation rows of the pilot.
* Effect on results: none.

## 2026-10-26 — Exploratory fairness analysis (post-hoc, not confirmatory)

* PREREG section: none changed. This is an analysis added after the pilot, not a departure from `PREREG.md`; it is recorded here so the dated record is complete. The confirmatory plan and the confirmatory analysis files are untouched.
* What changes: the pilot's existing held-out answers are judged a second time with a **fairness** rubric (`data_generation/trait_data_eval/fairness_exploratory.json`, same format and `logprob_0_100` scale as the upstream altruism template), by the same primary judge through the same client, and the secondary judge on its existing 30-row subset. Contrasts are paired by question against the existing trait (altruism) scores. The full plan, rubric, anchors and predictions are in `docs/fairness_exploratory_prereg.md` (committed before any fairness score exists).
* Why: the pilot judged only the altruistic dimension; regex shares in the pilot answers suggest the game-built vector moves fairness but not altruism. The operator asked for this measurement. Evidence: the pilot's judgments (`judgments/`), the regex shares in the prereg § 1.
* Decided by: operator (the item "exploratory fairness analysis", 2026-10-26).
* Data existing at the time: generation rows of the pilot (held-out set, 9 judged points, 1,080 rows); the pilot's trait and coherence judgments (primary and secondary).
* Effect on results: none on the confirmatory numbers. Exploratory verdicts only, labelled as such. Count note: the brief's "7 points × 120 = 840" does not match the pilot, which has 9 judged held-out points (1,080 rows); all of them are judged.

## 2026-11-03 — The "poor agreement" decision rule (D12)

* PREREG section: § 9.4 and § 0 D12 ("NOT pre-registered here ... fixed by the operator at checkpoint 2 ... before any full judging")
* What changes: the rule is fixed. **A judge fails on a rubric** if the lower bound of the 95% interval of the human–judge Spearman correlation on that rubric (question-clustered bootstrap, as in `scripts/measurement/agreement.py` and § 8.1; seed 20261028) is **< 0.5**. **The bridge fails** if the lower bound of the 95% interval of the bridge AUC (§ 9.3) is **≤ 0.5**. Per rubric: primary fails and secondary passes → the secondary judge becomes the primary for that rubric (its verdicts become the confirmatory ones for that rubric, with its own mode and subsets as § 6 sets them for the primary); both fail → that rubric's judging stops and the rubric is reported as failing the rule, with no verdicts. The rule applies to every rubric judged (altruism trait, coherence, and the fairness and generosity rubrics below).
* Why: § 9.4 leaves it to the operator at checkpoint 2; the operator adopted the draft's candidate thresholds (pilot report § 10, question 1: "Agree with you").
* Decided by: operator (2026-11-03).
* Data existing at the time: generation rows of the pilot; the pilot's judgments; no human labels yet.
* Effect on results: decides which judge's scores are confirmatory per rubric, or whether a rubric is reported at all. Applied in part 2, once both labellers' exports exist.

## 2026-11-03 — Human calibration sheet accepted at 164 items

* PREREG section: § 9.1 (about 200 items, stratified by arm role)
* What changes: the 164-item sheet of the pilot is the calibration sheet. The two empty strata (relaxed and Dictator-only arms, 18 items each) are **not** topped up.
* Why: pilot report § 10, question 2; the operator: "Agree with you".
* Decided by: operator (2026-11-03).
* Data existing at the time: the pilot rows and the exported sheet `human/calibration_sheet.csv`.
* Effect on results: human agreement is measured on 164 items without relaxed or Dictator-only answers.

## 2026-11-03 — Second labeller is AAC

* PREREG section: § 0 D10 (the colleague's identity is recorded in the provenance at checkpoint 2)
* What changes: the second labeller is **AAC** (recorded in `provenance/labellers.md`). AAC is sent **only** `human/grader.html`; never `provenance/human_sample_key.json` or `human/judge_reveal_*.json`.
* Why: pilot report § 10, question 3; the operator: "AAC".
* Decided by: operator (2026-11-03).
* Data existing at the time: as above.
* Effect on results: none (provenance).

## 2026-11-03 — Served model ids stay the model paths

* PREREG section: § 11 J10 (served ids set with `--alias`)
* What changes: nothing beyond the 2026-10-23 entry: the full run keeps the path ids (`SERVED_IDS` in `scripts/judging/gqt_judging.py`); the servers are not relaunched with `--alias`.
* Why: pilot report § 10, question 4; the operator: "Keep the file paths".
* Decided by: operator (2026-11-03).
* Data existing at the time: as above.
* Effect on results: none.

## 2026-11-03 — Full run generates under the earlier pins (torch 2.6.0+cu124, transformers 4.52.3)

* PREREG section: § 3 / § 4 (steering and generation as in the per-game study); the 2026-10-23 entry "Generation interpreter differs from the one the README names"
* What changes: the full run (all 13,200 rows) generates in a new virtual environment outside the repository, `~/venvs/pva-torch260-tf4523`, with **torch 2.6.0+cu124 and transformers 4.52.3** (the per-game study's pins), not the pilot's 2.7.1+cu126 / 4.54.0. Exact versions and `pip freeze`: `provenance/interpreter_full_run.md` and `provenance/interpreter_full_run_pip_freeze.txt`. The pilot's rows stay as they are, under the pilot's interpreter.
* Why: pilot report § 10, question 5; the operator: "Agree with you" — so the full run's k = 0 rows are comparable with the per-game study's.
* Decided by: operator (2026-11-03).
* Data existing at the time: the pilot rows (2.7.1 / 4.54.0).
* Effect on results: the pilot and the full run are under different interpreters and are never mixed (next entry).

## 2026-11-03 — Full-run rows go to their own directories

* PREREG section: § 4 (rows layout `rows/<arm>/<set>_k<k>.jsonl`); README layout
* What changes: the full run writes its rows to **`rows_full/<arm>/<set>_k<k>.jsonl`** and its per-point provenance and logs to **`provenance/generation_full/`** (`generation_points_<set>.json`, one log per process). The pilot's `rows/` and `provenance/generation_*` are left where they are (the pilot's judgments, the human sheet and its key refer to them). A resume of the full run reads only `rows_full/`, so it can never treat a pilot point as complete; the per-row `torch_version` / `engine_version` fields also tell the two apart.
* Why: the operator's brief: keep the pilot's rows (other interpreter) apart from the full run's.
* Decided by: operator (2026-11-03), the choice of directory by the implementation.
* Data existing at the time: the pilot rows.
* Effect on results: none; part 2 judges `rows_full/`.

## 2026-11-03 — Full-run generation details the PREREG leaves open

* PREREG section: § 4 (the `div8_main` table), § 5 (the in-distribution set)
* What changes: (1) The in-distribution set is `data_generation/trait_data_eval/altruistic.json` read in place (sha256 `a1a2d2b6…07dc`); its question ids are `in_distribution_00` … `in_distribution_39` in file order (the file has no ids). (2) The held-out set also gets the `unhooked_repeat` control point, as in the pilot (120 rows beyond the 13,200; it shows the unhooked run reproducible under the new interpreter and is never judged). (3) The run is split over the two A40s, two processes per GPU, each with `OMP_NUM_THREADS=8`; each point is still one process, one GPU, and its own seeded run, so the split changes no row's seeds or batch composition. Points of one set written by different processes are merged into `generation_points_<set>.json` under a file lock (`lab/general_questions.py` `merge_point_summary`); each point records its own device.
* Why: the PREREG names the set but not its ids; the repeat control is the pilot's; the split is for time.
* Decided by: implementation choice.
* Data existing at the time: the pilot rows.
* Effect on results: none.

## 2026-11-03 — Pairwise parameters approved

* PREREG section: § 6, § 11 J7; the 2026-10-23 entry "Pairwise judging parameters and client mode the PREREG left open"
* What changes: nothing; the parameters of that entry (`max_tokens` 32 primary / 2048 secondary, no format instruction appended, template as frozen) are approved for the full run.
* Why: pilot report § 10, question 6; the operator: "Approve".
* Decided by: operator (2026-11-03).
* Data existing at the time: the pilot's judgments.
* Effect on results: none.

## 2026-11-03 — The primary judge's pairwise pass on a 25% subset

* PREREG section: § 6 item 3 ("Primary judge on all pairs"), § 10 (10,560 primary pairwise calls)
* What changes: the primary judge's pairwise pass runs on a **pre-registered, seeded 25% subset**, both orders, at every steered arm and k = ±3 on the held-out set: 30 of the 120 (question, sample) pairs of a point, the same 30 for every point, `random.Random(20261024).sample(sorted pairs, 30)` — the secondary judge's pre-registered subset (§ 13), so both judges' pairwise passes cover the same pairs. The 30 pairs (26 distinct questions) and the seed: **`provenance/pairwise_primary_subset.json`**, committed with this entry, before any full-run pairwise call. Primary pairwise calls: 44 points × 30 × 2 orders = 2,640 (was 10,560).
* Why: pilot report § 10, question 7 (the primary's pairwise pass was about half its judging time at 8.05 s/call); the operator: "Agree, 25% subset". Assumption: "a 25% subset like the secondary" is read as the secondary's (question, sample)-pair subset, not a new subset of 10 whole questions; it covers 26 questions (more question clusters for the clustered bootstrap) and makes the two judges' pairwise calls directly comparable.
* Decided by: operator (2026-11-03); the reading of "25% subset" by the implementation.
* Data existing at the time: the pilot's judgments.
* Effect on results: the pairwise metric (secondary) is estimated on a quarter of the pairs; its intervals widen.

## 2026-11-03 — Score-token probability mass is a reported outcome

* PREREG section: § 8 (what is reported), § 11 J3/J4 (logprob mode)
* What changes: `lab/llama_server_client.py` already stores `detail.number_mass` (total probability on the integer tokens 0..100 among the top-20 logprobs) on every primary (Llama, `logprob_0_100`) call and marks `< 0.25` as `low_mass`. The full analysis **reports**, per rubric (altruism, fairness, generosity and coherence) and per point: mean and median `number_mass`; the share of calls with mass < 0.9 and < 0.5; the counts of `low_mass` and `refusal` statuses. It also reports a **sensitivity analysis** of the primary contrasts (§ 8.3) with calls of mass < 0.5 excluded. This is descriptive: the scoring and the confirmatory verdicts are unchanged.
* Why: the operator, on the frozen judge prompts: "can you measure also the percentage of the mass vocabulary related to a possible answer".
* Decided by: operator (2026-11-03).
* Data existing at the time: the pilot's judgments.
* Effect on results: none on the confirmatory verdicts; adds descriptive tables and one sensitivity analysis.

## 2026-11-03 — Fairness and generosity rubrics added as secondary outcomes

* PREREG section: § 6 (per answer, both judges score trait and coherence), § 8.3 (one primary Holm family)
* What changes: two trait rubrics are added **after the pilot and before any full judging**: **fairness** (`data_generation/trait_data_eval/fairness.json`) and **generosity** (`data_generation/trait_data_eval/generosity.json`), same format, 0–100 scale, REFUSAL clause and closing line as the altruism template. **Altruism (the frozen upstream template) stays the primary confirmatory outcome**; fairness and generosity are **secondary outcomes**, each with **its own Holm family** over the same decision-minus-null contrasts as the primary family (§ 8.3). Both judges score them on the same rows and subsets as the altruism trait rubric — primary: every held-out and in-distribution row plus the bridge AUC subset, `logprob_0_100`; secondary: the pre-registered 25% subset plus the bridge AUC subset, `text_0_100` with the J6 instruction. A rubric enters the full run only if it passes its anchor validity check (`analysis/rubric_validation/PREREG_anchor_check.md`, committed before the check is scored); a rubric that fails is recorded, kept out of the full run and reported as such, with no further iteration. The two templates' sha256 are appended to `provenance/freeze.md` once frozen. The exploratory fairness rubric (`fairness_exploratory.json`, 2026-10-26) is not used in the full run.
* Why: the operator, on the frozen judge prompts: "I also do not see the fairness evaluation. Measure three, fairness, generous and altruistic." This is a post-pilot addition, said so plainly: the pilot's exploratory fairness analysis (`analysis/fairness_exploratory/REPORT.md`) had already been seen when it was decided.
* Decided by: operator (2026-11-03).
* Data existing at the time: the pilot rows and judgments, including the exploratory fairness judgments of the pilot's held-out answers.
* Effect on results: adds two secondary outcomes; the primary family and its verdicts are unchanged.

## 2026-11-03 — Projected judging time of the full run

* PREREG section: § 10 (33,264 primary + 7,182 secondary calls, 37.3 h)
* What changes: with the primary's pairwise pass at 25% and two more trait rubrics, recomputed at the pilot's measured speeds (`analysis/pilot_validation.json` `throughput`: primary score calls 3.35 s/call, pairwise 8.05 s/call; secondary trait 0.87 s, coherence 0.92 s, pairwise 1.35 s):

| judge | calls | hours |
|---|---|---|
| primary (Llama-3.3-70B) | trait 3 × 8,040 = 24,120 + coherence 13,080 + pairwise 2,640 + 5% re-send 1,992 = **41,832** | 34.6 + 5.9 + 2.8 ≈ **43.3 h** |
| secondary (gpt-oss-20b) | trait 3 × 2,280 = 6,840 + coherence 1,920 + pairwise 2,640 + 5% re-send 570 = **11,970** | 1.65 + 0.49 + 0.99 + 0.17 ≈ **3.3 h** |

  The re-send hours are priced at each judge's pilot overall speed (4.98 s and 1.05 s per call). If a rubric fails its anchor check, its 8,040 primary and 2,280 secondary calls (≈ 7.5 h and 0.55 h) drop out.
* Why: the operator's decisions 7 and "measure three".
* Decided by: operator (2026-11-03); the arithmetic by the implementation.
* Data existing at the time: the pilot's judgments.
* Effect on results: none.

## Pending, to be recorded here before it is used

* Nothing. (The "poor agreement" rule, D12, was fixed on 2026-11-03, above.)

## 2026-11-03 — Anchor check of the fairness and generosity rubrics: both pass

* PREREG section: none beyond the entry "Fairness and generosity rubrics added as secondary outcomes" above; this records its check's outcome.
* What changes: nothing in the plan. On the 24 fresh held-out anchors the primary judge passes every pre-specified criterion for both rubrics (fairness: fair 93.6, generous 42.4, neutral 18.0, selfish 0.0; generosity: generous 94.8, fair 9.9, neutral 1.3, selfish 0.0). The version-1 wording was used unrevised (it passed on the development set, so the one allowed revision was not used). Both rubrics enter the full run as secondary outcomes. Report: `analysis/rubric_validation/REPORT.md`; sha256 in `provenance/freeze.md` (2026-11-03 section).
* Why: `analysis/rubric_validation/PREREG_anchor_check.md` (commit 81a4279, before scoring).
* Decided by: the pre-specified rule.
* Data existing at the time: no study answer was scored by either rubric.
* Effect on results: fairness and generosity are judged in part 2. Caveat stated in the report: the fairness rubric itself tells the judge to place over-giving mid-scale.
