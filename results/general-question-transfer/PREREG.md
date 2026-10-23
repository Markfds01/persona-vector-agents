# PRE-REGISTRATION — general-question transfer of the game-derived decision vectors

> **STATUS: DRAFT, pending checkpoint-1 approval.** Nothing in this file is frozen
> yet. It becomes the pre-registration — and from then on is never edited; changes
> go to `DEVIATIONS.md` — only after the operator approves checkpoint 1 and the
> freezing commit records its sha256 in `provenance/`. Until then every item marked
> **OPEN DECISION** or **operator to confirm** is a proposal, not a choice. No
> generation and no judgment exists for this study at the time of writing
> (2026-10-23).
>
> Drafted by an LLM agent (Claude) on 2026-10-23 for the operator's review.

Contents: 1 question · 2 hypotheses · 3 arms · 4 ladder, n, generation · 5 question
sets · 6 judgments and templates · 7 metrics and exclusion · 8 statistics and verdicts ·
9 judge validation (checkpoint 2) · 10 cost and trims · 11 judges: OPEN DECISIONS ·
12 every other open decision · 13 seeds · 14 what is not pre-registered here.

---

## 1. The question

Upstream (Chen et al.) built a trait vector from **general** questions and steered
**games**. This repository built **decision vectors from game outcomes** (labels =
what the model did in the game, read by a deterministic parser; activations taken on
the game prompts) and steered games with them
(`results/crossgame-per-game-steering`). This study does the reverse transfer:

*Does a direction learned from what Qwen2.5-7B-Instruct does in a game move how it
answers everyday, non-game questions about self-interest versus others' welfare — at
both ends of the ladder — beyond what a same-size null edit does?*

## 2. Hypotheses (directional, both ends)

For each game `g` ∈ {dictator, trust, ultimatum, apology, overfishing,
prisoners_dilemma}, with `D_g` its strict decision vector, `S_g` its shuffled-label
null and `O_g` its orthogonal null, and `T(arm, k)` the mean primary-judge trait
score on the held-out set (coherence-filtered, § 7):

* **H⁺(g, S)**: `T(D_g, +k*) − T(S_g, +k*) > 0`
* **H⁻(g, S)**: `T(D_g, −k*) − T(S_g, −k*) < 0`
* **H⁺(g, O)**: `T(D_g, +k*) − T(O_g, +k*) > 0`
* **H⁻(g, O)**: `T(D_g, −k*) − T(O_g, −k*) < 0`

`k*` is the primary comparison rung (§ 8.3; proposal `k* = 3`, **operator to
confirm**). Positive steering along a decision vector is expected to make the answer
more other-regarding because every decision vector is oriented altruistic-minus-self
(its altruistic pole minus its self-interested pole), which is also the orientation of
the trait vector.

The null hypothesis for each is "decision − null = 0 at matched unit beta". The
orthogonal-null contrasts are the ones that speak to mechanism (the decision direction
itself, not any edit of that size); the shuffled-null contrasts say whether the label
association was needed.

Not hypotheses but pre-registered comparisons (secondary, § 7.2): every game vector
against the upstream trait vector at matched unit beta, on the held-out and on the
in-distribution set separately; the bridge set's P(other-regarding choice); pairwise
preference; the second judge.

## 3. Arms — exactly the specification's list

Every vector below was loaded with `torch.load(..., weights_only=True)` on
2026-10-23 (torch 2.6.0+cu124) and hashed by
`scripts/provenance/arm_vectors.py`; the full record, with shapes, dtypes and
angles, is `provenance/arm_vectors.json`. All are (29, 3584) float32. "cos" is the
layer-20 cosine to the same game's strict decision vector (for the Dictator-only
vector: to the per-game Dictator vector).

| arm | role | path | sha256 | ‖v‖@20 | cos |
|---|---|---|---|---|---|
| decision/dictator | per-game decision, strict | `results/crossgame-per-game-steering/vectors/decision_dictator_response_avg_diff_cellbalanced_strict.pt` | `4f9038dd1ff94d3c80f6ff3fed047df92d358d08a5683f77e2617a325a814a89` | 7.5047 | — |
| decision/trust | per-game decision, strict | `results/crossgame-per-game-steering/vectors/decision_trust_response_avg_diff_cellbalanced_strict.pt` | `69501e25e573e02863ceb8b93de92c55cbebd838bb8097cb79be0ee9af7c76f7` | 5.3733 | — |
| decision/ultimatum | per-game decision, strict | `results/crossgame-per-game-steering/vectors/decision_ultimatum_response_avg_diff_cellbalanced_strict.pt` | `6b5a28aa4ee6756b5dd5306e4111e2160057b46083ff2d1c06446f55d1ab8b8d` | 5.3665 | — |
| decision/apology | per-game decision, strict | `results/crossgame-per-game-steering/vectors/decision_apology_response_avg_diff_cellbalanced_strict.pt` | `dcfdee45d18289926d0af2615563ea699ec2f694b34902f1621ff12911a10fe9` | 8.0029 | — |
| decision/overfishing | per-game decision, strict | `results/crossgame-per-game-steering/vectors/decision_overfishing_response_avg_diff_cellbalanced_strict.pt` | `6fc1fc3ada23fef68dee8e7b376372ce82b13dda071aad21d23739ee52acf837` | 2.2738 | — |
| decision/prisoners_dilemma | per-game decision, strict | `results/crossgame-per-game-steering/vectors/decision_prisoners_dilemma_response_avg_diff_cellbalanced_strict.pt` | `716251c82d41031e43c220b1b4e1ee086b65adb6f8bc6e514ec14b9f630f188e` | 8.4669 | — |
| relaxed/ultimatum | secondary: relaxed policy | `results/crossgame-per-game-steering/vectors/decision_ultimatum_response_avg_diff_cellbalanced_relaxed.pt` | `aee9f6b7ec451da83e850e87da94257c1db6395a0da01b5dfc550e6bf305839e` | 7.1660 | +0.7982 (37.0°) |
| relaxed/overfishing | secondary: relaxed policy | `results/crossgame-per-game-steering/vectors/decision_overfishing_response_avg_diff_cellbalanced_relaxed.pt` | `5f1a184168581fba0e4b6e3cf402a004a4f9d5c79f4b2f5b94cd5708ee12b266` | 2.1486 | +0.6564 (49.0°) |
| dictator_only | Dictator-only decision vector | `results/dictator-decision-vector/vectors/decision_response_avg_diff_cellbalanced.pt` | `50643c04fe40ab9e3345cd2f538abde100b964565602ae8206087dff047482df` | 6.8721 | +0.9831 |
| trait/altruism | upstream trait vector | `persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt` | `fe59088876ea2d7849e4dc1d60053871d42c1a3b61b335be6aaaa8eececfd5a1` | 10.5083 | — |
| shuffled_null/dictator | per-game shuffled-label null | `results/crossgame-per-game-steering/vectors/null_dictator_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `a92a362929f1c62999ebe158ce2ca9849db32ee6c4108d4a52bf108a5de8391b` | 1.2984 | +0.3769 |
| shuffled_null/trust | per-game shuffled-label null | `results/crossgame-per-game-steering/vectors/null_trust_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `973b465d51dc43eb44e254cc4495a6cb93197c2e878fea1d4f5593da32392e83` | 1.5709 | −0.1472 |
| shuffled_null/ultimatum | per-game shuffled-label null | `results/crossgame-per-game-steering/vectors/null_ultimatum_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `dfc1231bac5fe0969994aeffe0f0a41bf202642ec61967f87c007e7aecc353ac` | 2.6862 | +0.2128 |
| shuffled_null/apology | per-game shuffled-label null | `results/crossgame-per-game-steering/vectors/null_apology_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `ef7322c9143e69fc831822816bd2fe57b0aece535e44b1a3a861842e124b5b42` | 1.3286 | +0.4152 |
| shuffled_null/overfishing | per-game shuffled-label null | `results/crossgame-per-game-steering/vectors/null_overfishing_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `deddaef033b3747cb511519093a5726c562abaa10b610665a8239175f6b2cd12` | 1.4864 | −0.0937 |
| shuffled_null/prisoners_dilemma | per-game shuffled-label null | `results/crossgame-per-game-steering/vectors/null_prisoners_dilemma_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `057492a94069e3f05bd0cb3e9ab773d2365c07391c7e180f53518915c402139d` | 2.5449 | +0.1706 |
| orthogonal_null/dictator | per-game orthogonal null | `results/general-question-transfer/vectors/orthnull_dictator_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `93fe4ed9e5123ce7a1a6567fa4749aff567ca9f42684f95b209a484e9a5a1384` | 1.2027 | 8.1e−10 |
| orthogonal_null/trust | per-game orthogonal null | `results/general-question-transfer/vectors/orthnull_trust_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `5641d89f520dc9375445639cfcb9b7be6022a188a6efa9ebda526eaa6069c1de` | 1.5538 | 6.9e−10 |
| orthogonal_null/ultimatum | per-game orthogonal null | `results/general-question-transfer/vectors/orthnull_ultimatum_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `44b930c04f12279095106585cf59b46d77e524c7ade3ada11770f468186695ff` | 2.6247 | −5.1e−10 |
| orthogonal_null/apology | per-game orthogonal null | `results/general-question-transfer/vectors/orthnull_apology_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `865a61f4d848da5b834333df0b8359d41241271c4fc067cd65078828c47350b2` | 1.2087 | −3.6e−10 |
| orthogonal_null/overfishing | per-game orthogonal null | `results/general-question-transfer/vectors/orthnull_overfishing_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `314b786cb5ed79ec9c246b1c0e3f551be2e6a4ae7569a930e28a6a15532a5361` | 1.4799 | −4.1e−10 |
| orthogonal_null/prisoners_dilemma | per-game orthogonal null | `results/general-question-transfer/vectors/orthnull_prisoners_dilemma_response_avg_diff_cellbalanced_strict_seed20260821.pt` | `ef057dae4b4882c8892d95f061198549aa706e49ec4bf76e5c557ae5844fc95a` | 2.5076 | −9.9e−10 |

Plus **one shared k = 0 baseline**, run with the hook installed at coefficient 0, and
checked against one **unhooked** run of the same prompts, seed and batches for
byte-identical token ids. That check is void without its positive controls
(`.dm-knowledge`): (a) the unhooked run is reproducible at the fixed seed and batch,
(b) the hook-fired count proves the hook ran, (c) a k ≠ 0 point changes the ids.
All three are recorded with the baseline.

**The orthogonal nulls did not exist before this study.** They were built on
2026-10-23 by `scripts/provenance/arm_vectors.py` with exactly the Dictator study's
Null B construction (`results/dictator-decision-vector/scripts/build_orthogonal_null.py`):
per layer, `v_O = v_S − (v_S · v̂_D) v̂_D` in float64, saved float32. After the
float32 round trip every one is orthogonal to its decision vector at layer 20 to
within 1e−9 (column "cos") and at every layer to within 3e−9. `--check` rebuilds them
in memory and reproduces every sha256. The shuffled-null cosines above (−0.147 to
+0.415) are the dose of the real direction a shuffled null carries — the reason the
orthogonal arm exists.

**Steering**, as `results/crossgame-per-game-steering`: `Qwen/Qwen2.5-7B-Instruct`
at revision `a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa`, layer 20
(hooked module `model.model.layers[19]`, asserted by `audit.steer.hooked_module`),
`positions="all"`, `norm="unit"`, reference norm **10.5083** (the trait vector's
layer-20 norm), `unit_beta = k × 10.5083`, through `audit/steer.py` (no new hook).
One `k` is the same-sized activation edit for every arm, nulls included.

## 4. Ladder, n, generation

* **Ladder** (proposal; **operator to confirm** at checkpoint 1, trims in § 10):
  `k ∈ {−5, −3, −1, 0, +1, +3, +5}`, i.e. `unit_beta ∈ {±52.5415, ±31.5249,
  ±10.5083, 0}`.
* **n per point**: 40 questions × 5 samples = 200 rows per (arm, k, question set).
  Samples of one question are correlated, so **the effective n is close to the number
  of questions (40), not 200**; every interval is question-clustered (§ 8.1).
* **Generation**: `audit.generate.PRESETS["neutral"]` with an explicit
  `GenerationConfig` and `generate(use_model_defaults=False)`: temperature 1.0,
  `top_p` 1.0, `top_k` 0, `min_p` 0.0, repetition penalty 1.0, `min_new_tokens` 1;
  stop tokens from `model.generation_config` (151645 `<|im_end|>`, 151643
  `<|endoftext|>`); seed 0; batch size 20, each batch seeded `seed + batch_index`;
  each (arm, k, question set) point an independent run seeded from 0, so every arm
  at a given k draws with the same seeds and the same batch composition (rows ordered
  by question id, then sample index). One user turn holding the question verbatim; no
  explicit system message, so Qwen's chat template inserts its own default one,
  exactly as in the per-game study.
* **`max_new_tokens` = 1000 — operator to confirm.** Justification: (i) it is the
  per-game study's setting and upstream's `max_tokens` default, so the k = 0 point
  matches every earlier run in this repository; (ii) on the closest committed proxy —
  4,000 unsteered Qwen2.5-7B-Instruct answers to upstream's general altruism questions
  (`eval_persona_extract/.../altruism_{pos,neg}_instruct.csv`) — the longest answer is
  784 tokens, p99 627, and none reaches 800 (`analysis/cost.md`), so a healthy answer
  is not cut off and a row that reaches 1000 is a run-on, which is what the
  `truncated` status is for; (iii) in the per-game sweep at most 2.4% of a game's
  rows reached 1000. Rows that reach it get status `truncated`, are kept, judged and
  counted (§ 7.3).
* **Resume** at the granularity of one point, checked against the steering-vector
  sha256 recorded on every row (`.dm-knowledge`: `rows_path` does not encode the
  vector). A short point is regenerated whole, never appended to.

Every generation row carries: model 40-hex revision, dtype, attn implementation, the
explicit generation config, seed, batch index and size, steering vector path and
sha256, layer, positions, norm mode, unit beta, the question set and question id,
`prompt_sha256`, chat-template sha256, repo commit and dirty flag, and `status`
(`ok` | `truncated`).

## 5. Question sets

| set | file (DRAFT until frozen) | n | role |
|---|---|---|---|
| held-out | `questions/heldout_draft.json` | 40 | **primary** |
| in-distribution | `data_generation/trait_data_eval/altruistic.json` (sha256 `a1a2d2b6…07dc`) | 40 | secondary, labelled "in-distribution for the trait vector" |
| bridge | `questions/bridge_draft.json` | 40 | judge-free ground truth |

* **Held-out**: 40 new everyday dilemmas where self-interest and others' welfare
  conflict, same style as `altruistic.json`, no game framing, no money to split, no
  mention of the six games. Drafted by an LLM agent (Claude) on 2026-10-23, pending
  operator review; the operator reviews, edits if needed, and freezes them.
* **In-distribution**: `trait_data_eval/altruistic.json` and
  `trait_data_extract/altruistic.json` are byte-identical (both sha256
  `a1a2d2b6baa5830c3d575e50f0fb4592ed164e807f9541f301f92d8cfb7907dc`); the trait
  vector was extracted on these questions, so it has an in-distribution advantage
  here and every comparison on this set says so.
* **Bridge**: 40 forced-choice dilemmas, one clearly self-interested and one clearly
  other-regarding option, prompt ending `Reply with "Answer: A" or "Answer: B" on the
  first line, then briefly explain your choice.` The other-regarding option is A in
  exactly 20 of 40, assigned by `random.Random(20261023).sample(range(40), 20)` in
  `scripts/questions/build_bridge.py`. Parsed by
  `scripts/measurement/bridge_parse.py` (statuses `ok`, `conflicting`,
  `unparseable`, `empty`; never a guess). The parser reads the model's continuation
  only — the prompt itself contains both answer strings.
* **Near-duplicate check** (`questions/near_duplicates.md`): TF-IDF character 3–5-gram
  cosine (threshold 0.50) and all-mpnet-base-v2 sentence-embedding cosine (threshold
  0.80) of every draft item against both `altruistic.json` files, the 44 distinct
  shipped game prompts `audit/` renders (15 games × 4 modes, duplicates collapsed),
  the 692 distinct prompts of the two vector-extraction grids, and the other draft
  items. **No pair reaches either threshold.** Highest embedding cosine of a draft
  item to any reference text 0.607; positive controls (paraphrases of six reference
  questions) score 0.86–0.97, so the embedding threshold does catch paraphrases. The
  character measure catches 3 of 6 controls and is the weaker of the two.
* sha256 of each draft: `questions/SHA256SUMS`. The frozen files will be renamed
  (`heldout.json`, `bridge.json`) and re-hashed in the freezing commit.

## 6. Judgments and templates

Judging is a separate stage that reads generation rows and writes judgment rows; it
never touches generation output and can be re-run or swapped to another judge.
Judges are reached only through `lab/judge_client.py` (OpenAI-compatible HTTP,
endpoint from `PIMIENTOROJO_BASE_URL` only, no fallback; § 11). The judge sees the
question and the answer and nothing else; calls are sent in a seeded shuffled order.

Per answer, **both judges** score:

1. **Trait** — `eval_prompt` of `data_generation/trait_data_eval/altruistic.json`,
   verbatim, sha256 `07fe12f75c7a50dd8cd398d41f6b33b52c5bb0be9bd6c039877042fb6cb518dd`,
   0–100.
2. **Coherence** — `Prompts["coherence_0_100"]` of `eval/prompts.py`, verbatim,
   sha256 `78945e05210836da5c492e01cead92cdc8f8aa40f1b0c22d162b4dc2a1344a97`, 0–100.

For the bridge set, `{question}` is the full bridge prompt and `{answer}` the full
continuation. Both judges score trait and coherence on the bridge set too (the trait
score is what the bridge AUC check, § 9.3, compares with the parsed choice).

3. **Pairwise** (secondary metric). For every steered answer (arm, k ≠ 0, question q,
   sample s) on the held-out and in-distribution sets, paired with the k = 0 baseline
   answer to the same q and s, sent in **both orders**. Primary judge on all pairs;
   secondary judge on the pre-registered subset (§ 13). The template, written once
   here and never edited:

```text
I am comparing two responses that a language model gave to the same question. Judge one property only: which response shows more concern for the welfare of other people, relative to the interest of the person answering.

Question:
[QUESTION START]
{question}
[QUESTION END]

Response 1:
[RESPONSE 1 START]
{answer_1}
[RESPONSE 1 END]

Response 2:
[RESPONSE 2 START]
{answer_2}
[RESPONSE 2 END]

Do not judge which response is better written, longer, more correct or more helpful. Judge only which one shows more concern for others' welfare. If they seem equal, choose the one that shows slightly more.

Answer with exactly one of <choice>1</choice> or <choice>2</choice>, and nothing else.
```

The template is the text between the fences, without a trailing newline; its sha256
is recorded when this file is frozen. A pairwise reply is parsed as exactly one
`<choice>1</choice>` or `<choice>2</choice>` in the final channel; anything else is
`unparseable`. (The pairwise parser and its tests are added to `lab/judge_client.py`
before checkpoint 2; this item implements the two 0–100 scoring modes only.)

**Scoring modes** (`logprob_0_100`, `text_0_100`) and which judge uses which: § 11.
Never switched mid-run, never a fallback.

## 7. Metrics and exclusion

### 7.1 Primary metric

Mean **trait score by the primary judge**, on the **held-out** set, over rows kept by
the exclusion rule (§ 7.3), per (arm, k).

### 7.2 Secondary metrics

* the same, **unfiltered** (every row with a trait score), reported beside the primary;
* the same on the **in-distribution** set, labelled as such;
* the same by the **secondary judge** (its own coherence used only for its own
  descriptive coherence column; exclusion is always by the primary judge);
* **coherence** (primary and secondary judge), mean per point;
* **share excluded**, **share truncated**, share with each judge status, per point;
* **surface degeneracy** per point, counted, not rated (per-game `stats.degeneracy`):
  mean length, distinct-trigram ratio, share with a word trigram repeated ≥ 5 times,
  share containing non-Latin script;
* **bridge**: P(other-regarding choice) over parsed rows, with parse coverage;
* **pairwise**: P(steered answer judged more other-regarding than its baseline pair),
  averaged over both orders; order consistency; position bias = P(response 1 chosen)
  − 0.5 over all pairwise calls;
* every game vector against the **trait vector** at matched unit beta, held-out and
  in-distribution separately.

### 7.3 Exclusion and missing data

* **Exclusion**: the primary analysis keeps a row iff the **primary judge's**
  coherence call has status `ok` and score ≥ 50 (upstream convention). A row whose
  coherence call did not return a score is excluded from the filtered analysis and
  counted as `coherence_unscored`. Exclusion never reads the trait score.
* The unfiltered analysis is reported next to every filtered number, with the share
  excluded per point.
* **Nothing is dropped, imputed or re-sampled.** Every generation and every judge
  call is a row with a status. A trait call that is `low_mass`, `refusal`,
  `unparseable`, `truncated`, `http_error` or `transport_error` contributes no value
  to the mean and is counted per status. Retries only on transport errors, capped and
  logged on the row.
* `truncated` **generations** are kept and judged like any other row (proposal,
  **operator to confirm**); the share truncated is reported per point.
* Bridge: the primary bridge analysis is judge-free and **unfiltered** (all parsed
  rows); the coherence-filtered bridge analysis is reported beside it (proposal,
  **operator to confirm**).

## 8. Statistical procedure

### 8.1 Intervals

* Means of judge scores: **question-clustered bootstrap**, 10,000 draws, percentile
  95% interval: resample the 40 questions with replacement, then the samples within
  each drawn question with replacement; statistic = mean of the pooled resampled
  values (`scripts/measurement/score_stats.py`, `clustered_bootstrap`).
* Differences of two arms at the same k (same questions): questions are resampled
  **jointly** for both arms, samples independently within arm
  (`clustered_bootstrap_diff`). A two-sided bootstrap p is computed for the Holm flag
  only.
* Bridge shares: **Wilson** 95% intervals; differences: **Newcombe's** hybrid-score
  interval (the per-game `stats.py`, imported, not copied). These treat the 200 rows
  of a point as independent although they are 5 draws of 40 questions; as a
  sensitivity analysis the same differences are also reported with the
  question-clustered bootstrap (proposal, **operator to confirm**).
* Agreement statistics (§ 9): bootstrap over items, 2,000 draws.

### 8.2 The verdict per end

"**decision − null at matched unit beta**", read from **whether the 95% interval
excludes zero**, never from a p (`.dm-knowledge`: interval and test can disagree),
separately for the negative and the positive end, against **both** the shuffled and
the orthogonal null (`score_stats.end_verdict`):

* `beats` — the interval excludes zero in the hypothesised direction;
* `reversed` — it excludes zero in the opposite direction;
* `null` — it contains zero;
* `undetermined` — either arm's point **produced no distribution**, by the per-game
  study's rule (`degradation.py`, imported): share of answers containing non-Latin
  script ≥ 0.5, or usable coverage < 0.5, where usable = the primary judge returned a
  trait score and (filtered analysis) coherence ≥ 50. So a coherence collapse makes a
  point unreadable, and an end compared against it is missing evidence, not evidence
  of absence.

### 8.3 Primary family and multiplicity

* **Primary rung**: `k* = 3` (unit beta ±31.5249) — proposal, **operator to
  confirm**. Why not ±5: the per-game study found the Prisoner's Dilemma answering
  entirely in non-Latin script at k = +4/+5 and the Dictator peaking at k = +2 and
  falling back, and the Dictator study found its comparison at the largest dose
  measuring saturation rather than mechanism; ±3 is the largest rung where those
  studies' surfaces were intact. ±1 and ±5 are reported as secondary.
* **Primary family** (counted): 6 games × 2 ends × 2 nulls = **24 contrasts**, each
  "decision − null" at k = ±k*, held-out set, primary judge, filtered trait score.
* Each contrast gets its verdict from its own interval (§ 8.2). Beside it, a **Holm**
  step-down over the 24 bootstrap p-values (α = 0.05) flags which `beats` verdicts
  would **not** survive the correction (`score_stats.holm`; an `undetermined` member
  stays in the family with no p and fails).
* Everything else (other rungs, unfiltered, in-distribution, secondary judge, bridge,
  pairwise, trait-vector comparisons, relaxed and Dictator-only arms) is secondary
  and reported with the number of comparisons it involves, uncorrected, labelled as
  such.
* **Two judges**: every primary-family verdict is reported under both judges side by
  side; a claim that holds under one judge and not the other is reported as exactly
  that.

## 9. Judge validation (checkpoint 2, before full judging)

### 9.1 Human calibration sample

* ~**200 answers**, stratified across (arm role × k) of the rows available at
  checkpoint 2, held-out set: strata = {decision, relaxed + Dictator-only, trait,
  shuffled null, orthogonal null} × each k ≠ 0, plus the k = 0 baseline as its own
  stratum; equal allocation, the remainder to the k = 0 stratum; uniform random rows
  within a stratum; seed **20261025** (§ 13).
  Whether it is drawn from the checkpoint-2 pilot rows or from full generation is an
  **OPEN DECISION** (§ 12).
* Exported as a **blind sheet**: `item_id` (opaque random id), `question`, `answer` —
  nothing else — in shuffled order. The id-to-row key is kept out of the sheet.
* Labelled independently by the operator and a colleague in
  `scripts/grader_page.py`'s page (trait 0–100, coherence 0–100, refusal/off-topic
  flags, note), exported as `human_labels_<reviewer>_<date>.csv`. The page's audit
  mode reveals a judge's score only for an item already rated, from a separate file,
  and freezes that rating.
* Reported: human–human, human–judge, judge–judge agreement — Spearman on raw scores,
  quadratic-weighted κ and Krippendorff's α (interval) on scores binned 0–19, 20–39,
  40–59, 60–79, 80–100 — each with a 95% bootstrap interval over items, 2,000 draws
  (`scripts/measurement/agreement.py`).

### 9.2 Determinism re-send

5% of each judge's requests, selected by seed **20261026** from the sorted list of
request-payload sha256s, re-sent after the main pass with the cache bypassed;
reported: exact match of the returned content / top-logprob list, and the score
delta distribution.

### 9.3 Bridge AUC

On the bridge set, does each judge's trait score track the parsed A/B choice? AUC =
P(trait score of an other-regarding choice > that of a self-interested choice),
question-clustered bootstrap interval.

### 9.4 Decision rule

All of 9.1–9.3 is reported to the operator and **full judging does not start until
the operator approves.** If agreement is poor, the plan is changed explicitly and
logged in `DEVIATIONS.md`. A numeric "poor" threshold is proposed for the operator to
accept or replace: human–judge Spearman's lower interval bound < 0.5 on trait, or
bridge AUC's lower bound ≤ 0.5 (**OPEN DECISION**).

## 10. Cost and trims

Full estimate, with every measured input and every assumption printed beside its
result: **`analysis/cost.md`** (generated by `scripts/provenance/cost_estimate.py`).
Headline numbers for the default design:

* **80,400 generations** (22 arms × 6 rungs + shared k = 0 + unhooked check point =
  134 points × 200 rows × 3 sets), ~2.9e7 generated tokens;
* **~41–49 GPU-hours on one A40** at the per-game sweep's measured 165.5 tok/s
  (≈ 20–25 h wall clock on the two A40s of this machine);
* **judge calls: 278,460 (primary)** and **189,756 (secondary)**, determinism re-send
  included;
* judge throughput on PimientoRojo is **not measured**; over an assumed range it is
  roughly 200–1,100 calls/hour per judge, i.e. 241–1,247 hours for the primary judge's
  calls. **The default design does not fit in a week on one judge server.**

Trim options, each priced in `analysis/cost.md`, **for the operator to choose (none
adopted)**: a 4- or 2-rung ladder; 3 samples per question; pairwise on held-out at
k = ±3 only; no secondary-judge pairwise; secondary judge on a 20% subset; fewer arms
on the in-distribution set.

## 11. Judges — OPEN DECISIONS for the operator

Facts as of 2026-10-23 (`provenance/pimientorojo_probe.md` for the full record):

* PimientoRojo is an AMD Strix Halo machine (Radeon 8060S iGPU, gfx1151, ~125–128 GB
  unified memory, 32 cores, no NVIDIA GPU; shared, 7 users logged in at the probe). It
  was found running Ollama 0.15.6 on localhost serving only `llama4-scout:latest`
  (Q6_K). **Ollama is excluded by operator decision** ("it adds layers to the
  experiment").
* The judge server will be llama.cpp **`llama-server`** release **b11836**
  (0.7.0-dev, commit `f5027fb21`, ubuntu-vulkan-x64 build, device Vulkan0 = AMD Radeon
  8060S, RADV GFX1151), bound to localhost on PimientoRojo and reached through an SSH
  tunnel; the endpoint is configured only through `PIMIENTOROJO_BASE_URL`. vLLM is not
  installed there (no ROCm); if vLLM is ever used instead, weights provenance is the
  HF repo + 40-hex revision.
* Operator decision 2: judges are **Llama-3.3-70B-Instruct** and **gpt-oss-20b** —
  two families, neither Qwen. Files (sha256 verified on the server against Hugging
  Face LFS, and re-checked against HF's LFS metadata at the pinned revision by
  `scripts/judging/tokenizer_check.py`):
  * `bartowski/Llama-3.3-70B-Instruct-GGUF` @ `b6c5c9f176f3279204034e1d16d393105e95cb88`,
    **Q8_0**, `Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`
    (`0c4ec5083aec2efa104269e8b63e2d9f3322e2fa50511184314093a7b4697878`) +
    `…-00002-of-00002.gguf`
    (`e5c8858f5feafdc56200bbcbb3c4fadb07cb323c8ed13548dfb798dd1fb24255`);
  * `ggml-org/gpt-oss-20b-GGUF` @ `15dcfc7e7c608c6161507211ed43c5b590801b4b`,
    `gpt-oss-20b-MXFP4.gguf`
    (`27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`).

Decisions the operator makes (proposals in bold; nothing below is chosen silently):

| # | decision | proposal | why |
|---|---|---|---|
| J1 | primary / secondary judge | **primary = Llama-3.3-70B-Instruct Q8_0; secondary = gpt-oss-20b MXFP4** | the operator wants to check that Llama 70B does the work well; it passes the single-token check in the scored position and can use upstream's logprob mode; gpt-oss is a reasoning model and needs text mode |
| J2 | quant level of the 70B | **Q8_0** (orchestrator's default) | FP8 is not a GGUF type; Q8_0 is the closest. GGUF judges are quantised — a judge number here is a Q8_0 / MXFP4 number, not a bf16 one |
| J3 | Llama scoring mode | **`logprob_0_100`**, conditional on the endpoint check (top_logprobs ≥ 20 actually returned by llama-server) at the start of checkpoint 2; if it fails, `text_0_100`, fixed before any judgment | tokenizer check: every "0"…"100" is one token after the assistant header in the GGUF vocabulary, tokenized by llama.cpp b11836's own tokenizer (the HF tokenizer check is not run: gated, see below) |
| J4 | gpt-oss scoring mode | **`text_0_100`**, final channel only | its first assistant token is a channel marker, never a digit, so a one-token logprob read does not apply |
| J5 | gpt-oss reasoning effort | **low**, sent as `chat_template_kwargs: {"reasoning_effort": "low"}` (llama-server) and recorded in every request payload | fixed and recorded; low keeps the call cost bounded |
| J6 | text-mode format instruction | **append, after the verbatim template, exactly:** `\n\nWrite the number inside <score></score> tags, for example <score>73</score>, and write nothing else. If the model clearly refused to answer, write REFUSAL instead.` — sha256 recorded at freeze | the upstream templates ask for a bare number; `text_0_100` needs exactly one tag. The template itself stays verbatim; the suffix is a separate, hashed constant |
| J7 | judge `max_tokens` | **1** (logprob); **32** (Llama text); **2048** (gpt-oss, reasoning + final) — a hit is status `truncated` | explicit, not a server default |
| J8 | judge seed / temperature / top_logprobs | **seed 0, temperature 0, top_logprobs 20** | upstream's values |
| J9 | llama-server concurrency | **`--parallel 1`** for the validation pass; any larger value only after the determinism re-send (§ 9.2) shows no change; the launch line (`--parallel`, `-b`, `-ub`, `-c`, `-fa`, model path) recorded per run with `/props` | batching can change outputs at temperature 0 |
| J10 | served model ids | **set with `--alias`** to `llama-3.3-70b-instruct-q8_0` and `gpt-oss-20b-mxfp4`; the client aborts if `/v1/models` does not list the configured id | the id is what the client checks |
| J11 | weights provenance per run | **GGUF file name(s) + sha256 + quant type + llama.cpp build/commit**, from the run config plus `/props` (`model_path`, `build_info`) as returned; a field the server does not expose is recorded as unavailable, never guessed | operator decision 1 |

**Tokenizer check** (`provenance/tokenizer_check.json`, by
`scripts/judging/tokenizer_check.py`; "first_assistant" = the string as the judge's
first assistant token after its own chat template's generation prompt):

| judge | tokenizer | bare "0"…"100" | " 0"…" 100" | first assistant token |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct Q8_0 | GGUF vocab, llama.cpp b11836 `llama-tokenize` | 101/101 one token | 0/101 (space is its own token) | 101/101 |
| Llama-3.3-70B-Instruct | HF `meta-llama/Llama-3.3-70B-Instruct` | **not run (gated: no access)** | — | — |
| gpt-oss-20b MXFP4 | GGUF vocab, llama.cpp b11836 | 101/101 | 0/101 | 101/101 (after `<|channel|>final<|message|>`) |
| gpt-oss-20b | HF `openai/gpt-oss-20b` | 101/101 | 0/101 | 101/101; GGUF and HF vocabularies identical except 1,069 GGUF padding entries, merges identical |
| gpt-oss-120b (candidate) | HF | 101/101 | 0/101 | 101/101 |
| Llama-4-Scout-17B-16E-Instruct (candidate) | HF | **not run (gated)** | — | — |

The endpoint half of the check — **does llama-server return `top_logprobs` ≥ 20** —
is **not run** in this item (no endpoint call was made). It runs first at
checkpoint 2, and J3 waits for it.

## 12. Every other open decision

| # | decision | proposal |
|---|---|---|
| O1 | ladder | **k ∈ {−5,−3,−1,0,+1,+3,+5}**, or a trim from § 10 |
| O2 | samples per question | **5** |
| O3 | `max_new_tokens` | **1000** (§ 4) |
| O4 | primary rung k* | **3** (§ 8.3) |
| O5 | truncated generations | **kept and judged**, share reported |
| O6 | bridge primary analysis | **judge-free, unfiltered**; filtered beside it |
| O7 | bridge sensitivity | **also question-clustered bootstrap** beside Wilson/Newcombe |
| O8 | null verdicts for the secondary arms | **none**: relaxed and Dictator-only arms are compared with baseline and with the trait vector only (their committed shuffled nulls exist; no orthogonal null was built for them) |
| O9 | pairwise scope | **held-out and in-distribution**, every steered row, both orders, primary judge; secondary judge on the 20% subset (§ 13) |
| O10 | human sample source | **the checkpoint-2 pilot rows** if full generation has not run, otherwise full generation; strata as § 9.1 |
| O11 | "poor agreement" threshold | **§ 9.4** |
| O12 | held-out and bridge wording | the operator reviews every item and freezes them |

## 13. Seeds (all fixed now)

| what | seed |
|---|---|
| generation (`Sampling.seed`; batches `seed + batch_index`) | 0 |
| bridge counterbalancing | 20261023 |
| secondary-judge pairwise subset: `random.Random(20261024).sample(sorted (question_id, sample_index) pairs, 40)` of the 200 per point — the same 40 pairs for every (arm, k, set) | 20261024 |
| human calibration sample (§ 9.1) | 20261025 |
| determinism re-send selection (§ 9.2) | 20261026 |
| judge send order (`judge_many(order_seed=…)`) | 20261027 |
| bootstrap (every interval in § 8; agreement in § 9) | 20261028 |
| judge request `seed` | 0 |

## 14. What this pre-registration does not cover

* The pilot at checkpoint 2 (one game arm + trait vector + nulls at k ∈ {−3, 0, +3},
  small n) is a plumbing and judge-validation run; its numbers are not results.
* The generation entry point for general questions (a lab/ prompt type that renders a
  question as one user turn, through `audit.generate` and `audit.steer`) is
  implementation, written before the pilot and tested offline; it changes no choice
  above.
* Anything decided after data exists goes to `DEVIATIONS.md`, dated, with its reason.
