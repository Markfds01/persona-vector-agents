# PRE-REGISTRATION — general-question transfer of the game-derived decision vectors

> **STATUS: DRAFT, pending checkpoint-1 approval.** Nothing in this file is frozen
> yet. It becomes the pre-registration — and from then on is never edited; changes
> go to `DEVIATIONS.md` — only after the operator approves checkpoint 1 and the
> freezing commit records its sha256 in `provenance/`. Until then every item marked
> **OPEN DECISION** or **operator to confirm** is a proposal, not a choice. No
> generation and no judgment exists for this study at the time of writing
> (2026-10-23).
>
> Drafted by an LLM agent (Claude) on 2026-10-23 for the operator's review. **Revision 2
> (2026-10-23)** after the operator's feedback: a ÷8 design (§ 4, § 10), the measured judge
> results (§ 11), the local-llama-server client wording, a single human labeller (§ 9.1), and
> every remaining open decision gathered in § 0.

Contents: 0 remaining decisions for the operator · 1 question · 2 hypotheses · 3 arms · 4 ladder, n, generation · 5 question
sets · 6 judgments and templates · 7 metrics and exclusion · 8 statistics and verdicts ·
9 judge validation (checkpoint 2) · 10 cost and trims · 11 judges: OPEN DECISIONS ·
12 every other open decision · 13 seeds · 14 what is not pre-registered here.

---

## 0. Remaining open decisions and questions for the operator

Everything still undecided, each with the answer recommended here. Nothing below is chosen
until the operator says so; "ready to freeze" means only that no further change is planned.
Details are in the section named in the last column.

| # | decision / question | recommended answer | § |
|---|---|---|---|
| D1 | **The ÷8 configuration** (total judge work ≈ 1/8 of the default's 348 h) | **`div8_main`**: ladder {−3, 0, +3} for all 22 arms (trait vector also at ±1, ±5 on held-out), 3 samples per question, held-out for all arms, in-distribution for the 6 decision vectors + trait vector, bridge for all arms, pairwise on held-out at ±3 by the primary judge, secondary judge on a pre-registered 25% subset. 13,200 generations; 33,264 primary + 7,182 secondary calls; **37.3 h** of judging (1/9.3), 44.3 h if pairwise calls are priced by length. The one alternative, `div8_alt` (no pairwise, 5 samples on held-out): 34.1 h. | 4, 10 |
| D2 | primary / secondary judge | **primary Llama-3.3-70B-Instruct Q8_0 (`logprob_0_100`); secondary gpt-oss-20b MXFP4 (`text_0_100`)** — measured, not assumed | 11 |
| D3 | quant level of the 70B | **Q8_0** (the only quant measured). Q4_K_M would be faster but is unmeasured and a coarser judge; it would need its own measurement and calibration | 11 |
| D4 | gpt-oss reasoning effort | **low** (measured 1.00 s/call; medium/high not measured, slower roughly in proportion to reasoning tokens) | 11 |
| D5 | `max_new_tokens` | **1000** (the per-game study's value; longest proxy answer 784 tokens) | 4 |
| D6 | **Llama system header** "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" | the header comes from the GGUF-embedded Llama-3.x chat template, which defaults `date_string` to "26 Jul 2024" when the caller passes none (llama-server passes none). **Recommended: pass a fixed explicit `date_string` (the freeze date) via `chat_template_kwargs`**, so the rendered prompt is fully specified by this repository's own config, and hash the rendered system header with the prompt template. The alternative (record the template default as-is) is identical on every call and also valid | 11 |
| D7 | gpt-oss "Current date" in its own system header | its template writes `Current date: <today>` with `strftime_now` and takes no override, so the secondary judge's prompt carries the day it runs. **Recommended: accept, record the call timestamp (already on every row), and run the whole secondary pass in one sitting** (about 2 h) so it sees one date | 11 |
| D8 | primary rung k* | **3** | 8.3 |
| D9 | secondary-judge subset | **25%** of the (question, sample) pairs, one fixed subset for every point. Giving the secondary judge all held-out rows costs 2.6 h more (39.9 h total) | 4, 10 |
| D10 | human labels | **the operator is the only labeller** (no colleague). The human–human agreement of the original specification cannot be computed; **declared substitute**: intra-rater test–retest on a hidden random 25% of the calibration sheet re-labelled after ≥ 3 days, plus human–judge and judge–judge agreement. **A deviation from the original specification; operator to confirm.** A second labeller can be added later without changing anything else | 9.1 |
| D11 | question sets | the 40 held-out and 40 bridge drafts are accepted as-is ("draft is okay") and are **ready to freeze**. Authorship to record in the freezing commit: drafted by an LLM agent (Claude) on 2026-10-23, accepted unedited by the operator on 2026-10-23 | 5 |
| D12 | "poor agreement" threshold | human–judge Spearman lower bound < 0.5 on trait, or bridge AUC lower bound ≤ 0.5 (§ 9.4); with one labeller the test–retest Spearman must itself reach 0.7, otherwise the labels are too noisy to calibrate against | 9.4 |
| D13 | pairwise secondary judge | on the same 25% subset (above) | 6 |
| D14 | minor items | truncated generations kept and judged; bridge judge-free and unfiltered with the filtered analysis beside it; bridge sensitivity by clustered bootstrap; no null verdicts for the relaxed and Dictator-only arms; human sample from the pilot rows (§ 12, O5–O10) | 12 |

When the operator approves, the freezing commit records this file's sha256 in `provenance/`
together with the answers to D1–D14.

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

* **Design (proposal `div8_main`; operator to confirm, D1).** The operator asked for judge
  work of about 1/8 of the default design's (the default — 7-rung ladder, 5 samples, every
  arm on every set — is 80,400 generations and 348 h of judging at the measured speeds and
  is **not** proposed; `analysis/cost.md`).
  * **Ladder**: `k ∈ {−3, 0, +3}` for **every** arm, i.e. `unit_beta ∈ {±31.5249, 0}`;
    plus the **trait vector only** at `k ∈ {−5, −1, +1, +5}`
    (`unit_beta ±52.5415, ±10.5083`) on the held-out set, a dose-response reference.
  * **n per point**: 40 questions × **3** samples = 120 rows per (arm, k, question set).
    Samples of one question are correlated, so **the effective n is close to the number
    of questions (40), not 120**; every interval is question-clustered (§ 8.1).
  * **Which arms on which set**:

| set | arms | k | points | generations |
|---|---|---|---|---|
| held-out (primary) | all 22 steered arms (§ 3) | ±3 (trait vector also ±1, ±5) | 22 × 2 + 4 + shared k = 0 baseline = 49, plus the unhooked check point | 6,000 |
| in-distribution | the 6 decision vectors and the trait vector | ±3 | 7 × 2 + baseline = 15 | 1,800 |
| bridge | all 22 steered arms | ±3 | 22 × 2 + baseline = 45 | 5,400 |
| **total** | | | | **13,200** |

  The relaxed and Dictator-only arms are held-out and bridge arms only; the nulls are
  held-out and bridge arms only. Judge calls per kind: § 6 and § 10.
  One alternative is priced in § 10 (`div8_alt`: no pairwise, 5 samples on the held-out
  set).
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
| in-distribution | `data_generation/trait_data_eval/altruistic.json` (sha256 `a1a2d2b6…07dc`) | 40 | secondary, decision vectors and trait vector only, labelled "in-distribution for the trait vector" |
| bridge | `questions/bridge_draft.json` | 40 | judge-free ground truth |

* **Held-out**: 40 new everyday dilemmas where self-interest and others' welfare
  conflict, same style as `altruistic.json`, no game framing, no money to split, no
  mention of the six games. Drafted by an LLM agent (Claude) on 2026-10-23; the operator
  accepted both drafts as they stand ("draft is okay", 2026-10-23), so they are **ready to
  freeze** (D11); the freezing commit renames and re-hashes them and records the
  authorship (LLM-drafted, operator-accepted unedited).
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
Judges are reached only through `lab/llama_server_client.py` (HTTP client for the local
llama-server (llama.cpp) endpoint, `/v1/chat/completions`; endpoint from
`PIMIENTOROJO_BASE_URL` only, no fallback; every request sends `cache_prompt: false`; § 11). The judge sees the
question and the answer and nothing else; calls are sent in a seeded shuffled order.

Per answer, **both judges** score:

1. **Trait** — `eval_prompt` of `data_generation/trait_data_eval/altruistic.json`,
   verbatim, sha256 `07fe12f75c7a50dd8cd398d41f6b33b52c5bb0be9bd6c039877042fb6cb518dd`,
   0–100.
2. **Coherence** — `Prompts["coherence_0_100"]` of `eval/prompts.py`, verbatim,
   sha256 `78945e05210836da5c492e01cead92cdc8f8aa40f1b0c22d162b4dc2a1344a97`, 0–100.

For the bridge set, `{question}` is the full bridge prompt and `{answer}` the full
continuation. The bridge set's choice is judge-free (parsed), so in the ÷8 design the
**primary judge scores coherence on every bridge row** (for the filtered bridge analysis)
and **both judges score trait only on the AUC subset** — the k = 0 baseline and the trait
vector at ±3, 3 × 120 rows — which is what the bridge AUC check (§ 9.3) needs. The
secondary judge does not score bridge coherence.

**Which judge scores what (`div8_main`)**: primary — trait and coherence on every held-out
and in-distribution row, coherence on every bridge row, trait on the bridge AUC subset,
pairwise on held-out at ±3. Secondary — trait and coherence on a pre-registered random 25%
subset of the held-out and in-distribution rows (§ 13), trait on the bridge AUC subset,
pairwise on the same 25% subset.

3. **Pairwise** (secondary metric). For every steered answer (arm, k = ±3, question q,
   sample s) on the **held-out** set, paired with the k = 0 baseline answer to the same q
   and s, sent in **both orders**. Primary judge on all pairs; secondary judge on the
   pre-registered 25% subset (§ 13). The template, written once
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
`unparseable`. (The pairwise parser and its tests are added to `lab/llama_server_client.py`
before checkpoint 2; the client implements the two 0–100 scoring modes only.)

**Scoring modes** (`logprob_0_100`, `text_0_100`) and which judge uses which: § 11.
Never switched mid-run, never a fallback.

## 7. Metrics and exclusion

### 7.1 Primary metric

Mean **trait score by the primary judge**, on the **held-out** set, over rows kept by
the exclusion rule (§ 7.3), per (arm, k).

### 7.2 Secondary metrics

* the same, **unfiltered** (every row with a trait score), reported beside the primary;
* the same on the **in-distribution** set (decision vectors and trait vector only),
  labelled as such;
* the same by the **secondary judge**, on its 25% subset only (its own coherence used only
  for its own descriptive coherence column; exclusion is always by the primary judge);
* **coherence** (primary and secondary judge), mean per point;
* **share excluded**, **share truncated**, share with each judge status, per point;
* **surface degeneracy** per point, counted, not rated (per-game `stats.degeneracy`):
  mean length, distinct-trigram ratio, share with a word trigram repeated ≥ 5 times,
  share containing non-Latin script;
* **bridge**: P(other-regarding choice) over parsed rows, with parse coverage;
* **pairwise** (held-out, ±3): P(steered answer judged more other-regarding than its baseline pair),
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
  interval (the per-game `stats.py`, imported, not copied). These treat the 120 rows
  of a point as independent although they are 3 draws of 40 questions; as a
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
  studies' surfaces were intact. In the ÷8 design ±1 and ±5 exist for the trait vector
  only (reported as secondary); **no other arm has a rung to fall back on**, so an arm
  whose ±3 point produces no distribution is `undetermined` at that end (§ 8.2).
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
  that. Under the ÷8 design the secondary judge scores a 25% subset (30 of the 120 rows of a
  point, about 0.75 sample per question), so its 24 verdicts are **lower-precision and
  descriptive**: an interval that contains zero there is not evidence against the primary
  judge's verdict, and the secondary judge's verdicts are not part of the Holm family.

### 8.4 Power and what the ÷8 design costs (stated, not hidden)

* **The effective n is about 40 questions in every design**; going from 5 to 3 samples per
  question does not change it. What it changes is the within-question noise of each
  question's mean: relative to the default design, the standard error of a difference grows
  by a factor between 1.00 (all variance between questions) and √(5/3) = 1.29 (all variance
  within questions). The truth is in between and is estimated from the checkpoint-2 pilot
  rows (plumbing numbers, not results).
* Approximate planning figures, normal approximation, paired difference of two arms on the
  same 40 questions with s_d the standard deviation across questions of the per-question
  mean difference: the 95% interval excludes zero when |Δ| ≳ 0.31 s_d; the minimum
  detectable difference at 80% power is ≈ 0.44 s_d (uncorrected), and ≈ 0.62 s_d at the
  smallest Holm threshold of the 24-member family (0.05/24); each multiplied by up to 1.29
  for 3 samples. A percentile bootstrap over 40 clusters is mildly liberal.
* **Preserved from the default**: the primary hypotheses (§ 2) at both ends against both
  nulls on the held-out set at k* = ±3, all 24 contrasts, the same family and the same
  Holm correction, all 22 arms on the held-out and bridge sets, the judge-free bridge.
* **Lost versus the default**: (1) the rungs ±1 and ±5 for every arm but the trait vector —
  no dose-response, no non-monotonic shape (the Dictator's peak at k = +2), no fallback when
  ±3 is degenerate; (2) 40% fewer rows per point (120 vs 200); (3) the in-distribution set
  for the nulls, the relaxed and Dictator-only arms — no in-distribution decision-minus-null
  contrast; (4) pairwise on the in-distribution set and the secondary judge's pairwise on
  more than 25%; (5) the secondary judge on all rows (its verdicts become descriptive);
  (6) bridge trait scores outside the AUC subset and the secondary judge's bridge coherence;
  (7) tighter bridge intervals (Wilson n = 120 per point, not 200).

## 9. Judge validation (checkpoint 2, before full judging)

### 9.1 Human calibration sample

* ~**200 answers**, stratified across (arm role × k) of the rows available at
  checkpoint 2, held-out set: strata = {decision, relaxed + Dictator-only, trait,
  shuffled null, orthogonal null} × k ∈ {−3, +3}, plus the k = 0 baseline as its own
  stratum (11 strata); 18 per steered stratum, the remaining 20 to the k = 0 stratum;
  uniform random rows within a stratum; seed **20261025** (§ 13).
  Whether it is drawn from the checkpoint-2 pilot rows or from full generation is an
  **OPEN DECISION** (§ 12).
* Exported as a **blind sheet**: `item_id` (opaque random id), `question`, `answer` —
  nothing else — in shuffled order. The id-to-row key is kept out of the sheet.
* Labelled by **the operator, who is the only human labeller** (no colleague), in
  `scripts/grader_page.py`'s page (trait 0–100, coherence 0–100, refusal/off-topic
  flags, note), exported as `human_labels_<reviewer>_<date>.csv`. The page's audit
  mode reveals a judge's score only for an item already rated, from a separate file,
  and freezes that rating.
* **Declared substitute for human–human agreement (operator to confirm, D10; a deviation
  from the original specification, which had two labellers).** With one human the
  inter-rater reliability cannot be computed, so the labels' reliability is measured by
  **intra-rater test–retest**: after **≥ 3 days**, the operator re-labels a **random 25%**
  of the calibration sheet (about 50 items; seed **20261029**), built by
  `grader_page.py retest` — re-shuffled, every item under a **fresh opaque item_id**, no
  marker that it is a repeat, the id-to-original key kept in a separate file and read only
  by `agreement.py --retest-key`. The operator may still recognise an answer; the ≥ 3-day
  gap is the mitigation and the limitation is reported. Reported: Spearman and
  quadratic-weighted κ / Krippendorff's α (binned) between the two passes, with bootstrap
  intervals over items. A test–retest Spearman below 0.7 on either score is reported as
  "the human labels are too noisy to calibrate against" (D12).
* Reported: test–retest (above), human–judge, judge–judge agreement — Spearman on raw scores,
  quadratic-weighted κ and Krippendorff's α (interval) on scores binned 0–19, 20–39,
  40–59, 60–79, 80–100 — each with a 95% bootstrap interval over items, 2,000 draws
  (`scripts/measurement/agreement.py`).

### 9.2 Determinism re-send

5% of each judge's requests, selected by seed **20261026** from the sorted list of
request-payload sha256s, re-sent after the main pass (every request already carries
`cache_prompt: false`, so a re-send is a cold repeat);
reported: exact match of the returned content / top-logprob list, and the score
delta distribution. Measured on the bench (`provenance/judge_measurement_2026-10-23.md`):
Llama logprob 24/24 identical with the cache off, 10/24 differing (up to 0.30) with it on;
Llama text 24/24; gpt-oss text 23/24 (one 80 ↔ 70 flip) — the re-send reports the study's
own rates.

### 9.3 Bridge AUC

On the bridge set, does each judge's trait score track the parsed A/B choice? AUC =
P(trait score of an other-regarding choice > that of a self-interested choice),
question-clustered bootstrap interval.

### 9.4 Decision rule

All of 9.1–9.3 is reported to the operator and **full judging does not start until
the operator approves.** If agreement is poor, the plan is changed explicitly and
logged in `DEVIATIONS.md`. A numeric "poor" threshold is proposed for the operator to
accept or replace: human–judge Spearman's lower interval bound < 0.5 on trait, or
bridge AUC's lower bound ≤ 0.5, or — with a single labeller — a test–retest Spearman
below 0.7 on trait (**OPEN DECISION**, D12).

## 10. Cost and the ÷8 design

Full estimate, with every measured input and every assumption printed beside its result:
**`analysis/cost.md`** (generated by `scripts/provenance/cost_estimate.py`). The judge speeds
in it are **measured** (§ 11); the one assumption is the length of a pairwise call.

**Default design (reference only, not proposed)**: 80,400 generations (22 arms × 6 rungs +
shared k = 0 + unhooked check, × 200 rows × 3 sets), ~2.9e7 generated tokens, ~41–49 GPU-hours
on one A40; **278,460 primary and 189,756 secondary judge calls**, which at the measured
speeds is 295.5 h + 52.7 h = **348.2 h (14.5 days)** of serial judging at `--parallel 1`.

**The operator asked for ÷8.** Proposed configuration **`div8_main`** (§ 4 for the sets and
rungs; D1):

| quantity | `div8_main` | default |
|---|---|---|
| generations | **13,200** (6,000 held-out + 1,800 in-distribution + 5,400 bridge) | 80,400 |
| generation GPU-hours, one A40 | 6.7–8.1 | 41–49 |
| primary judge calls | **33,264** = trait 8,040 + coherence 13,080 + pairwise 10,560 + 5% re-send 1,584 | 278,460 |
| secondary judge calls | **7,182** = trait 2,280 + coherence 1,920 + pairwise 2,640 + 5% re-send 342 | 189,756 |
| primary hours (3.82 s/call) | **35.3 h** | 295.5 h |
| secondary hours (1.00 s/call) | **2.0 h** | 52.7 h |
| total judging | **37.3 h = 1/9.3** of the default | 348.2 h |

With the pairwise calls priced by prompt length instead of at the measured 3.82 s (a pairwise
prompt holds two answers: ≈ 6.2 s on the primary, ≈ 1.17 s on the secondary; no pairwise call
was measured), `div8_main` takes 42.2 h + 2.1 h = **44.3 h = 1/7.9** — so the design sits
between 1/8 and 1/9 whichever way that assumption falls.

The trait counts: primary trait = 5,880 held-out + 1,800 in-distribution + 360 bridge-AUC
rows; coherence = every judged row (5,880 + 1,800 + 5,400); pairwise = 22 arms × 2 rungs × 120
steered rows × 2 orders. Secondary = 25% of the held-out and in-distribution rows (trait and
coherence), the 360 bridge-AUC rows, 25% of the pairwise calls.

**One alternative, `div8_alt`** — as `div8_main` but **no pairwise anywhere** and **5 samples
per question on the held-out set** (200 rows per held-out point): 17,200 generations, 30,408
primary + 6,468 secondary calls, 32.3 h + 1.8 h = **34.1 h (1/10.2)**. It buys back the
within-question precision on the primary set (the SE inflation of § 8.4 disappears there) and
gives up pairwise entirely. Recommended: `div8_main`, because pairwise is the one
corroboration in a different judging format; if the operator values the primary contrasts'
precision above that, `div8_alt`.

Also priced: `div8_main` with the secondary judge on **all** held-out rows (D9): 16,443
secondary calls, 4.6 h, total 39.9 h (1/8.7). What the ÷8 design preserves and loses, and the
power caveats: **§ 8.4**. Other single trims of the default are priced in `analysis/cost.md`
for reference; none is adopted.

## 11. Judges — measured, recommended values (operator to confirm)

Measured on PimientoRojo on 2026-10-23, bench items only (no study data), full report
`provenance/judge_measurement_2026-10-23.md`; the probe of the host is
`provenance/pimientorojo_probe.md`.

* PimientoRojo is an AMD Strix Halo machine (Radeon 8060S iGPU, gfx1151, ~125–128 GB
  unified memory, 32 cores, no NVIDIA GPU; shared). It was found running Ollama 0.15.6 on
  localhost serving only `llama4-scout:latest` (Q6_K). **Ollama is excluded by operator
  decision** ("it adds layers to the experiment").
* The judge server is llama.cpp **`llama-server` b11836** (0.7.0-dev, commit `f5027fb21`,
  Vulkan build, device Vulkan0 = AMD Radeon 8060S, RADV GFX1151), bound to the loopback
  interface of PimientoRojo and reached through an SSH tunnel; the client talks only to this
  server and the endpoint is configured only through `PIMIENTOROJO_BASE_URL`. Launch lines as
  run: Llama `-c 8192 --parallel 1 --jinja`, gpt-oss `-c 16384 --parallel 1 --jinja`, both
  `-ngl 999`.
* Operator decision 2: judges are **Llama-3.3-70B-Instruct** and **gpt-oss-20b** — two
  families, neither Qwen. Files (sha256 verified on the server against Hugging Face LFS, and
  re-checked against HF's LFS metadata at the pinned revision by
  `scripts/judging/tokenizer_check.py`):
  * `bartowski/Llama-3.3-70B-Instruct-GGUF` @ `b6c5c9f176f3279204034e1d16d393105e95cb88`,
    **Q8_0**, `Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`
    (`0c4ec5083aec2efa104269e8b63e2d9f3322e2fa50511184314093a7b4697878`) +
    `…-00002-of-00002.gguf`
    (`e5c8858f5feafdc56200bbcbb3c4fadb07cb323c8ed13548dfb798dd1fb24255`);
  * `ggml-org/gpt-oss-20b-GGUF` @ `15dcfc7e7c608c6161507211ed43c5b590801b4b`,
    `gpt-oss-20b-MXFP4.gguf`
    (`27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`).

**Measured results that fix the recommendations** (12 bench items, 24 distinct prompts per
mode, temperature 0, seed 0, `cache_prompt=false`, one request at a time):

| | Llama-3.3-70B Q8_0, `logprob_0_100` | Llama-3.3-70B Q8_0, text | gpt-oss-20b MXFP4, `text_0_100`, effort low |
|---|---|---|---|
| usable as a scorer | **yes**: 20 `top_logprobs` on 48/48; integer tokens 0–100 with mass ≥ 0.999 on 48/48 | yes | yes; its logprob mode is **not** usable (first token `<\|channel\|>`, digit mass 0 on 48/48) |
| seconds per call, mean / median | **3.82 / 3.71** (≈ 940 calls/h) | 6.09 / 5.97 (≈ 590/h) | **1.00 / 0.94** (≈ 3,600/h) |
| determinism | **24/24 identical only with `cache_prompt=false`**; with the cache on, 10/24 pairs differed, up to 0.30 | 24/24 | 23/24 (one 80 ↔ 70 flip); not bitwise reproducible |
| agreement with upstream's own scores, n = 12 (Spearman, trait / coherence) | 0.74 / 0.91 | 0.58 / 0.72 | 0.54 / 0.41 |

Parallelism gives at most 1.3× (gpt-oss with `--parallel 4`; the 70B logprob workload does
not scale at all), so planning uses `--parallel 1`. The two servers share one GPU: hours add.
A cached-prefix call is ~24% faster, but the study sends `cache_prompt: false`, so the cold
figure is the planning figure.

**Recommended values** (each still **operator to confirm**, § 0):

| # | decision | recommended | why |
|---|---|---|---|
| J1 | primary / secondary judge | **primary = Llama-3.3-70B-Instruct Q8_0; secondary = gpt-oss-20b MXFP4** | the operator wants Llama 70B to do the work; it is the only one of the two that gives a continuous, reproducible score; gpt-oss is the cheap (3.8× faster) second family |
| J2 | quant level of the 70B | **Q8_0** | measured; FP8 is not a GGUF type. GGUF judges are quantised — a judge number here is a Q8_0 / MXFP4 number, not a bf16 one. Q4_K_M unmeasured |
| J3 | Llama scoring mode | **`logprob_0_100`**, temperature 0, seed 0, `top_logprobs` 20, `max_tokens` 1, **`cache_prompt: false`**, **`--parallel 1`** | the endpoint check passed (above); `cache_prompt=false` is what makes it reproducible; it is sent by the client on every request. No fallback to text mode: if the endpoint stopped returning logprobs the run would stop |
| J4 | gpt-oss scoring mode | **`text_0_100`, final channel only** | its first assistant token is `<\|channel\|>`, never a digit |
| J5 | gpt-oss reasoning effort | **low**, `chat_template_kwargs: {"reasoning_effort": "low"}`, recorded in every request payload | measured 1.00 s/call; effort medium/high not measured |
| J6 | text-mode format instruction | **append, after the verbatim template, exactly:** `\n\nWrite the number inside <score></score> tags, for example <score>73</score>, and write nothing else. If the model clearly refused to answer, write REFUSAL instead.` — sha256 recorded at freeze | the upstream templates ask for a bare number; `text_0_100` needs exactly one tag. The bench used a shorter instruction (`Respond with only your final score inside tags, exactly like <score>NN</score>.`); the few extra tokens do not move the measured speeds beyond their noise, and checkpoint 2 re-measures on the real prompts |
| J7 | judge `max_tokens` | **1** (Llama logprob); **2048** (gpt-oss, reasoning + final; measured 35–75 generated tokens at low); **32** (Llama text, not used) — a hit is status `truncated` | explicit, not a server default |
| J8 | judge seed / temperature / top_logprobs | **seed 0, temperature 0, top_logprobs 20** | upstream's values |
| J9 | llama-server concurrency | **`--parallel 1`**; any larger value only after the determinism re-send (§ 9.2) shows no change; the launch line (`--parallel`, `-b`, `-ub`, `-c`, `-fa`, model path) recorded per run with `/props` | batching can change outputs at temperature 0; measured gain ≤ 1.3× |
| J10 | served model ids | **set with `--alias`** to `llama-3.3-70b-instruct-q8_0` and `gpt-oss-20b-mxfp4`; the client aborts if `/v1/models` does not list the configured id | the bench servers were started without an alias (their id is the model path); the study relaunches them with the alias |
| J11 | weights provenance per run | **GGUF file name(s) + sha256 + quant type + llama.cpp build `b11836` / commit `f5027fb21`**, from the run config plus `/props` (`model_path`, `build_info`) as returned; a field the server does not expose is recorded as unavailable, never guessed | operator decision 1 |
| J12 | **Llama system header** (D6) | **pass a fixed explicit `chat_template_kwargs: {"date_string": "<freeze date, D Mon YYYY>"}`** on every Llama request, recorded in the payload (and therefore in the cache key), and **hash the rendered system header together with the prompt template** | see the next paragraph |
| J13 | gpt-oss "Current date" (D7) | **accept; record the call timestamp; run the whole secondary pass in one sitting** | see the paragraph after the next |

**Where "Today Date: 26 Jul 2024" comes from** (verified on 2026-10-23 against the template text
that llama-server exposes at `/props` → `chat_template`, 4,613 characters, taken from the GGUF
with `--jinja`, and against `provenance/tokenizer_check.json`'s rendered `chat_prefix`). The
Llama-3.x chat template embedded in the GGUF (Meta's official Jinja template) always emits a
system block, `Cutting Knowledge Date: December 2023` then `Today Date: <date_string>`, and
begins with `{%- if not date_string is defined %}{%- set date_string = "26 Jul 2024" %}`. So
when the caller passes no `date_string` — and llama-server passes none unless a request sets
`chat_template_kwargs` — the date is the template's own default. The judge prompt then sits in
the user turn after that header. Nothing in it comes from this study. Two options: (a) keep the
template default — identical on every call, so harmless for comparisons across arms, but the
rendered prompt is partly specified by a file this repository does not own; (b) **recommended**:
pass an explicit `date_string` (the freeze date), so the rendered prompt is fully specified by
the study's own configuration. Whether this llama-server build forwards `chat_template_kwargs`
to the template is checked at the start of checkpoint 2 by rendering one prompt and reading the
header back; if it does not, (a) is used and the template default is recorded.

**gpt-oss's own date.** The gpt-oss harmony template writes `Current date:` into its system
header from `strftime_now("%Y-%m-%d")` and takes no override (`reasoning_effort` and
`model_identity` are the only template variables it reads); the rendered prefix in
`provenance/tokenizer_check.json` shows the day of the check. The secondary judge's prompt
therefore carries the date of the call. It cannot be pinned without editing the template, which
this study will not do; it is recorded by every row's timestamp, and the secondary pass is
about 2 h, so it can run in one sitting under one date.

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

The endpoint half of the check was **run on 2026-10-23** by the judge-measurement bench
(`provenance/judge_measurement_2026-10-23.md`, 12 bench items, no study data): llama-server
returned exactly 20 `top_logprobs` on 48/48 Llama calls, and the first token's digit mass was
≥ 0.999 (minimum 1.000) on 48/48 trait and coherence calls, each score token a whole integer
(`100`, `90`, `95`, `80`, `0`, …). For gpt-oss-20b the first token is `<|channel|>` on 48/48
calls and the digit mass is 0 — its logprob mode is **not usable**, consistent with J4. The
HF-tokenizer half for the Llama model is still not run (gated repository).

## 12. Every other open decision

(The decisions that need the operator's attention are in § 0. These are the remaining
parameters, each with its recommended value; unless the operator objects at checkpoint 1 they
are frozen as written.)

| # | decision | proposal |
|---|---|---|
| O1 | ladder | **`div8_main`** (§ 4; D1): {−3, 0, +3} for every arm, trait vector also ±1, ±5 on held-out |
| O2 | samples per question | **3** (5 on the held-out set under `div8_alt`) |
| O3 | `max_new_tokens` | **1000** (§ 4) |
| O4 | primary rung k* | **3** (§ 8.3) |
| O5 | truncated generations | **kept and judged**, share reported |
| O6 | bridge primary analysis | **judge-free, unfiltered**; filtered beside it |
| O7 | bridge sensitivity | **also question-clustered bootstrap** beside Wilson/Newcombe |
| O8 | null verdicts for the secondary arms | **none**: relaxed and Dictator-only arms are compared with baseline and with the trait vector only (their committed shuffled nulls exist; no orthogonal null was built for them) |
| O9 | pairwise scope | **held-out at ±3 only**, every steered row, both orders, primary judge; secondary judge on the 25% subset (§ 13) |
| O10 | human sample source | **the checkpoint-2 pilot rows** if full generation has not run, otherwise full generation; strata as § 9.1 |
| O11 | "poor agreement" threshold | **§ 9.4** (D12) |
| O12 | held-out and bridge wording | **accepted as drafted; ready to freeze** (D11) |
| O13 | human labelling | **single labeller; intra-rater test–retest substitute** (§ 9.1; D10) |

## 13. Seeds (all fixed now)

| what | seed |
|---|---|
| generation (`Sampling.seed`; batches `seed + batch_index`) | 0 |
| bridge counterbalancing | 20261023 |
| secondary-judge subset: `random.Random(20261024).sample(sorted (question_id, sample_index) pairs, 30)` of the 120 per point (25%) — the same 30 pairs for every (arm, k, set); used for the secondary judge's trait, coherence and pairwise calls | 20261024 |
| human calibration sample (§ 9.1) | 20261025 |
| determinism re-send selection (§ 9.2) | 20261026 |
| judge send order (`judge_many(order_seed=…)`) | 20261027 |
| bootstrap (every interval in § 8; agreement in § 9) | 20261028 |
| test–retest subset of the calibration sheet (§ 9.1; `grader_page.py retest`) | 20261029 |
| judge request `seed` | 0 |

## 14. What this pre-registration does not cover

* The pilot at checkpoint 2 (one game arm + trait vector + nulls at k ∈ {−3, 0, +3},
  small n) is a plumbing and judge-validation run; its numbers are not results.
* The generation entry point for general questions (a lab/ prompt type that renders a
  question as one user turn, through `audit.generate` and `audit.steer`) is
  implementation, written before the pilot and tested offline; it changes no choice
  above.
* Anything decided after data exists goes to `DEVIATIONS.md`, dated, with its reason.
