# Task: steer general (non-game) questions with the game-derived decision vectors, and grade them with LLM judges served by PimientoRojo

## Context

Repo: `persona-vector-agents` (fork of johnathansun/persona-vector-agents), branch from `main`.
Read first: `README.md`, `results/dictator-decision-vector/{README,METHOD}.md`,
`results/crossgame-per-game-steering/{README,METHOD}.md`, `audit/README.md`, `.dm-knowledge/`.

Upstream (Chen et al.) builds trait vectors from **general** questions and steers **games**.
We did the reverse half: we built **decision vectors from game outcomes** (labels = what the model
actually did, scored by a parser, activations taken on the game prompts). This task completes it:
**steer general, open-ended questions with the game-derived vectors** and measure whether the
trait moves, against matched nulls and against the upstream trait vector.

The scientific question: *does a direction learned from what the model does in games transfer
to how it behaves on everyday questions, in both directions (positive and negative steering),
beyond what a same-size null edit does?*

This is a science deliverable. The standard is: **robust, faithful, nothing decided silently.**
Every choice that can move a number is either fixed in this prompt, written into a
pre-registration before data exists, or put to me at a checkpoint. If you are about to make a
choice that is in none of those three places, stop and ask.

## Non-negotiable principles

1. **Pre-register before generating.** Write `PREREG.md` (hypotheses, arms, ladder, n, primary
   and secondary metrics, exclusion rules, statistical tests, what counts as "beats its null")
   and commit it before any generation or judging. Deviations later go in a `DEVIATIONS.md`
   with date and reason. Never edit PREREG.md after data exists.
2. **Generation and judging are separate stages.** Generation writes immutable rows. Judging
   reads rows and writes separate judgment rows. Judging never touches generation output and
   can be re-run or swapped to another judge without regenerating.
3. **Judges are plain API calls to PimientoRojo.** OpenAI-compatible HTTP endpoint(s)
   (vLLM-style `/v1/chat/completions`, `/v1/models`). No local judge model, no in-process
   judge, no hidden fallback to OpenAI or any other provider. If PimientoRojo is unreachable,
   the run stops with an error and writes nothing.
4. **No silent missing data.** Every generation and every judge call produces a row, with a
   `status` (`ok`, `unparseable`, `refusal`, `low_mass`, `transport_error`, `truncated`, …).
   Nothing is dropped, imputed, or re-sampled to obtain a "better" answer. Retries are allowed
   **only** for transport errors (connection, 5xx, timeout), are capped, and are logged.
5. **Blind judging.** The judge sees only the question and the answer. Never the arm, the
   vector, k, the game, or anything else in the row. Send judge calls in shuffled order.
6. **Provenance on every row**, following repo conventions (see `.dm-knowledge`): model 40-hex
   revision sha, dtype, attn implementation, explicit generation config, seed, batch index,
   steering vector sha256, layer, positions, norm, unit beta; and for judgments: endpoint,
   served model id as reported by `/v1/models`, judge weights revision (ask PimientoRojo's
   config to expose it; if unavailable, record that it is unavailable, do not guess),
   prompt template sha256, full request payload sha256, decoding params, raw response
   (top logprobs or text), timestamp.
7. **Fail closed.** Assertions over warnings. A check that cannot run is reported as not run,
   never as passed.

## PimientoRojo connection (fill in / confirm with me before use)

- Base URL(s): `<PIMIENTOROJO_BASE_URL>` (env var `PIMIENTOROJO_BASE_URL`; key in
  `PIMIENTOROJO_API_KEY` if needed). Never hard-code, never commit secrets or hostnames
  that are not public.
- Judges served: **primary** `<JUDGE_A>` and **secondary** `<JUDGE_B>`, which must come from
  **different model families, and neither from the Qwen family** (the steered model is
  Qwen2.5-7B-Instruct). Suggested: Llama-3.3-70B-Instruct (FP8) and gpt-oss-120b. Which is
  primary goes in PREREG.md, chosen before any judgment exists.
- At start of every run, call `/v1/models` and record the response. Abort if the served id
  does not match the configured one.

Write one small client module (e.g. `lab/judge_client.py`): OpenAI-compatible, async with a
bounded concurrency, `temperature=0`, fixed `seed`, explicit `max_tokens`. Cache keyed on the
request payload hash, append-only. Do **not** import or modify upstream `judge.py` /
`eval/eval_persona.py` (`.dm-knowledge` documents why: engine switching between coef 0 and
steered points, `activation_steer.py` rewriting `HF_HOME` at import).

## Scoring modes: verify the tokenizer, do not assume it

Upstream reads a 0–100 score from **one token's top logprobs** (expected value over integer
tokens, `None` if their mass < 0.25). That is valid only if every integer 0–100 is a single
token for that judge's tokenizer.

- Before using `logprob_0_100` with a judge, **test it**: load the judge's tokenizer and
  assert every string "0"…"100" encodes to exactly one token, and that the endpoint returns
  `top_logprobs` ≥ 20. Commit the check's output.
- If the check fails, that judge uses a **declared** alternative, fixed in PREREG.md:
  `text_0_100` (generate, parse exactly one integer in a tagged field like `<score>NN</score>`,
  anything else → `unparseable`). For reasoning models (gpt-oss), use a fixed, recorded
  reasoning-effort setting and parse only the final answer channel.
- Never switch modes mid-run. Never fall back automatically.

## Data: held-out general questions

**Known issue:** `data_generation/trait_data_eval/altruistic.json` and
`data_generation/trait_data_extract/altruistic.json` contain the **same 40 questions**. The
upstream trait vector was extracted on them. Using them as the test set gives the trait vector
an in-distribution advantage over our game vectors.

Therefore:
- **Primary test set: new held-out open-ended questions** (target 40), same style and domain
  as `altruistic.json` (everyday dilemmas where self-interest and others' welfare conflict),
  **no** game framing, no dollar amounts to split, no mention of the six games. Draft them,
  check for near-duplicates against both existing files and against every game prompt in
  `audit/` (record the similarity method and threshold), then **stop and let me review and
  freeze them** (checkpoint 1). Record who wrote them.
- **Secondary set: the existing 40**, reported separately and labelled "in-distribution for
  the trait vector".
- **Bridge set (judge-free ground truth):** ~40 forced-choice everyday dilemmas, each with one
  clearly self-interested and one clearly other-regarding option, answer format
  `Answer: A` / `Answer: B` + free-text justification. Counterbalance: the other-regarding
  option is A in exactly half the items, assigned by a recorded seed. Parse deterministically
  in `audit/`-style code; unparseable is a status, not a guess. Also frozen at checkpoint 1.

## Arms and ladder

Steer **Qwen/Qwen2.5-7B-Instruct**, pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`,
bf16, `sdpa`, layer 20, `positions="all"`, `norm="unit"`, reference norm 10.5083 (the upstream
altruism trait vector's layer-20 norm), `unit_beta = k × 10.5083`, exactly as in
`results/crossgame-per-game-steering`. Use `audit/steer.py` (do not write a new hook), the
neutral preset with an explicit GenerationConfig and `use_model_defaults=False`, stop tokens
from `model.generation_config`. Fixed `max_new_tokens` (state it in the prereg); rows that hit
it get `truncated` status.

Arms, each with its committed vector (load with `weights_only=True`, record sha256):
- 6 per-game decision vectors, strict policy (`results/crossgame-per-game-steering/vectors/`)
- relaxed vectors for Ultimatum and Overfishing (37° and 49° from strict) as a secondary arm
- the Dictator-only decision vector (`results/dictator-decision-vector/vectors/`)
- upstream trait vector `persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt`
- per-game **shuffled-label null** (the committed ones)
- per-game **orthogonal null** (shuffled null with the decision direction projected out, the
  construction in `results/dictator-decision-vector/METHOD.md`). This is the arm that speaks
  to mechanism, and it matters most for a transfer claim.
- one shared k=0 baseline (byte-identity check against an unhooked run, plus the positive
  controls from `.dm-knowledge`: hook-fired count, and k≠0 changing ids).

Ladder and n: propose them in PREREG.md with a cost estimate (generations, tokens, GPU hours,
judge calls) **before running anything** (checkpoint 1). Default proposal: k ∈ {−5,−3,−1,0,+1,+3,+5},
40 questions × 5 samples = 200 per point. Note that samples within a question are correlated:
the effective n is closer to the number of questions, and the analysis must cluster on question.

## Judgments per answer

All prompt templates are **verbatim from the repo** where one exists, with sha256 recorded:
1. **Trait score**: `eval_prompt` from `trait_data_eval/altruistic.json`, 0–100.
2. **Coherence**: `Prompts["coherence_0_100"]` from `eval/prompts.py`, 0–100.
3. **Pairwise** (secondary metric): for each steered answer, pair it with a baseline (k=0) answer
   to the **same question and sample index**, ask which response shows more concern for
   others' welfare. Present **both orders**, and report position bias. Write the pairwise
   template once, put it in PREREG.md, never edit it.
4. Bridge set: parsed choice (no judge needed); judges score coherence of the justification.

Both judges score everything (trait + coherence). Pairwise: primary judge, plus the
secondary judge on a pre-registered random subset.

Exclusion: primary analysis keeps answers with coherence ≥ 50 (upstream convention) **by the
primary judge**; report the unfiltered analysis next to it, and the share excluded per point.
Exclusion never depends on the trait score.

## Judge validation (checkpoint 2, before full judging)

1. **Human calibration set**: a stratified random sample (~200 answers across arms × k,
   seed recorded) exported as a blind sheet (question, answer, nothing else, shuffled). I and
   a colleague label trait 0–100 and coherence 0–100 independently. Report human–human,
   human–judge and judge–judge agreement (Spearman, plus Krippendorff's α or quadratic-weighted κ
   on binned scores), with bootstrap CIs.
2. **Determinism check**: re-send 5% of judge requests, report exact-match and score deltas
   (vLLM batching can be nondeterministic even at temperature 0, so measure it).
3. **Bridge check**: on the bridge set, does the judge's trait score track the parsed A/B
   choice (AUC)?
4. Stop and report all of this to me. **Do not run full judging until I approve.** If agreement
   is poor, we change the plan explicitly and log it in DEVIATIONS.md, not quietly.

## Analysis

Mirror `results/crossgame-per-game-steering`:
- Per arm and k: mean trait score with **question-clustered bootstrap** CIs (resample
  questions, then samples within), coherence, share excluded, share truncated, non-Latin
  share, trigram-repeat degeneracy.
- Bridge: P(other-regarding choice) with Wilson intervals; differences with Newcombe intervals.
- The verdict is "**decision − null at matched unit beta**", read from whether the interval
  excludes zero (not from p, as documented in `.dm-knowledge`), separately for the negative
  and positive ends, against **both** the shuffled and the orthogonal null. Mark ends where
  either arm is degenerate (coverage or coherence collapse) as `undetermined`, using the same
  rule as the per-game study.
- Compare game vectors with the upstream trait vector at matched unit beta, on the held-out set
  and on the in-distribution set separately.
- Report results from both judges side by side. A claim that holds under one judge and not
  the other gets reported as exactly that.
- State the number of comparisons. Flag anything that would not survive a Holm correction
  across the pre-registered primary family.

## Deliverables

`results/general-question-transfer/` in the house style: `README.md` (the result, including
negatives and nulls stated plainly; a section "what this does not establish"), `METHOD.md`,
`PREREG.md`, `DEVIATIONS.md`, `questions/` (frozen sets + sha256), `rows/` (generations),
`judgments/<judge>/` (raw judge rows), `analysis/`, `figures/`, `provenance/`, `scripts/`
with one entry-point shell script per stage that owns all its output paths and supports resume
checked against vector sha (see `.dm-knowledge` on `--resume` pitfalls).

Offline tests (no GPU, no network) for: the judge client against a fake OpenAI-compatible
server (cache, retry-only-on-transport, status codes, no fallback), the score aggregators,
the bridge parser, the tokenizer check, the clustered bootstrap. Add them to `pytest.ini`.
Run the whole suite. Report the real pass/skip counts, per interpreter.

## Checkpoints (stop and wait for me at each)

1. PREREG.md + drafted question/bridge sets + cost estimate + PimientoRojo `/v1/models`
   output + tokenizer check results.
2. Pilot (one game arm + trait vector + nulls at k ∈ {−3, 0, +3}, small n) + judge validation
   (human sheet exported, then the agreement report once labels are in).
3. Full results draft before opening the PR.

Work on a branch `feat/x/general-question-transfer`, open a PR to `main` of
`Markfds01/persona-vector-agents` at the end. Never push to `main`. Never commit hostnames,
keys, or absolute machine paths.
