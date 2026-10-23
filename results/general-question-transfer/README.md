# General-question transfer of the game-derived decision vectors — frozen pre-registration

**There is no result in this directory yet.** Nothing has been generated and nothing
has been judged. Checkpoint 1 was approved by the operator on 2026-10-23, and this
directory holds the **frozen** pre-registration (`PREREG.md`, never edited again: later
changes go to `DEVIATIONS.md`), the frozen question sets, the arm vectors pinned by sha256,
the llama-server client, the tokenizer check, the cost estimate and the human grader page.
`provenance/freeze.md` records the sha256 of everything the pre-registration pins.

The question the study answers: does a direction learned from what
Qwen2.5-7B-Instruct **does in a game** (`results/crossgame-per-game-steering`,
`results/dictator-decision-vector`) move how it answers **everyday, non-game**
questions about self-interest versus others' welfare, at both ends of the ladder,
beyond a same-size shuffled-label null and an orthogonal null? `PREREG.md` § 1–2.

## What to review

| item | file |
|---|---|
| the frozen pre-registration; **§ 0 lists every decision as the operator decided it** (D1–D14) | `PREREG.md` (§ 0 decisions, § 4 and § 10 the ÷8 design, § 11 judges, § 12 the rest) |
| the freeze record: sha256 of the pre-registration, the question sets, the templates (incl. the rendered Llama system header) and the arm-vectors table, and the commit it was frozen on | `provenance/freeze.md` |
| 40 held-out dilemmas (drafted by an LLM agent, Claude, 2026-10-23; accepted unedited by the operator, 2026-10-23) | `questions/heldout.json` |
| 40 forced-choice bridge items, counterbalanced by seed | `questions/bridge.json` (built by `scripts/questions/build_bridge.py`) |
| near-duplicate report (no pair at either threshold) | `questions/near_duplicates.md` / `.json` |
| question-set checksums | `questions/SHA256SUMS` |
| every arm's vector, path, sha256, norm, angle | `provenance/arm_vectors.json`; the six orthogonal nulls built here are in `vectors/` |
| tokenizer single-token check for the judges | `provenance/tokenizer_check.json` |
| what is known about the judge host | `provenance/pimientorojo_probe.md` |
| the measured judge results (speed, logprob usability, determinism) | `provenance/judge_measurement_2026-10-23.md` |
| cost estimate, the ÷8 design and its alternative | `analysis/cost.md` / `cost.json` |
| the human grader page, demonstrated on SYNTHETIC items | `grader_demo/grader_SYNTHETIC.html` (open it in a browser; load `judge_reveal_SYNTHETIC.json` for audit mode) |
| deviations log (empty; the "poor agreement" rule is fixed here at checkpoint 2) | `DEVIATIONS.md` |

## Layout

```
PREREG.md, DEVIATIONS.md, README.md
questions/        frozen question sets, near-duplicate report, SHA256SUMS
vectors/          the six per-game orthogonal nulls (built 2026-10-23, reproducible)
provenance/       freeze.md, arm_vectors.json, tokenizer_check.json, pimientorojo_probe.md
analysis/         cost.md, cost.json
grader_demo/      the grader page on synthetic placeholder items, and its inputs
scripts/
  grader_page.py              blind grader page, two independent reviewers (page | retest | demo)
  judging/tokenizer_check.py  GGUF-vocab + HF tokenizer single-token check
  measurement/bridge_parse.py the bridge set's A/B reader (no judge)
  measurement/score_stats.py  point summaries, clustered bootstrap, verdicts, Holm
  measurement/agreement.py    Spearman, quadratic-weighted kappa, Krippendorff's alpha
  provenance/arm_vectors.py   arm hashes + orthogonal-null build (--check rebuilds)
  provenance/cost_estimate.py the cost estimate
  provenance/freeze.py        writes / checks provenance/freeze.md
  questions/build_bridge.py   bridge counterbalancing (--check rebuilds)
  questions/near_duplicates.py the near-duplicate report
  tests/                      offline tests, in `python -m pytest -q`
```

The judge client is `lab/llama_server_client.py` (tests in
`lab/tests/test_llama_server_client.py`): an HTTP client for the local llama-server
(llama.cpp) endpoint that takes its endpoint only from `PIMIENTOROJO_BASE_URL`, checks
`/v1/models` before sending anything, sends `cache_prompt: false` on every request, caches
append-only on the request-payload sha256, retries only transport errors, and gives every
call a status. The study has not called it; the judge measurement
(`provenance/judge_measurement_2026-10-23.md`) was a separate bench on 12 items.

## How to re-run what is here

From the repository root. Scripts that need `torch`/`transformers` were run with the
repository's measured interpreter (torch 2.6.0+cu124, transformers 4.52.3).

```
python results/general-question-transfer/scripts/provenance/arm_vectors.py --check
python results/general-question-transfer/scripts/questions/build_bridge.py --check
python results/general-question-transfer/scripts/provenance/freeze.py --check
HF_HUB_OFFLINE=1 python results/general-question-transfer/scripts/questions/near_duplicates.py
HF_HUB_OFFLINE=1 python results/general-question-transfer/scripts/provenance/cost_estimate.py
python results/general-question-transfer/scripts/judging/tokenizer_check.py \
    --llama-tokenize <llama.cpp b11836>/llama-tokenize --work <scratch> \
    --out results/general-question-transfer/provenance/tokenizer_check.json
python results/general-question-transfer/scripts/grader_page.py demo \
    --out results/general-question-transfer/grader_demo
python -m pytest -q
```

## What this does not establish

* **Anything about transfer.** No steered text exists for this study.
* **Anything about the judges' quality.** The judges have not scored study answers; whether
  Llama-3.3-70B-Instruct scores these answers the way a person would is exactly what
  checkpoint 2's human calibration measures.
* **The judges' behaviour on study answers.** Speed, logprob usability and determinism
  were measured on 12 bench items only; the length of a pairwise call is still an
  assumption (`analysis/cost.md`).
