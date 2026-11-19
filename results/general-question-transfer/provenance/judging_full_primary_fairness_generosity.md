# Full run, primary judge, fairness and generosity passes: provenance and run quality

Governed by DEVIATIONS 2026-11-17 ("Remaining primary judging starts before the human labels"); no new deviation. Run-quality numbers only; no score, by any grouping, is reported here (PREREG 9.1).

* Judge: Llama-3.3-70B-Instruct Q8_0, llama-server b11836 on PimientoRojo (already running, not touched), via SSH tunnel 127.0.0.1:18090 -> PimientoRojo 127.0.0.1:8090. Ollama not used. `cache_prompt` false. Frozen templates `fairness.json` and `generosity.json` (eval_prompt sha256 checked against provenance/freeze.md; `freeze.py --check` passed before each kind).
* Command: `PRIMARY_URL=http://127.0.0.1:18090 KINDS="fairness generosity" bash results/general-question-transfer/scripts/run_judging_full.sh`; logs `judging_full_primary_{fairness,generosity}.log`. Order seed 20261027, re-send seed 20261026. 20-call smoke test per rubric first (20/20 ok), outside the repo.
* Fairness: judge 2026-11-18 06:31 - 16:37 UTC (4.52 s/call), then re-send. Generosity: judge 17:08 - 2026-11-19 02:21 UTC (4.13 s/call), re-send to 02:49.
* `scripts/judging/run_quality.py` now takes an optional comma-separated kind list (default `trait`).

## Run quality

* Fairness: 8,040 of 8,040 calls, 0 failed; status `ok` 8,040, `low_mass` 0, `refusal` 0, `unparseable` 0. `number_mass`: min 0.99915, 5th percentile 0.9999995, median 1.0000000; share below 0.9: 0.
* Generosity: 8,040 of 8,040 calls, 0 failed; status `ok` 8,040, `low_mass` 0, `refusal` 0, `unparseable` 0. `number_mass`: min 0.99955, 5th percentile 0.99999, median 1.0000000; share below 0.9: 0.
* Determinism re-send (PREREG 9.2): fairness 402 of 8,039 distinct payloads, status identical 402/402, top-logprobs identical to the bit 193/402 (48.0%), max |score difference| 1.92 points, 278/402 differ by less than 0.01. Generosity 402 of 8,039 distinct payloads, status identical 402/402, identical 201/402 (50.0%), max |score difference| 2.35 points, 265/402 below 0.01.
