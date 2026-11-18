# Full run, primary judge, coherence and pairwise passes: provenance and run quality

Governed by DEVIATIONS 2026-11-17 ("Remaining primary judging starts before the human labels"). Run-quality numbers only; no score or preference, by any grouping, is reported here (PREREG 9.1).

* Judge: Llama-3.3-70B-Instruct Q8_0, llama-server b11836 on PimientoRojo (started before this run, not touched by it; flags and `/props` in `judgments/full/llama-3.3-70b-instruct-q8_0/run_20261117T113917Z.json` and `run_20261118T002626Z.json`), reached by an SSH tunnel 127.0.0.1:18090 -> PimientoRojo 127.0.0.1:8090. Ollama not used. `cache_prompt` false.
* Command: `PRIMARY_URL=http://127.0.0.1:18090 KINDS="coherence pairwise" bash results/general-question-transfer/scripts/run_judging_full.sh` (freeze check passed before each kind; logs `judging_full_primary_{coherence,pairwise}.log`). Order seed 20261027, re-send seed 20261026. 20-call smoke test per rubric first (all ok), outside the repo.
* Coherence: judge 2026-11-17 11:39:16 - 23:49 UTC (3.35 s/call), re-send to 2026-11-18 00:26:25. Pairwise: judge 00:26:25 - ~06:06 (7.72 s/call), re-send to 06:22:41.

## Run quality

* Coherence: 13,080 of 13,080 calls, 0 failed, 0 transport errors; status `ok` 13,080, `low_mass` 0, `refusal` 0, `unparseable` 0. `number_mass`: min 0.99765, 5th percentile 0.9999998, median 1.0000000; share below 0.9: 0; below 0.5: 0.
* Pairwise: 2,640 of 2,640 calls (both orders of the 25% subset), 0 failed, 0 transport errors; status `ok` 2,640 (parse rate 100%).
* Determinism re-send (PREREG 9.2): coherence 647 of 12,932 distinct payloads, status identical 647/647, top-logprobs identical to the bit 323/647 (49.9%), max |score difference| 1.84 points, 531/647 differ by less than 0.01. Pairwise 132 of 2,640 distinct payloads, status identical 132/132, replies identical 132/132.
* Fairness and generosity not started (B2).
