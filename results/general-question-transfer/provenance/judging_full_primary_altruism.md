# Full run, primary judge, altruism trait pass: provenance and run quality

Governed by DEVIATIONS 2026-11-04 ("Primary altruism judging starts before the human labels").
Run-quality numbers only; no score, by any grouping, is reported here (PREREG 9.1).

* Judge: Llama-3.3-70B-Instruct Q8_0 (`models/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`, file sha256 in `pimientorojo_probe.md`), llama-server b11836 (`f5027fb21`), launched on PimientoRojo (pid started 2026-10-23 or later, still the same process, uptime >12 days at launch) as
  `llama-server -m models/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf --host 127.0.0.1 --port 8090 -ngl 999 -c 8192 --parallel 1 --jinja --no-webui`; reached by an SSH tunnel `127.0.0.1:18090 -> PimientoRojo 127.0.0.1:8090`. Ollama not used.
* Mode `logprob_0_100`, `max_tokens` 1, `top_logprobs` 20, seed 0, `cache_prompt` false, date string "23 Oct 2026"; server `/v1/models` and `/props` are in `judgments/full/llama-3.3-70b-instruct-q8_0/run_20261104T152031Z.json`.
* Template: `data_generation/trait_data_eval/altruistic.json`; `freeze.py --check` passed ("freeze record reproduces") immediately before the send (first lines of `judging_full_primary.log`).
* Command: `PRIMARY_URL=http://127.0.0.1:18090 bash results/general-question-transfer/scripts/run_judging_full_trait.sh` (judge phase, then re-send); output `judgments/full/llama-3.3-70b-instruct-q8_0/{trait,determinism_resend}.jsonl`. Rows come from `rows_full/`; the plan extension that adds the in-distribution set for the primary judge is in `scripts/judging/gqt_judging.py`.
* Order seed 20261027, re-send seed 20261026. Judge phase 2026-11-04 15:20:31 – 23:47:46 UTC (30,436 s, 3.79 s/call); re-send 23:47:47 – 00:12:51 UTC.

## Run quality

* Calls: 8,040 planned = 5,880 held-out + 1,800 in-distribution + 360 bridge-AUC; 8,040 done, 0 failed, 0 transport errors.
* Status: `ok` 8,040; `low_mass` 0; `refusal` 0; `unparseable` 0.
* `number_mass`: min 0.99964, 5th percentile 0.9999992, median 1.0000000, mean 0.9999997; share below 0.9: 0; share below 0.5: 0 (the pilot's trait calls: median 1.0, min 0.9999).
* Determinism re-send (PREREG 9.2): 402 of 8,039 distinct payloads (the plan holds one repeated payload); status identical in 402/402; top-logprobs identical to the bit in 215/402 (53.5%); max |score difference| between original and re-send 1.26 points on the 0–100 scale; 314/402 differ by less than 0.01. The server is not bit-reproducible across requests (Vulkan, batch-position effects); this is the figure the analysis reports as the judge's re-send noise.
