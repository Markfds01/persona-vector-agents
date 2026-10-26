<!-- Copied 2026-10-23 from the judge-measurement report written on the judge host. It contains
     only the 12 bench items' timings and scores, no study data. Edits: loopback addresses and
     local paths replaced (the endpoint is `PIMIENTOROJO_BASE_URL` only). Section 8 is appended. -->

# LLM-judge measurement on PimientoRojo (llama.cpp b11836, Vulkan)

Measured 2026-10-23 (13:05–13:45 local on PimientoRojo). Everything ran **on PimientoRojo against the server's loopback address**; only the 12 provided benchmark items were sent (no study data).
All numbers below are measured, except the workload-hour table, which multiplies measured seconds/call by the PREREG call counts.

## 1. Verdicts

| | Llama-3.3-70B-Instruct Q8_0 (:8090) | gpt-oss-20b MXFP4 (:8091) |
|---|---|---|
| Logprob endpoint | **Yes.** `top_logprobs=20` returns exactly 20 entries on 48/48 calls. | 20 entries returned on 48/48 calls, **but unusable for scoring** (see below). |
| First-token distribution is digits 0–100 with mass ≥ 0.25 | **Yes, 48/48 trait + coh calls**: mass ≥ 0.999 (min 1.000). Tokens are whole integers (`100`, `90`, `95`, `80`, `0`, …), so one token = one score. | **No, 0/48**: first token is always `<\|channel\|>` (harmony header), digit mass 0.000. |
| Soft (expected-value) score in 1 token | Yes, ~3.8 s/call cold | Not available in a single pass. Side probe: with reasoning at effort `low` and `logprobs` on, the token after `<score>` has a 20-entry top list of whole numbers, but it is conditional on a sampled reasoning chain and is ≥ 0.94 concentrated on one value (`80`: 0.937, `90`: 0.9992, `80`: 0.957, `90`: 0.9997 on 4 items), so it adds little. |
| Determinism (temperature 0, seed 0) | Logprob mode: **deterministic when the prompt cache is bypassed** (24/24 prompts, two repeats, identical to 4 decimals). With default prompt-cache reuse, the repeat differs slightly (max \|Δscore\| 0.30, 10/24 pairs differ). Text mode: 24/24 identical raw text and score. | Text mode (reasoning `low`): score identical on 23/24 pairs; 1 flip (80 → 70, trait item `pos-altruism_31_pos_4`); reasoning length differed in 2/24 pairs. The same request re-sent later gave 70, then 80. Not bitwise reproducible. |
| Output format | `<score>NN</score>` parsed 24/24 | parsed 24/24 (reasoning is returned separately; final content is just `<score>NN</score>`) |

**Recommendation from the measurements:** use Llama-3.3-70B in logprob mode as the judge that can give a continuous, deterministic score. gpt-oss-20b can only be used as a sampled text/reasoning judge, with integer scores on a coarse grid and small run-to-run variation. The only gpt-oss route to a soft score is a custom prefill past the reasoning, which was not tried (the task said not to force it).

## 2. Throughput (seconds per call)

Two measurement regimes, because llama-server re-uses the KV prompt cache by default:

* **Cold / no-cache (planning figure).** `cache_prompt=false`, 24 distinct prompts (12 trait + 12 coherence), one at a time (`conc.py … 1`, file `conc_serial_nocache.jsonl`). This is the full prefill cost and is what a study with all-different prompts pays.
* **Bench as-run** (`bench.py`, default `cache_prompt`): the first call of each pair (rep 0) already benefits from prefix sharing with earlier items (same template and question), and the repeat (rep 1) is an exact cache hit. These per-call numbers are in the `result_*.json` files but **should not be used for planning**; `prompt_per_second` in them is computed on the uncached remainder only and is not a true prefill speed.

| Judge / mode | n | mean s | median s | p90 s | max s | calls/h (3600/mean) at `--parallel 1` | tokens |
|---|---|---|---|---|---|---|---|
| Llama-3.3-70B, logprob (max_tokens 1) | 24 | **3.82** | 3.71 | 4.20 | 5.19 | **942** | ~570–630 prompt, 1 gen |
| Llama-3.3-70B, text (`<score>NN</score>`, 8 gen tokens) | 24 | **6.09** | 5.97 | 6.47 | 7.40 | **591** | ~590–630 prompt, 8 gen (gen ≈ 3.0 tok/s) |
| gpt-oss-20b, text, reasoning `low` | 24 | **1.00** | 0.94 | 1.19 | 1.44 | **3,600** | ~610–650 prompt, 35–75 gen (≈ 76 tok/s; prefill ≈ 1,800–2,000 tok/s) |
| gpt-oss-20b, logprob (max_tokens 1, not usable as score) | 24 | 0.42 | 0.42 | 0.45 | 0.51 | 8,600 | 1 gen |

Cross-check from the additional 48-call no-cache determinism run on the 70B (logprob): median 3.65 s, p90 4.18 s. Bench as-run (prefix-warm) numbers, for information only: 70B logprob rep 0 median 2.91 s / p90 3.49 s; 70B text rep 0 median 2.90 s; gpt-oss text-low rep 0 median 0.83 s / p90 1.15 s. A study whose prompts share long prefixes could land between the two regimes; I did not measure that.

### Concurrency (does `--parallel` / concurrent requests help?)

24 prompts, `cache_prompt=false`, wall-clock / 24 (effective seconds per call).

| Server config | mode | clients | effective s/call | speed-up vs 1 client |
|---|---|---|---|---|
| gpt-oss-20b `--parallel 4 -c 65536` (temporary server on :8092, since killed) | text-low | 1 / 2 / 4 | 0.997 / 0.842 / 0.759 | 1.00 / 1.18 / **1.31** |
| same | logprob | 1 / 4 | 0.430 / 0.354 | 1.00 / 1.22 |
| gpt-oss-20b live `--parallel 1` (:8091) | text-low | 1 / 4 | 0.984 / 0.979 | 1.00 / 1.01 (requests just queue) |
| Llama-70B `--parallel 2 -c 16384` (temporary restart, then restored) | logprob | 1 / 2 | 3.78 / 3.80 | **1.00 / 1.00** |
| same | text | 1 / 2 | 6.08 / 5.00 | 1.00 / **1.22** |

Concurrency scales poorly on this iGPU (shared memory bandwidth): at best ×1.2–1.3. The 70B logprob workload (prefill bound) does not scale at all. Planning should therefore use the `--parallel 1` numbers; a ≤ 20–30 % gain is possible for text mode only. The single-slot servers remain the recommended configuration.

## 3. Workload-hours (serial, `--parallel 1`, cold no-cache mean s/call)

hours = calls × mean_s / 3600. The PREREG gives 278,460 primary-judge calls and 189,756 secondary-judge calls; which model plays which role is not specified in my task, so every judge/mode is shown against both workloads.

| Judge / mode (mean s/call) | 278,460 calls: full | ÷2 | ÷4 | 189,756 calls: full | ÷2 | ÷4 |
|---|---|---|---|---|---|---|
| Llama-3.3-70B logprob (3.82) | **295.5 h** (12.3 d) | 147.7 h | 73.9 h | **201.4 h** (8.4 d) | 100.7 h | 50.3 h |
| Llama-3.3-70B text (6.09) | **471.1 h** (19.6 d) | 235.5 h | 117.8 h | **321.0 h** (13.4 d) | 160.5 h | 80.3 h |
| gpt-oss-20b text-low (1.00) | **77.3 h** (3.2 d) | 38.7 h | 19.3 h | **52.7 h** (2.2 d) | 26.4 h | 13.2 h |

Notes:
* Running both servers at the same time on the one GPU was not measured; the iGPU and memory bandwidth are shared, so wall-clock hours of the two judges should be treated as **additive**, not overlapping.
* Best case with concurrency (measured gains above): gpt-oss text × 0.76 → 58.8 h / 40.1 h at full size; 70B text × 0.82 → 386 h / 263 h. 70B logprob unchanged.
* Full-size 70B-logprob primary + gpt-oss-text secondary, serial: 295.5 + 52.7 ≈ **348 h (14.5 days)**; at ÷4: ≈ 87 h. The alternative role assignment (gpt-oss primary, 70B secondary) is 77.3 + 201.4 ≈ 279 h at full size.
* The prefix-warm figure (~2.9 s/call for 70B logprob) would cut the 70B rows by ~24 % if the study's prompts really share long prefixes.
* If prompts grow beyond the 8,192-token context slot of the 70B server (`-c 8192`), the server must be restarted with a larger `-c`; bench prompts are only ~600 tokens.

## 4. Agreement with upstream judge scores (n = 12 items, indicative only)

Spearman ρ over the 12 items (rep 0 scores vs upstream's own `upstream_altruism` / `upstream_coherence`), Pearson r in brackets. With n = 12, |ρ| ≥ 0.58 is needed for two-sided p < 0.05, and the scores are heavily tied (see "distinct values"), so treat all as rough. The coherence upstream scores are almost all 94–100, so ρ there is dominated by small differences.

| Judge / mode | Trait ρ (r) | distinct trait values | Coherence ρ (r) | distinct coh values |
|---|---|---|---|---|
| Llama-70B logprob (expected value) | **0.743** (0.951) | 11 | **0.914** (0.701) | 8 |
| Llama-70B logprob, cache bypassed (`extra_llama70b_nocache.json`) | 0.748 (0.951) | – | 0.939 (0.696) | – |
| Llama-70B text | 0.576 (0.919) | 4 | 0.722 (0.630) | 3 |
| gpt-oss-20b text, low | 0.536 (0.877) | 5 | 0.412 (0.267) | 3 |
| gpt-oss-20b logprob | n/a (0/24 scored) | – | n/a | – |

Per-item scores are in the `result_*.json` files (`rows[].score`, `rows[].upstream`). The Llama logprob scores are on a continuous scale (e.g. 89.96, 90.01), while the text-mode judges produce coarse values (gpt-oss trait values: 0, 60, 70, 80, 90). Both judges rate trait scores systematically a bit higher than upstream for the lowest-rated items (e.g. upstream 58.8 / 68.4 / 71.1 → Llama logprob ≈ 80–90).

## 5. Provenance

**Machine:** PimientoRojo, Fedora Linux 43 (Server), kernel 7.0.12-101.fc43.x86_64; AMD Radeon 8060S (RADV GFX1151) Vulkan0, unified memory 128,062 MiB total; 32 CPU threads (`nproc` 32; servers use n_threads 16). Shared machine: 9–11 ssh sessions logged in during the run, load average 0.0–0.7 throughout (final snapshots: `0.68, 0.44, 0.27` at 13:38 with both servers idle-ish, `0.10, 0.28, 0.25` at 13:43), no foreign heavy jobs seen. Full snapshot: `machine_state.txt`. (Its `ps` line errored on a trailing comma in my command; the `pgrep` output in this report replaces it.)

**GPU/unified memory in use** (from `/sys/class/drm/card0/device`, with both servers loaded): GTT used 90.58 GB of 137.4 GB total (`mem_info_gtt_used = 90,578,546,688`), VRAM carve-out used 0.31 GB of 0.54 GB; system `free -m`: 91,472 MiB used, 34,922 MiB buff/cache, 36,590 MiB available. The llama-server logs at the default verbosity do not print per-buffer sizes (logs contain only load/slot/listen lines), so a per-server split was not obtained; model sizes are 74.97 GB (70B Q8_0, from `/v1/models` meta `size`) and 12.10 GB (gpt-oss), plus KV caches. The 70B takes ≈ 67 s to load once its files are in the page cache.

**Build:** llama.cpp `b11836-f5027fb21` (Vulkan), binary `~/judges/llama.cpp/llama-b11836/llama-server`.

**Server command lines (as running at the end; PIDs 2441399 and 2424903):**

```
cd ~/judges && setsid nohup ~/judges/llama.cpp/llama-b11836/llama-server -m models/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf --host <loopback> --port 8090 -ngl 999 -c 8192 --parallel 1 --jinja --no-webui > server_llama70b.log 2>&1 < /dev/null &
llama-server -m models/gpt-oss-20b-MXFP4.gguf --host <loopback> --port 8091 -ngl 999 -c 16384 --parallel 1 --jinja --no-webui   (log: server_gptoss20b.log)
```

Both listen on `loopback` only (`ss -ltn`: loopback port 8090, loopback port 8091). The llama-server log warns "no API key is set and CORS allows all origins"; irrelevant while bound to loopback.

**GGUFs and sha256** (from `models/MANIFEST.txt`, verified against Hugging Face before this task):

| File | Repo | sha256 |
|---|---|---|
| Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf (39,927,004,064 B) | bartowski/Llama-3.3-70B-Instruct-GGUF | `0c4ec5083aec2efa104269e8b63e2d9f3322e2fa50511184314093a7b4697878` |
| Llama-3.3-70B-Instruct-Q8_0-00002-of-00002.gguf (35,048,050,944 B) | same | `e5c8858f5feafdc56200bbcbb3c4fadb07cb323c8ed13548dfb798dd1fb24255` |
| gpt-oss-20b-MXFP4.gguf (12,109,566,624 B) | ggml-org/gpt-oss-20b-GGUF | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` |

I did not re-hash the files in this session (that would read 87 GB); I relied on the manifest.

**`/v1/models` meta** (`models_8090.json`, `models_8091.json`): 70B: `n_params` 70,553,706,560, `n_ctx` 8192, `n_ctx_train` 131,072, `n_embd` 8192, `n_vocab` 128,256, `ftype` Q8_0. gpt-oss: `n_params` 20,914,757,184, `n_ctx` 16,384, `n_ctx_train` 131,072, `n_embd` 2880, `n_vocab` 201,088, `ftype` MXFP4 MoE. Model id used in requests is the path string, e.g. `models/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf`.

**`/props`** (`props_8090.json`, `props_8091.json`): `total_slots` 1 on both; `build_info` b11836-f5027fb21; default-generation-settings `n_ctx` 8192 / 16384 (the `n_ctx` key is under `default_generation_settings`); `endpoint_props` false, `endpoint_metrics` false, `endpoint_slots` true. Server-side defaults are overridden per request by `bench.py` (`temperature 0`, `seed 0`).

**Chat template** (exposed by `/props`, taken from the GGUF with `--jinja`):
* Llama-3.3: the stock Llama-3.x template (4,613 chars; BOS `<|begin_of_text|>`, EOS `<|eot_id|>`). It injects a default system header with `Cutting Knowledge Date: December 2023` / `Today Date: 26 Jul 2024`, so the judge prompt sits in the user turn after that header. `supports_reasoning_effort` false.
* gpt-oss-20b: harmony template accepting `reasoning_effort` (default "medium"); `bench.py` passes `chat_template_kwargs {"reasoning_effort": "low"}` and max_tokens 2048. Generation prompt ends `<|start|>assistant`, so the first generated token is the channel header `<|channel|>`, which is why logprob mode sees no digits.

**Request details:** `/v1/chat/completions`, one user message = the item's verbatim `trait_prompt` or `coh_prompt` (plus, in text mode, `"\n\nRespond with only your final score inside tags, exactly like <score>NN</score>."`), `temperature 0`, `seed 0`. Logprob mode: `max_tokens 1`, `logprobs true`, `top_logprobs 20`; score = Σ p·int over tokens that parse as integers 0–100, divided by their total mass, required to be ≥ 0.25 (otherwise null). Each call was sent twice (rep 0 / rep 1); 12 items × 2 prompt kinds × 2 reps = 48 calls per run; zero HTTP errors in all runs.

**Per-call timings:** every call's latency, prompt/completion tokens, `prompt_per_second`, `predicted_per_second`, score, upstream score, finish reason and raw text are in `result_llama70b_logprob.json`, `result_llama70b_text.json`, `result_gptoss20b_text_low.json`, `result_gptoss20b_logprob.json` (plus the `bench_*.out` console logs). No-cache timing runs: `conc_serial_nocache.jsonl`; concurrency runs: `conc_gptoss20b.jsonl`, `conc_llama70b_par2.jsonl`; extra probes: `extra_llama70b_nocache.json`, `extra_gptoss20b_think.json`. Server log tails: `server_*_tail.log`.

## 6. Deviations, changes and caveats

* `bench.py` is **unchanged**. I added three helper scripts (kept alongside, and copied to `~/judges/` on PimientoRojo): `extra.py` (no-cache determinism probe for logprob mode; gpt-oss "think" probe with logprobs), `conc.py` (concurrency/no-cache timing probe) and `analyze.py` (statistics used above; Spearman with average ranks for ties). `extra.py` had a first run in which the "token after `<score>`" search matched the closing tag; fixed (search the first occurrence) and re-run; only the fixed run is reported.
* Temporary servers: (1) a gpt-oss `--parallel 4 -c 65536` server on loopback port 8092, killed by PID afterwards; (2) the 70B was stopped and restarted once with `--parallel 2 -c 16384` for the concurrency test, then restarted with the **original** command (`-c 8192 --parallel 1`), so its PID changed from 2428287 to 2441399. My first attempt at that restart script (`par_test.sh`) targeted a wrong PID (a wrapper shell) and failed to bind the port ("couldn't bind HTTP server socket"); I stopped it by PID before it produced any measurement, the original 70B server was never interrupted, and nothing from that attempt is reported. The corrected script is the one saved as `par_test.sh`. `server_llama70b.log` now reflects the restored server (load 1:08; log from the second launch).
* After the final restart both servers were verified `{"status":"ok"}` on `/health`.
* The ~13:01 "killed earlier start" of the 70B mentioned in the task was not reproduced; the start I did took ~1 min because the GGUFs were in the page cache.
* Statistics are from 12 items and 24 distinct prompts per judge/mode: medians, p90 and Spearman values carry wide uncertainty. p90 over 24 values is effectively the 3rd-largest.
* Upstream's trait scores contain a single zero (item `neg-altruism_0_neg_4`, upstream ≈ 0); every judge also scores it 0, so trait Spearman is partly driven by that one item plus the gradient among the other eleven (upstream 58.8–95.0).
* Latency includes HTTP and JSON handling on localhost only (negligible).
* Not measured: simultaneous load on both servers, prompts longer than ~650 tokens, other reasoning efforts for gpt-oss (`medium`/`high` would raise gen tokens and seconds/call roughly in proportion), the `llama-server` `--cache-reuse`/`--cont-batching` options, flash-attention flags.

## 7. State left behind

* **Both servers are RUNNING, bound to loopback:** Llama-3.3-70B Q8_0 on :8090 (`-c 8192 --parallel 1`, original command line) and gpt-oss-20b MXFP4 on :8091 (`-c 16384 --parallel 1`).
* On PimientoRojo, only files under `~/judges/` were created or changed (scripts, result/extra/conc JSON, logs). Nothing outside `~/judges/` was touched; no packages installed; no sudo.
* Locally everything is in a scratch directory on the generation machine (the raw per-call JSON files are not committed here). No git operations were run.

## 8. Addendum: how the study reads these results (added when copied into the repository)

* The Llama server was run with `cache_prompt` at its default for the as-run bench and with
  `cache_prompt=false` for the timing and determinism probes. The study's client sends
  `cache_prompt: false` on **every** request (`lab/llama_server_client.py`), so the study's
  per-call cost is the "cold / no-cache" column of section 2 and its Llama logprob scores are
  the reproducible ones (24/24; with the cache 10/24 pairs differed, up to 0.30).
* gpt-oss text mode is not bitwise reproducible (23/24 identical, one 80 to 70 flip). The study
  records this per call and treats the secondary judge's scores as sampled, which is why the
  determinism re-send (PREREG 9.2) reports the flip rate for it.
* **The `Today Date: 26 Jul 2024` header** (section 5, "Chat template"). The Llama-3.x chat
  template embedded in the GGUF (Meta's official Jinja template, exposed by `/props` as
  `chat_template`, 4,613 characters) always writes a system block
  `Cutting Knowledge Date: December 2023` / `Today Date: <date_string>`, and starts with
  `{%- if not date_string is defined %}{%- set date_string = "26 Jul 2024" %}`. llama-server
  passes no `date_string` unless the request sets `chat_template_kwargs: {"date_string": ...}`,
  so the default date comes from the template, not from the server or the study. Whether
  `chat_template_kwargs` reaches the template for this build is checked at the start of
  checkpoint 2 by rendering one prompt and reading it back (not done here: no endpoint call
  was made by the item that wrote this addendum).
