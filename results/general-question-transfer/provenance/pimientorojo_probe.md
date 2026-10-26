# PimientoRojo — what is known about the judge host (2026-10-23)

No hostname, address or credential is recorded in this repository. The endpoint is
configured only through the environment (`PIMIENTOROJO_BASE_URL`, optional
`PIMIENTOROJO_API_KEY`), and `lab/llama_server_client.py` has no default and no fallback.

**This checkpoint-1 item made no call to the judge endpoint.** Everything below was
probed read-only by the orchestrator, or decided by the operator, on 2026-10-23, and
is recorded here as reported.

## Hardware and access (orchestrator probe)

* Reachable over SSH from the generation machine.
* AMD Strix Halo: Radeon 8060S integrated GPU (gfx1151), ~125 GB unified memory
  (~128 GB as reported by llama.cpp's device list), 32 cores. No NVIDIA GPU, no
  ROCm, no vLLM installed; the RADV Vulkan driver is present.
* Shared machine: 7 users were logged in at the probe. The operator: contention "does
  not matter for now, we are doing trials".

## What was found running, and why it is not used

* Ollama 0.15.6, bound to localhost only (not exposed on the network). Its
  `/v1/models` returned exactly one model, `llama4-scout:latest`
  (Ollama tag; Q6_K, 88.4 GB, family llama4). Llama-3.3-70B-Instruct and
  gpt-oss-120b were not installed. Whether that endpoint returns `top_logprobs` was
  not verified.
* **Excluded by operator decision (2026-10-23): do not use Ollama — "it adds layers to
  the experiment".** No Ollama-specific code path, tag or default exists in this
  repository.

## The judge server that will be used (operator decisions 1 and 2)

* llama.cpp **`llama-server`**, release **b11836** (version 0.7.0-dev, commit
  `f5027fb21`, full sha `f5027fb21307eb264aed483242603d68c87b0fe3`), the
  `ubuntu-vulkan-x64` build, installed in the user's home without sudo;
  `--list-devices` → `Vulkan0: AMD Radeon 8060S Graphics (RADV GFX1151)`, ~128 GB.
  Bound to localhost on PimientoRojo, reached through an SSH tunnel from the
  generation machine; nothing exposed.
* `/v1/chat/completions` and `/v1/models`; `/props` describes the
  loaded model and slots and is recorded by the client when present.
* Judges (operator decision 2: two families, neither Qwen):

| judge | repository @ revision | file(s) | sha256 (verified on the server against HF LFS) |
|---|---|---|---|
| Llama-3.3-70B-Instruct, Q8_0 | `bartowski/Llama-3.3-70B-Instruct-GGUF` @ `b6c5c9f176f3279204034e1d16d393105e95cb88` | `Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf` | `0c4ec5083aec2efa104269e8b63e2d9f3322e2fa50511184314093a7b4697878` |
| | | `Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00002-of-00002.gguf` | `e5c8858f5feafdc56200bbcbb3c4fadb07cb323c8ed13548dfb798dd1fb24255` |
| gpt-oss-20b, MXFP4 | `ggml-org/gpt-oss-20b-GGUF` @ `15dcfc7e7c608c6161507211ed43c5b590801b4b` | `gpt-oss-20b-MXFP4.gguf` | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` |

  The revisions and the sha256s were also checked from the generation machine against
  Hugging Face's LFS metadata at those revisions (`provenance/tokenizer_check.json`,
  `hf_lfs_sha256` / `sha256_matches_expected`). The GGUF files may still have been
  downloading on the server when this was written.
* Quantisation: GGUF judges are quantised. FP8 (the task's suggestion) is not a GGUF
  type; Q8_0 is the closest and is the orchestrator's default for the 70B. The quant
  level is an operator decision (PREREG § 11, J2).

## What each judge run must record (PREREG § 11, J9–J11)

* the `/v1/models` response, verbatim, and abort if the configured id is not listed;
* `/props` as returned (`model_path`, `build_info`, slots), or "unavailable";
* GGUF file name(s) + sha256 + quant type + llama.cpp build and commit;
* the `llama-server` launch settings that can change outputs at temperature 0:
  `--parallel`, `--batch-size`, `--ubatch-size`, `--ctx-size`, flash attention, and the
  `--alias` the served id comes from;
* judge weights revision: for a GGUF that is the file sha256 above; a field the server
  does not expose is recorded as unavailable, never guessed.

## Not yet available

* The llama-server `/v1/models` output (the server was being installed when this was
  written), and whether it returns `top_logprobs` ≥ 20 on `/v1/chat/completions`.
  Both are the first thing run at checkpoint 2, after the operator confirms the
  connection.
