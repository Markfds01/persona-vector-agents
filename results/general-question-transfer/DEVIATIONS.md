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
* What changes: `lab/llama_server_client.py` gains a third mode `choice_1_2` (exactly one `<choice>1</choice>` or `<choice>2</choice>` in the final channel, else `unparseable`; no logprobs; tests in `lab/tests/test_llama_server_client.py`); the two 0-100 modes are untouched. Pairwise requests use `max_tokens` **32** on the primary and **2048** on the secondary (the reasoning budget J7 already sets for gpt-oss), temperature 0, seed 0, `cache_prompt: false`, the same `chat_template_kwargs` as that judge's other calls. The pairwise template is sent as the whole user message with no format instruction appended (it already fixes the reply format). The pairwise call's score is 1.0 or 2.0 (the response position chosen); the row stores which position the steered answer held.
* Why: these values are not in the PREREG; they follow J7's reasoning and the template's own reply format.
* Decided by: implementation choice, flagged for the operator at checkpoint 2.
* Data existing at the time: none.
* Effect on results: pairwise only (secondary metric).

## 2026-10-23 — Judge order and the secondary judge's one-sitting rule

* PREREG section: § 11 D7/J13 ("run the whole secondary pass in one sitting")
* What changes: the pilot judging runs the secondary judge's whole pass first, then its determinism re-send, then the primary judge's trait/coherence, pairwise and re-send, so the two servers (one GPU) never compete and the secondary pass sits under one date. Nothing about what is judged changes.
* Why: D7 and the shared GPU.
* Decided by: implementation choice within D7.
* Data existing at the time: generation rows of the pilot.
* Effect on results: none.

## Pending, to be recorded here before it is used

* **The "poor agreement" decision rule (PREREG § 9.4, D12).** Not pre-registered. The operator
  fixes it at checkpoint 2, after seeing the pilot's agreement numbers and before any full
  judging, and it is entered here with its date and reason. Nothing beyond the pilot runs
  before that.
