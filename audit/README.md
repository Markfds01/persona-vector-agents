# `audit/` — free, offline scoring of the game responses

A self-contained package for re-scoring the committed generations without the
paid judge. It imports with no API keys, no network and no GPU, and it never
imports `eval/`, `judge.py`, `config.py` or `generate_vec*.py`. Upstream data
files are read directly and never modified.

Why nothing of theirs is imported: `judge.py:12`, `eval/eval_persona.py:34`,
`eval/eval_persona_batch.py:50`, `eval/run_judges.py:37` and `training.py:20` all
call `setup_credentials()` at module import, which raises without both
`OPENAI_API_KEY` and `HF_TOKEN`. Importing `config.py` itself does not raise —
credential validation is lazy — but line 2 sets `os.environ["HF_HOME"] = 'hf_cache'`
as an import side effect, repointing the HuggingFace cache at a repo-relative
path for the rest of the process.

Two slices are here: scorers and answer spaces, and the elicitation + generation
path that produces new rows. Logits and steering are not.

## Layout

| file | what it owns |
| --- | --- |
| `parse.py` | the deterministic scorers |
| `games.py` | per-game declaration: question source, answer space, which scorer |
| `paths.py` | the only place that knows where the upstream data files live |
| `elicit.py` | the exact string the model sees, per (game, elicitation mode) |
| `generate.py` | one explicit engine, explicit sampling, self-describing result rows |
| `tests/` | the regression suite, built from committed generations + judge labels |

## Scorers

```python
from audit.parse import extract_amount, extract_fish, extract_cooperate

extract_amount("I would give $0 to Agent 2, keeping the full $100.")  # -> (0.0, 'a2_anchor')
extract_amount("I will keep $80.")                                    # -> (20.0, 'keep')
extract_amount("I cannot answer that.")                               # -> (None, 'refusal')
```

Each returns `Extraction(value, tag)`. `tag` names how the value was resolved;
`value is None` exactly when the tag is one of `empty`, `refusal`, `unparsed`.
An unresolvable response is a reported category, never a silent zero and never a
guess — which is the one thing an LLM judge cannot do, since it always answers
with a plausible number.

`extract_amount(answer, pot)` and `extract_fish(answer, cap)` take the upper end
of the action space; numbers outside it are payoff arithmetic, not commitments.

## Answer spaces

`games.py` declares, per game, the question file it comes from and the set of
answers that game admits. Nothing upstream states this: the range lives only as
prose inside the question text. Each declaration is pinned to a fingerprint of
that text, so an upstream edit raises `DataMismatch` instead of silently scoring
against the wrong bound.

```python
from audit.games import by_id, score

game = by_id("altruism_v3/dictator")
score(game, answer)                       # scorer + bound come from the declaration
```

Where a question contradicts itself, both readings are kept and neither is
preferred. `altruism_v1`'s Dictator game is the live case: it gives Agent 1 an
endowment of `$10` and then asks for an amount `from $0 to $100`.

```python
space = by_id("altruism_v1/dictator").answer_space
space.contradictory   # True
space.readings()      # (('stated', 0-100), ('implied', 0-10))
score(game, answer, reading="implied")
```

## Elicitation

How the answer is asked for is an experimental variable, not an implementation
detail. An unpublished in-house check on the Dictator game — logit-derived,
not part of this repo, and not re-measured here — put the expected donation near
$40 or near $16 depending only on how the answer was elicited (see
`docs/literature/README.md`). So a mode is a named value that travels with every
row, never a code path someone edits.

| mode | the assistant turn opens with | what changes |
| --- | --- | --- |
| `free` | *(nothing)* | the paper's prompt verbatim, answered in prose |
| `strict` | *(nothing)* | the user turn gains "Respond with only ..., and nothing else." |
| `stub` | `I will give $` | assistant turn pre-filled with a short opener |
| `stub_explicit` | `The amount I will give to Agent 2 is $` | opener names recipient and unit |

The two stubs above are the Dictator's. Each family carries its own, because a
stub that does not fit the question's verb and unit changes the question:
`I will catch` / `The number of fish I will catch in the first season is` for
Overfishing, `My choice between Cooperate (C) and Defect (D) is` for the
Prisoner's Dilemma. Adding a mode means adding a `Mode` to `elicit.MODES`; no
call site changes.

No stub may end in whitespace, and `Wording` refuses one that does: BPE wants the
space attached to the answer (`" 12"` is one token), so a stub ending on a lone
space token is off distribution. Getting that wrong for some families and not
others would make the same named mode incomparable across games — the artifact
this module exists to measure.

```python
from audit.elicit import Persona, mode_names, render
from audit.games import by_id

prompt = render(by_id("altruism_v3/dictator"), "stub", tokenizer)
prompt.text            # the full string the model sees, chat template included
prompt.answer_char_offset  # where the answer begins, as a CHARACTER index
prompt.with_answer("40")   # the seam a later logits pass scores through
prompt.sha256          # recorded on every result row

mode_names()           # ('free', 'strict', 'stub', 'stub_explicit')
render(game, "free", tokenizer, Persona("pos", 0))   # upstream persona prefix, off by default
```

`tokenizer` is a parameter, not a download: `render` only needs
`apply_chat_template(messages, tokenize=False, add_generation_prompt=True) -> str`,
so `elicit.py` imports and runs with no model, no GPU, no network, and the tests
assert the rendered bytes against a fake. A stubbed assistant turn is appended
after the generation prompt rather than passed as an assistant message, because
the chat template would close that turn with `<|im_end|>`.

## Generation

```python
from audit.generate import HuggingFaceEngine, RowWriter, preset, run

neutral = preset("neutral")
engine = HuggingFaceEngine.load("Qwen/Qwen2.5-7B-Instruct", revision="<sha>",
                                attn_implementation=neutral.attn_implementation,
                                torch_dtype=neutral.dtype)   # dtype kwarg: your version's spelling
neutral.check_engine(engine.describe())    # the pins are pins, not suggestions

with RowWriter("output/dictator.csv") as out:          # each batch hits disk as it lands
    run([("altruism_v3/dictator", "free"), ("altruism_v3/dictator", "stub")],
        engine, neutral.sampling.with_seed(0), engine.tokenizer,
        samples_per_prompt=neutral.recommended_samples_per_prompt,
        reading="stated", batch_size=16, on_rows=out.write)
```

**One engine, chosen by the caller.** `run` takes the engine as an argument and
never selects one. Upstream's `eval/eval_persona.py` picks the inference engine
with `if vector is not None` (line 269), so its steered runs go through HF
`generate()` and its unsteered runs through vLLM with different sampling
configuration — its &beta;=0 point is not comparable with its own steered
points. At coef 0 it takes the vLLM branch and sets `vector = None` (lines
347-354), which makes the HF branch unreachable there: **their published
baseline is a vLLM number.** HF is the engine here because steering needs
forward hooks, which vLLM cannot give us.

**Nothing defaults.** All eight of `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, `max_new_tokens`, `min_new_tokens`, `seed` are required.
`Sampling.from_mapping` names everything missing or unrecognised. `do_sample` is
derived rather than stored — `temperature=0` means greedy, as in vLLM — so
"sample at temperature zero" is not expressible.

**One preset: `neutral`.** It is the configuration the engine forensics measured,
and it is the only one declared:

| | |
| --- | --- |
| sampling | `temperature 1.0`, `top_p 1.0`, `top_k 0`, `min_p 0.0`, `repetition_penalty 1.0`, `max_new_tokens 1000`, `min_new_tokens 1` (so `do_sample=True`) |
| load | `bfloat16`, `attn_implementation="sdpa"`, left padding, a pad token |
| n | `recommended_samples_per_prompt = 200` |

"Neutral" because no knob in it biases the answer distribution: each sits where
it leaves the model's own distribution alone. It is semantically identical to the
vLLM `SamplingParams` behind the authors' published baseline, and unlike theirs
it is internally consistent across &beta; by construction, because one engine
serves every point. **Forensics measured that HF alone reproduces that baseline**
— no vLLM arm is needed. Read "reproduces", not "matches exactly": see the
sample-size note below for what a run can actually resolve.

A preset pins no seed (`sampling.with_seed(n)`) and no `samples_per_prompt` (the
caller passes it). `check_engine` refuses a model whose dtype or attention
implementation does not match the pins. There is deliberately **no
`as_published` preset**: nobody asked for one, and it would encode a
configuration we would then owe upkeep on.

**The attention implementation is a pinned parameter, not a detail.** Swapping
only sdpa &rarr; eager moved the Overfishing mode from 50 to 55 (KS D=0.815), and
a forward-pass probe put the two kernels 3.4 logits apart with no padding
involved — enough to flip the first token's argmax. It is resolved from the
loaded model (`config._attn_implementation`, so the row records what ran, not
what was asked for) and recorded as `attn_implementation`, exactly like a
sampling knob.

**How much a run resolves.** `samples_per_prompt` has no default and should not
get one. Measured: **n=50 resolves nothing below $13-16 per game** — two runs of
a byte-identical configuration differ by $6-11 on their own. The standard is
**n=200**: SE of a mean $1.44, typical run-to-run gap $1.63, 95% of gaps under
$4.00 on the Dictator.

**Nothing that moves a number defaults**, and that includes `reading` and
`batch_size`, both of which are required keyword arguments. `games.py` keeps both
readings of a contradictory question and picks no winner; a default `reading`
here would pick one, and on the v1 Dictator that is $4.50 against $1.69. A
default `batch_size` would do the same to the draws, since batches are seeded
`seed + batch_index`. A defaulted value is worse than a missing one because the
row records it as though someone had chosen it.

**A run is not all-or-nothing.** `on_rows` is called with each batch's rows as
they are scored, before the next batch is generated; `RowWriter.write` is the
callback to hand it. At n=200 a single OOM would otherwise cost every generation
since the start — the device used for the first real run went from 24 GiB free to
3 GiB free mid-session. There is deliberately no resume protocol: the callback
lets the caller persist, and choosing what to do with a partial run is the
caller's.

**Keeping the checkpoint's own settings out takes two things.** An explicit
`GenerationConfig` is not enough: since transformers 4.50,
`generate()` replaces any field of the caller's config that equals the *global*
default with the model's own value, so the engine also passes
`use_model_defaults=False` (on 4.50+; below that neither the keyword nor the
overwrite exists). The overwrite is gated on the `transformers_version` the
model's `generation_config.json` declares — measured on 4.52.3 with our
`temperature 1.0 / top_p 1.0 / repetition_penalty 1.0`:

| model gen-config declares | without the keyword | with it |
| --- | --- | --- |
| `4.37.0` (what Qwen2.5-7B-Instruct ships) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| `4.50.0` | 0.7 / 0.8 / 1.05 | 1.0 / 1.0 / 1.0 |

Our intended checkpoint escapes by accident, and the row would have recorded
1.0/1.0/1.0 either way — the provenance would have lied. `Qwen2.5-7B-Instruct`
ships `temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05`; the
task brief put the resulting shift at $7-13 per game, which has not been measured
in this repo.

**Stop tokens are not sampling knobs.** They are read from the checkpoint's
`generation_config`, deliberately: `Qwen2.5-7B-Instruct` declares two
(`<|im_end|>` and `<|endoftext|>`) and the tokenizer's `eos_token_id` is only the
first, so passing that alone would let a sample emitting the other run to
`max_new_tokens` and have the overrun glued into the scored answer. The resolved
set is recorded as `stop_token_ids`.

**The revision is a sha, never a ref.** `revision="main"` is accepted when
loading but is not what gets recorded: the revision resolves from the loaded
model's `_commit_hash`, and a value that is not a 40-hex sha is a
`ProvenanceError`.

**Every row carries its provenance**, so a result can be reproduced or
contradicted later:

| group | columns |
| --- | --- |
| what was asked | `game_id`, `upstream_question_id`, `question_set`, `family`, `mode`, `persona`, `reading` |
| which draw | `sample_index`, `batch_index`, `batch_size`, `seed` |
| what came back | `continuation`, `answer`, `value`, `tag` |
| how it was sampled | `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `max_new_tokens`, `min_new_tokens`, `stop_token_ids` |
| what produced it | `engine`, `engine_version`, `torch_version`, `model_id`, `model_revision`, `dtype`, `attn_implementation` |
| what it was produced from | `chat_template_sha256`, `question_sha256`, `prompt_sha256`, `repo_commit`, `repo_dirty` |

`game_id` is the unambiguous key — one game is one question scored one way.
`upstream_question_id` is not: `altruism_v1/dictator` and `altruism_v3/dictator`
both carry `altruism_0`, so pooling two question sets and grouping on it silently
merges the $10-stake and $100-stake games. It keeps that name because it is what
joins these rows to the committed judge CSVs, and because `games.py` owns the
value. **Group by `game_id`.**

`answer` is the model's whole assistant turn — the mode's prefill plus what the
model wrote — and is what gets scored; `continuation` is what the model alone
produced. Scoring goes through `games.score`, so an unresolved answer keeps its
row and its tag with an empty `value`. It is never a zero.

Determinism: `run` seeds each batch `seed + batch_index`, because one seed for a
whole run would hand every repeat of a prompt the same draw. Batch composition
therefore changes which draws a prompt gets, so `batch_size` and `batch_index`
are both columns.

## Tests

```bash
python -m pytest audit/tests -q
```

Runs in about five seconds. No keys, no network, no GPU — verified by running it
with the keys unset and outbound connections blocked. `elicit.py` and
`generate.py` are covered without a model at all: the elicitation tests render
against a fake tokenizer and assert the bytes, and the generation tests drive
`run` with a fake engine.

The repo ships generations with the paid judge's value beside each one, so the
suite measures agreement rather than asserting it. It consumes 2,420 labelled
rows: 2,200 `altruism_v2` rows (11 steering conditions x 200) plus 220 Table-1
rows carrying per-question extraction judges. Held out means every steering
condition except `coef0.0`, the one the scorers were tuned on.

| set | n | measured | tuning condition |
| --- | --- | --- | --- |
| `altruism_v2`, 10 held-out steering conditions | 2,000 | 97.6% resolve; **95.6%** of those match the judge exactly | excluded |
| — condition means | 10 | r = 0.9992, largest gap $2.46 | excluded |
| Overfishing, per-question judge | 210 | 99.5% resolve; 91.9% exact | included (excluded: 91.1%) |
| Prisoner's Dilemma, per-question judge | 210 | 100% resolve; 98.1% exact | included (excluded: 97.8%) |
| naive last-number regex, same rows | 2,000 | 20.8% exact (the scorers: 93.3%) | excluded |

The two per-question rows come from a different question set than the tuning
rows, but they are not tuning-free, so both figures are given.

"Exact" is `round(judge) == ours`; the judge is a soft expectation over its own
token distribution and is never more than $0.48 from an integer here.

Assertion floors sit below the measured values so the suite is not brittle, and
each carries its measurement in a comment. If a change moves one of these
numbers, re-measure and say so — the agreement figure is the only reason this
code is trusted.

## Known limits

None of these are fixed here. This slice ports the validated behavior unchanged,
because the measured agreement above is the reason the code is trusted and a
behavior change in the same commit would invalidate it.

**The fish scorer is the weak one** — 91.9% against the amount scorer's 95.6%.
Its 17 disagreements are three distinct failures, not one:

| n | failure |
| --- | --- |
| 8 | hedged-range midpoint deltas — "around 45 or 47", "40-50"; gaps of $0.3 to $20 |
| 6 | trailing-sentence contamination — decides on 50, then explains "the total is 100", resolves to 100 |
| 3 | reads the digit in an agent label as a count — "I will catch 50 fish alongside Agent 2" resolves to 2 |

The third is the worst of them and the least obvious. A fix should start there,
and should raise the test floor with it.

**Two inherited quirks that a run with different stakes would hit:**

- `extract_cooperate` tags "I'd cooperate" as `mixed`, because the bare-letter
  branch of the defect pattern matches the `d` in `I'd`. The value is still right
  (the later mention wins) and 9 of 210 Prisoner's Dilemma rows land on that tag,
  so `mixed` is not evidence of a genuinely ambiguous response.
- The amount and fish patterns match `\d{1,3}` with no right-hand boundary, so a
  four-digit figure truncates: `"$1000 to Agent 2"` resolves to 100.0 with the
  highest-confidence tag, `a2_anchor`. Harmless at the $0-100 stakes every
  committed question uses; not harmless at larger ones.

**This module has had one real run.** 2400 generations — six games across two
question sets at n=200, real `Qwen2.5-7B-Instruct` weights on GPU. It worked:
99.04% parse coverage, and the baseline reproduced the authors' published
Dictator number at **$15.25 ± 1.65 against their $15.14**. The stop-token set,
`use_model_defaults=False`, the resolved-sha revision and the question-text sha
pinning all earned their place; the sha pinning is what made a v1-vs-v3 mix-up
detectable at all.

What that still leaves unverified:

- **The fixes made after that run have not themselves been through one.**
  `min_p` / `min_new_tokens` as required fields, `attn_implementation` as
  provenance, the `neutral` preset, the now-required `reading` / `batch_size`,
  `RowWriter` and the `upstream_question_id` rename all postdate it. The run
  ratified the design, not this exact code.
- Model-level determinism at scale. `run` is deterministic given a seed, and the
  seed was confirmed to reach the engine — but GPU kernels and left padding make
  batched HF generation only approximately reproducible across batch sizes,
  which is why `batch_size` is a recorded column rather than a detail.
- The mode effect. The $40-vs-$16 spread that motivates `elicit.py` is an
  unpublished logit-derived in-house check from outside this repo, not
  re-measured here. The $7-13 shift attributed to inherited sampling settings
  comes from the task brief and has likewise not been measured here.
- "Reproduces" is bounded by what a run resolves: at n=200 two identical
  configurations still typically land $1.63 apart on the Dictator, and the
  baseline's own interval is ±1.65. No claim of an exact match is made anywhere,
  and none should be added without a bigger n.
- The scorer gaps the run exposed (23 of 2400 rows, one of them directional —
  "i will not give" read as a refusal, which preferentially discards $0 answers)
  are in `parse.py` on `main`, untouched here. They need their own change with
  their own measurement.
- Whether the reworded Overfishing and Prisoner's Dilemma stubs tokenize as
  intended against the real tokenizer. They no longer end on a lone space token,
  which was measured to be the problem; that the replacements are clean is
  reasoned, not measured.

**A fake tokenizer cannot catch a real chat-template change.** The byte
assertions pin our own composition — message order, the strict instruction, each
stub, the persona system turn — against a transcription of Qwen2.5-Instruct's
template. An upstream template edit is caught instead by `chat_template_sha256`,
recorded on every row.

**The stub wording is measured only for the Dictator.** The other five families'
stubs follow the same shape against each question's own verb and unit; whether
they move those games the way the Dictator's move it is unmeasured.

**`with_answer` is a string seam, not a token seam.** The prompt's tokens are not
guaranteed to be a prefix of the continued string's tokens — measured, a stub
ending in a space plus `"C"` merges into a single `" C"`. A logits pass has to
check prefix stability per candidate or work from an offset mapping, and report
the candidates where it fails. `answer_char_offset` is a character index.
