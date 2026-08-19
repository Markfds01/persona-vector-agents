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

Three slices are here: scorers and answer spaces, the elicitation + generation
path that produces new rows, and activation steering on top of that path. A
logits pass is not.

## Layout

| file | what it owns |
| --- | --- |
| `parse.py` | the deterministic scorers |
| `games.py` | per-game declaration: question source, answer space, which scorer |
| `paths.py` | the only place that knows where the upstream data files live |
| `elicit.py` | the exact string the model sees, per (game, elicitation mode) |
| `generate.py` | one explicit engine, explicit sampling, self-describing result rows |
| `steer.py` | the authors' activation steering, applied to that engine |
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

## Steering

`steer.py` applies the authors' activation steering to the engine above. It is a
*reproduction*, not a reimplementation: if the vector lands anywhere other than
where `activation_steer.ActivationSteerer` puts it, every steered number is
incomparable with theirs and nothing downstream notices.

```python
from audit import generate, steer

steering = steer.Steering("persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt",
                          layer=20, coeff=2.0, positions="all")
engine = generate.HuggingFaceEngine(model, tokenizer, "Qwen/Qwen2.5-7B-Instruct")

steer.sweep([("altruism_v3/dictator", "free")], engine, model,
            [steer.Steering(path, 20, c) for c in (0.0, 5.0, -5.0, 2.0, -2.0)],
            generate.preset("neutral").sampling, tokenizer, samples_per_prompt=200,
            reading="stated", batch_size=20, out_dir="data/audit-steer/rows")
```

`layer` is upstream's 1-based `--layer`, kept in their spelling: it indexes the
vector file directly (`vector[20]`) and the layer list at `layer - 1`
(`model.model.layers[19]`), and those are the same tensor — `hidden_states[20]`.
`Steering` validates at construction; `sweep` writes one flushed CSV per
coefficient and never defaults `reading`, `batch_size` or `samples_per_prompt`.

### What is matched exactly

- **The dtype order.** The vector is cast to the model's parameter dtype *before*
  it is scaled: `coeff * bfloat16(v)`, not `bfloat16(coeff * v)`. The two differ
  in the last bf16 mantissa bit. A test pins the order against
  `ActivationSteerer`'s own arithmetic.
- **`positions="all"`** — the mode `scripts/eval_steering.sh` uses — adds at every
  position of every forward pass, prompt tokens and each decode step alike. The
  hook is installed around the whole `generate()` call, where they install it.
- **Tuple layer outputs.** A decoder block returning `(hidden, attn_weights)`
  keeps its second entry; only the first is steered.
- **`positions="prompt"`'s flaw.** They detect a decode step as `t.shape[1] == 1`,
  which misfires on a one-token prompt. Reproduced as-is and commented, because
  the point is parity. Unused by their steering script.

### Where it deliberately differs

- **Hook site resolution.** `ActivationSteerer` searches five architecture
  spellings and takes the first match; it reaches `model.layers` on Qwen2 only
  because there is no `transformer.h`. `hooked_module` resolves the explicit path
  and *raises* unless it is the same object their search returns.
  `upstream_layer_list` reproduces their search so a test can assert identity
  rather than assume it.
- **The vector is copied.** `torch.as_tensor` returns the *same* object when dtype
  and device already match, so theirs can alias a caller's tensor and carry its
  `requires_grad` into the hook. `detach().clone()` removes both. Cannot move a
  number.
- **`activation_steer` is never imported by a run.** Importing it sets
  `HF_HOME=hf_cache`, repointing the model cache mid-process. Only the tests
  import it, in a process that loads no weights; a test asserts `steer.py`'s
  source never does.

### The four checks, run on real weights before any sweep number was used

`Qwen2.5-7B-Instruct`, bf16, sdpa, one GPU.

| check | result |
| --- | --- |
| **beta = 0 is a byte-exact no-op** | token ids identical to no hook at all, at a fixed seed and batch. Positive controls: the unhooked run is itself reproducible, the hook fired 366 times at coeff 0, and coeff 2 changes the ids — without those the check is void, because an uninstalled hook also passes it. Also identical through `SteeredEngine` at the string level. |
| **hook site** | the hooked module *is* `model.model.layers[19]`, and its captured output is byte-equal to `hidden_states[20]` and to neither `[19]` nor `[21]`. 29 hidden states for 28 layers; the vector file is 29 x 3584. |
| **projection cross-check** | `cal_projection.py`'s `(a·b).sum()/b.norm()` recomputed from our activations, teacher-forced on their exact committed prompt+answer text, against their committed `..._proj_layer20` column: **r = 0.99997** over 300 rows, OLS slope 0.9991, mean offset −0.0025 (SD 0.012, max 0.043). The same check at layer 19 gives r = 0.940 and at layer 21 r = 0.991, so it discriminates. Layer, vector and orientation all confirmed; the residual is bf16 kernel noise. |
| **direction** | the positive arm is strictly increasing at every step, $12.96 at beta 0 to $76.68 at +5 (+12.90 per unit beta, r = 0.999). See the sweep below for the negative arm, which is not monotonic — in ours or in theirs. |

The hook site and the dtype order are pinned permanently and offline: the tests
import `ActivationSteerer` itself on a few-hundred-parameter CPU model and assert
object identity of the hook target and bitwise equality of the delta.

### The norm convention

`Steering(..., norm=...)` chooses how the vector's length is treated. **`"raw"` is
the default and is upstream's behaviour** — they have no such switch — so nothing
about the reproduction changes unless you ask for it.

| mode | delta | when |
| --- | --- | --- |
| `raw` (default) | `coeff * v` | reproducing them; comparing coefficients within one vector |
| `unit` | `coeff * v / ‖v‖` | comparing directions built by *different* methods |

Neither is the correct one; they answer different questions.

`raw` keeps the measured norm, and that norm is a measurement rather than an
accident: it is how far the positive and negative prompts moved the layer-20
activations apart. It gives beta a real anchor — **beta = 1 adds exactly the
separation the prompt manipulation produced** — and it is on-distribution by
construction.

`unit` divides that out, which is what you want when the norms of the vectors
being compared measure *different quantities* and so are not a common scale: a
trait vector's norm measures prompt-induced separation, a decision vector's
measures decision-conditioned separation, and a shared beta axis silently compares
them at whatever ratio their norms happen to have.

As a fact about the shipped data rather than a verdict on their method: the six
`*_response_avg_diff.pt` vectors at layer 20 range from **9.0739** (retaliation) to
**14.3173** (forgiveness), so one beta axis spans a 57.8% difference in applied
magnitude across traits. Altruism, the one steered here, is **10.5083**. Measured
in this repo, not quoted.

Both the mode and the measured norm go on every row (`steer_norm_mode`,
`steer_vector_norm`), so a finished sweep converts between conventions without
regenerating anything: **`beta_unit = beta_raw * ‖v‖`**, in either direction.
`steer_vector_norm` is always the shipped length, never 1.0, because that is the
number the conversion needs.

The two are equal in exact arithmetic but **not bitwise** at the equivalent
coefficient: `unit` divides in float32 and then casts, `raw` casts and then scales,
so they round a different number of times. Measured on the altruism vector at
beta_raw 1, 2 and 5, in bf16: `‖unit − raw‖ / ‖raw‖` peaks at 3.3e-3 and the angle
between them at 1 − cos = 4.6e-6. In float32 the same comparison gives 4.6e-8 and
exactly zero, which is what shows the bf16 gap is the cast and not the algebra.

### The one game this has been run on

`altruism_v3/dictator`, mode `free`, `neutral` preset, seed 0, batch 20, reading
`stated`, n=200 per coefficient — 2,200 generations in 59.4 minutes (590,078
generated tokens, 165.5 tok/s). Coefficients were run **sign-paired, widest pair
first**: 0, then ±5, ±2, ±3, ±1, ±4. The pairing is what would have left both arms
of the curve intact had the run been evicted part-way; it is not an outward
progression, and nothing depends on it being one. No eviction occurred, and every
coefficient completed 200/200 on its first attempt.

| beta | parsed/200 | ours | theirs `v1/` n=50 | theirs `v2/` n=50 |
| ---: | ---: | --- | --- | --- |
| −5 | 196 | 30.11 ± 3.69 (SD 26.20, SE 1.87) | 25.41 [18.63, 32.19] | 36.48 [28.36, 44.60] |
| −4 | 198 | 26.06 ± 3.62 (SD 25.82, SE 1.84) | 25.18 [18.26, 32.11] | 32.81 [25.14, 40.48] |
| −3 | 197 | 23.87 ± 3.68 (SD 26.19, SE 1.87) | 24.06 [16.75, 31.37] | 19.44 [12.31, 26.57] |
| −2 | 199 | 13.52 ± 3.04 (SD 21.74, SE 1.54) | 14.57 [8.13, 21.01] | 16.52 [10.36, 22.68] |
| −1 | 196 | 11.36 ± 2.73 (SD 19.36, SE 1.38) | 14.73 [8.58, 20.89] | 8.82 [3.90, 13.74] |
| 0 | 200 | 12.96 ± 2.87 (SD 20.59, SE 1.46) | 15.14 [9.34, 20.94] | 15.14 [9.34, 20.94] |
| +1 | 196 | 26.62 ± 3.47 (SD 24.63, SE 1.76) | 31.73 [24.39, 39.08] | 29.83 [21.59, 38.07] |
| +2 | 198 | 38.14 ± 3.63 (SD 25.93, SE 1.84) | 41.31 [34.61, 48.00] | 44.82 [38.21, 51.42] |
| +3 | 194 | 53.89 ± 3.61 (SD 25.47, SE 1.83) | 48.88 [42.39, 55.36] | 53.15 [45.82, 60.47] |
| +4 | 194 | 65.71 ± 4.49 (SD 31.68, SE 2.27) | 64.62 [56.75, 72.50] | 66.67 [58.21, 75.12] |
| +5 | 190 | 76.68 ± 4.73 (SD 33.05, SE 2.40) | 75.30 [65.23, 85.37] | 80.78 [71.31, 90.25] |

**Which of their files is the comparator, established by question text and not by
filename.** `v1/` and `v2/` are different question *sets*, but their Dictator
question is byte-identical to the one we ran (`ac8f242ed284c309`), so both are
valid n=50 runs of exactly our prompt and both are given. `v1/` is what
`scripts/eval_steering.sh` writes. `deprecated/v1/` is **not** a comparator — that
one really is the `altruism_v1` Dictator (`fa72234d97e5c4b6`, $10 stake, n=10).

**Every one of the eleven intervals overlaps theirs, in both sets** (r = 0.9905
against `v1/`, mean gap $2.56; r = 0.9866 against `v2/`, mean gap $3.72). Their own
two n=50 runs of the identical prompt differ from *each other* by $4.40 on average
and $11.07 at worst — more than our distance to either. At n=50 "their published
number" is not one target, which is the documented reason the standard here is
n=200.

**The negative arm is not monotonic, in ours or in theirs.** Ours bottoms out at
beta −1 and climbs back to 30.11 at −5; their `v1/` run bottoms at −2 and their
`v2/` at −1. Reproducing that turning point is stronger evidence of a correct hook
than the monotonic positive arm is — a wrong layer or a mis-scaled vector would not
put the kink in the same place.

What the turnaround is: their committed `coherence` judge falls monotonically down
the negative arm (95 → 79 in `v1/`, 95 → 77 in `v2/`) while the positive arm holds
at 92–96, and the amount starts climbing at exactly beta −3, where coherence drops
below ~89. What it is **not** shown to be: coherence was not measured on *our*
generations — that needs the paid judge, and this run made no API calls — and our
parse coverage does not deteriorate on the negative arm (98% at −5 against 95% at
+5), so the answers are not becoming unreadable; they still name a number. The
coherence-collapse reading is consistent with their labels, not demonstrated by
this run.

### Result rows

Steered rows carry `generate.ROW_FIELDS` plus ten columns — `steer_coeff`,
`steer_layer`, `steer_module_index`, `steer_module_path`, `steer_positions`,
`steer_norm_mode`, `steer_vector`, `steer_vector_sha256`, `steer_vector_norm`,
`steer_delta_dtype` — written by `SteeredRowWriter`, which flushes per batch like
`RowWriter`. The tuple is pinned literally in the tests and checked disjoint from
`ROW_FIELDS` at import, since a collision would be silently overwritten.

A sweep varies the coefficient and nothing else: `sweep` refuses steerings that
differ in vector, layer, positions or norm mode, because its results are keyed by
coefficient and would otherwise conflate two experiments. The norm mode is in the
output filename for the same reason.

Two caveats on provenance, neither of which moves a number.

The rows above were produced by the pre-review `steer.py`, which rebuilt the
steering (and reread the vector file) once per batch; the committed module builds
it once per engine. Same file, same coefficient, same cast, so the delta tensor is
identical — but the committed code is not byte-identical to what ran, and the sweep
has not been repeated.

They also predate the norm switch, so they carry the earlier eight-column steering
schema without `steer_norm_mode` / `steer_vector_norm`, and their filenames lack
the `_raw_` segment the current `rows_path` writes. Every one of them is `raw` by
construction, that being the only behaviour the code then had. Re-running is the
only way to get the two new columns onto them, and it was not worth the GPU time.

**These ten columns belong on `generate.Row`.** They live in `steer.py` only
because the task that added this module was scoped not to touch `generate.py`,
and the cost is that `SteeredRowWriter` duplicates about twenty lines of
`RowWriter` and has to be kept in step with it. Moving them is the first thing to
do here.

## Tests

```bash
python -m pytest audit/tests -q
```

Runs in about nine seconds. No keys, no network, no GPU — verified by running it
with the keys unset and outbound connections blocked. `elicit.py` and
`generate.py` are covered without a model at all: the elicitation tests render
against a fake tokenizer and assert the bytes, and the generation tests drive
`run` with a fake engine.

`tests/test_steer.py` is the one module that needs torch, because a forward hook
cannot be exercised against a stub. It **skips whole** (visibly, as a skip) when
torch is absent, so the rest of the suite still runs in a torch-free environment.
Most of it drives a few-hundred-parameter CPU model; four tests build a real
`Qwen2ForCausalLM` of a few thousand parameters and run `generate()` with a KV
cache, so the prompt forward and every width-1 decode step are exercised the way
they are at 7B. Those four also need `transformers` and skip without it. Still no
GPU, no network, no keys — the whole module runs in about four seconds.

The semantics are mutation-tested, not just asserted. Each row is a real edit to
`steer.py` and the tests that then fail:

| mutation | tests failed |
| --- | ---: |
| `__enter__` registers a no-op instead of the hook | 9 (incl. all three no-op tests) |
| `positions="all"` skips a width-1 decode step | 2 |
| scale before the dtype cast instead of after | 4 |
| hook `layers[layer]` instead of `layers[layer - 1]` | 42 |
| `norm="unit"` silently falls back to raw | 4 |
| `steer_vector_norm` reported as 1.0 | 2 |
| `raw` normalises too | 7 |
| `unit` casts to bf16 before dividing, not after | 1 |

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

**The eight steering columns are in the wrong module.** `SteeredRowWriter`
duplicates about twenty lines of `generate.RowWriter` so that a steered row can
carry its coefficient, layer and vector. They belong on `generate.Row`; until they
move, the two writers have to be kept in step by hand.

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
