# How the decision vector is computed

This document describes the construction of an **outcome-defined steering vector**
for the Dictator game, and the two null vectors used to test it. It assumes no
prior context beyond a working knowledge of transformer activations.

Everything here is for `Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa` attention.

---

## 1. What problem this solves

The reference method (Chen et al.'s persona vectors) builds a steering direction
for a trait by writing two *system prompts* — one instructing the model to be
altruistic, one instructing the opposite — running both over a set of everyday
questions, and taking the difference of mean activations. The resulting direction
encodes **how a prompt manipulation moved the residual stream**.

That has two properties worth naming:

* The label is the experimenter's instruction, not the model's behaviour. A
  response labelled "altruistic" is one the model produced *while told to be
  altruistic*, whether or not it acted altruistically.
* The extraction prompts are off-distribution from the games the vector is later
  steered on. The shipped altruism vector was extracted from advice questions
  ("I just got a large unexpected…"), then applied to economic games.

The vector described here inverts both choices. Its labels come from **what the
model actually did** in the game, scored deterministically, and the activations
are captured **on the game prompts themselves**. Nothing about it references a
trait word, a persona instruction, or a judge.

---

## 2. The prompt grid

1800 generations: **6 endowments × 5 wordings × 60 samples**.

**Endowments** — $10, $20, $50, $100, $250, $500. A 50× range, 1.7 orders of
magnitude. The ceiling is a scorer constraint, not an arbitrary choice: the
deterministic amount parser matches at most three digits, so a $1000 endowment
would be misread as $100. $999 is the largest scoreable stake.

**Wordings** — five neutral rephrasings that vary the verb (*give / transfer /
allocate / send / hand over*), the sentence order, the grammatical person, and
how payoffs are presented. They do **not** vary the strategic situation: every
cell is one-shot, anonymous, no further rounds, no observation, no reputation,
the recipient makes no decision, and there is no hint of need. The recipient is
called "Agent 2" in all five deliberately — the scorer anchors amounts on the
agent named in the answer, so renaming agents would change the *scorer* per
wording and put a parser artifact inside the labels.

One cell (`upstream` at $100) is byte-identical to the reference paper's own
Dictator question, which anchors the grid to an external comparison point.

Sampling is the `neutral` preset: temperature 1.0, top_p 1.0, top_k 0, min_p 0.0,
repetition_penalty 1.0, max_new_tokens 1000, seed 0, batch size 20. No knob in it
biases the answer distribution.

### What the grid measured on the way past

Giving is **scale-invariant over 50×** — mean fraction given runs 0.155 / 0.208 /
0.182 / 0.178 / 0.182 / 0.197 across the six endowments. That is why labels are
assigned by **fraction of the endowment**, never by dollars.

Surface wording moves the decision far more than the stake does: the share giving
at least half runs from 0.106 (second-person "allocate") to 0.489 ("hand over"),
a 4.7× spread with the payoff structure held identical. This is the confound that
forces cell-balancing in §5.

---

## 3. The labels, and why they are the right poles

In a one-shot anonymous Dictator game with no observation and no future, giving
is **strictly dominated**: every dollar transferred reduces the giver's payoff and
buys nothing back. The game therefore has an unambiguous game-theoretic reference
point — keep everything — and any deviation from it is the behaviour of interest.
That makes the two ends of the payoff decision natural, objective poles:

| pole | definition | rows |
|---|---|---|
| self-interested | transferred exactly 0 | **939** |
| altruistic | transferred ≥ half the endowment | **452** |
| discarded — middle | 0 < fraction < 0.5 | 362 |
| discarded — subtraction-derived score | see below | 32 |
| discarded — unresolved | 12 unparsed + 3 refusal | 15 |

Parse coverage over all 1800 generations: **99.17%**. An answer the scorer cannot
resolve is a reported category, never a silent zero.

**Why the middle is discarded.** The contrast is between two decisions the model
demonstrably reached, not a regression on a continuum. Rows between the poles are
excluded by design; nothing in this construction speaks to them.

**Why two scorer tags are excluded.** Two of the parser's resolution paths
(`complement`, `keep`) do not read the amount off the response — they subtract a
read amount from the pot. That inverts a refusal: a response reading "the rational
response for Agent 1 would be to send $0" can score as the whole pot, putting a
self-interested row in the altruistic pole. Those 32 rows are excluded rather than
trusted. Rebuilding the vector with them included moves it by cosine 0.99996, so
the choice is principled but immaterial.

**Hand audit of the labelling.** 56 rows sampled stratified over all 8 resolution
tags × both poles, each read against its response: **0 pole errors**. Rule of
three puts the upper 95% bound on the pole error rate at ~5%. The audit measures
the parser; no label used in the vector comes from a human or a judge.

---

## 4. Teacher-forced activation extraction

For each labelled row, the prompt it was generated from is re-rendered and
verified byte-for-byte against the fingerprint recorded at generation time, then
**one forward pass** is run over `prompt + the model's own response` with
`output_hidden_states=True`, batch size 1, no padding. Batch size 1 matters: with
padding, position slices do not mean what they say.

All 29 hidden states (28 layers + embeddings) are pooled three ways per layer:

| pooling | positions |
|---|---|
| `prompt_avg` | mean over `[0, prompt_len)` |
| `prompt_last` | the single position `prompt_len - 1` |
| `response_avg` | mean over `[prompt_len, end)` |

Two checks the reference implementation does not make were made, and both came
back clean over all 1785 scored rows: the prompt's tokens really are a prefix of
the prompt+response tokens (BPE can merge across that seam and silently shift
`prompt_len`), and no response is empty. **0 rows dropped.**

`dtype` and `attn_implementation` are pinned and recorded at extraction time, not
only at generation time. A persona vector is the output of a forward pass, and
this model's sdpa and eager kernels diverge by ~3.4 logits at bf16.

The vector is the per-layer mean difference, **altruistic minus self-interested**,
saved as `(29, 3584)` float32.

### Only `response_avg` is usable here

With causal attention, activations at prompt positions are a function of the
prompt alone. Every row in one grid cell shares a prompt, so **within a cell the
prompt-side vectors are the same vector**. Measured over 57–60 differently-sampled
answers to one prompt at layer 20, the maximum deviation from the cell mean as a
fraction of its norm is 0.7% for `prompt_avg` and 1.3% for `prompt_last` — kernel
jitter, not information — against 32% for `response_avg`.

A prompt-side pole difference can therefore only encode *which cells fell into
each pole*. Given the 4.7× wording spread, it does exactly that. The
cell-balanced control proves it: `prompt_avg`'s layer-20 norm collapses from 2.81
to 0.064 and `prompt_last` from 3.86 to 0.067, both into the jitter floor, while
`response_avg` holds 7.04 → 6.87.

The reference method's prompt-side variants are meaningful because *their*
manipulation is the prompt. Ours is not, so ours are not.

**The caveat that matters.** At layer 0 — the token-embedding average of the
response, involving no computation at all — the two poles already separate at
AUC 0.903. An answer that gives half the pot literally contains different words
and digits from one that gives nothing. So a large share of `response_avg`
separation at every layer is the *lexical content of the answer*, not an internal
decision state. This is a property of the `response_avg` construction in general,
not of the outcome labelling.

---

## 5. Cell-balancing, and why pooling is confounded

A naive difference of means pools all 452 altruistic rows against all 939
self-interested rows. But the poles are **not evenly distributed across cells**:
the "hand over" wording produces 4.7× the altruistic share that "allocate" does.
So a pooled difference partly measures *which prompt text* was more represented in
each pole — a surface-wording direction wearing a decision label.

The cell-balanced vector removes this. A **cell** is one (wording, endowment)
pair; there are 30, and all 30 contain rows in both poles. The vector is

```
v = mean over cells of ( mean activation of altruistic rows in that cell
                       - mean activation of self-interested rows in that cell )
```

Every cell contributes equally and every difference is taken **within a fixed
prompt**, so prompt composition cancels exactly rather than approximately.

Layer 20: the naive vector's norm is 7.0422, the cell-balanced 6.8721, cosine
between them 0.974. Composition was therefore not driving the naive vector — but
that is a *result* of the control, not a reason to skip it, and the cell-balanced
version is the one used for steering.

### Choice of layer

Held-out separation (fit on half the rows, score the disjoint other half) peaks at
**layer 20**: AUC 0.932, Cohen's d 2.32, with layers 19 and 21 within noise and a
broad plateau from 14 to 27. The reference method's inherited choice of layer 20
is independently the right one for this contrast — nothing in this construction
referenced it.

---

## 6. The two nulls, and the difference between them

A steering result needs a control that answers: *would a direction carrying no
decision information, of the same length, move the game just as much?* There are
two different such controls and they answer different questions.

### Null A — the shuffled-label null

Rebuild the vector from the **same activations** with the pole labels **permuted
within each cell**, preserving each cell's pole counts exactly. Within-cell
permutation is the matched null for a cell-balanced vector: it holds prompt
composition and every per-cell count fixed and destroys only the association
between a row's activation and the decision it recorded.

Before trusting it, the real vector was **rebuilt from the raw activations** and
checked against the shipped artifact — maximum relative per-layer deviation
2.7 × 10⁻⁸. A null is only a null for a construction you can reproduce.

Layer-20 norm of the shuffled vector: **0.5799**, against the real vector's
**6.8721**. Steering is done at unit norm, so the length of the intervention is
identical regardless.

### Why a permutation null is not sufficient

This is the part most worth carrying to another experiment, so it is spelled out
with the arithmetic.

**A permutation null is null with respect to LABELS, not orthogonal to the
target.** Permuting the pole labels destroys the association between an activation
and the decision it recorded. It does *not* produce a direction unrelated to the
real one, because the two directions are built from the same rows.

Measured here, Null A's cosine to the real vector at layer 20 is **+0.2423**. That
is not an unlucky draw. The deviations of these activations are anisotropic,
occupying roughly 8 effective dimensions, so *any* difference of means over this
row set lands partly in the same subspace. The empirical cosine null for these
vectors has sd **0.0808** — about 5x the theoretical `1/sqrt(3584) = 0.0167` — and
it is target-dependent. **Never quote the theoretical cosine null for activation
difference vectors.**

Now combine that with the second measured fact, which only appears once the sweep
is run: **the real vector saturates.** It reaches its full behavioural effect at
unit beta 10.51 ($38.75) and is flat or slightly declining out to 52.54 ($38.70).

The two facts together make the extreme comparison uninformative:

```
component of Null A along the real direction, at unit beta 52.54
    = cos x beta
    = 0.2423 x 52.54
    = 12.73 unit-beta

real vector's saturation point
    = 10.51 unit-beta

12.73 > 10.51
```

At the top of the sweep, Null A delivers a **supra-saturating dose of the real
direction** on top of its orthogonal remainder. Whatever else it is doing, it is
pushing the real direction harder than the real arm's own effect ceiling. A "null"
in that configuration cannot distinguish *"the decision direction did it"* from
*"any direction did it"* — and if it happens to match or exceed the real arm there,
that is the expected outcome, not evidence against the direction.

### Null B — the orthogonalised null

Null A with the real direction projected out, per layer:

```
v_B = v_A - (v_A . v_real_hat) v_real_hat
```

Cosine to the real vector after the float32 round-trip that produces the file
actually steered: **-8.9 x 10^-10**, i.e. exactly orthogonal at the steered layer.
Layer-20 norm 0.5626, against Null A's 0.5799 — removing a 0.242 component of a
unit-cosine pair barely shortens it, and unit-norm steering makes the length
irrelevant anyway.

**Null B removes the ambiguity by construction.** With the real component
projected out there is no dose along the real direction at any beta, so any
movement it produces is attributable to the rest of the edit.

What the two controls answer:

| control | question it answers |
|---|---|
| Null A (permuted labels) | Was the label association necessary? |
| Null B (orthogonalised) | Was this direction necessary, or would any edit of this size do? |

Both are needed and neither substitutes for the other. Null A is the right control
for "did I just fit noise in the labels"; Null B is the right control for "is the
direction doing the work".

**Two rules this generalises to**, both cheap:

1. **Report the cosine between every null and its target.** One line of code. A
   permutation null is not orthogonal, and the amount by which it is not is exactly
   what determines whether it is usable at the coefficient you chose.
2. **Find each arm's saturation point before choosing where to compare.** If any
   arm is flat at that coefficient, the comparison measures the flatness, not the
   mechanism. Where one curve saturates and another does not, no single coefficient
   is a fair comparison and the one you pick decides the answer.

In this run both rules mattered: read at unit beta 52.54 the evidence says the
decision separation is not the mechanism; read at 10.51, where the real arm is
still responding and Null B carries none of it, the same experiment says it is. The
measurement that resolves the question is the small one, not the dramatic one.

---

## 7. Steering, and matching intervention size

The vector is added at every position of every forward pass — prompt tokens and
each decode step alike — at the output of `model.model.layers[19]`, whose output
is byte-equal to `hidden_states[20]`. This is the reference implementation's hook
site and arithmetic, including the detail that the vector is cast to the model's
parameter dtype *before* being scaled (`coeff * bfloat16(v)`, not
`bfloat16(coeff * v)`; the two differ in the last bf16 mantissa bit).

**Raw coefficients are not comparable across vectors of different norms.** The
reference altruism vector has layer-20 norm 10.5083; the decision vector 6.8721;
the nulls 0.5799 and 0.5626. Sweeping all of them at the same raw beta would make
the decision vector a 35% weaker intervention and the nulls a 95% weaker one, and
any difference would be an artifact of vector length.

Every arm is therefore steered at **unit norm**, at coefficients

```
unit beta = k × 10.5083        for k = 0, ±1 … ±5
```

so that each point applies exactly the delta magnitude the reference vector's raw
beta *k* produced. Conversions:

| their raw beta *k* | unit beta | our raw beta | Null A raw beta |
|---|---|---|---|
| ±1 | ±10.5083 | ±1.5291 | ±18.120 |
| ±2 | ±21.0166 | ±3.0582 | ±36.240 |
| ±3 | ±31.5249 | ±4.5874 | ±54.361 |
| ±4 | ±42.0332 | ±6.1165 | ±72.481 |
| ±5 | ±52.5415 | ±7.6456 | ±90.601 |

`beta_raw = beta_unit / ‖v‖` in either direction, so a finished sweep can be
relabelled between conventions without regenerating anything.

### Gates run before any sweep

1. **beta = 0 is a byte-exact no-op** against no hook at all — with the three
   controls that make that statement mean anything: the unhooked run is
   reproducible at a fixed seed and batch, the hook demonstrably fired (48
   invocations), and a nonzero coefficient changes the token ids. A no-op check
   without positive controls is satisfied by a hook that was never installed.
2. **The hook site is correct**: the module resolved is `model.model.layers[19]`,
   and its output is byte-equal to `hidden_states[20]` — and to neither
   `hidden_states[19]` nor `hidden_states[21]`.
3. **The vector loaded is the file intended**: sha256 and layer-20 norm recorded
   on every result row.

Because delta is exactly zero at beta = 0 for every vector, all arms share one
no-op run. This was verified: the decision arm's beta = 0 rows are byte-identical
to the reference sweep's beta = 0 rows, all 200 answers. The curves share an
origin by construction, not by coincidence.

---

## 8. Scoring

Every answer is scored by a deterministic parser, never an LLM judge. It returns
a value and a **tag** naming how the value was resolved; the value is `None`
exactly when the tag is `empty`, `refusal`, or `unparsed`. An unresolvable
response keeps its row and its tag with an empty value — it is a reported
category, never a silent zero and never a guess.

The parser was validated against the reference paper's own paid `gpt-4.1-mini`
judge: 95.6% exact agreement over 1951 held-out generations, reproducing its
per-condition means at r = 0.999.

Two known parser failure modes are relevant when reading results:

* `complement` / `keep` derive the amount by subtraction and can invert a refusal
  (see §3).
* `a2_near` can anchor on a number in the payoff sentence. A response reading
  "I give Agent 2 $0; my payoff remains $100" can be scored as $100. This is the
  mirror image of the first, and it is the dominant scorer error on strongly
  negative steering, where answers frequently end by restating the payoff.

Neither is corrected automatically. Where they materially affect a reported
number, the affected rows are read by hand and the correction is stated.

---

## 9. Artifacts

Committed alongside this file:

| path | what |
|---|---|
| `vectors/decision_response_avg_diff_cellbalanced.pt` | the vector under test, (29, 3584) fp32 |
| `vectors/decision_response_avg_diff_cellbalanced_shuffled_seed20260819.pt` | Null A |
| `vectors/decision_shuffled_orthogonalised_seed20260819.pt` | Null B |
| `rows/decision/`, `rows/shuffled-null/`, `rows/orthogonal-null/` | the steering output: one self-describing CSV per coefficient, 4400 rows in total |
| `extraction/grid_seed0.csv` | the 1800-generation grid of section 2, scored, one self-describing row each - the input the poles were read off |
| `scripts/` | the code for the grid, the extraction and the analysis, committed as it ran |
| `summary.csv` | per-arm, per-coefficient n, parsed, mean, SD, SE and 95% CI |
| `steering_comparison.png` | the figure |
| `README.md` | the result, its bounds, and what it does not establish |

Not committed, and named so it is clear what a rebuild would need: the 2.1 GB of
captured activations (`acts_seed0/`), the null-construction and sweep scripts, and
the gate/provenance records (`gates.json`, `null_vector.json`,
`orthogonal_null_vector.json`, `sweep_provenance.json`, `negative_arm_audit.json`).
The gate results and the hand audit of the negative arm are summarised in
`README.md`, which also states what does and does not run about the committed
scripts.

Every result row carries its own provenance: model id and revision, dtype,
attention implementation, sampling parameters, stop tokens, chat-template hash,
question hash, prompt hash, repo commit, steering vector path, sha256, norm and
coefficient.
