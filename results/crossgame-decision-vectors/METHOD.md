# How the cross-game decision vectors are computed

`README.md` is the result. This file is the construction: the prompt grid, the
pole definitions, the two balancings, the nine pooling weightings, the nulls, and
what "out of sample" means for each number.

Everything here is observational. **No steering was run.** Nothing in this
directory adds a vector to the residual stream, and no number below is a causal
effect. The construction of a steering-ready vector and the test of whether
steering it does anything are separate questions; only the first is touched here,
and the answer to it is negative.

The method is deliberately the same as
`results/dictator-decision-vector/METHOD.md` wherever it can be — same model,
same revision, same dtype and attention kernel, same teacher-forced extraction,
same cell balancing, same label-null construction — so that the six-game vectors
and the Dictator-only vector are directly comparable. Since this revision the
extraction is not merely the same *method*, it is the same *code*: §5.

---

## 1. The question

The Dictator-only decision vector — built in `results/dictator-decision-vector/`,
whose METHOD.md is the source of the first figure below and whose README is the
source of the second — separates its poles at layer 20 with out-of-sample AUC
0.932, but at **layer 0**, the mean of the raw input
embeddings of the response tokens, it already reaches 0.903. Layer 0 carries no
computation. A direction that separates the poles there is separating the *tokens
of the answer*, not a decision, and both Dictator poles are dollar amounts.

The proposed fix was to pool across games whose answers share no tokens: $0 versus
$50, 0 fish versus 8 fish, "Defect" versus "Cooperate". If the shared surface is
what layer 0 is reading, pooling should destroy it, and whatever survives at depth
would be something else.

That is the experiment. It has three measurements, all of them observational:

1. **layer-0 versus layer-20 separation** — does pooling collapse layer 0?
2. **the 6x6 agreement matrix** — do the six per-game directions point the same way?
3. **leave-one-game-out transfer** — does a direction fit on five games separate
   the sixth game's poles, having never seen it?

A fourth was added after the first three came back: **reweighting**, to check
whether the pooled direction's dependence on the four dollar games is an artefact
of how the pool was weighted. Section 6.

## 2. The prompt grid

`scripts/prompting/crossgame_grid.py`, question set `crossgame_grid_v1`. Six
game families x six stakes x five wordings = **180 cells, 30 per game**.

| family | scorer | stakes | `pole_scale` |
|---|---|---|---|
| dictator, trust, ultimatum, apology | `amount` | endowment 10, 20, 50, 100, 250, 500 | the endowment |
| overfishing | `fish` | (max catch M, season capacity K) = (20,20) (20,10) (100,100) (100,50) (200,200) (200,100) | K |
| prisoners_dilemma | `binary_choice` | (R, g) = (10,0.1) (10,0.5) (10,1.0) (100,0.1) (100,0.5) (100,1.0) | 1.0 |

The five wordings are `upstream`, `rules_first`, `second_person`, `payoff_lines`,
`narrative`. Wording varies surface phrasing only; nothing varies the strategic
type of the game. Every game stays one-shot (per season, for Overfishing),
anonymous, with no communication, reputation or observation, and Overfishing keeps
its indefinite horizon because a known finite horizon unravels a commons dilemma
by backward induction.

**Why the stakes are payoff parameters and not just scale.** The Dictator run
varied one number, the endowment, because both its poles fill at every endowment.
Two games here are not like that: the grid module records that on the unsteered
baseline the Prisoner's Dilemma defected 50/50 and Overfishing took the greedy
catch 5/50, and the calibration pass committed here reproduces the same corner at
its own sample size — PD 0 cooperations in 48, overfishing 2 self-interested rows
in 48. A pole with no rows in it produces no vector, so for those two the ladder
varies the payoff parameter that governs the temptation to defect:

* **Prisoner's Dilemma** — payoffs `P=0, R, T=R(1+g), S=-R*g`. `T>R>P>S` holds for
  any `R>0, g>0`, and `2R > T+S` holds identically because `T+S = R`. Payoff
  matrices are defined only up to a positive affine transform, so `(R=10, g=0.5)`
  — T=15, R=10, P=0, S=-5 — *is* upstream's (3, 2, 0, -1) game multiplied by 5.
  That cell anchors the grid to the published question.
* **Overfishing** — the sustainability threshold is a catch of `K/2`; if both
  agents take more than that the total exceeds `K` and the lake collapses. Varying
  `K/M` moves that threshold relative to the range the model is choosing inside.

$500 is the endowment ceiling because `audit.parse._AMOUNT_RX` matches at most
three digits: a $1000 endowment would be read as $100. The recipient is always
"Agent 2" because `audit.parse` anchors an amount on the agent named in the
answer's own phrase; renaming the agents would change the scorer per wording and
put a parser artefact inside the labels.

## 3. Generation

`scripts/prompting/gen_crossgame.py`, one model load per invocation, `neutral`
preset, seed 0, batch size 32, `cuda:0`. `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa`, temperature 1.0,
top_p 1.0, top_k 0, min_p 0.0, repetition_penalty 1.0, max_new_tokens 1000, stop
tokens 151645 and 151643.

**This stage was not re-run for this revision.** The rows it produced are
committed under `extraction/`, and everything downstream runs from them.

Samples per cell differ per game, and the reason is the calibration pass rather
than taste:

| family | samples per cell | rows generated |
|---|---|---|
| overfishing | 48 | 1440 |
| dictator, trust, ultimatum, apology | 24 | 720 each |
| prisoners_dilemma | 8 | **240** |

Overfishing got the most because its self-interested pole ran at 2 of 48 in the
committed calibration pass. The Prisoner's Dilemma got the fewest because it
cooperated 0 times in 48 there and was expected to be a census, not a vector — it
then produced 18 cooperations over the full grid and a vector was built from them
anyway. **That is a design imbalance, not a behavioural one**, and it is the
single cheapest thing to fix in any follow-up. It is why every PD number in
`README.md` carries its `n` beside it.

Every row carries its own provenance — model id and revision, dtype, attention
implementation, sampling parameters, stop tokens, chat-template hash, question
hash, prompt hash, repo commit — so a row can be audited without reference to
anything outside the file. 4560 generated, 4503 scored and captured, 0 dropped.

## 4. Labels: the poles, under two policies

`scripts/prompting/poles.py`. The label is always what the model **did**, read
off the scorer's resolved value — never a rating, never the wording of the answer,
never an LLM judge. What differs per family is what "the payoff-maximising choice"
and "restraint" mean in that game's own units, so each family declares both.

**STRICT** — the definition the Dictator-only run used, so a vector built under it
is directly comparable with the Dictator-only vector:

| family | self-interested | altruistic |
|---|---|---|
| amount games | transferred exactly 0 | transferred at least half the endowment |
| overfishing | caught the maximum M | caught at most the sustainable share K/2 |
| prisoner's dilemma | Defect | Cooperate |

**RELAXED** — the same poles widened to the choice that actually maximises own
payoff, which for one game is not the strict pole at all:

| family | self-interested | altruistic |
|---|---|---|
| amount games | kept at least 90% of the endowment | unchanged |
| overfishing | caught *more* than K/2 (the defection in a commons dilemma, maximal or not) | unchanged |
| prisoner's dilemma | unchanged — the answer space has two points and no middle | unchanged |

RELAXED exists because in the Ultimatum the proposer's subgame-perfect offer is
the *smallest positive* amount a rational responder accepts: an offer of exactly
$0 leaves the responder indifferent and free to reject, so a pole pinned at $0
excludes precisely the self-interested play. Under STRICT the Ultimatum's self
pole is 29 rows across 10 cells; under RELAXED it is 235 rows across 24. RELAXED
applies one uniform threshold to all four amount games rather than a rule invented
for the Ultimatum, so no game gets a hand-fitted boundary.

**Both policies are run end to end and both are reported.** Everything in
`README.md` is strict unless it says otherwise, and every conclusion holds under
both.

Anything between the two poles is `middle` and is discarded by construction.

**Tag policy.** `audit.parse` names the path it resolved a value along. Paths that
read the number straight off the response are the primary poles. Paths that
*derive* it — the amount scorer's `complement` and `keep` subtract a read amount
from the pot, and the binary scorer's `mixed` picks between two choices the
response named both of, by position — can land a row in the wrong pole, and land
it exactly on the boundary. Those are excluded from the poles, counted, and re-run
as a sensitivity (`cos_sensitivity_vs_primary` in the analysis JSON).

Two known `audit.parse` bugs bear on the counts and are reported rather than
corrected here: a refusal misclassification that discards some $0 answers, and an
`a2_near` case that can read a figure out of a payoff sentence as the transfer.
Both are directional and both are named in `README.md`.

## 5. Teacher-forced activation extraction

`lab/extract.py` — **one implementation, shared by every grid in this
repository**, matching upstream `generate_vec.py:28-38`:

1. re-render the prompt the row was generated from and verify the rendered bytes
   against the `prompt_sha256` the row carries;
2. run **one** forward pass over `prompt + the model's own response` with
   `output_hidden_states=True`, batch size 1, no padding;
3. pool all 29 hidden states three ways per layer — `prompt_avg` over
   `[0, prompt_len)`, `prompt_last` at `prompt_len-1`, `response_avg` over
   `[prompt_len, end)`.

**Nothing in that loop is per game.** A grid declares its own games and registers
them into `audit.games.GAMES_BY_ID` — `scripts/prompting/crossgame_grid.py` here,
`results/dictator-decision-vector/scripts/decision_grid.py` for the earlier
Dictator-only run — and every row names its own `game_id` and `mode`, so which
prompt is re-rendered, which stakes it carried and which answer space scored it
are read off the data. The grid module is named on the command line
(`--grid <path>`). Adding a grid is a new declaration, never a second copy of the
capture loop. Tests are in `lab/tests/test_extract.py` and need no weights.

Batch size 1 and no padding is what makes the position slices mean what they say.
Two failures are reported rather than absorbed: prefix instability (BPE merging
across the prompt/response seam, which would put the wrong tokens in the response
slice) and an empty response (which would make `response_avg` NaN). Neither
occurred, on any of the 6288 rows.

The reduction order inside the pooling is load-bearing and is upstream's — mean
first, in the activation's own dtype, then cast to float32. Casting first gives
different bits, so the tests pin it.

dtype and `attn_implementation` are pinned and resolved FROM the loaded model into
each game's `meta.json`, never recorded as what was asked for: this model's sdpa
and eager kernels diverge at bf16, so a vector built under one is not comparable
with a vector built under the other. The model revision is recorded as the sha the
weights came from, never a moving ref. Every run in this project is sdpa/bf16 at
revision `a09a35458c702b33eeacc393d103063234e8bc28`.

Shards are written in complete blocks of 250 rows and a run resumes from them
(`--resume`), reading the row indices back out of the shards themselves so there
is no sidecar index to drift. An interruption has to cost a shard rather than a
corpus. The **load** is retried on OOM; a partially captured run never is, and
nothing about dtype, kernel, batch semantics or revision is relaxed to make a run
fit.

Output is `acts/<game>/shard_*.pt` — 5.3 GB for the six games, 2.1 GB for the
Dictator-only grid, neither committed. See `README.md` §12.

**Only `response_avg` is used for anything.** Causal masking makes every
prompt-side activation identical within a prompt cell, so `prompt_last` at layer 0
gives AUC exactly 0.500 — every row in a cell ties — and `prompt_avg` reaches
0.787, which is between-cell pole composition rather than decision information.
All three are captured so that stays checkable instead of asserted.

### The equivalence check that made the refactor safe

The shared module supersedes two files that did the same thing: this directory's
`extract_crossgame.py`, which is deleted, and `extract_acts.py` in the Dictator
directory, which is left in place because that directory is a published artefact
this work does not modify — it is now the frozen record of what produced an
earlier result, not a tool. Its docstring said it was "identical in method" to
this one, and "identical in method" is not a measurement, so before any
GPU was spent on a full re-extraction, 50 rows of each of the seven corpora were
re-extracted with the shared module and compared element by element against the
archived activations — same rows, same revision, same dtype, same kernel, batch 1,
so agreement should be exact.

It was: **109,132,800 of 109,132,800 elements exactly equal**, all three poolings,
all 29 layers, maximum absolute difference 0.0. That pre-check was a scratch run
and **its report is not committed** — it bought the decision to spend the GPU, not
the result. What is committed is the thing it predicted: the full re-extraction
over all 6288 rows, in `analysis/activation_equivalence.json`, which supersedes it
and is the record to read. No tolerance was needed and none is claimed.
`scripts/compare_activations.py` is the code.

## 6. The vectors

### Per game, cell-balanced

A **cell** is one (family, stake, wording). A cell is **usable** when it holds
rows in both poles; only such a cell can contribute a within-cell difference. The
game's vector is the unweighted mean over its usable cells of
`mean(altruistic) - mean(self-interested)`, computed in float64 and saved as a
`(29, 3584)` float32 tensor.

Cell balancing is what stops pole *composition* faking a direction: within a fixed
prompt, the difference is between two answers to the same question, so anything
that varies with the prompt cancels.

The 30-cell grid was balanced **by design**. The imbalance in usable cells is
**behavioural** — a cell only enters when the model populated both of its poles:

| game | usable cells | of |
|---|---|---|
| dictator | 30 | 30 |
| apology | 25 | 30 |
| trust | 20 | 30 |
| overfishing | 17 | 30 |
| ultimatum | 10 | 30 |
| prisoners_dilemma | 9 | 30 |

An unbalanced (whole-game, no cell weighting) vector is also built and kept in the
analysis JSON for the norms and split-half reliability, but no claim in
`README.md` rests on it and it is not committed.

### Pooled, nine weightings

`scripts/pooling/build_vectors.py` through `common.py`. Two-level: build each
game's cell-balanced vector first, then combine the six under one explicit, stated
weighting.

| scheme | across-game weight | balances |
|---|---|---|
| `cell_balanced` | mean over **all** usable cells of all six games — weight proportional to usable-cell count | nothing |
| `game_precision_raw` / `_unit` | proportional to effective n (inverse-variance) | nothing |
| `game_equal_raw` | plain mean of the six | the game axis, in raw units |
| `game_equal_unit` | unit-normalise each game per layer, then a plain mean | the game axis, in direction |
| `family_balanced_raw` / `_unit` | one third to each ANSWER FORMAT, split evenly inside it: 1/12 to each dollar game, 1/3 to overfishing, 1/3 to PD | **the answer-format axis** |
| `non_dollar_raw` / `_unit` | the dollar games dropped; overfishing 1/2, PD 1/2 | the extreme case |

`_unit` normalises each game to unit length per layer before weighting, `_raw`
does not. The six per-game vectors have layer-20 norms from 2.27 to 8.47, so a raw
mean is norm weight, not equal weight; `_unit` is the primary of each pair.

**Why the format axis is the one that matters, and why the game axis is not.**
Four of the six games answer in dollars, so one sixth each still puts 66.7% of the
weight on dollar-format answers. The confound is the answer surface, not the game,
so the axis to equalise is the surface. An earlier pass measured only the five
game-axis schemes, saw them land between 16° and 26° of the Dictator direction,
and said it made no claim about weightings balancing a different axis. Adding the
format axis reaches 66°, so the nine now cover 16° to 66°. That extends the
earlier pass rather than retracting it — `README.md` §5.

**Family balancing is post-hoc.** It was chosen after the agreement matrix showed
the dollar / non-dollar split — that split is the reason to equalise the format
axis, and it came from this same data. It is not pre-registered, and the nulls do
not price in the search that produced it.

**`non_dollar_raw`/`_unit` are thin.** They rest on 39.1 of the 244.3 total
effective rows under strict (16%). Every positive from them is provisional.

**Effective n** is `C^2 / sum_cells (1/n_alt + 1/n_self)` — the number of rows a
single unpartitioned two-group difference would need to be as precise as that
game's cell-balanced vector under a common per-row variance. It is the honest
denominator behind a game's contribution, and under format balancing a third of
the vector rests on PD's 10.2.

`cell_balanced`, rebuilt through this second code path, reproduces the six-game
study's own pooled vector at **cosine 1.000000 at every layer under both
policies**, and reproduces its leave-one-game-out table exactly
(`analysis/pooled_build.json`). Any difference between the schemes is therefore
the weighting and not the pipeline.

### The Dictator-only vector

`scripts/pooling/dictator_vector.py` rebuilds it from the re-extracted activations
of the earlier Dictator-only grid, under the construction that directory used
(cells are (wording, endowment), strict poles, direct-read tags only), and checks
it against the committed
`results/dictator-decision-vector/vectors/decision_response_avg_diff_cellbalanced.pt`.
It is not duplicated into this directory. `README.md` §6 and §7 use it, and the
point of §6 is that it was fit on a **separate, earlier generation run** — no row
of this grid entered it.

## 7. Separation, and what "out of sample" means

`auc` is the Mann-Whitney AUC with tie correction: P(a random altruistic
projection > a random self-interested one) along the direction, per layer.

In-sample separation is circular — the direction is by construction the thing that
maximises it — so every AUC published in `README.md` is scored on rows that were
not used to fit the direction it is scored against. Three constructions appear,
and they are **not** interchangeable:

| name | fit on | scored on |
|---|---|---|
| **split-half (whole-game)** | a random half of a game's pole rows, unbalanced | the other half |
| **split-half within cell** | half the rows of *every* cell, cell-balanced rebuild | the other half of every cell |
| **leave-one-game-out (LOGO)** | five games' cells | the held-out game's own poles — the direction never saw that game |

LOGO is the strongest of the three: a cosine says two directions point the same
way, a transferred AUC says a direction built without ever seeing a game still
tells that game's poles apart.

Split-half figures from the six-game battery and from the reweighting stage are
**not** comparable with each other: the first halves a game's whole pole set and
rebuilds an unbalanced vector, the second splits inside each cell and rebuilds a
balanced one. The valid comparison is always within one table.

A split-half figure also depends on the draw. `scripts/verify_committed.py`
re-runs the within-cell version at a seed the run did not use, and reports it as a
robustness check rather than a reproduction: the ordering across schemes is the
same, the magnitudes are not identical.

## 8. The nulls, and what each one is a null *of*

Every null here is a **label null**: the pole labels are permuted and both sides
are rebuilt from the same activations. It answers "could a random split of these
same activations produce this much agreement". It is **not** a null of two
unrelated directions — shuffled vectors keep whatever structure the activations
share.

Three constructions appear:

| null | permutation | used for |
|---|---|---|
| **game-wide** | labels permuted across a whole game, unbalanced rebuild | the 6x6 agreement matrix (300 draws), pooled vs the Dictator-only and shipped altruism vectors (1000 draws) |
| **within-cell** | labels permuted *inside each cell*, so every cell keeps its pole counts, cell-balanced rebuild | the reweighting stage's cosines (300 draws) |
| **within-cell, leave-one-out** | as above, with the pool rebuilt from the other five games | the LOGO cosines (300 draws) |

**Where each one sits matters, and the two cases behave differently.** Measured, at
layer 20, strict:

* Where the two sides **share no rows** — two different games' vectors, or a pooled
  vector against the Dictator-only vector fit on a separate generation run — the
  null is centred at **zero** (mean −0.007 to +0.011 across the 15 game pairs) but
  is **much wider** than an orthogonality null: sd 0.13–0.19 per pair and 0.22 for
  pooled-vs-Dictator, against the theoretical `1/sqrt(3584) = 0.0167`. That is 8x
  to 13x. So the bar a cosine has to clear is the null's p97.5 — 0.26 to 0.36 per
  pair — and **not** the ~0.03 that a random-direction argument would give.
* Where the two sides **do share rows** — a pooled vector against a game that is
  inside it — the null is centred at **+0.40** (sd 0.12–0.14, p97.5 0.60–0.65),
  because the shared rows are in both vectors under every permutation. The real
  cosine is inflated by the same sharing, which is what makes the comparison fair.
  It also rises with a game's weight: under `non_dollar_unit`, where PD is half
  the pool, its own null p97.5 reaches 0.82. That is why `README.md` §5 reads the
  leave-one-out version for the format-balanced schemes.

Use the measured null, never the theoretical one. This is the same lesson the
Dictator directory records and it is restated here because both of this run's
headline claims — which pairs agree, and which games the pool fails to represent
— are readings against a null, not against zero.

Null construction differs between the two stages for the pooled-vs-Dictator
comparison, and both are computed: game-wide gives p97.5 0.424 (1000 draws) and
0.490 (300 draws), within-cell gives 0.539, for the same cell-balanced pool. The
real cosine (+0.959) clears every version of the bar by a wide margin; the bar is
not what is in question.

## 9. Attenuation

A cosine between two noisily estimated directions is attenuated toward zero, so a
low cosine could in principle be a measurement failure rather than a real
disagreement. The correction is `cos / sqrt(rel_a * rel_b)`, where `rel` is the
split-half reliability at that layer — the cosine between the same vector rebuilt
from two disjoint halves of its own rows.

Reliability is **not** a fixed quantity: it depends on the random half, and on
whether the rebuild is whole-game or within-cell. Both are reported in
`README.md`, and the correction is applied only as a magnitude comparison. It is
never mixed with the nulls, which are computed on uncorrected cosines.

## 10. The layer-0 token decode

Layer 0 of `response_avg` is literally the mean **input embedding** of the response
tokens. No attention, no MLP, no computation of any kind has run. So a layer-0
direction lives in embedding space and can be read back in token space directly:
cosine against every row of `model.embed_tokens.weight`. CPU only, no forward
pass, the embedding matrix read straight off the safetensors shard.

Three scalar summaries of the same thing, because eyeballing a token list is not a
measurement:

* **digit-span share** — the fraction of the layer-0 direction's norm lying in the
  span of the ten tokens `'0'`–`'9'` (a QR basis of their embeddings).
* **signed digit alignment** — the cosine with `mean('4','5','6','7') - '0'`, i.e.
  "the answer is a mid-size digit rather than zero". The unsigned share cannot see
  polarity, and Overfishing's numeric polarity is inverted, so the sign matters.
* **the per-digit decomposition, and the share carried by `'0'` alone.** The
  aggregate cannot tell "'5' marks generous" from "'0' marks the extreme", and
  that is exactly what changes across weightings: the dollar signature is
  dismantled by format balancing while the aggregate barely moves, because a
  different surface replaces it. Reporting only the aggregate would be misleading.

**The reference is not the spherical figure** `sqrt(10/3584) = 0.053`. Token
embeddings are not isotropic, so any direction overlaps any ten of them more than
a sphere would predict. The control is empirical: random ten-token subspaces drawn
from the 5657 distinct tokens the model actually emitted in these responses. A
random-vocabulary control is also computed and is the wrong comparison for a
direction built out of response text, so it is reported and not used.

The run drew 32 control subspaces. 32 draws is few for a p97.5, and one draw that
happens to contain a digit token inflates it, so `README.md` quotes both the run's
32-draw band and a 1000-draw re-estimate from `scripts/verify_committed.py`. No
conclusion turns on the difference.

## 11. Orthogonalisation against the Dictator direction

The decisive test in section 6 of `README.md`. Per layer, project the
Dictator-only direction out of a leave-one-game-out pool and re-run the held-out
separation:

```
u     = dictator[layer] / ||dictator[layer]||
resid = pool[layer] - (pool[layer] . u) u
```

The Dictator-only vector was fit on a **separate, earlier Dictator-only generation
run**, not on any row in this grid. Projecting it out therefore removes a
direction estimated from independent data, which makes a collapse more meaningful
rather than less: it is not the pooled vector being made orthogonal to part of
itself.

## 12. Validation, and what the code has to reproduce

Three separate checks, and they answer different questions.

**Before the data.** `scripts/measurement/selftest_analysis.py` runs the whole
battery on **synthetic** activations with a known answer, so the analysis code was
checked before it saw this data. It plants a shared direction and requires the
battery to recover it (pooled AUC 1.000, agreement 0.837, LOGO 1.000), runs it on
pure noise and requires chance (0.513, 0.012), and confirms a game that filled
only one pole is excluded from every pooled structure. All 7 checks pass from the
committed copy, run from where it sits.

**That the committed code produces the committed numbers.**
`scripts/run_extraction.sh` and `scripts/run_analysis.sh` regenerate every vector
and every analysis file from the committed rows CSVs, and
`scripts/compare_published.py` diffs the result against what is committed, leaf by
leaf (`analysis/rebuild_deltas.json`).

**That the numbers are right, by a code path that shares nothing with the
analysis.** `scripts/verify_committed.py` reads the activation shards directly,
imports only `crossgame_grid` and `poles` — the grid and pole *definitions*, not
the computation — and does its own linear algebra. `README.md` §11 lists what it
covers and what it found.
