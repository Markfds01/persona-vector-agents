# How the per-game steering sweep is run

`README.md` is the result; this file is the construction. It states every choice
that can move a number, and for the two that were open questions — which pole
policy, and what the beta axis is measured against — it states what the other
choice would have been and what taking it would change.

---

## 1. The question

`results/crossgame-decision-vectors` built six outcome-defined decision vectors,
one per economic game, and asked whether they point the same way. That study is
**extraction and alignment only**: no vector was ever added to a residual stream
there, and it says so in its own first section.

`results/dictator-decision-vector` did add one — but only the Dictator's, and only
to the Dictator game.

This directory closes the missing cell. Each of the six games is steered with
**its own** vector, on the game's own evaluation prompt, over one shared ladder of
intervention sizes, with a matched null per game. It answers "does each game's
decision direction move that game's decisions, and by how much", one game at a
time. It makes **no** cross-vector claim: no game is steered with another game's
vector here, and nothing in this directory bears on whether the six directions
agree — that is what the vector study measured, and its answer was no.

---

## 2. What is steered, and what it is steered on

**The vectors** are the committed per-game cell-balanced vectors,
`results/crossgame-decision-vectors/vectors/decision_<game>_response_avg_diff_cellbalanced_strict.pt`,
at **layer 20**. Nothing here rebuilds, re-fits or edits them; they are loaded by
path and their SHA-256 is recorded on every generated row.

**The prompts** are the paper's Table-1 evaluation suite, `altruism_v3`, one
question per game, elicitation mode `free` — the model answers in prose and a
deterministic parser reads the number or the choice out of it. This is the same
prompt and the same mode the Dictator-only sweep steered, which is what puts the
two runs on one axis.

The prompts are therefore **out of sample** for the vectors. The vectors were fit
on a 30-cell grid per game (5 wordings x 6 stakes,
`results/crossgame-decision-vectors/scripts/prompting/crossgame_grid.py`); the
`altruism_v3` question is none of those thirty cells. A vector that only moved the
prompts it was fit on would show nothing here.

`scripts/prompting/eval_games.py` declares the six games. It does not re-register,
shadow or edit the shipped `audit.games` declarations — it reads them, and adds
exactly one number per game, the pole scale (§4).

---

## 3. The axis: intervention size, not raw coefficient

This is the choice the whole run turns on.

Upstream steering scales a shipped vector as-is: `beta * v`. The six per-game
vectors have six different lengths at layer 20 —

| game | layer-20 norm | ratio to the smallest |
|---|---|---|
| prisoners_dilemma | 8.4669 | 3.72x |
| apology | 8.0029 | 3.52x |
| dictator | 7.5047 | 3.30x |
| trust | 5.3733 | 2.36x |
| ultimatum | 5.3665 | 2.36x |
| overfishing | 2.2738 | 1.00x |

— so a shared raw beta would push the Prisoner's Dilemma **3.7 times harder** than
Overfishing, and any difference between the games would be partly that. The point
of this run is to compare games, so the games must be pushed the same amount.

Every point is therefore placed at a **unit beta**, with `norm="unit"`:

    unit_beta = k * REFERENCE_NORM,     k = 0, +-1, +-2, +-3, +-4, +-5

`norm="unit"` scales the vector to length 1 in float32 before the cast, so the
coefficient on the row **is** the length of the activation edit added at layer 20.
One k is the same sized edit in every game. Each row also records
`steer_vector_norm`, the shipped length, so the equivalent raw beta is
`unit_beta / steer_vector_norm` and the manifest carries both per game.

### What the reference norm is, and why it is that one

**`REFERENCE_NORM = ||altruism_response_avg_diff.pt[20]|| = 10.5083084106`** — the
layer-20 norm of the repo's shipped altruism *trait* vector.

Any shared constant would make the six games comparable to each other. This one
was chosen because it makes them comparable to three things at once:

1. **to each other** — the requirement, satisfied by construction;
2. **to the Dictator-only sweep**, which used exactly this reference, so its eleven
   points sit on this axis without rescaling and this run extends it rather than
   opening a second convention;
3. **to the authors' own beta**, because unit beta `k * 10.5083` is exactly the
   edit their raw beta `k` produced with the trait vector.

The alternative — a constant derived from these six vectors, say their mean norm —
would satisfy (1) and lose (2) and (3), and would additionally make the axis a
property of which games happen to be in the set, so adding a seventh game would
move every published number. The reference is a fixed external length instead.

The alternative that is **not** available is per-game raw beta, i.e. steering each
game at `beta = k` with its own vector. That is the 3.7x error above.

`--reference-vector` changes the axis, and the resolved norm is written to
`provenance/sweep_provenance_strict.json` under `axis.reference_layer_norm`, so a run on
a different reference is self-describing rather than silently different.

### Where the ladder stops

`k` runs to +-5 because that is where the Dictator-only sweep ran, and because that
sweep measured what happens past it: at `|unit_beta| = 52.54` the answers are
already degrading (its orthogonal null showed 40% of answers with a word trigram
repeated five or more times). Beyond +-5 the measurement stops being about the
direction and starts being about the size of any edit, so the ladder ends there.
Every point's surface-degeneracy statistics are reported next to its mean so a
reader can see where that begins for each game.

---

## 4. The poles, and what the decision measure is

Two measures are reported per point, and they are not interchangeable.

**The game's own measure** — dollars given, fish caught, P(Cooperate). It is what
a reader wants to see and it is **not comparable across games**: the units differ,
and in Overfishing a *higher* number is the *less* altruistic answer. It is never
summed or averaged across games anywhere in this directory. Its figure gives a
game **one y axis across both pole policies** — read from the two analyses
together, so the strict and the relaxed panel for that game can be compared — and
still a different axis per game, because the units are not shared.

**The pole shares** — P(altruistic) and P(self-interested) — are the same
contrast, oriented the same way, in all six games, and they are the contrast the
vectors were built from. Every cross-game table and every monotonicity claim is
made on P(altruistic).

Both measures, and the decision-minus-null contrast, also get a **both-policies**
figure that puts a game's strict and relaxed arms — and both of their nulls — in
one panel. It exists because a verdict can change between the policies through
the null rather than through the decision arm, which is invisible across two
images. It is a comparison aid: the verdict bands, ceilings and struck-out spans
stay on the per-policy figures.

The pole definitions are the vector study's, imported from its
`scripts/prompting/poles.py` rather than restated, under the **strict** policy:

| family | self-interested | altruistic |
|---|---|---|
| amount games (dictator, trust, ultimatum, apology) | transferred exactly $0 | transferred at least half the $100 endowment |
| overfishing | caught the maximum, 100 | caught at most the sustainable share, 50 |
| prisoner's dilemma | Defect | Cooperate |

Anything between is `middle` and is counted, never folded into a pole.

`poles.classify` needs a `pole_scale` per game — the endowment for an amount game,
the season capacity for Overfishing, 1.0 for the binary PD score. The shipped
`audit.games.Game` has no such field, because the pole definition belongs to this
investigation and not to the upstream data, so `eval_games.py` declares it, names
the phrase in the question text it was read off, and refuses at import if an
amount game's scale is not the endowment its declared answer space names. The
question text itself is fingerprinted by `audit.games`, so an upstream edit fails
loudly rather than silently re-scoring against the wrong bound.

Overfishing is the one whose scale is not simply the answer bound: the
`altruism_v3` question is sustainable while the two agents together take at most
100, so the capacity is 100, one agent's restrained share is 50, and the maximum
catch is the full 100.

### Why strict is the primary policy

Each game ships two vectors, `..._cellbalanced_strict.pt` and `..._relaxed.pt`.
**Strict is primary here.** Three reasons, in order of weight:

1. **It is the policy the Dictator-only sweep steered under.** Comparability with
   the run this one extends is the whole reason the axis is what it is, and a
   relaxed arm would not sit on it.
2. **It is the vector study's own primary**, and every number in that study's
   `README.md` is strict unless it says otherwise. A steering run whose primary
   disagreed with the extraction study's primary would need a reason, and there
   isn't one.
3. **It is the policy whose pole is the same shape in all six games** — "the
   payoff-maximising extreme" — where relaxed widens four of the six by a
   threshold (kept at least 90%) that has no analogue in the PD, whose answer
   space has two points and no middle.

### What running the relaxed arm would change

Measured, not asserted. The angle between each game's strict and relaxed
cell-balanced vector at layer 20:

| game | strict norm | relaxed norm | cosine | angle |
|---|---|---|---|---|
| prisoners_dilemma | 8.4669 | 8.4669 | +1.0000 | 0.0 deg |
| apology | 8.0029 | 7.7990 | +0.9995 | 1.7 deg |
| dictator | 7.5047 | 6.9216 | +0.9983 | 3.4 deg |
| trust | 5.3733 | 4.7255 | +0.9672 | 14.7 deg |
| ultimatum | 5.3665 | 7.1660 | +0.7982 | **37.0 deg** |
| overfishing | 2.2738 | 2.1486 | +0.6564 | **49.0 deg** |

So for three of six games the policy is immaterial: the PD's two vectors are the
same tensor (its answer space has no middle for a policy to move), and Apology and
Dictator differ by under 3.5 degrees, which at these betas is far below the
run-to-run spread. For **Ultimatum and Overfishing the relaxed arm would be a
different experiment, not a rescaling of this one** — 37 and 49 degrees apart, and
the vector study records why: the Ultimatum's self pole widens from 29 rows across
10 cells to 235 across 24 (effective n 15.1 -> 92.3) and Overfishing's from 50 rows
to a much larger set (28.9 -> 73.6). Both are better-estimated directions under
relaxed. Trust, at 14.7 degrees, is a borderline case.

The honest statement is therefore: the strict result here is a result about the
strict directions, and for Ultimatum and Overfishing it does **not** transfer to
the relaxed ones. Running the relaxed arm is one more invocation of the same
pipeline (`POLICY=relaxed`, and `GAMES` restricted to the five that are not a
reuse), costs another full sweep, and was the obvious next thing to spend a card
on. `POLICY` owns every output path in `run_steering.sh` — rows, analysis, tables,
logs and the null-build report are all suffixed by it — because a row's filename
carries its game, arm and coefficient but not its pole policy, so a shared output
root would have the second round land on the first. `README.md` § 8 gives both
invocations. **It has since been run** — same ladder, same reference
norm, same n, same null construction, same seeds, only the vectors different —
and `README.md` § 11 reports it. It did not simply confirm the strict arm: the
Ultimatum's verdict changes, and so does the Dictator's, whose two vectors are
3.4 degrees apart.

Both policies are nonetheless *scored* on every point, from the same values,
because reclassifying a finished row costs nothing; `analysis/steering.json`
carries `poles.relaxed` beside `poles.strict` throughout. Measured since, over
all 154 points of both arms: **`P(altruistic)` is identical under both policies**
— relaxed widens only the SELF pole — so the two arms are read on the same
measure and the only difference between them is the vector. Steering with the
relaxed *vectors* needed another run; it was made, on five of the six games, and
`README.md` § 11 reports it.

### One recorded artifact in the relaxed policy

`poles.RELAXED_KEEP_FRACTION` is 0.9 and the test is `fraction <= 1.0 - 0.9`. In
binary that right-hand side is 0.09999999999999998, so a transfer of **exactly** a
tenth of the endowment — $10 of $100 — falls outside the relaxed self-interested
pole although the prose ("kept at least 90%") includes it. Reported, not
corrected: it is the boundary the committed relaxed vectors were built under, and
changing it would change them. It does not touch any strict number.
`scripts/tests/test_steering_sweep.py` pins the behaviour as it is.

---

## 5. The two arms

Both arms are steered at the same unit betas, and they are written to separate
directories because a shared one would have the second overwrite the first.

**`decision`** — that game's committed vector, all 11 points.

**`shuffled-null`** — that game's matched null, at `k = 0, +5, -5`, the extremes
and zero, exactly as the Dictator-only sweep ran its null.

The null vector is built by `scripts/measurement/build_nulls.py` from the **same
activations and the same cells** as the real vector, with the pole labels
**permuted within each cell**, both counts preserved. Within-cell permutation is
the matched null for a cell-balanced vector: it holds the prompt composition and
every per-cell pole count fixed and destroys only the association between a row's
activation and the decision it recorded. A global permutation would additionally
reshuffle composition across cells, which cell balancing exists to cancel.

Before writing a null the script **rebuilds the committed real vector** from those
activations and refuses to continue unless it reproduces it: if it cannot
reproduce the artifact, its null is not a null *of* that artifact. The rebuild
runs through `analyze_crossgame.balanced_vector` — the function that produced the
committed vectors — rather than a transcription of it. Measured max relative
per-layer deviation: **2.77e-08** over the six strict games and **2.92e-08** over
the six relaxed ones (float32 storage round-trip), against a refusal threshold of
1e-6. `provenance/null_vectors_strict.json` and `null_vectors_relaxed.json` record
it per game; the run log prints the same numbers rounded to `2.7e-08`.

The nulls are much shorter than the real vectors (1.30 to 2.69 against 2.27 to
8.47) — which is the expected consequence of destroying the systematic
difference, and exactly why the comparison has to be at matched **unit** beta. At
matched raw beta the null would be a far smaller push and would prove nothing.

`k = 0` is a shared no-op: the delta is exactly zero for both vectors, so the two
arms must have generated the same text there. That is checked, not assumed —
every row's continuation, scored answer, value and tag — and reported as
`beta0_generations_identical_across_arms` per game. The row files are **not**
byte-identical and cannot be: three columns name which vector file was loaded,
and that is the one thing the two arms differ in at `k = 0` by design.

---

## 6. Generation

One configuration, `audit.generate.PRESETS["neutral"]`: temperature 1.0, `top_p`
1.0, `top_k` 0, `min_p` 0.0, no repetition penalty, `max_new_tokens` 1000. Every
knob sits where it leaves the model's own distribution alone, and the preset is
checked against the loaded engine before any generation — asking for bf16 + sdpa
and silently getting something else is a wrong result, not a warning.

`Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa`, one A40, seed 0,
batch size 20, `reading="stated"`. `attn_implementation` is pinned and recorded on
every row like a sampling knob, because it is one: the vector study measured
sdpa -> eager alone moving the Overfishing mode from 50 to 55.

**n per beta point is 100**, against the repo's standard of 200. That is a
deliberate, reported reduction, made before the run rather than discovered after
it: the full design is 6 games x (11 + 3) points = 84 points, which is 16,800
generations at n=200 and, at the measured 4.75-6.25 minutes per 200-sample point
on this card, about 7 hours of exclusive GPU — on a card shared with another
tenant. At n=100 it is 8,400 generations and about 3.5 hours.

What that costs, in the repo's own measured terms: the SE of a game mean goes from
about $1.44 to about $2.04, the smallest difference between two points that can be
resolved goes from about $4.0 to about $5.7, and the Wilson half-width on a pole
share at p = 0.5 goes from +-0.069 to +-0.098. Every effect the Dictator-only sweep
found is far larger than that — its move was +25.78 on the mean and 0.61 -> 0.00 on
the pole — so the headline per-game answer is unaffected. What n = 100 cannot
resolve is the **fine shape** of a curve: that sweep's "still rising at +2, then
declining" contrasts were +5.39 (p = 6.2e-05) and -1.92 (p = 0.030), and at n = 100
those would be borderline. Claims of that kind are not made in `README.md`, and
where the shape looks like it has one, the report says the sample size cannot
settle it.

Every share and every interval in this directory carries its own `n`.

---

## 7. Running on a shared card

The A40 this ran on is shared with another tenant that loads and unloads models on
its own schedule. Three things follow, all of them in the code rather than in a
wrapper:

* **A free-memory gate before the load.** The run refuses to start until the card
  reports 16,600 MiB free — the ~15.2 GiB of weights plus a working set — and
  polls until it does, with a bounded wait. Batch size is never lowered to fit: the
  peak above the weights at batch 20 is about 1.4 GiB, against a multi-GiB swing
  from next door, so shrinking the batch would change which draws happen
  (`audit.generate.run` seeds each batch `seed + batch_index`) without buying
  meaningful room.
* **The load, and only the load, is retried.** A load that OOMs is transient. A
  generation that OOMs is not retried in place.
* **Resume at the granularity of one beta point.** Each point is an independent
  `generate.run` whose batches are seeded from zero, so a point regenerated on its
  own is bit-identical to the same point inside an uninterrupted run — which is
  what makes skipping a finished CSV safe. A **short** CSV is never appended to:
  its remaining rows would land in differently composed batches. It is regenerated
  whole. Each game also lands its own manifest entry the moment it finishes.

The device is pinned explicitly. No process other than this run's own is ever
signalled, at any point, for any reason.

---

## 8. What the analysis computes

`scripts/measurement/analyze_game.py`, per game, per arm, per point:

* parse coverage and the full tag census. **An answer the scorer cannot resolve
  keeps its row and its tag and is never counted as a zero**; every mean and share
  is over parsed rows and carries its own n.
* the game's own measure: mean, SD, median and a 95% interval — **Student's t for
  an amount, Wilson for a 0/1 outcome.** The estimator follows the units, not the
  column name, and `eval_games.measure_is_binary` reads it off the game's scorer.
  Only the Prisoner's Dilemma is the second kind: its own measure IS P(Cooperate),
  the same quantity as its altruistic pole share, and a t interval on it runs below
  zero at the negative end of the ladder and has width exactly zero at `k = 0`.
  Every summary in `analysis/steering.json` records the `estimator` it was built
  on, so the file says which path each number came down.
* P(altruistic), P(self-interested) and P(middle) with **Wilson** intervals, under
  both pole policies. Wilson because three of the six games park a pole on a single
  value, and a Wald interval at p = 0.985 is not an interval.
* surface degeneracy — mean length, distinct-trigram ratio, the share of answers
  with a trigram repeated five or more times, the share containing non-Latin
  script. These are **counted, not rated**: no judge is involved anywhere in this
  directory.
* **every parsed row counts, whatever path the scorer read it along.** This
  differs from the vector construction, deliberately. Building a vector, a row
  resolved along a *derived* path (`complement`, `keep`, `mixed`) is excluded from
  the poles, because a derived value can land a row in the wrong pole and the
  vector would then be fit on a mislabelled activation. Measuring a steered
  distribution there is no fitting: the question is what the model did, and a
  correctly derived value is what it did. The full tag census is on every point in
  `analysis/steering.json`, so the derived share at each beta is visible rather
  than folded away. It matters most at the Prisoner's Dilemma's far positive end,
  where `mixed` reaches 7 of 100.
* the move from that arm's own `k = 0`: **Welch** on the mean (the extremes have
  several times the baseline's SD, so a pooled-variance test would be wrong in the
  direction that matters) and **Newcombe's** hybrid-score interval on each pole
  share. A 0/1 own measure takes the second path instead — Newcombe, with the
  pooled two-proportion score test reported beside it. **The interval and the p are
  two procedures, not one**: Newcombe's hybrid-score interval is not the inversion
  of that score test and the two can disagree. At n = 100 per arm sixteen (k1, k2)
  pairs do, always with the interval as the more conservative of the two, and one
  of them is in this directory — the Prisoner's Dilemma's decision arm at k = +1
  moves 0/100 to 4/100, `p = 0.0434` against an interval reaching −0.0043. Nothing
  here reads the p for a verdict: every one is `excludes_zero`, i.e. the interval.
  `stats.difference` routes on the summary rather than on the caller, and
  `stats.welch` refuses a share outright.
* the real arm against its own null at every matched unit beta, the same two ways.

**Monotonicity** is reported three ways, because a bare yes/no over 11 noisy points
is not an answer:

1. the Spearman rank correlation of P(altruistic) against k, which says which
   direction the ladder runs;
2. whether the observed series *never* steps backwards — `strictly_monotone`, a
   fact about the sample, not a test;
3. the list of adjacent steps that go against the overall direction **by more than
   their own 95% Newcombe interval**. `monotone_up_to_noise` is true when that list
   is empty, and it is the form of the claim `README.md` makes.

No p-value anywhere in this directory is corrected for multiplicity, and the
report says so where it quotes one.

`scripts/pooling/crossgame_tables.py` computes nothing new. It lays the six games
side by side on the shared ladder and writes `analysis/points.csv`, one flat row
per (game, arm, k).

---

## 9. What this design cannot answer

* **Nothing about cross-game transfer.** No game is steered with another game's
  vector, or with the pooled vector. Whether the six directions agree was measured
  in `results/crossgame-decision-vectors`, and the answer there was no.
* **Nothing about the relaxed directions for Ultimatum and Overfishing**, which are
  37 and 49 degrees away from the ones steered here (§4). Those were steered as a
  separate arm and are reported in `README.md` § 11; no claim in this file is a
  claim about them.
* **Nothing about how stable a shuffled-label null is.** One null seed per policy,
  20260821. `README.md` § 11.3 shows a verdict — the Dictator's positive arm,
  margin +0.116 — turning on which null was drawn rather than on the direction.
  Separating the two needs several seeds, and that is not run here.
* **Nothing about mechanism.** The Dictator-only study answered "is the decision
  separation what is doing the work" with an orthogonal null — the shuffled null
  with the decision direction projected out. That arm is **not run here**: it is
  another six games' worth of GPU, and the shuffled-label null is the control the
  brief asked for. A game whose real arm beats its shuffled null has shown the
  effect is not a generic consequence of an edit of that size along a
  same-data direction; it has **not** shown that the decision separation
  specifically is the mechanism.
* **Nothing that follows from a single elicitation mode.** Only `free` is run. The
  vector study measured the same Dictator question giving a very different answer
  under a different mode, so these are results about `free`.
* **The Prisoner's Dilemma is underpowered by construction** and no sample size
  here fixes it: its vector rests on **18 cooperations in 231 direct-tag responses**
  across 9 usable cells, effective n 10.2, a design imbalance the vector study
  documents at length. Its numbers are reported with that n beside them and should
  not be read as the equal of the other five.
