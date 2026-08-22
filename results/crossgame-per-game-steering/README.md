# Six games, each steered with its own decision vector

**Every one of the six games moves under its own direction, and for three of them
that is the whole story. For the other three it is not.** One game's movement is
not distinguishable from its own null; two games are pinned against a ceiling or
a floor at baseline and can only be steered one way; and the largest single effect
in the run arrives together with the model abandoning English.

`results/crossgame-decision-vectors` built the six vectors and says in its own
first section that **nothing there is steering** — no vector was ever added to a
residual stream. `results/dictator-decision-vector` added one, on one game. This
directory closes the missing cell: each game is steered with **its own** vector,
at layer 20, on its own evaluation prompt, over one shared ladder of intervention
sizes, with a matched shuffled-label null per game.

It makes **no cross-game claim**. No game is steered here with another game's
vector or with the pooled vector. Whether the six directions agree is what the
vector study measured, and its answer was no.

Everything below is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa`, layer 20,
`positions="all"`, `norm="unit"`, `altruism_v3` questions in mode `free`, seed 0,
**n = 100 per point**, strict pole policy. 84 points, 8,400 generations, one
uninterrupted run. `METHOD.md` is the construction and carries the justification
for the two open choices; this file is the result.
§ 10 is the figures, and they carry every verdict below. § 11 is the relaxed
arm, run afterwards, which changes two of the six verdicts.

---

## 0. The two choices, and what the other one would have cost

**Pole policy: strict.** It is what the Dictator-only sweep steered under (so this
run extends that axis rather than opening a second one), it is the vector study's
own primary, and it is the only policy whose pole has the same shape in all six
games. What running relaxed instead would change is measured rather than asserted
— the angle between each game's strict and relaxed vector at layer 20:

| game | strict norm | relaxed norm | cosine | angle | what that means |
|---|---|---|---|---|---|
| prisoners_dilemma | 8.4669 | 8.4669 | +1.0000 | 0.0° | the same tensor; its answer space has no middle for a policy to move |
| apology | 8.0029 | 7.7990 | +0.9995 | 1.7° | immaterial |
| dictator | 7.5047 | 6.9216 | +0.9983 | 3.4° | immaterial |
| trust | 5.3733 | 4.7255 | +0.9672 | 14.7° | borderline |
| **ultimatum** | 5.3665 | 7.1660 | +0.7982 | **37.0°** | a different direction |
| **overfishing** | 2.2738 | 2.1486 | +0.6564 | **49.0°** | a different direction |

So for three of six games the policy is immaterial. **For Ultimatum and
Overfishing a relaxed arm would be a different experiment, not a rescaling of this
one**, and nothing here transfers to it — which matters, because those are exactly
the two games whose result below is qualified. Under relaxed both are
better-estimated directions (the vector study puts the Ultimatum's effective n at
15.1 → 92.3 and Overfishing's at 28.9 → 73.6). Running it is one more
`POLICY=relaxed` invocation of the same pipeline — which owns every output path,
so the two rounds sit side by side rather than one landing on the other — and
another full sweep of card time. **It has since been run — § 11**, on five games
of the six; § 8 has the invocation. It did not leave the strict result standing:
the Ultimatum's verdict changes, and so does the Dictator's.

**The reference norm for the ladder: the altruism trait vector's layer-20 norm,
10.5083.** The six vectors' own norms span **3.7×** (Overfishing 2.2738 to PD
8.4669), so a shared *raw* beta would push the six games by six different amounts
— which would destroy exactly the comparison this run exists to make. Every point
is placed at `unit_beta = k × 10.5083` with `norm="unit"`, so one `k` is the same
sized activation edit in every game. Any shared constant achieves that; this one
also puts the six games on the same axis as the Dictator-only sweep and as the
authors' own raw beta `k`, for free, and does not change if a seventh game is ever
added. Both the unit beta and the per-game equivalent raw beta are in
`provenance/coefficients.csv`.

---

## 1. The whole result on one axis

`P(altruistic pole)`, strict, decision arm. Same definition and same orientation
in every game — this is the one row that can honestly sit in a shared table.
n = 100 per cell; the game's own measure is in §3.

| game | k=−5 | −4 | −3 | −2 | −1 | **0** | +1 | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dictator | 0.020 | 0.060 | 0.040 | 0.020 | 0.050 | **0.140** | 0.465 | **0.526** | 0.412 | 0.266 | 0.116 |
| trust | 0.011 | 0.031 | 0.084 | 0.051 | 0.062 | **0.337** | 0.333 | 0.367 | 0.449 | 0.418 | **0.520** |
| ultimatum | 0.071 | 0.100 | 0.060 | 0.101 | 0.061 | **0.212** | 0.469 | 0.490 | 0.510 | **0.541** | 0.515 |
| apology | 0.000 | 0.000 | 0.011 | 0.031 | 0.052 | **0.400** | 0.850 | **1.000** | 0.980 | 0.970 | **1.000** |
| overfishing | 0.023 | 0.034 | 0.253 | 0.542 | 0.853 | **0.959** | 0.960 | 0.980 | 0.980 | 0.970 | 0.950 |
| prisoners_dilemma | 0.030 | 0.030 | 0.010 | 0.020 | 0.000 | **0.000** | 0.040 | 0.100 | 0.240 | 0.622 | **0.907** |

**Read the baseline column first.** Two of the six games start at an extreme:
Overfishing is already at 0.959 (the model almost always catches exactly the
sustainable 50) and the Prisoner's Dilemma at 0.000 (it never cooperates). Neither
of those games can move in both directions from where it starts, and the flat half
of each of their curves is a ceiling and a floor, not a failed intervention.

## 2. Is it monotone in beta?

On `P(altruistic)`, strict. `monotone up to noise` means no adjacent step goes
against the overall direction by more than its own 95% Newcombe interval — the
form of the claim this data supports; a bare yes/no over 11 noisy points does not.

| game | Spearman ρ vs k | direction | monotone up to noise | significant reversals |
|---|---|---|---|---|
| dictator | +0.706 | increasing | **no** | +3→+4 (−0.146), +4→+5 (−0.150) |
| trust | +0.955 | increasing | yes | none |
| ultimatum | +0.891 | increasing | yes | none |
| apology | +0.950 | increasing | yes | none |
| overfishing | +0.834 | increasing | yes | none |
| prisoners_dilemma | +0.703 | increasing | yes | none |

**All six run the same way**: more positive beta, more of the altruistic pole.
**Five of six are monotone up to noise. The Dictator is not** — it peaks at k=+2
and then falls back significantly, twice, which is the same shape the Dictator-only
sweep reported for a *different* Dictator vector fit on a different prompt grid.

## 3. Per game

Each block: the vector and its norm, the betas, what moved, and whether it beats
its own shuffled-label null. "beats its null" is the honest bar — moving away from
baseline is not enough, because a same-sized push along a randomly relabelled
direction moves several of these games a long way too (§4).

### Dictator — the effect is real, and it peaks

`decision_dictator_response_avg_diff_cellbalanced_strict.pt`, ‖v‖@20 = 7.5047,
raw-beta equivalent ±1.400 to ±7.002. n = 100 per point, parse coverage 0.94–1.00.

| | k=−5 | k=−1 | **k=0** | k=+1 | **k=+2** | k=+5 |
|---|---|---|---|---|---|---|
| P(gives ≥ half) | 0.020 | 0.050 | **0.140** | 0.465 | **0.526** | 0.116 |
| mean $ given | 2.02 | 5.21 | **11.39** | 38.10 | **43.20** | 35.45 |

Both arms move and both beat the null: at k=+5 decision−null is **+0.116
[+0.052, +0.196]** on the pole and **+14.31 (p = 2.8e−28)** on the mean; at k=−5,
**−0.155 [−0.244, −0.074]** and **−15.51 (p = 2.7e−04)**. The negative arm is
saturated from the first step — k=−1 already carries −0.090 of the −0.120 total.
Past the k=+2 peak the pole share falls back while the mean stays near $35–43,
which is mass parking just below half rather than at half: the same behaviour the
Dictator-only study described for its own vector.

That similarity is not a coincidence and it is worth stating: this vector and the
Dictator-only one are **+0.9831 (10.5°) apart at layer 20** despite being fit on
different prompt grids, and their steering curves land in the same place. This arm
is effectively an independent replication of that sweep.

### Trust — a strong negative arm, a weak positive one

‖v‖@20 = 5.3733. Parse coverage 0.95–0.99.

| | k=−5 | k=−1 | **k=0** | k=+1 | k=+3 | k=+5 |
|---|---|---|---|---|---|---|
| P(sends ≥ half) | 0.011 | 0.062 | **0.337** | 0.333 | 0.449 | 0.520 |
| mean $ sent | 0.53 | 6.73 | **33.45** | 39.27 | 43.23 | 48.16 |

The negative arm is large, immediate and beats the null decisively (k=−5:
**−0.691 [−0.777, −0.578]**, mean **−52.50**, p = 1.2e−40). The positive arm is
the weakest of the four dollar games on the pole metric — it does not clear its
own baseline significantly until **k=+5** (+0.184) — while the mean climbs
steadily and significantly from k=+2 (+8.64, p = 5.2e−03) to k=+5 (+14.71,
p = 4.8e−06). So the model is sending more, without crossing the half-endowment
line as often. Against the null the positive arm is nonetheless decisive
(**+0.498 [+0.386, +0.596]**), because the null pushes trust in the *opposite*
direction (§4).

### Ultimatum — it moves, but not more than its own null. **This is the null result.**

‖v‖@20 = 5.3665. Parse coverage 0.96–1.00.

| | k=−5 | k=−1 | **k=0** | k=+1 | k=+4 | k=+5 |
|---|---|---|---|---|---|---|
| P(offers ≥ half) | 0.071 | 0.061 | **0.212** | 0.469 | **0.541** | 0.515 |
| mean $ offered | 7.10 | 6.90 | **19.74** | 40.35 | 46.38 | 46.13 |

Against its own baseline this looks like every other game: significant in both
directions from the first step, monotone up to noise, +0.329 at the top.
**Against its own shuffled-label null it is nothing.**

| | decision − null, P(altruistic) | decision − null, mean |
|---|---|---|
| k=−5 | −0.179 [−0.522, +0.012] — contains 0 | −9.15, p = 0.32 |
| k=+5 | +0.021 [−0.117, +0.157] — contains 0 | −2.07, p = 0.68 |

A same-sized edit along a randomly relabelled version of the same activations does
what the Ultimatum's decision direction does. **On this evidence the Ultimatum's
movement is not attributable to its decision direction**; it is what an edit of
that size at layer 20 does to this prompt. That is the plainest null in the run
and it should not be softened.

Two things bound it rather than explain it away, and neither is a rescue: the
Ultimatum has the **thinnest strict poles of the six** (self pole 29 rows across 10
of 30 cells, effective n 15.1 — the vector study's own number), and it is one of
the two games whose relaxed vector is a **different direction** (37.0°) fit on a
self pole of 235 rows. Whether the relaxed Ultimatum vector beats its null is
**not established here.** It was the single most worthwhile follow-up in this
directory, it has been run, and the answer is **yes** — § 11.

### Apology — the cleanest effect in the run

‖v‖@20 = 8.0029. Parse coverage 0.92–1.00.

| | k=−5 | k=−1 | **k=0** | k=+1 | **k=+2** | k=+5 |
|---|---|---|---|---|---|---|
| P(transfers ≥ half) | **0.000** | 0.052 | **0.400** | 0.850 | **1.000** | **1.000** |
| mean $ transferred | 0.00 | 4.18 | **25.23** | 46.10 | 50.61 | 51.16 |

Saturating in both directions: every parsed answer transfers at least half from
k=+2 outward, and none does at k=−4 and k=−5. It beats its null by **+0.978
[+0.913, +0.994]** at k=+5 and **−0.570 [−0.663, −0.465]** at k=−5, and the
surface stays intact throughout (trigram-repeat share ≤ 0.09 at every point). If
one game in this set demonstrates an outcome-defined direction steering its own
game, it is this one.

### Overfishing — a real negative arm, and no room at all on the positive side

‖v‖@20 = 2.2738, the smallest of the six by a factor of 2.4. Parse coverage
0.86–1.00.

| | k=−5 | k=−3 | k=−2 | k=−1 | **k=0** | k=+1 | k=+5 |
|---|---|---|---|---|---|---|---|
| P(catches ≤ 50) | 0.023 | 0.253 | 0.542 | 0.853 | **0.959** | 0.960 | 0.950 |
| mean fish caught | 97.67 | 84.77 | 70.03 | 52.72 | **50.04** | 49.13 | 50.56 |

Note that for this game a **higher** own-measure is the **less** altruistic
answer, which is why the report is made on the pole share.

**The positive arm is flat and every point on it is non-significant** — but the
baseline is already 0.959, with a mean of exactly the sustainable share. There is
nowhere to go. This is a ceiling, and reading the flat positive arm as "the vector
does not work" would be wrong.

**The negative arm is one of the largest effects in the run and it is graded, not
a step**: 0.959 → 0.853 → 0.542 → 0.253 → 0.034 → 0.023 across k = −1…−5, with the
mean rising from 50.0 to 97.7 fish. It beats its null by **−0.944 [−0.972, −0.862]**
at k=−5 (the null itself does not move at all: 0.968 at k=−5 against 0.959 at
k=0). So the weakest-normed vector of the six, fit on the game with the worst
split-half reliability, still drives its own game hard in the one direction that
has room.

The caveat: parse coverage falls to 0.86 at k=−5 and 13% of answers there contain
non-Latin script, so the far negative end is measured on degrading text.

### Prisoner's Dilemma — the largest move in the run, and the one to trust least

‖v‖@20 = 8.4669. **Its vector rests on 18 cooperations in 231 direct-tag responses
across 9 of 30 usable cells, effective n 10.2** — the vector study's own numbers,
and a design imbalance it documents at length. Every figure below carries that.

| | k=−5 | k=−1 | **k=0** | k=+1 | k=+2 | k=+3 | k=+4 | k=+5 |
|---|---|---|---|---|---|---|---|---|
| P(Cooperate) | 0.030 | 0.000 | **0.000** | 0.040 | 0.100 | 0.240 | 0.622 | 0.907 |
| parse coverage | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 1.00 | 0.98 | 0.86 |
| share non-Latin script | 0.01 | 0.00 | **0.01** | 0.03 | 0.03 | 0.03 | **1.00** | **1.00** |

**The negative arm has no room**: the baseline is 0.000, and no negative point is
significantly different from it. **The positive arm moves further than anything
else in this run** — 0.000 to 0.907, beating its null by **+0.697 [+0.577,
+0.779]** — but the last two rungs of that climb happen while the model is not
answering in English. At k=+4 and k=+5 **every** answer contains non-Latin script:
the model switches wholesale to Chinese, reasons about the payoff matrix there,
and concludes `选择 合作（C）`. The scorer reads those correctly, but coverage
falls to 0.86 and 14% of k=+5 answers are unparsed.

So the claim this game supports is the smaller one: **P(Cooperate) rises
significantly and with the surface intact at k=+2 (+0.100, p = 1.2e−03) and k=+3
(+0.240, p = 1.8e−07)**, where non-Latin script is at 3%. The 0.622 and 0.907 at
k=+4 and +5 are past the point where this game separates the direction from the
size of the edit, and they are reported for completeness, not as the headline.

P(Cooperate) is a 0/1 outcome, so it is the one own measure in this package read
with Wilson and Newcombe rather than a t interval and Welch — `METHOD.md` § 8. It
is the same quantity as this game's altruistic pole share, and `analysis/points.csv`
and `analysis/steering.json` carry the same interval for both.

## 4. The nulls are not inert, and that is the point

`P(altruistic)` on the shuffled-label arm — the same activations, the same cells,
the pole labels permuted within each cell, steered at the same unit betas.
`*` marks a significant move from that arm's own k=0.

| game | k=−5 | k=0 | k=+5 |
|---|---|---|---|
| dictator | 0.175 | 0.140 | 0.000 * |
| trust | **0.701** * | 0.337 | 0.022 * |
| ultimatum | 0.250 | 0.212 | 0.495 * |
| apology | **0.570** * | 0.400 | 0.022 * |
| overfishing | 0.968 | 0.959 | 0.989 |
| prisoners_dilemma | 0.050 * | 0.000 | 0.210 * |

Four of six nulls move significantly, and **two of them move the wrong way**:
Trust's and Apology's nulls push generosity *up* at k=−5 (0.337→0.701 and
0.400→0.570), and the Ultimatum's null pushes it up at k=+5 to 0.495 — which is
essentially where the real Ultimatum vector gets to (0.515). Only Overfishing's
null does nothing anywhere.

This is why "moved from baseline" is not the finding and "beat its own null" is.
It is also why the comparison has to be at matched **unit** beta: the null vectors
are much shorter than the real ones (1.30–2.69 against 2.27–8.47), so at matched
*raw* beta they would be a far smaller push and would prove nothing.

## 5. Summary

| game | negative arm | positive arm | beats its null | monotone up to noise |
|---|---|---|---|---|
| dictator | real, saturated from k=−1 | real, peaks at k=+2 then declines | **yes**, both ends | no (2 reversals) |
| trust | real and large | real on the mean, weak on the pole until k=+5 | **yes**, both ends | yes |
| ultimatum | moves from baseline | moves from baseline | **no — null at both ends** | yes |
| apology | real, saturates to 0.000 | real, saturates to 1.000 | **yes**, both ends | yes |
| overfishing | real and graded | **no room** (baseline 0.959) | **yes** on the negative end only | yes |
| prisoners_dilemma | **no room** (baseline 0.000) | real from k=+2; text collapses by k=+4 | **yes** on the positive end only | yes |

**Where it is null, plainly:** the Ultimatum against its own null, at both
extremes. Overfishing's entire positive arm. The Prisoner's Dilemma's entire
negative arm. The last two are floor and ceiling effects with an obvious reading;
the Ultimatum is not, and it is a real negative.

## 6. What this does not establish

* **Nothing about mechanism.** The Dictator-only study answered "is the decision
  separation what does the work" with an *orthogonal* null — the shuffled null with
  the decision direction projected out. **That arm is not run here.** Beating a
  shuffled-label null shows the effect is not a generic consequence of an edit of
  that size along a same-data direction. It does **not** show the decision
  separation specifically is the mechanism.
* **Nothing about cross-game transfer.** No game is steered with another game's
  vector or with a pooled vector.
* **Nothing about the relaxed directions** for Ultimatum and Overfishing, which are
  37° and 49° away from what was steered. Those were steered separately and are
  reported in § 11; nothing in §§ 1–5 is a claim about them.
* **Nothing beyond mode `free`.** One elicitation mode; the vector study measured
  the same Dictator question answering very differently under another.
* **Nothing at n = 200.** n = 100 was chosen before the run and its cost is stated
  in `METHOD.md` §6: it resolves every effect above with a wide margin, and it
  cannot settle fine curve shape. The one place that bites is the Dictator's
  peak-and-decline, which is reported as observed with its intervals rather than
  as a fitted shape.
* **Nothing that repairs the Prisoner's Dilemma's 18 rows.** No sample size on the
  steering side fixes a thin extraction.

## 7. What is in this directory

```
README.md                 this file
METHOD.md                 the construction and the two justified choices
vectors/                  every vector steered with, both policies, plus their
                          matched nulls and MANIFEST.json (§ 11)
rows/<game>/<arm>/         one CSV per beta point, 100 rows each, full provenance per
                           row. 47 MB; lands in the follow-up PR, not this one
rows_relaxed/<game>/<arm>/ the same, for the relaxed arm (§ 11); five games, not six.
                           Same follow-up PR
analysis/steering.json    every number in this file, per game / arm / point
analysis/points.csv       one flat row per (game, arm, k)
analysis/steering_relaxed.json, analysis/points_relaxed.csv   the same, relaxed (§ 11)
figures/                  the three figures of § 10, regenerated from steering.json
figures/relaxed/          the same three for the relaxed arm
provenance/               the sweep manifest, the null-vector report, and the run logs
scripts/prompting/        eval_games.py (the games and their pole scales), run_sweep.py
scripts/measurement/      build_nulls.py, analyze_game.py, stats.py
scripts/pooling/          crossgame_tables.py — the six games side by side
scripts/figures/          make_figures.py — the figures, from the analysis alone
scripts/tests/            83 offline tests, in `python -m pytest -q`
scripts/run_steering.sh   the whole pipeline, parameterised by environment
```

The real vectors are **not** copied here. They are loaded by path from
`results/crossgame-decision-vectors/vectors/`, and every generated row records the
path and the SHA-256 of the file that was actually used.

## 8. How to check these numbers

Everything downstream of the rows is CPU-only and runs from the checkout. The row
CSVs land in the follow-up PR (§ 7), so until that lands `analyze_game.py` has
nothing to read; the test suite and the figures (§ 10) run either way, and every
number below was produced from the rows named in `provenance/sweep_provenance.json`.

```sh
python -m pytest -q                       # 513 pass, 6 skipped, repo-wide

PY=<python with torch>
$PY results/crossgame-per-game-steering/scripts/measurement/analyze_game.py \
    --rows-root results/crossgame-per-game-steering/rows \
    --coefficients results/crossgame-per-game-steering/provenance/coefficients.csv \
    --out /tmp/steering.json
$PY results/crossgame-per-game-steering/scripts/pooling/crossgame_tables.py \
    --analysis /tmp/steering.json --out-csv /tmp/points.csv
```

Rebuilding the null vectors needs the activation shards (~7 GB, not committed);
`build_nulls.py` refuses to write one unless it can first rebuild the committed
real vector it is a null of.

Regenerating the rows needs a GPU and about four hours. **`POLICY` owns every
output path**, so each arm reproduces into its own, and running one cannot land on
the other's committed artifacts:

```sh
# the strict round: rows/, analysis/steering.json, analysis/points.csv, provenance/
SAMPLES=100 ACTS=<shards> PY=<python> \
    bash results/crossgame-per-game-steering/scripts/run_steering.sh

# the relaxed round exactly as it was run — five games, not six (§ 11):
# rows_relaxed/, analysis/steering_relaxed.json, analysis/points_relaxed.csv,
# provenance/relaxed/ and provenance/null_vectors_relaxed.json
POLICY=relaxed GAMES=ultimatum,overfishing,dictator,trust,apology \
    SAMPLES=100 DEVICE=1 ACTS=<shards> PY=<python> \
    bash results/crossgame-per-game-steering/scripts/run_steering.sh
```

## 9. Provenance

One uninterrupted run, 2026-08-22 08:01–12:18 UTC, on one NVIDIA A40 shared with
another tenant, first attempt, no OOM, no resume. Per game: dictator 30.0 min,
trust 63.6, ultimatum 32.7, apology 29.7, overfishing 56.0, prisoners_dilemma
44.7. transformers 4.52.3, torch 2.6.0+cu124.

Checked rather than assumed: at `k = 0` the delta is exactly zero for both
vectors, so the decision and null arms must have generated the **same text**, and
they did — every row's continuation, scored answer, value and tag match, in all
six games of both rounds. The row files themselves are not identical and cannot
be: three columns name which vector file was loaded, which is the one thing the
two arms differ in by design. Each null vector was written only after its
committed real vector was rebuilt from the archived activations through the same
function that produced it; max relative per-layer deviation across all six games
was **2.7e−08**, against a 1e−06 refusal threshold
(`provenance/null_vectors.json`).

`provenance/steering_run_notes.md` records the machine and the schedule and is not
part of the pipeline.

## 10. The figures

Three, in `figures/`, all written by `scripts/figures/make_figures.py` from
`analysis/steering.json` alone — no torch, no GPU, no second analysis path, so a
figure cannot disagree with the table it came from.

```sh
python results/crossgame-per-game-steering/scripts/figures/make_figures.py
```

| file | what it is |
|---|---|
| `steering_pole_shares.png` | the headline: `P(altruistic)` for all six games, each real arm with **its own null on the same axes** |
| `steering_own_measure.png` | the same six arms on each game's own measure (dollars, fish, `P(Cooperate)`) |
| `steering_vs_null.png` | decision minus its own null at both extremes, six games on one axis — the bar itself |

Four things are decided by a function rather than written into a caption, so the
rule can be checked instead of the picture trusted. `scripts/tests/test_figures.py`
pins each one, and pins that together they reproduce § 5 exactly.

* **The null is drawn for every game.** § 4 is the reason: four of six nulls move
  significantly and two move the wrong way, so a figure of real arms alone would
  misread as six successes.
* **A supported band is coloured by whether that end beat its null** — blue where
  it did, red where it did not, amber where the comparison settles nothing. The
  band marks significance against the arm's own `k = 0`, and it is computed per
  side, so an arm that works one way only draws one-sided rather than being
  smoothed into a symmetric band. The Ultimatum's bands are red and amber, never
  blue, which is the point: it moves and it still shows nothing.
* **A ceiling and a floor are marked as such.** An arm has no room when the
  distance from its baseline to the bound is inside the baseline's own Wilson
  half-width. That selects Overfishing's positive arm (baseline 0.959) and the
  Prisoner's Dilemma's negative arm (baseline 0.000), and nothing else.
* **`k = +4` and `k = +5` of the Prisoner's Dilemma are struck out on the figure.**
  Every answer there is non-Latin. They are excluded from its supported band, which
  therefore covers exactly `k = +2` and `k = +3` — the claim § 3 makes. A point is
  struck out when its non-Latin share reaches 0.5; those two are the only points in
  the run that qualify, and both sit at 1.00.

Separately, a point whose parse coverage is under 0.90 is drawn hollow with the
number of answers the scorer actually resolved. That is a caveat, not a
disqualification — Overfishing's `k = -4` and `k = -5` (89 and 86 rows) carry a
large, graded, real effect.

### What drawing it surfaced

**The Ultimatum's null at `k = -5` resolved 8 answers of 100.** Its 92 unparsed
rows are not gibberish — they are long, fluent English that discusses the
bargaining problem and never names an offer, so the scorer has nothing to read.
It is the worst-covered point in the run by a wide margin, and it is one half of
the comparison § 3 quotes as `-0.179 [-0.522, +0.012]`.

That interval is therefore **missing evidence, not evidence of absence**, and the
figures label it `undetermined` rather than folding it in with the null. It does
not rescue the Ultimatum and it is not meant to: the `k = +5` comparison is
measured on healthy arms at both ends (coverage 0.97 and 0.99) and is a clean
`+0.021 [-0.117, +0.157]`. **The Ultimatum still beats its null at no end, and
that verdict now rests on the end that was actually measured.**

## 11. The relaxed arm

The same design, the same ladder, the same reference norm, the same n, the same
null construction and the same seeds. **The only thing that changes is the
vector.** Nothing in §§ 1–9 moves; this is a second arm beside it.

That the comparison really is like-for-like is checked rather than asserted:
**`P(altruistic)` is identical under both pole policies at all 154 points of both
rounds** — relaxed widens only the *self-interested* pole. So the two halves are
read on the same measure, and the difference between them is the direction that
was added, not the way the answer was scored.

Five games were run. **The Prisoner's Dilemma was not**, and that is a reuse, not
a result: its relaxed decision vector and its relaxed null are element-wise
identical to the strict ones — max absolute difference exactly **0.0** across all
29 layers, its answer space being two points with no middle — so every row would
have been a bit-identical regeneration. `vectors/MANIFEST.json` marks both pairs
`identical_to_strict`. **Its § 3 numbers are its relaxed numbers. They are not an
independent run.**

### 11.1 What changed

| game | angle to strict | strict verdict | relaxed verdict |
|---|---|---|---|
| dictator | 3.4° | beats its null at **both** ends | **negative end only** |
| trust | 14.7° | beats at both ends | beats at both ends, far harder |
| **ultimatum** | **37.0°** | **beats at neither end** | **beats at both** (positive end unusable — 11.3) |
| apology | 1.7° | beats at both ends | beats at both ends |
| overfishing | 49.0° | negative end only | negative end only |
| prisoners_dilemma | 0.0° | positive end only | *reused, identical vector* |

`P(altruistic)`, decision arm, relaxed vectors:

| game | k=−5 | −4 | −3 | −2 | −1 | **0** | +1 | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dictator | 0.010 | 0.031 | 0.040 | 0.010 | 0.030 | **0.140** | 0.612 | 0.612 | 0.438 | 0.351 | 0.234 |
| trust | 0.056 | 0.075 | 0.054 | 0.065 | 0.075 | **0.337** | 0.490 | 0.690 | 0.776 | 0.879 | **0.939** |
| ultimatum | 0.100 | 0.090 | 0.130 | 0.080 | 0.040 | **0.212** | 0.398 | 0.415 | 0.511 | 0.571 | 0.697 |
| apology | 0.011 | 0.011 | 0.021 | 0.010 | 0.071 | **0.400** | 0.910 | 0.980 | 0.990 | 1.000 | 0.990 |
| overfishing | 0.056 | 0.143 | 0.374 | 0.553 | 0.918 | **0.959** | 0.959 | 0.970 | 0.980 | 0.960 | 0.990 |

### 11.2 The Ultimatum: the answer is yes, on the end that was measured

This was the question the round existed to settle, and it settles it.

| | decision − null, `P(altruistic)` | both arms healthy? |
|---|---|---|
| k=−5 | **−0.324 [−0.432, −0.205]** | **yes** — 100 and 99 answers parsed, 0% non-Latin |
| k=+5 | +0.697 [+0.326, +0.789] | **no** — the null parsed 7 of 100, all non-Latin |

**The negative end is a clean, unqualified win, and it is the finding.** The
relaxed direction — 37.0° away from the strict one, fit on 487 rows against 281 —
moves the Ultimatum's decisions further than a same-sized push along a random
relabelling of the same data. The strict direction did not.

The positive end is **not** reported as a second win. Its interval clears zero by
a distance, but the null arm there stopped answering in English and stated an
offer in 7 answers of 100; that is a missing distribution, not a weak one, and an
interval computed across it is not evidence about the direction either way.

Symmetrically, § 3's strict `−0.179 [−0.522, +0.012]` was never a null: **its**
null arm parsed 8 of 100 (§ 10). Each policy has exactly one usable extreme for
this game, and they are opposite ends. On the usable end, strict shows nothing
(`+0.021 [−0.117, +0.157]`, both arms healthy) and relaxed shows a real effect.

The relaxed vector also degrades the text faster on the positive arm: coverage
0.90, 0.84, 0.76 at k=+3, +4, +5, against 0.98, 0.98, 0.97 for the strict one.

### 11.3 The null is not a fixed reference, and the Dictator shows it

**The Dictator's two real vectors are 3.4° apart — the same direction for any
practical purpose — and its verdict still changes.**

| dictator, k=+5 | decision arm | its null | contrast |
|---|---|---|---|
| strict | 0.116 | **0.000** | **+0.116 [+0.052, +0.196]** — beats |
| relaxed | 0.234 | **0.351** | −0.116 [−0.240, +0.013] — does not |

Both arms are healthy at both ends, so this is a real disagreement and not a
degraded comparison. **The mover is the null.** The decision arm shifts 0.116 →
0.234; the null shifts 0.000 → 0.351. The relaxed null is a different artifact —
a different within-cell permutation over the wider relaxed self pole, 612 rows
against 562 — and re-drawing it is enough to overturn a `+0.116` margin.

This does not touch the large margins: Apology's `+0.802`, Trust's `+0.839`,
Overfishing's `−0.394` and the Ultimatum's `−0.324` all survive a re-drawn null
comfortably. What it does say is that **"beats its own null" rests on one null
draw at n = 100**, and a verdict whose margin is of order 0.1 is not robust to
that draw. The Dictator's positive arm was such a verdict. Both rounds use one
null seed (20260821); neither can separate the direction from the draw.

Round 1's § 4 said four of six nulls move significantly. Under relaxed the nulls
move at least as much, and **Overfishing's — the one § 4 singles out as inert —
is no longer inert**: 0.959 → 0.450 at k=−5, against 0.968 under strict. That is
why the same qualitative win shrinks from −0.944 to −0.394.

### 11.4 Per game, plainly

* **Ultimatum** (‖v‖ 7.1660, 37.0° from strict, 487 rows). **Beats its null on
  the negative end, −0.324 [−0.432, −0.205], both arms healthy.** Monotone up to
  noise, ρ = +0.845. The positive end's +0.697 is not usable (null parsed 7/100,
  all non-Latin). **A change from strict, and the round's result.**
* **Trust** (‖v‖ 4.7255, 14.7°, 406 rows). Beats its null at both ends and much
  harder than strict: **−0.857 [−0.910, −0.756]** and **+0.839 [+0.740, +0.895]**.
  The positive arm that strict could barely move off baseline until k=+5 (0.520)
  reaches **0.939**. Monotone up to noise, ρ = +0.943. The clearest gain of the
  round after the Ultimatum.
* **Apology** (‖v‖ 7.7990, 1.7°, 651 rows). Beats its null at both ends,
  −0.815 [−0.881, −0.710] and +0.802 [+0.703, +0.868]. Reproduces strict, as a
  1.7° change should. Its null at k=−5 parses 86 of 100 with 30% non-Latin — a
  caveat on that point, not a collapse.
* **Overfishing** (‖v‖ 2.1486, 49.0°, 1425 rows). Same verdict as strict:
  **−0.394 [−0.497, −0.279]** on the negative end, nothing on the positive one,
  where the baseline is still 0.959 and there is still no room. Monotone up to
  noise, ρ = +0.970. Cleanest coverage of any Overfishing arm so far (0.90 at
  k=−5 against 0.86). One difference worth naming: on the game's **own measure**
  the relaxed positive arm does move — mean catch 50.04 → 35.20, −14.84 fish
  (p = 3.2e−13), where strict did nothing (+0.52, p = 0.75). The ceiling on the
  pole share was hiding real movement. It is still not a win: its null pushes
  catch further down (27.83), so decision − null on the mean is **+7.37 the wrong
  way** (p = 6.8e−03). Unhiding the movement made the positive-arm null more
  decisive, not less.
* **Dictator** (‖v‖ 6.9216, 3.4°, 612 rows). **Beats its null on the negative end
  only**, −0.230 [−0.324, −0.143]. The positive end does not (11.3). Still not
  monotone up to noise (ρ = +0.752), still peaking early — at k=+1/+2 (0.612)
  rather than strict's k=+2 — and still declining after.
* **Prisoner's Dilemma.** Not run. Identical vector and identical null; § 3
  stands unchanged as its relaxed result too.

### 11.5 What the relaxed arm does not establish

Everything in § 6 still applies, and two things are specific to this arm:

* **It is not a replication of the strict arm.** Where the two disagree — the
  Ultimatum and the Dictator — the vector, the null, and the rows all differ.
  Only for the Prisoner's Dilemma is the vector held fixed, and there nothing was
  re-run at all.
* **Nothing about null stability.** One null seed per policy. 11.3 shows a
  verdict turning on the draw; measuring how often that happens needs several
  seeds, and that was not run.
