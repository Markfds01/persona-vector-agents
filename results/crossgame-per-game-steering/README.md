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
`POLICY=relaxed` invocation of the same pipeline and another full sweep of card
time; it was not run.

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
**not established here and is the single most worthwhile follow-up in this
directory.**

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
significantly and with the surface intact at k=+2 (+0.100, p = 1.3e−03) and k=+3
(+0.240, p = 2.0e−07)**, where non-Latin script is at 3%. The 0.622 and 0.907 at
k=+4 and +5 are past the point where this game separates the direction from the
size of the edit, and they are reported for completeness, not as the headline.

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
  37° and 49° away from what was steered.
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
vectors/                  the six matched shuffled-label null vectors (strict, seed 20260821)
rows/<game>/<arm>/         one CSV per beta point, 100 rows each, full provenance per row
analysis/steering.json    every number in this file, per game / arm / point
analysis/points.csv       one flat row per (game, arm, k)
provenance/               the sweep manifest, the null-vector report, and the run logs
scripts/prompting/        eval_games.py (the games and their pole scales), run_sweep.py
scripts/measurement/      build_nulls.py, analyze_game.py, stats.py
scripts/pooling/          crossgame_tables.py — the six games side by side
scripts/tests/            39 offline tests, in `python -m pytest -q`
scripts/run_steering.sh   the whole pipeline, parameterised by environment
```

The real vectors are **not** copied here. They are loaded by path from
`results/crossgame-decision-vectors/vectors/`, and every generated row records the
path and the SHA-256 of the file that was actually used.

## 8. How to check these numbers

Everything downstream of the rows is CPU-only and runs from the checkout:

```sh
python -m pytest -q                       # 473 pass, 2 skipped, repo-wide

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
real vector it is a null of. Regenerating the rows needs a GPU and about four
hours: `SAMPLES=100 ACTS=<shards> PY=<python> bash
results/crossgame-per-game-steering/scripts/run_steering.sh`.

## 9. Provenance

One uninterrupted run, 2026-08-22 08:01–12:18 UTC, on one NVIDIA A40 shared with
another tenant, first attempt, no OOM, no resume. Per game: dictator 30.0 min,
trust 63.6, ultimatum 32.7, apology 29.7, overfishing 56.0, prisoners_dilemma
44.7. transformers 4.52.3, torch 2.6.0+cu124.

Checked rather than assumed: each game's `k = 0` decision and null runs are
**byte-identical** (the delta is exactly zero for both vectors, so the arms share
an origin) — true for all six. Each null vector was written only after its
committed real vector was rebuilt from the archived activations through the same
function that produced it; max relative per-layer deviation across all six games
was **2.7e−08**, against a 1e−06 refusal threshold
(`provenance/null_vectors.json`).

`provenance/steering_run_notes.md` records the machine and the schedule and is not
part of the pipeline.
