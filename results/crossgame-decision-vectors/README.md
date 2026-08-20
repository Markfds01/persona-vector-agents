# Six games, six directions: there is no shared decision axis

**The negative result is the result.** Building an outcome-defined decision vector
for six different economic games and asking whether they point the same way gives:
no. What the per-game directions agree about is the **surface of the answer**, not
the decision. The four games answered in dollars agree strongly with each other;
the two answered in anything else — fish counts, and the words Cooperate/Defect —
sit inside their own null against everything, including each other. Pooling the
six does not remove the confound, and **balancing the pool by game makes one
measure of it slightly worse**. Projecting out a direction fit on the Dictator
game alone destroys essentially all of the pooled vector's separating power.

One thing does survive, and it is the only positive here: a direction fit on five
games separates the sixth game's poles **at depth but not at layer 0** — the
Prisoner's Dilemma goes 0.595 at layer 0 to 0.969 at layer 20 under a direction
that never saw it. That is cross-surface and it is real, but it does not show up
as vector alignment, which points at a shared *subspace* rather than a common
axis.

**Nothing here is steering.** No vector was added to the residual stream, no
generation was run under an intervention, and no number in this file is a causal
effect. This is extraction, alignment and transfer measurement only. `METHOD.md`
is the construction; this file is the result.

Everything below is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa`, `response_avg`
activations, seed 0 for generation and 20260820 for analysis. 4560 generations,
4503 scored and captured, 0 dropped. **Strict pole policy unless stated**; the
relaxed policy is carried end to end and agrees on every conclusion.

---

## 1. The agreement matrix

Per-game cell-balanced vectors, `response_avg`, **layer 20**, cosine.

```
          dict   trust   ultim    apol    fish      pd
 dict     1.000   0.791   0.511   0.842   0.217   0.339
 trust    0.791   1.000   0.488   0.643   0.203   0.396
 ultim    0.511   0.488   1.000   0.419   0.128   0.135
 apol     0.842   0.643   0.419   1.000   0.164   0.228
 fish     0.217   0.203   0.128   0.164   1.000   0.066
 pd       0.339   0.396   0.135   0.228   0.066   1.000
```

Each pair against its own **label null** — the pole labels permuted within each
game and both vectors rebuilt, 300 draws (`METHOD.md` §8):

| pair | cos | null mean | null sd | null p97.5 | verdict |
|---|---|---|---|---|---|
| dictator–apology | +0.842 | −0.003 | 0.186 | 0.349 | **above** |
| dictator–trust | +0.791 | +0.011 | 0.169 | 0.344 | **above** |
| trust–apology | +0.643 | +0.005 | 0.164 | 0.328 | **above** |
| dictator–ultimatum | +0.511 | −0.007 | 0.165 | 0.287 | **above** |
| trust–ultimatum | +0.488 | −0.001 | 0.166 | 0.324 | **above** |
| ultimatum–apology | +0.419 | +0.007 | 0.181 | 0.338 | **above** |
| trust–PD | +0.396 | +0.006 | 0.173 | 0.332 | above, marginal |
| dictator–PD | +0.339 | −0.003 | 0.148 | 0.263 | above, marginal |
| apology–PD | +0.228 | −0.005 | 0.139 | 0.275 | within null |
| dictator–overfishing | +0.217 | −0.004 | 0.141 | 0.297 | within null |
| trust–overfishing | +0.203 | +0.010 | 0.188 | 0.359 | within null |
| apology–overfishing | +0.164 | +0.009 | 0.139 | 0.291 | within null |
| ultimatum–PD | +0.135 | −0.003 | 0.141 | 0.264 | within null |
| ultimatum–overfishing | +0.128 | +0.006 | 0.132 | 0.256 | within null |
| **overfishing–PD** | **+0.066** | +0.005 | 0.163 | 0.329 | within null |

**The split is exactly the answer-surface split.** All six pairs among the four
dollar games clear their null. Every pair involving overfishing falls inside it.
The two games that share no answer tokens with the dollar games share nothing with
*each other* either: overfishing–PD is +0.066, the lowest cell in the matrix.

### These are label nulls, and reading them wrong changes the answer

The null permutes the **labels** and rebuilds both sides from the same
activations. It asks "could a random split of these same rows produce this much
agreement". It is **not** a null of two unrelated directions.

Two things follow, and both are measured rather than assumed:

* **The null is centred at zero** — mean −0.007 to +0.011 across the 15 pairs —
  **but it is 8x to 11x wider than an orthogonality null.** Its sd is 0.13 to
  0.19, against the theoretical `1/sqrt(3584) = 0.0167` for two random directions
  in this space. So the bar a cosine has to clear is the null's **p97.5 of 0.26 to
  0.36**, not the ≈0.03 a random-direction argument would give. Under the wrong
  bar, +0.128 and +0.164 look like agreement. They are not.
* Where two vectors **share rows** — a pooled vector against a game inside it —
  the same construction is centred at **+0.40**, not zero, and its p97.5 is 0.60
  to 0.65. Section 5 uses that version, and the difference is the whole reason
  those thresholds look so much higher.

This is the same lesson the Dictator directory records. It is restated because
both headline claims here are readings against a null.

### It is not attenuation

Each game's vector was rebuilt from a disjoint random half of its own rows; the
split-half reliability at layer 20 is dictator 0.969, apology 0.978, trust 0.948,
PD 0.899, ultimatum 0.895, overfishing 0.866. Correcting each cosine by
`cos / sqrt(rel_a * rel_b)` barely moves it — dictator–overfishing 0.217 → 0.237,
overfishing–PD 0.066 → 0.075. The vectors are measured well enough to say they
genuinely do not align.

(Reliability depends on the random half. An independent split during packaging,
different seed, gave 0.785 to 0.988 across the six with the same ordering
conclusion. It is a magnitude correction, never mixed with the nulls above, which
are computed on uncorrected cosines.)

### The same matrix at layer 0

A weaker version of the same shape: dictator–apology +0.748, dictator–trust
+0.581, trust–apology +0.467, ultimatum–overfishing −0.059, apology–PD −0.028,
**overfishing–PD −0.081**. That is what you expect if layer 20 is partly carrying
the same surface information layer 0 carries outright.

## 2. Separation per game, and the pooled layer-0 test that failed

All AUCs are **out of sample**: the direction is fit on half the rows and scored
on the other half, per layer (`METHOD.md` §7).

| | layer 0 | layer 20 | best layer |
|---|---|---|---|
| **pooled (cell-balanced)** | **0.906** | **0.917** | 0.928 @10 |
| dictator | 0.933 | 0.947 | 0.947 @20 |
| trust | 0.830 | 0.936 | 0.939 @19 |
| ultimatum | 0.861 | 0.871 | 0.878 @26 |
| apology | 0.918 | 0.982 | 0.985 @24 |
| overfishing | 0.678 | 0.724 | 0.724 @20 |
| prisoners_dilemma | 0.931 | 0.934 | 0.950 @2 |

**Layer 0 did not collapse.** It went from 0.903 (Dictator-only, archived) to
0.906 (pooled across six games). That was the test the whole design existed to
run, and the confound survived it.

The reason is in the cell counts. The pooled cell-balanced vector is built from
**111 usable cells, and 85 of them are dollar cells** (dictator 30, apology 25,
trust 20, ultimatum 10) against 17 overfishing and 9 PD. Pooling did not remove
the shared token signature because the pool never stopped being mostly dollars:
`cos(pooled, archived Dictator-only) = +0.959` at layer 20, against a label null
p97.5 of 0.424. The pooled vector is very nearly the Dictator vector.

The PD row of that table rests on **18 cooperating responses** — see §8b before
reading anything into it.

## 3. The mechanism, shown rather than asserted

Layer 0 of `response_avg` is literally the mean **input embedding** of the response
tokens: no attention, no MLP, nothing has computed. So the direction lives in
embedding space and can be read back in token space by cosine against the
embedding matrix — CPU only, no forward pass.

The lists below are the **top ten and top eight in order**, not a selection:

* **pooled, cell-balanced** — altruistic: `'5'`+0.372, `'6'`+0.257, `'4'`+0.242,
  `'7'`+0.234, `' fair'`+0.217, `' a'`+0.196, `'2'`+0.188, `'A'`+0.183,
  `'8'`+0.178, `'５'`+0.158. Self-interested: `' Agent'`−0.289, `' no'`−0.223,
  `' any'`−0.223, `','`−0.183, `' '`−0.161, `'、'`−0.154, `' payoff'`−0.151,
  `'，'`−0.151.
* **archived Dictator-only** — the same shape: `'5'`+0.335, `'6'`+0.225,
  `' fair'`+0.210, `'A'`+0.206, `'7'`+0.193, `'4'`+0.189, `' a'`+0.178,
  `'2'`+0.166, `' common'`+0.158, `'1'`+0.156; against `' payoff'`−0.259,
  `' Agent'`−0.220, `','`−0.219, `' '`−0.197, `' rational'`−0.192, `' to'`−0.184,
  `' no'`−0.171, `')'`−0.159.
* **overfishing** — `' fish'`+0.291, `' catch'`+0.215, `'**:'`+0.168,
  `' risk'`+0.162, `' caught'`+0.158, `'###'`+0.156, `' **'`+0.152, `'Fish'`+0.150,
  `' stock'`+0.134, `')**'`+0.132; against `','`−0.208, `' and'`−0.195,
  `'0'`−0.187, `' for'`−0.170, `' a'`−0.167, `' their'`−0.155, `' the'`−0.153,
  `' will'`−0.149. Note the **inverted numeric polarity**: in the dollar games
  digits point at the altruistic pole, here `'0'` points at the self pole, because
  restraint is the SMALL number in a fishery and the LARGE number in a transfer.
* **prisoners_dilemma** — self-interested: `' Agent'`−0.340, `' Def'`−0.285,
  `'ect'`−0.253, `'D'`−0.232, `' $'`−0.231, `' defect'`−0.217, `' agent'`−0.213,
  `' the'`−0.209. **The altruistic side is not clean and should not be quoted as
  if it were**: its top tokens are rare CJK fragments (`'选择'`+0.183,
  `'브'`+0.171) with `' Cooper'`+0.145 only third, then more CJK. That is what the
  mean of **18** input embeddings looks like — the defect side is 213 rows and is
  crisp, the cooperate side is 18 rows and is mostly noise.
* **shipped `altruism`** — `' others'`+0.252, `' community'`+0.246,
  `' communities'`+0.226, `' to'`+0.224, `' helping'`+0.197, `' altru'`+0.192,
  `' support'`+0.181, `' can'`+0.177, `' of'`+0.165, `' comunità'`+0.163; against
  formatting tokens (`' **'`, `' If'`, `'**'`). Trait vocabulary, no digits.

**Each game's layer-0 direction is its own answer vocabulary.** That is the
"surface form of its own answers" hypothesis confirmed directly rather than
inferred, and it explains §1: the dollar games align because they share a digit
signature, and the fishery's inverted polarity is why overfishing aligns with
nobody. Every list also carries punctuation and formatting tokens — layer 0 is the
average of a whole response, not just its answer — which is why the scalar below
exists.

### The scalar version

Fraction of a layer-0 direction's norm lying in the span of the ten digit tokens
`'0'`–`'9'`. The reference is **not** the spherical figure `sqrt(10/3584)=0.053` —
embeddings are not isotropic — but an empirical control: random ten-token
subspaces drawn from the 5657 distinct tokens the model actually emitted in these
responses.

| vector | digit-span share | control mean | control p97.5 (32 draws / 1000) | signed cos with mean(`'4','5','6','7'`) − `'0'` |
|---|---|---|---|---|
| pooled, cell-balanced | 0.419 | 0.102 | 0.145 / 0.191 | +0.275 |
| pooled, game-balanced | **0.426** | 0.098 | 0.140 / 0.177 | **+0.322** |
| archived Dictator-only | 0.387 | 0.123 | 0.166 / 0.200 | +0.253 |
| shipped `altruism` | **0.106** | 0.119 | 0.165 / 0.202 | −0.071 |
| trust | 0.489 | 0.100 | 0.172 / 0.213 | +0.324 |
| apology | 0.422 | 0.107 | 0.164 / 0.205 | +0.130 |
| dictator | 0.342 | 0.119 | 0.157 / 0.211 | +0.208 |
| overfishing | 0.307 | 0.109 | 0.185 / 0.189 | +0.240 |
| ultimatum | 0.287 | 0.109 | 0.163 / 0.189 | +0.162 |
| prisoners_dilemma | 0.246 | 0.134 | 0.223 / 0.220 | +0.068 |

The run drew 32 control subspaces. That is few for a p97.5 and a single draw
containing a digit token inflates it, so the 1000-draw re-estimate made during
packaging is given beside it. No conclusion turns on the difference: the two
pooled vectors and the archived Dictator vector sit at 3.1x to 4.3x their control
mean and clear both p97.5s.

**The shipped `altruism` vector is the clean negative control**: 0.106, *below*
its own control mean, i.e. no digit content at all, and a signed alignment of
−0.071. Together with its token list above, that is a different object from
anything built here.

Every individual game vector clears its own control p97.5 too, but by very
different margins, and the ordering is the expected one. The Prisoner's Dilemma —
whose answers contain no digits at all — sits at 0.246 against a control p97.5 of
0.220 (1000 draws) and 0.223 (32), i.e. 1.8x its control mean against 2.6x to 4.9x
for the five games whose answers are numbers. That is the floor, and it is a
useful check that the statistic is measuring what it claims.

## 4. Leave-one-game-out: the one positive

Fit the cell-balanced direction on five games and score the held-out game's own
poles. The direction never saw that game.

| held out | layer 0 | layer 20 | best | n alt / self |
|---|---|---|---|---|
| trust | 0.931 | 0.968 | 0.969 @19 | 300 / 79 |
| dictator | 0.870 | 0.940 | 0.940 @20 | 167 / 395 |
| apology | 0.824 | 0.937 | 0.937 @20 | 422 / 217 |
| **prisoners_dilemma** | **0.595** | **0.969** | 0.977 @23 | **18 / 213** |
| ultimatum | 0.618 | 0.713 | 0.780 @2 | 252 / 29 |
| overfishing | 0.599 | 0.666 | 0.689 @25 | 1297 / 50 |

The three dollar games transfer well, but they are being scored by a direction fit
mostly on *other dollar games*, so that is largely the surface again. The
interesting rows are the last three, where layer 20 rises well above layer 0:
PD 0.595 → 0.969, overfishing 0.599 → 0.666, ultimatum 0.618 → 0.713.

**There is genuine cross-surface transfer at depth.** A direction that never saw
the Prisoner's Dilemma still separates Cooperate from Defect at 0.969 at layer 20
while managing only 0.595 on the same rows at layer 0. That gap cannot be the
answer tokens, because layer 0 *is* the answer tokens and it is close to the floor
there.

It coexists with, and is smaller than, the surface component, and **it does not
show up as vector alignment** (§1). A shared component that is invisible in the
cosine but visible in the transfer is what a small shared *subspace inside*
game-specific directions looks like, not a single common axis. §6 then shows this
particular result does not survive removing the Dictator direction.

**The PD row is 18 cooperating responses**, 14 of them from a single wording. It
is the most cited number in this file and the thinnest. §8b.

## 5. Balancing the pool by game does not fix it

The obvious objection to §2 is that the pool was weighted by usable cells, and 85
of 111 cells are dollar cells. So the pool was rebuilt giving each game an equal
sixth (`game_equal_unit`: unit-normalise each game per layer, then average), plus
three other weightings as a bracket.

| scheme | cos vs archived Dictator @20 | null p97.5 | cos vs the cell-balanced pool @20 | norm @20 |
|---|---|---|---|---|
| **game_equal_unit** | **+0.901** | 0.488 | **+0.966** | 0.690 |
| game_equal_raw | +0.916 | 0.446 | +0.980 | 4.493 |
| game_precision_unit | +0.957 | 0.574 | +0.994 | 0.802 |
| game_precision_raw | +0.961 | 0.575 | +0.995 | 5.628 |
| cell_balanced (the §1–§4 pool) | +0.959 | 0.539 | 1.000 | 5.066 |

0.959 → 0.901 is a real move — the angle to the Dictator vector opens from 16.4 to
25.7 degrees — but the two pooled vectors are +0.966 to *each other*, and every
downstream measurement says the change is not the one that was needed. At layer 0
it is +0.890 → +0.819. The profile is flat: `cos(game_equal_unit, Dictator)` is
about 0.90 at every layer from 10 to 27 and never drops below 0.82.

**The digit signature did not dilute; on the signed measure it concentrated.**
From the §3 table: digit-span share 0.419 → 0.426 against a control band of
0.14–0.19, and signed alignment with "a mid-size digit rather than zero"
+0.275 → +0.322. Under the relaxed policy the unsigned measure moves the other way
(0.442 → 0.432) — call it unchanged — while the signed measure rises under both
(+0.142 → +0.195).

Why averaging concentrates rather than removes it: averaging six vectors cancels
what is idiosyncratic to each game and keeps what they share, and the digit
component is the shared part. Consistent with that, the game-balanced pool at
0.426 scores *higher* than five of the six individual game vectors it is made of
(§3 table) and is beaten only by trust at 0.489. **That reading is inference, not
a separate measurement** — the alternative, that the pooling arithmetic happens to
favour digits for another reason, is not excluded by anything here.

### Two of the six games still do not align with the pool

`cos(pool, that game's own vector)` at layer 20, against the **within-cell** label
null. The pool *contains* the game, so both the cosine and its null are inflated by
the game's own share — which makes a failure below stronger, not weaker.

| game | game_equal_unit | null p97.5 | verdict | cell_balanced | null p97.5 | verdict |
|---|---|---|---|---|---|---|
| dictator | +0.893 | 0.608 | above | +0.960 | 0.727 | above |
| trust | +0.851 | 0.651 | above | +0.851 | 0.683 | above |
| apology | +0.796 | 0.601 | above | +0.898 | 0.671 | above |
| ultimatum | +0.647 | 0.633 | above, marginal | +0.569 | 0.624 | **within null** |
| **prisoners_dilemma** | **+0.523** | 0.623 | **within null** | +0.446 | 0.670 | **within null** |
| **overfishing** | **+0.430** | 0.616 | **within null** | +0.274 | 0.657 | **within null** |

Leave-one-game-out sharpens it — the pool rebuilt from the other five, so no
shared rows, and the null drops back to being centred at zero:

| held out | cos(pool without it, its own vector) | null p97.5 | verdict | attenuation-corrected |
|---|---|---|---|---|
| dictator | +0.824 | 0.308 | above | 0.911 |
| trust | +0.757 | 0.382 | above | 0.848 |
| apology | +0.676 | 0.320 | above | 0.740 |
| ultimatum | +0.470 | 0.266 | above | 0.662 |
| **prisoners_dilemma** | **+0.314** | 0.315 | **within null** | 0.374 |
| **overfishing** | **+0.204** | 0.348 | **within null** | 0.319 |

Game balancing **moved** the two failing games (overfishing +0.274 → +0.430, PD
+0.446 → +0.523) and moved the Ultimatum from inside its null to just outside it.
It did not move either failing game past its null, under either policy. A vector
that gives overfishing a full sixth of its weight still does not point where
overfishing's own vector points.

The attenuation-corrected column uses within-cell reliabilities (apology 0.969,
dictator 0.965, trust 0.910, PD 0.790, ultimatum 0.543, overfishing 0.436 — lower
than §1's because they measure a *cell-balanced* vector rebuilt from half of each
cell, a much noisier estimate). Even on the most generous correction available the
two non-dollar games are less than half as aligned as the dollar games. It is a
magnitude comparison, not a significance test.

### Nor does it help the transfer

Leave-one-game-out AUC, game-balanced against cell-balanced, strict:

| held out | game_equal_unit L0 | L20 | best | cell_balanced L0 | L20 | best |
|---|---|---|---|---|---|---|
| trust | 0.929 | 0.967 | 0.967 @19 | 0.931 | 0.968 | 0.969 @19 |
| dictator | 0.834 | 0.922 | 0.923 @17 | 0.870 | 0.940 | 0.940 @20 |
| apology | 0.763 | 0.929 | 0.932 @15 | 0.824 | 0.937 | 0.937 @20 |
| prisoners_dilemma | 0.659 | 0.957 | 0.971 @21 | 0.595 | 0.969 | 0.977 @23 |
| ultimatum | 0.626 | 0.696 | 0.801 @28 | 0.618 | 0.713 | 0.780 @2 |
| overfishing | 0.556 | 0.651 | 0.675 @25 | 0.599 | 0.666 | 0.689 @25 |

**Game balancing is worse at layer 20 on all six games.** What it does is lower the
layer-0 AUC on the dollar games (apology 0.824 → 0.763, dictator 0.870 → 0.834),
which is the surface component shrinking — but the layer-20 numbers shrink with
it, so nothing is gained. The §4 positive is *narrower* under game balancing, not
wider: PD's layer-0-to-layer-20 gap goes from 0.595 → 0.969 to 0.659 → 0.957.

Split-half within cell tells the same story: pooled layer 0 goes 0.855 → 0.831 for
the rebalancing, on a direction built out of nothing but token embeddings, and
layer 0 out-scores every layer from 1 to 13 — layer 2 is 0.677, and layer 14
(0.835) is the first to beat it. Per game under `game_equal_unit`, layer 0 / layer
20: trust 0.885 / 0.928, dictator 0.861 / 0.932, apology 0.812 / 0.942,
overfishing 0.554 / 0.641, ultimatum 0.481 / 0.670, and **PD 0.871 / 1.000 on 7
altruistic against 10 self-interested rows, which means nothing** — an AUC of
1.000 on 17 rows is what 17 rows do.

### Decomposition

Split the pool into a dollar-only sub-pool (dictator, trust, ultimatum, apology)
and a non-dollar sub-pool (overfishing, PD), same weighting:

| | cos(pool, dollar-4) | cos(pool, non-dollar-2) | cos(dollar-4, non-dollar-2) |
|---|---|---|---|
| game_equal_unit | 0.945 | 0.652 | 0.368 |
| cell_balanced | 0.990 | 0.508 | 0.379 |

Equal weighting lifted the non-dollar contribution from 0.508 to 0.652. The pool is
still 0.945 to the dollar four, because the dollar four agree with each other and
the other two agree with nobody, including each other.

### Scope of the weighting claim

Five weightings are measured here, and all five balance along the **game** axis or
do not balance at all. They span +0.901 to +0.961 against the Dictator vector, and
every conclusion in this file holds across that span. **This directory makes no
claim about weightings that balance a different axis** — for example equalising
the three answer *formats* (dollar, fish, word) rather than the six games, which
would move weight off the dollar games much harder because four of six games are
dollar games. That was still being measured when this was packaged and is not
included; treat "no weighting changes the answer" as unproven and out of scope
here.

## 6. The decisive test: project the Dictator direction out

Take the leave-one-game-out pool, remove the archived Dictator direction per layer,
and re-run the held-out separation. Strict, `game_equal_unit`:

| held out | L20 before → after | best after | residual norm fraction @20 |
|---|---|---|---|
| apology | 0.929 → **0.313** | 0.482 @10 | 0.527 |
| dictator | 0.922 → **0.500** | 0.543 @27 | 0.546 |
| trust | 0.967 → **0.811** | 0.855 @5 | 0.478 |
| prisoners_dilemma | 0.957 → **0.314** | 0.820 @28 | 0.421 |
| ultimatum | 0.696 → **0.329** | 0.597 @1 | 0.429 |
| overfishing | 0.651 → **0.484** | 0.514 @25 | 0.408 |

**About 41–55% of the vector's length survives and almost none of its separating
power does.** Several games land below 0.5, meaning the residual orders the poles
backwards. The pooled vector adds nothing over the Dictator vector.

The archived Dictator vector was fit on a **separate, earlier Dictator-only
generation run** — no row of this grid entered it — so this removes a direction
estimated from independent data. That makes the collapse more meaningful, not
less: it is not the pool being made orthogonal to a piece of itself.

Two honest qualifications:

* The PD residual reaches 0.820 at layer 28, and under the **relaxed** policy the
  PD residual at layer 20 is 0.821 rather than 0.314 — an enormous swing for a
  pole set that is *identical* under both policies (PD is unaffected by the pole
  policy; only the other five games' poles change). An estimate that moves from
  0.31 to 0.82 because of what happened in other games is 18 rows talking, not a
  finding.
* Trust survives best (0.811). Whatever residual structure exists is not uniformly
  zero.

## 7. Against the vectors that already exist

Layer 20, cell-balanced pooled vector, game-wide label null from 1000 draws.

| against | cos | label null p97.5 | verdict |
|---|---|---|---|
| archived Dictator-only | **+0.959** | 0.424 | above; the pooled vector ≈ the Dictator vector |
| the repo's shipped `altruism` | **+0.110** | 0.184 | **within null — indistinguishable from nothing** |

Per game, against the archived Dictator vector / the shipped altruism vector at
layer 20: dictator +0.983 / +0.109, apology +0.842 / +0.163, trust +0.802 /
−0.013, ultimatum +0.501 / +0.026, PD +0.359 / +0.136, overfishing +0.242 /
−0.143. The game-balanced pool against shipped altruism is +0.067 (null 0.169).

**Nothing built here aligns with the shipped altruism vector.** For reference,
`cos(archived Dictator vector, shipped altruism) = +0.088` at layer 20 — the same
non-relationship — and §3 shows why they are different objects.

**Use the measured null, never the theoretical one.** `1/sqrt(3584) = 0.0167`. The
measured label-null sd at layer 20 is 0.224 against the Dictator vector and 0.100
against the shipped altruism vector: 6x to 13x wider.

## 8. Pole census, and the limits that go with it

| game | rows generated | scored | alt | self | middle | tag-excluded | usable cells | unresolved |
|---|---|---|---|---|---|---|---|---|
| dictator | 720 | 712 | 167 | 395 | 140 | 10 | 30/30 | 1.11% |
| trust | 720 | 699 | 300 | 79 | 309 | 11 | 20/30 | 2.92% |
| ultimatum | 720 | 713 | 252 | 29 | 396 | 36 | 10/30 | 0.97% |
| apology | 720 | 714 | 422 | 217 | 73 | 2 | 25/30 | 0.83% |
| overfishing | 1440 | 1425 | 1297 | 50 | 78 | 0 | 17/30 | 1.04% |
| **prisoners_dilemma** | **240** | 240 | **18** | 213 | 0 | 9 | 9/30 | 0.00% |

Tags: dictator `a2_anchor` 549, `a2_near` 64, `verb_obj` 42, `bare` 37, `bare_int`
8, `unparsed` 8, `keep` 5, `complement` 5, `answer_is` 2. trust `a2_anchor` 439,
`a2_near` 93, `verb_obj` 73, `bare_int` 47, `bare` 35, `unparsed` 21, `complement`
8, `keep` 3, `answer_is` 1. ultimatum `a2_anchor` 520, `a2_near` 82, `verb_obj` 40,
`complement` 27, `bare` 27, `keep` 9, `unparsed` 7, `bare_int` 6, `answer_is` 2.
apology `a2_anchor` 587, `bare` 43, `verb_obj` 43, `a2_near` 37, `unparsed` 6,
`complement` 2, `answer_is` 1, `bare_int` 1. overfishing `fish` 1404, `fish_bare`
21, `unparsed` 15. PD `defect_decl` 184, `defect` 29, `coop_decl` 16, `mixed` 9,
`coop` 2.

### a. Effective n — read this column before any other number

`effective_n = C^2 / sum_cells (1/n_alt + 1/n_self)`: the number of rows a single
unpartitioned two-group difference would need to be as precise as that game's
cell-balanced vector.

| game | usable cells | alt / self rows | rows inside usable cells | **effective n** | equal weight | cell weight | precision weight | norm @20 |
|---|---|---|---|---|---|---|---|---|
| apology | 25 | 422 / 217 | 306 / 217 | **75.8** | .167 | .225 | .310 | 8.00 |
| dictator | 30 | 167 / 395 | 167 / 395 | **75.2** | .167 | .270 | .308 | 7.50 |
| trust | 20 | 300 / 79 | 180 / 79 | **39.2** | .167 | .180 | .161 | 5.37 |
| overfishing | 17 | 1297 / 50 | 735 / 50 | **28.9** | .167 | .153 | .118 | 2.27 |
| ultimatum | 10 | 252 / 29 | 90 / 27 | **15.1** | .167 | .090 | .062 | 5.37 |
| **prisoners_dilemma** | 9 | **18** / 213 | **18** / 52 | **10.2** | .167 | .081 | .042 | 8.47 |

Equal weighting gives the Prisoner's Dilemma one sixth of the vector on an
effective 10.2 rows and the Ultimatum one sixth on 15.1. Three of the six games
carry their share on thin evidence — PD 10.2, ultimatum 15.1, overfishing 28.9 —
and **two of those three, PD and overfishing, are exactly the two that fail every
alignment test in §1 and §5**. Those two facts are not independent and this data
cannot separate them: PD's and overfishing's non-alignment could be about their
answer surface or about their thin poles. That is inference either way, and it is
the single largest unresolved ambiguity in this directory.

Under RELAXED the Ultimatum's self pole widens from 29 rows across 10 cells to 235
across 24, so its effective n goes 15.1 → 92.3 and it becomes the heaviest game
under precision weighting. PD is unchanged by the policy — its answer space has
two points and no middle — and stays at 10.2 either way.

### b. The Prisoner's Dilemma is 18 rows, 14 of them one wording

PD cooperated **18 times in 231 direct-tag responses (7.8%)**, and the split by
wording is not even:

| wording | cooperations |
|---|---|
| `rules_first` | **14 / 47 (29.8%)** |
| `narrative` | 2 / 47 |
| `payoff_lines` | 1 / 46 |
| `second_person` | 1 / 45 |
| `upstream` | **0 / 46** |

A single-wording probe over six *additional* payoff matrices — spanning temptation
`T−R ∈ {1,2}` and sucker risk `R−S ∈ {2,3,9,10,11}`, including matrices where
cooperating cannot lose money — produced **0 cooperations in 96 draws**
(`evidence/pd_probe_6matrices_16samples.csv`). So PD cooperation in this model is
driven by how the question is worded, not by the payoff matrix.

Every PD figure in this file — the +0.066 matrix cell, the 0.969 leave-one-out
transfer, the 0.957 → 0.314 collapse, the 1.000 split-half AUC on 17 rows — rests
on those 18 rows across 9 cells. It is the weakest row and column of every table
here.

### c. PD was generated at 240 rows against 720 and 1440

That part is a **design** imbalance rather than a behavioural one: the run was
planned when PD looked like a corner rather than a vector. The committed
calibration pass (`evidence/calibration_upstream_wording_8samples.csv`, the
`upstream` wording, 8 samples in each of a game's 6 stakes) has PD at **0
cooperations in 48**, and it is the reason for the whole sample allocation —
overfishing got 48 per cell because its self-interested pole ran at **2 of 48**,
the amount games got 24 because both their poles filled (dictator 13 alt / 23
self, apology 17 / 24, trust 16 / 9, ultimatum 13 / 1), and PD got 8 as a census.
It is the cheapest thing in this study to fix.

### d. Both poles are focal points, and pooling cannot remove it

Share of each pole sitting on one single value, recomputed from the row CSVs,
strict, direct-read tags only:

| game | altruistic pole | self-interested pole |
|---|---|---|
| dictator | 80.2% at exactly half (n=167) | 100% at zero (n=395) |
| trust | 66.7% at half (n=300) | 100% at zero (n=79) |
| ultimatum | 73.4% at half (n=252) | 100% at zero (n=29) |
| apology | 60.4% at half (n=422) | 100% at zero (n=217) |
| overfishing | 86.3% at exactly K/2 (n=1297) | 94.0% at exactly K (n=50) |
| prisoners_dilemma | 100% Cooperate (n=18) | 100% Defect (n=213) |

The contrast being measured is therefore closer to **"the answer is the halfway
focal point" versus "the answer is the extreme"** than to graded generosity. Every
game has that same structure, so pooling across games removes a shared *token* but
cannot remove this — it is the one thing balancing by game preserves perfectly.
Any shared direction found in this data could be the focal-point contrast rather
than a decision.

(The overfishing self-interested cell needs care. Under STRICT that pole is "caught
the maximum M", so it is 100% at the maximum *by construction*; the 94.0% is the
share sitting on one single number, `K`, across the six fisheries — 47 of 50, with
the other 3 at `2K` where `M = 2K`. The largest single *raw* catch value holds 23
of the 50.)

### e. Overfishing is the weakest-separating game by a distance

Own layer-20 AUC 0.724 against 0.87–0.98 for the rest, and its self pole is 50 rows
of 1425. Its non-alignment is measured, not an artefact of a failed extraction, but
it rests on a small pole and it has the worst within-cell split-half reliability of
the six (0.436 at layer 20).

### f. Both pole policies were run end to end and they agree

Relaxed pooled layer 0 = 0.877 against layer 20 = 0.874; the same
dollar-games-agree / non-dollar-games-do-not split (dictator–apology 0.841,
overfishing–PD 0.149, all overfishing pairs within null). Where the policies differ
it is the Ultimatum, whose self pole widens from 29 rows to 235, lifting its
leave-one-out separation from 0.696 to 0.887. Nothing about the digit signature,
the two failing games, or the Dictator collapse changes. Full numbers for both are
in `analysis/crossgame_analysis.json` under `by_policy` and in
`analysis/pooled_analysis.json` under `policies`.

### g. Two known scorer bugs

Reported, not corrected here, and both directional: a refusal misclassification in
`audit.parse` that discards some $0 answers, and an `a2_near` case that can read a
figure out of a payoff sentence as the transfer. The first shrinks self-interested
poles; the second can put a row in the wrong pole. Rows resolved along a *derived*
path (`complement`, `keep`, `mixed`) are excluded from the poles entirely, counted
above, and re-run as a sensitivity (`cos_sensitivity_vs_primary`).

### h. Prompt-side variants are a check, not a result

`prompt_last` at layer 0 gives AUC exactly 0.500 — every row in a prompt cell has
an identical activation there, so the measurement is all ties. `prompt_avg` reaches
0.787, which is between-cell pole composition, not decision information. Only
`response_avg` is used for anything above.

### i. The analysis code was validated before it saw this data

`scripts/extraction/selftest_analysis.py` runs the whole battery on synthetic
activations: it recovers a planted shared direction (pooled AUC 1.000, agreement
+0.837 against a null p97.5 of +0.032, leave-one-out 1.000 on both games),
returns chance on pure noise (pooled 0.513, agreement +0.012, leave-one-out 0.539
and 0.537), and confirms a single-poled game is excluded from every pooled
structure and gets no vector. **Re-run during packaging from the committed copy:
all 7 checks pass**, with the numbers above.

## 9. What this does not establish

* **Nothing about steering.** No intervention was run. "The pooled vector is
  0.96 to the Dictator vector" is a statement about two directions, not about what
  either does to behaviour.
* **Not that PD and overfishing fail *because* of their answer surface.** They are
  also the two thinnest games (§8a). The two explanations are confounded in this
  data and cannot be separated with it.
* **Not that the digit component survives *because* it is the shared component.**
  That reading is consistent with the pooled vectors scoring above five of six
  individual ones, but it was not tested separately.
* **Not that a decision direction does not exist.** What is measured is that
  difference-of-means over these six games, under these five weightings, does not
  find one, and that what it does find is dominated by a direction the Dictator
  game alone already gives. §5's last paragraph bounds the weighting claim.
* **Not that the focal-point reading is wrong.** §8d is unresolved and is the
  cheapest thing to test next: build a contrast whose altruistic answer is not the
  halfway focal point and see whether the layer-0 signature follows the focal point
  or the decision.

## 10. What is in this directory

```
README.md                  this file — the result
METHOD.md                  the construction: grid, poles, balancings, nulls
vectors/                   22 vectors, (29, 3584) float32
extraction/                THE GENERATIONS THE VECTORS WERE BUILT FROM — 4560 scored rows
evidence/                  the PD payoff-matrix probe and the calibration pass
analysis/                  every number above, both policies, all layers, all nulls
provenance/                the manifest and the execution logs of every stage
scripts/                   the code as it ran, plus the packaging-time verifier
```

There is **no `rows/` directory and no steering output**, because no steering was
run. `extraction/` is the analogue of the Dictator directory's `extraction/`: the
unsteered answers whose outcomes became the pole labels.

### `vectors/`

22 files, all `(29, 3584)` float32, all cell-balanced, both pole policies — every
vector any claim in this file rests on, and nothing else:

* `decision_<game>_response_avg_diff_cellbalanced_<policy>.pt` — the six per-game
  vectors (12 files)
* `decision_pooled_cell_balanced_response_avg_diff_<policy>.pt` — the pool of §1–§4
* `decision_pooled_game_equal_unit_response_avg_diff_<policy>.pt` — the primary
  game-balanced pool of §5
* `decision_pooled_{game_equal_raw,game_precision_raw,game_precision_unit}_response_avg_diff_<policy>.pt`
  — the weighting bracket (6 files)

**Two sets are deliberately not committed.** The unbalanced (non-cell-balanced)
per-game and pooled vectors exist in the run and their layer-20 norms are quoted
in §8a and in the analysis JSON, but no claim here rests on them and they are 14
more files at 417 KB each. And the `family_balanced` / `non_dollar` sub-pools
belong to the reweighting pass that was still in flight when this was packaged
(§5, "Scope of the weighting claim") — they are not part of this result and none
of the committed analysis describes them.

All 22 were checked bit-identical, tensor by tensor, against the copies this
packaging regenerated from the activations.

### `extraction/`

One CSV per game, 4560 rows total, 35 columns, one row per generation, scored by
`audit.parse` with the resolution path named in its `tag`. These make the pole
counts, the focal-point shares and the PD wording breakdown checkable rather than
merely stated.

### `evidence/`

* `pd_probe_6matrices_16samples.csv` — 96 generations over six additional PD payoff
  matrices, all resolving to Defect. The evidence for §8b.
* `calibration_upstream_wording_8samples.csv` — the calibration pass that set the
  per-game sample counts.

**What is *not* here: the corrupted PD CSV.** During the run, one PD generation
completed and reported success, and the extraction that followed it died with
`_csv.Error: line contains NUL` (`provenance/finish.log`, 15:03:27Z). PD was
regenerated with the same parameters and the second extraction succeeded
(`provenance/finish_pd.log`, 15:13:47Z). A copy was archived under the name
`prisoners_dilemma.corrupt.csv`, **but that file contains zero NUL bytes and is
byte-identical (sha256) to the landed `extraction/prisoners_dilemma.csv`**, and its
mtime precedes the regenerated file's, so it is not a later copy of it either. The
corrupted bytes were not preserved; the archived copy cannot be used to inspect the
corruption, so it is not committed under a name that would claim otherwise. What is
committed is the two logs, which record the failure and the regeneration.

### `analysis/`

| file | what it holds |
|---|---|
| `crossgame_analysis.json` | §1–§4 and §7–§8: both policies, all layers, per-game and pooled, all three agreement matrices with their nulls, leave-one-game-out, split-half, the pole census |
| `crossgame_layer0_tokens.json` | the §3 token decode as the six-game study recorded it: the cell-balanced pool, overfishing and PD |
| `pooled_build.json` | the §8a census, effective n, the five weightings, and the `cell_balanced` rebuild check |
| `pooled_analysis.json` | §5: cosines by layer for all five schemes, in-pool and leave-one-out, both label nulls, leave-one-game-out AUC by layer, split-half within cell, both policies |
| `pooled_addendum.json` | §5's leave-one-out nulls, §6's Dictator-orthogonalised transfer, and the dollar / non-dollar decomposition |
| `pooled_reliability.json` | within-cell split-half reliability, the attenuation-corrected cosines, and the §8d focal-point census |
| `pooled_layer0_tokens.json` | the §3 token decode of seven more: four pooled schemes under strict, two under relaxed, and the archived Dictator vector |
| `verification.json` | the independent recomputation run during packaging, and the only place holding every number in one file — see "How to check these numbers" |

`pooled_digit_share.json` — the run's own record of the §3 scalar table — **could
not be preserved.** It was overwritten in place by a later job before packaging
finished. Its numbers are not lost: `verification.json` reproduces the run's exact
32-draw procedure and returns its values, and `scripts/verify_committed.py` is the
code that does it. The same later job rewrote five of the six pooling scripts, so
`scripts/pooling/` holds only the one that is still byte-identical to what ran; see
"How to rebuild it".

### `provenance/`

`manifest.json` is the per-game record: rows CSV path, activation directory,
generated-row count, tag counts, the activation meta (model id and revision, dtype,
attention implementation, torch and transformers versions, hidden size, layer
count, chat-template hash, device, batch size, padding, shard count, dropped rows,
capture time), and the per-policy pole census, usable cells, per-layer norms and
split-half for both policies.

The logs are the execution trace, and together they cover the whole generation and
extraction run: `main_run.log` (the first pass: overfishing, then the four amount
games, ending in the CUDA OOM that killed `trust`), `resume_run.log` (trust
regenerated from scratch, then the rest), `finish.log` (trust landed, ultimatum,
and the PD generation whose extraction hit the NUL error), `finish_pd.log` (PD
regenerated and landed), `analysis.log` (both policies analysed).
`pooled_addendum.log` is the reweighting stage's addendum run.
`pooled_evaluate.log` is **not** here: the only copy on disk had already been
overwritten by the later job when packaging reached it, so committing it would have
misattributed another run's output to this one.

### `scripts/`

`scripts/extraction/` is the six-game study's code, byte-identical to what ran
(verified against the worktree copies the run actually invoked).
`scripts/pooling/` is the reweighting stage; only `build_vectors.py` survives
byte-identical, and it is the only one of the six that the later job left alone.
`scripts/verify_committed.py` was written during packaging, is not part of either
run, and carries its own usage in its docstring.

## 11. How to check these numbers

**Everything in §1 through §8 was recomputed from the committed data during
packaging, by code that does not import either run's analysis module.**
`scripts/verify_committed.py` reads the activation shards directly, imports only
`crossgame_grid` and `poles` (the grid and pole *definitions*, not the
computation), and rebuilds:

| check | result |
|---|---|
| the six per-game cell-balanced vectors, against the committed `.pt` files | cosine **1.000000** at every layer, all six |
| the 6x6 agreement matrix at layer 20 | matches to 4 decimal places, all 15 pairs |
| leave-one-game-out AUC, cell-balanced pool | matches L0, L20, best and best-layer for all six |
| leave-one-game-out AUC, `game_equal_unit` | matches L0, L20, best and best-layer for all six |
| `cos(pool, Dictator)` and `cos(pool, pool)` at layers 0 and 20 | matches all five schemes |
| the layer-0 digit-span share, the signed digit alignment, and the run's 32-draw control band | matches the run's own record for all 13 vectors, to 4 decimal places |
| the layer-0 token decode | same tokens, same order, same values |
| the Dictator-projected-out collapse, all six games | matches AUC before, AUC after, best-after layer and residual norm fraction |
| the pole census, tag counts, parse coverage, focal shares, effective n | matches all six games |
| the PD wording breakdown and the payoff-matrix probe | 18/231; 14/47 `rules_first`; 0/96 in the probe |

The output is `analysis/verification.json`. The digit-share row compares against
the run's `digit_share.json`, read before that file was overwritten; the
comparison itself is no longer re-runnable and the reproducing code is committed
instead.

Three things the recomputation found that the runs' own write-ups state loosely,
all corrected in this file:

* the agreement-matrix label null is centred at **zero**, not at 0.26–0.35 — that
  range is its p97.5, and the distinction is the whole content of §1's second
  subsection;
* overfishing's self-interested pole is 100% "at the maximum catch" by the strict
  definition, so the 94.0% is the share on one single value, `K` (§8d);
* the PD wording split is 14/47 `rules_first` against 2/47, 1/46, 1/45 and
  **0/46** — one of the four other wordings produced no cooperation at all, where
  the write-up said "1–2" for all four (§8b).

**Independently of that**, the four pooling JSON files here were regenerated
during packaging from the pooling scripts *as they then stood* — before the
overwrite described under "What does not work about the committed code" — and came
back **byte-identical**, as did all 22 vectors (bit-identical tensors). That is
what makes the copies here the verified originals rather than whatever is on disk
now. It is **not** a claim that they re-derive from committed code: five of the
six pooling scripts are not here. The activation cache that stage uses does rebuild
**bit-identically** from the shards, in 7.2 s.

To check anything by hand from what is committed:

* **Pole counts and focal shares** — from `extraction/*.csv` alone. Drop rows whose
  `tag` is `unparsed`/`refusal`/`empty` and rows whose tag is a *derived* path
  (`complement`, `keep`, `mixed`), then apply the §4 rules of `METHOD.md` using
  each row's `game_id` to look up its `pole_scale`.
* **The PD wording breakdown** — group `extraction/prisoners_dilemma.csv` by the
  wording segment of `game_id` and count `value == 1.0`.
* **The vectors' identity** — `torch.load(path)[20].double().norm()` reproduces the
  §8a norm column, and the pairwise cosines reproduce §1.
* **Everything downstream of the activations** — needs `acts/`, which is not
  committed. See below.

## 12. How to rebuild it

| stage | what it does | code | needs |
|---|---|---|---|
| 1. the grid | the 180 prompt cells, with each cell's fingerprint checked against its own text | `scripts/extraction/crossgame_grid.py` | nothing |
| 2. generation | 4560 generations, `neutral` preset, seed 0, batch 32. Output is `extraction/*.csv`. | `scripts/extraction/gen_crossgame.py`, driven by `run_all.sh` / `resume_run.sh` / `main_run.sh` / `finish.sh` / `finish_pd.sh` | GPU + model weights |
| 3. activations | teacher-forced forward pass over prompt+response, batch 1, no padding, three poolings x 29 layers. Writes `acts/<game>/shard_*.pt`, **5.3 GB, not committed**. | `scripts/extraction/extract_crossgame.py` | GPU + model weights |
| 4. per-game landing | one game's vectors + its `manifest.json` entry, landed as soon as it is extracted so a GPU eviction costs one game | `scripts/extraction/land_family.py` | `acts/` |
| 5. the six-game analysis | §1–§4, §7, §8: vectors, agreement matrices, nulls, leave-one-game-out, split-half, census. CPU only. Writes `crossgame_analysis.json`. | `scripts/extraction/analyze_crossgame.py` | `acts/` |
| 6. the token decode | §3's token lists. CPU only, reads the embedding matrix off the safetensors shard, no forward pass. | `scripts/extraction/decode_layer0.py` | model weights on disk |
| 7. the reweighting | §5–§6: the five pooled vectors, their cosines and nulls, the orthogonalisation, the reliabilities, the digit share. CPU only. | `scripts/pooling/build_vectors.py` and the five scripts named below that could not be committed | `acts/` |
| the self-test | the whole §5 battery on synthetic activations with a known answer | `scripts/extraction/selftest_analysis.py` | nothing |
| the PD probe | §8b's six extra payoff matrices | `scripts/extraction/pd_probe.py` | GPU + model weights |
| the calibration | the pass that set the per-game sample counts | `scripts/extraction/calib_report.py` | `evidence/calibration_*.csv` |
| the verification | §11: recomputes §1–§8 from the activations without either run's analysis code. CPU only. Writes `analysis/verification.json`. | `scripts/verify_committed.py` | `acts/`, and the model weights for the §3 part |

### The two big directories, and how to regenerate them

**`acts/` — 5.3 GB, not committed.** Six directories of activation shards plus a
`meta.json` each. Regenerate with stage 3 above, one game at a time, from the
committed `extraction/*.csv`:

```
python scripts/extraction/extract_crossgame.py \
    --rows extraction/<game>.csv --out-dir <somewhere>/acts/<game> --device 0
```

It re-renders each row's prompt and verifies it byte-for-byte against the
`prompt_sha256` the row already carries, so a mismatch is caught rather than
silently producing activations for a different prompt. It needs the model weights
and a GPU; on the hardware this ran on it was 0.4 to 6.2 minutes per game (240 to 1425 rows). The
`rows_csv` and `acts_dir` paths inside `provenance/manifest.json` are the run's
absolute paths and have to be repointed.

**`cache/response_avg.pt` — 1.8 GB, not committed.** Purely derived: the
`response_avg` slice of all six games' shards concatenated into one
`(4503, 29, 3584)` float32 tensor plus `labels.json`, so the reweighting stage does
not re-read 5.3 GB on every invocation. `scripts/pooling/build_vectors.py` (through
`common.load_response_avg`) writes it on first use if it is absent. Measured during
packaging: **7.2 s to rebuild, bit-identical** to the run's copy, labels included.
Do not commit it; delete it freely.

### What does not work about the committed code, and why it was left that way

Three separate problems, none of them patched away.

**1. Five of the six pooling scripts are not here, and it is not a choice.** While
this packaging was running, a *later* job re-ran the reweighting stage in place —
adding two new weighting schemes — and overwrote `common.py`, `evaluate.py`,
`addendum.py`, `reliability.py` and `digit_share.py` in the source directory,
together with `digit_share.json`, `analysis.json` and the evaluate log. Only
`build_vectors.py` is still byte-identical to what produced the §5–§6 numbers, and
it is the only one committed. The originals were not recoverable at packaging time.

The consequence is bounded and stated plainly: **every §5 and §6 number in this
file was independently recomputed by `scripts/verify_committed.py` and matched**,
and `pooled_analysis.json`, `pooled_addendum.json`, `pooled_reliability.json` and
`pooled_build.json` are the verified originals (regenerated from the then-current
scripts during packaging and byte-identical). What is missing is the ability to
re-derive those JSON files from committed code. `verify_committed.py` re-derives
the *numbers* from the activations, which is the check that matters, but it is not
the same thing and it is not being presented as such.

**2. The extraction scripts do not run from where they sit.** They were a package
named `scratch/` directly under the repo root. Every entry point does
`sys.path.insert(0, Path(__file__).resolve().parents[1])` to reach the repo root so
it can `import audit`, and several import each other as `from scratch import ...`.
From `results/crossgame-decision-vectors/scripts/extraction/` neither resolves.

Checked, not assumed — each script with an argument parser was run with `--help`
from the repo root, as committed and again copied back to `scratch/` at the repo
root:

| script | as committed | copied to `scratch/` at the repo root |
|---|---|---|
| `analyze_crossgame.py` | fails, `No module named 'scratch'` | runs |
| `land_family.py` | fails, `No module named 'scratch'` | runs |
| `calib_report.py` | fails, `No module named 'scratch'` | runs |
| `selftest_analysis.py` | fails, `No module named 'scratch'` | runs, and all 7 checks pass |
| `gen_crossgame.py` | fails, `No module named 'audit'` | runs |
| `extract_crossgame.py` | fails, `No module named 'audit'` | runs |
| `pd_probe.py` | fails, `No module named 'audit'` | runs |
| `decode_layer0.py` | **runs** | runs |
| `summarize.py` | **runs** (stdlib only; takes a positional path, no `--help`) | runs |
| `poles.py` | **runs** — stdlib only, import-only module | runs |
| `crossgame_grid.py` | import-only module; needs `audit` on the path, which its importers put there | — |

Two of the eleven work as they sit and one is import-only; the other eight need
the `scratch/` layout. `calib_report.py` and `summarize.py` take a positional path
and do their work at import time, so they were checked by invoking them with no
argument and reading which error came back — an import error means the layout is
wrong, an `IndexError` on `sys.argv[1]` means it got past its imports.

So: copy `scripts/extraction/` to `scratch/` at the repo root and invoke it from
there. `analyze_crossgame.py` additionally defaults `--their-vectors` to the
relative path `persona_vectors/Qwen2.5-7B-Instruct`, so the repo root must also be
the working directory.

**3. The shell wrappers are committed for their invocations, not for re-running.**
`run_all.sh`, `resume_run.sh`, `main_run.sh`, `finish.sh` and `finish_pd.sh` all
hardcode `R=/home/marco/dockmaster/state/worktrees/crossgame-vector`, a worktree
that no longer exists, a `.venv` interpreter inside it, and
`OUT=/home/marco/dockmaster/data/crossgame-vector`. They are the record of what was
actually launched — the sample counts, the batch size, the seed, the device, the
family order and the retry policy — which is why they are here. `main_run.sh`
carries the reasoning for the per-game sample counts in its own header.
`scripts/pooling/build_vectors.py` likewise hardcodes absolute paths to the source
and output directories through `common.py`, which is not committed.

## 13. Provenance

Model `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa` (both pinned,
resolved from the loaded model, and recorded on every row), transformers 4.52.3,
torch 2.6.0+cu124, chat template sha256 prefix `cd8e9439f0570856`, stop tokens
151645 and 151643. `neutral` preset: temperature 1.0, top_p 1.0, top_k 0, min_p
0.0, repetition_penalty 1.0, max_new_tokens 1000, min_new_tokens 1. Seed 0 for
generation, batch size 32, device `cuda:0`. Analysis seed 20260820, 1000 shuffles
for the pooled nulls and 300 per pair for the matrix and the reweighting nulls.
4560 generations, 4503 scored and captured, 0 dropped. Hidden size 3584, 29 hidden
states (embedding + 28 layers).

**The recorded commit does not pin the code that ran.** Every row records
`repo_commit = 4d10f19dfe7436845456f0f8b67d2adef25776a4` with `repo_dirty = true`,
across all 4560 rows. The working tree had uncommitted changes — the scripts were
an untracked `scratch/` directory at the time, which is exactly why they are being
committed here — so the commit identifies the baseline the run started from, not
the code it executed. "The extraction scripts are byte-identical to what ran" is a
statement about the copy they were taken from, verified against the worktree the
run invoked; there is no committed tree from that moment to diff against. This is
the weakest link in this directory's provenance and it cannot be repaired after
the fact.

**The run was interrupted twice and one game was regenerated twice.** The GPU is
shared with another tenant that swings between roughly 21.6 and 27.4 GiB and cycles
models without warning. `main_run.log` records the model load OOMing twice and
retrying before the first pass started, and then a CUDA OOM inside a forward pass
that killed `trust`; `resume_run.log` records `trust` regenerated from scratch, and
a second OOM on its first retry. `finish.log` and `finish_pd.log` record the PD
generation that completed and reported success but whose CSV could not be read
(`_csv.Error: line contains NUL`), and the regeneration that followed. **Each game
is internally seed-consistent** — one run, seed 0, batch 32 — but the set was not
produced by a single uninterrupted invocation, and two games were produced by their
second attempt rather than their first.

The other five CSVs were verified NUL-free with row counts and SHA-256 recorded at
landing time. All six were re-verified NUL-free during packaging, and each
`extraction/<game>.csv` is byte-identical to the `rows.csv` beside the activations
it was extracted from.

**No GPU was used during packaging**, and no model was loaded: the verification in
§11 is CPU-only linear algebra over the archived activations, plus reading the
embedding matrix off the safetensors shard for the §3 decode.

Nothing under `audit/`, `persona_vectors/` or `results/dictator-decision-vector/`
was modified by this work.
