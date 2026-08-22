# Six games, six directions: there is no shared decision axis

**The negative result is the result.** Building an outcome-defined decision vector
for six different economic games and asking whether they point the same way gives:
no. What the per-game directions agree about is the **surface of the answer**, not
the decision. The four games answered in dollars agree strongly with each other;
the two answered in anything else — fish counts, and the words Cooperate/Defect —
sit inside their own null against everything, including each other.

Pooling the six does not remove the confound, and **no weighting of them removes
it either**. Nine weightings are measured here, reaching **66 degrees** away from
the Dictator-only direction — +0.961 (16°) at the near end down to +0.403 (66°) at
the far end, a 50-degree spread — and the whole curve moves as one: every step
away from the dollar games costs surface separation and deep separation together,
and the layer-0 digit signature never falls into its control band at any point on
it. Projecting out a direction fit on the Dictator game alone destroys essentially
all of the pooled vector's separating power **under the strict pole policy**;
under relaxed the same collapse is real but much smaller (§6).

One thing does survive, and it is the only positive here: a direction fit on five
games separates the sixth game's poles **at depth but not at layer 0** — the
Prisoner's Dilemma goes 0.595 at layer 0 to 0.969 at layer 20 under a direction
that never saw it. That is cross-surface and it is real, but §5 shows it is the
four dollar games doing the transferring: drop them and it falls to chance under
strict (PD 0.509) — though that particular cell reads 0.817 under relaxed and
should not be quoted on its own (§5).

**Nothing here is steering.** No vector was added to the residual stream, no
generation was run under an intervention, and no number in this file is a causal
effect. This is extraction, alignment and transfer measurement only. `METHOD.md`
is the construction; this file is the result.

Everything below is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa`, `response_avg`
activations, seed 0 for generation and 20260820 for analysis. 4560 generations,
4503 scored and captured, 0 dropped. **Strict pole policy unless stated**; the
relaxed policy is carried end to end and agrees on every conclusion, though two
individual cells move a long way with the policy and are flagged where they
appear (§5, §6).

---

## 0. Where these numbers come from

Every number in this file was produced by the code in this directory and in
`lab/` and `audit/`, from the rows committed under `extraction/`. That was not
true of the first version of this directory, and closing the gap is what this
revision is.

**One extractor, for every grid.** The teacher-forced capture used to exist twice
— once for the Dictator-only study and once here — and the two copies were only
asserted to agree. Every activation behind every number in this file now comes off
one implementation, `lab/extract.py`, with tests under `lab/tests/`. Nothing in it
is per game: a grid declares its own games and every row names its own `game_id`,
so which prompt is re-rendered and which stakes it carried are read off the data
(`METHOD.md` §5). That is a statement about the pipeline, not the repository:
`results/dictator-decision-vector/scripts/extract_acts.py` is still committed as
the record of the run that produced that directory, and nothing calls it any more
— its corpus was re-extracted through `lab/extract.py` and is in the table below.

**The whole corpus was re-extracted with it**, from the committed rows CSVs — the
six games here (4503 rows) and the earlier Dictator-only grid (1785 rows), so
every vector compared in this file comes off one extractor. The check that this
is the same computation as before is exact rather than distributional, because
the same rows went through it:

| | rows | elements compared | exactly equal | max abs difference |
|---|---|---|---|---|
| the six games, against the archived activations | 4503 | 1,404,071,424 | **all of them** | 0.0 |
| the Dictator-only grid, against its archived activations | 1785 | 556,577,280 | **all of them** | 0.0 |
| **total** | **6288** | **1,960,648,704** | **all of them** | **0.0** |

Bit-identical, all three poolings, all 29 layers, every row — so the shared module
is provably the same computation as both files it replaced, and every vector,
matrix and AUC published before this revision is unaffected by the refactor.
`analysis/activation_equivalence.json` is the record;
`scripts/compare_activations.py` is the code.

**Then everything downstream was rebuilt from the new activations** by
`scripts/run_analysis.sh`, and diffed leaf by leaf against what the first version
published (`scripts/compare_published.py`, output in
`analysis/rebuild_deltas.json`). What moved is reported in §12; nothing moved
enough to touch a conclusion.

It is the same activations throughout — one directory, written by `lab/extract.py`
and reported bit-identical in the table above, read by `scripts/run_analysis.sh`
for every number in this file and read again independently by
`scripts/verify_committed.py` for §11. `provenance/manifest.json` records that
directory per game, together with the `meta.json` the extractor wrote beside the
shards: the module that produced them, the revision, the kernel, and the
`rows_csv_sha256` that pins each rows CSV by content rather than by path.

**And the reweighting is now complete.** The first version of this file measured
five weightings, all of which balance the *game* axis or do not balance at all.
They ran +0.901 to +0.961 against the Dictator vector — 25.7° to 16.1°, under ten
degrees of spread — and that version explicitly declined to generalise from them:
it said this directory made "no claim about weightings that balance a different
axis" and that "no weighting changes the answer" should be treated as unproven and
out of scope. This revision closes that scope rather than correcting a claim.
Balancing the **answer-format** axis — a third each to dollar, fish and word
answers — moves the pooled direction to 44°, and dropping the dollar games
entirely to 66°. §5 carries all nine. The headline is unchanged and is still a
negative.

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
  the same construction is **not centred at zero**, and how far off zero depends
  on how much of the pool that game is. Across all nine weightings and all six
  games at layer 20, strict, its mean runs −0.004 to +0.877 and its p97.5 runs
  0.255 to 0.954: a game carrying half the pool (PD under `non_dollar_raw`) drags
  its own null to +0.877, and a game carrying none of it (a dollar game under the
  `non_dollar` schemes) sits back at zero. The often-quoted "+0.40, p97.5 0.60 to
  0.65" is `game_equal_unit`'s band specifically. §5 therefore reads every cell
  against **its own** null and never against one number.

This is the same lesson the Dictator directory records. It is restated because
both headline claims here are readings against a null.

### It is not attenuation

Each game's vector was rebuilt from a disjoint random half of its own rows. The
matrix above is **cell-balanced**, so the reliability that matches it is the
**within-cell** split-half, and that is the one used here: at layer 20 dictator
0.965, apology 0.969, trust 0.910, PD 0.790, ultimatum 0.543, overfishing 0.436.
Correcting by `cos / sqrt(rel_a * rel_b)` lifts the two failing cells to
dictator–overfishing 0.217 → **0.334** and overfishing–PD 0.066 → **0.113** —
a bigger move than the whole-game reliabilities (0.237 and 0.075) would suggest,
because overfishing's within-cell vector is the noisiest of the six. It is still
nowhere near the dollar games, which are barely moved: dictator–apology 0.842 →
0.870. The correction changes the size of the gap, not its direction.

(§5 reports the same overfishing↔PD correction, 0.113, from the leave-one-out
construction. The whole-game reliabilities are the higher figures because a
whole-game rebuild is free to lean on between-cell composition, which cell
balancing removes; they are the wrong denominator for this matrix. The correction
is a magnitude comparison and is never mixed with the nulls, which are computed on
uncorrected cosines.)

### The same matrix at layer 0

A weaker version of the same shape: dictator–apology +0.748, dictator–trust
+0.581, trust–apology +0.467, ultimatum–overfishing −0.059, apology–PD −0.028,
**overfishing–PD −0.081**. That is what you expect if layer 20 is partly carrying
the same surface information layer 0 carries outright.

**Read the layer-0 row of the three thin games with the reliabilities in hand.**
Within-cell split-half at layer 0 is overfishing **0.158**, ultimatum **0.230**
and PD 0.599, against dictator 0.884 and apology 0.858. Any layer-0 cell involving
overfishing or ultimatum is measured too poorly for its size to mean much — the
attenuation correction on such a cell is a division by a number near zero and is
not quoted here for that reason. The shape of the matrix is the claim; the
individual layer-0 magnitudes of those rows are not.

## 2. Separation per game, and the pooled layer-0 test that failed

All AUCs are **out of sample**: the direction is fit on half the rows and scored
on the other half, per layer (`METHOD.md` §7). Every row here is the **whole-game**
split-half — a random half of the pole rows, unbalanced rebuild — including the
pooled one, so the table is read down a single column. The cell-balanced pool
under the within-cell split is 0.855 at layer 0 and 0.908 at layer 20 (§5); the two
constructions are not comparable and `METHOD.md` §7 says why.

| | layer 0 | layer 20 | best layer |
|---|---|---|---|
| **pooled (whole-game)** | **0.906** | **0.917** | 0.928 @10 |
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
`cos(pooled, Dictator-only) = +0.959` at layer 20, against a label null p97.5 of
0.424. The pooled vector is very nearly the Dictator vector.

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
* **Dictator-only** — the same shape: `'5'`+0.335, `'6'`+0.225,
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
  `'븜'`+0.171) with `' Cooper'`+0.145 only third, then more CJK. That is what the
  mean of **18** input embeddings looks like — the defect side is 213 rows and is
  crisp, the cooperate side is 18 rows and is mostly noise.
* **shipped `altruism`** — `' others'`+0.252, `' community'`+0.246,
  `' communities'`+0.226, `' to'`+0.224, `' helping'`+0.197, `' altru'`+0.192,
  `' support'`+0.181, `' can'`+0.177, `' of'`+0.165, `' comunità'`+0.163; against
  formatting tokens (`' **'`, `' If'`, `'**'`). Trait vocabulary, no digits.

**Each game's layer-0 direction is its own answer vocabulary.** That is the
"surface form of its own answers" hypothesis read directly off the decode rather
than inferred from the cosines. It lines up with §1: the four games that align are
the four that share a digit signature, and the game that aligns with nobody is the
one whose numeric polarity is inverted. **That is a correspondence, not a
mechanism.** The same four games are also the four with the thickest poles, and
this data cannot say which of the two is doing the work — §9 states that
limitation and nothing here overrides it. Every list also carries punctuation and
formatting tokens — layer 0 is the average of a whole response, not just its
answer — which is why the scalar below exists.

### The scalar version

Fraction of a layer-0 direction's norm lying in the span of the ten digit tokens
`'0'`–`'9'`. The reference is **not** the spherical figure `sqrt(10/3584)=0.053` —
embeddings are not isotropic — but an empirical control: random ten-token
subspaces drawn from the 5657 distinct tokens the model actually emitted in these
responses.

| vector | digit-span share | control mean | control p97.5 (32 draws / 1000) | signed cos with mean(`'4','5','6','7'`) − `'0'` |
|---|---|---|---|---|
| pooled, cell-balanced | 0.419 | 0.102 | 0.145 / 0.191 | +0.275 |
| pooled, game-balanced (`game_equal_unit`) | **0.426** | 0.098 | 0.140 / 0.177 | **+0.322** |
| pooled, format-balanced (`family_balanced_unit`) | 0.368 | 0.102 | 0.170 / 0.177 | +0.313 |
| pooled, dollar games dropped (`non_dollar_unit`) | 0.334 | 0.110 | 0.194 / 0.198 | +0.227 |
| Dictator-only | 0.387 | 0.123 | 0.166 / 0.200 | +0.253 |
| shipped `altruism` | **0.106** | 0.119 | 0.165 / 0.202 | −0.071 |
| trust | 0.489 | 0.100 | 0.172 / 0.213 | +0.324 |
| apology | 0.422 | 0.107 | 0.164 / 0.205 | +0.130 |
| dictator | 0.342 | 0.119 | 0.157 / 0.211 | +0.208 |
| overfishing | 0.307 | 0.109 | 0.185 / 0.189 | +0.240 |
| ultimatum | 0.287 | 0.109 | 0.163 / 0.189 | +0.162 |
| prisoners_dilemma | 0.246 | 0.134 | 0.223 / 0.220 | +0.068 |

The run drew 32 control subspaces. That is few for a p97.5 and a single draw
containing a digit token inflates it, so a 1000-draw re-estimate is given beside
it. No conclusion turns on the difference: **no vector built here falls into its
control band under any weighting**, and the shipped `altruism` vector is the only
one that does.

**The shipped `altruism` vector is the clean negative control**: 0.106, *below*
its own control mean, i.e. no digit content at all, and a signed alignment of
−0.071. Together with its token list above, that is a different object from
anything built here.

### The aggregate hides the thing that actually changes

Reporting only the aggregate would be misleading, because the *dollar* signature
does largely go — a different surface takes its place and holds the total up. Per
digit, cosine with the layer-0 direction (positive = toward the altruistic pole):

| vector | '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | share carried by `'0'` alone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dictator-only | +0.01 | +0.16 | +0.17 | +0.06 | +0.19 | **+0.34** | +0.22 | +0.19 | +0.13 | +0.11 | 0.008 |
| cell_balanced | +0.03 | +0.13 | +0.19 | +0.13 | +0.24 | **+0.37** | +0.26 | +0.23 | +0.18 | +0.14 | 0.030 |
| game_equal_unit | −0.04 | +0.08 | +0.11 | +0.10 | +0.20 | **+0.36** | +0.24 | +0.20 | +0.16 | +0.12 | 0.042 |
| family_balanced_unit | −0.18 | −0.03 | −0.07 | +0.00 | +0.05 | **+0.22** | +0.12 | +0.06 | +0.06 | +0.06 | 0.181 |
| non_dollar_unit | **−0.26** | −0.11 | −0.21 | −0.08 | −0.09 | **+0.04** | −0.01 | −0.07 | −0.05 | −0.02 | 0.256 |
| overfishing alone | −0.19 | +0.02 | −0.09 | +0.02 | +0.00 | +0.11 | +0.03 | +0.03 | +0.03 | +0.08 | 0.187 |
| PD alone | −0.16 | −0.17 | −0.20 | −0.13 | −0.13 | −0.05 | −0.04 | −0.13 | −0.09 | −0.11 | 0.159 |
| shipped `altruism` | +0.00 | −0.01 | −0.03 | −0.06 | −0.06 | −0.06 | −0.05 | −0.07 | −0.06 | −0.06 | 0.004 |

**The dollar signature does go.** `'5'` toward the altruistic pole runs +0.34
(Dictator-only) → +0.37 (cell-balanced) → +0.36 (game-balanced) → **+0.22**
(format-balanced) → **+0.04** (no dollar games). The "a mid-size digit means
generous" direction is genuinely dismantled by balancing the format axis.

**What replaces it is another surface, not a decision.** In the non-dollar vector
the only substantial digit is `'0'` at −0.26, on the self-interested side, and it
alone accounts for 0.256 of the 0.334 aggregate share. `'0'` marks the
self-interested pole there because overfishing's self pole is the maximum catch
and PD's payoff sentences are numeric. The token lists confirm it:

* **family_balanced_unit**, top ten each side in order: altruistic `'5'`+0.219,
  `' fish'`+0.167, `' risk'`+0.128, `' catch'`+0.126, `'6'`+0.123, `'太过'`+0.116,
  `'缺乏'`+0.113, `' tăng'`+0.112, `' comfortable'`+0.112, `'５'`+0.112;
  self-interested `' Agent'`−0.350, `' '`−0.189, `','`−0.184, `'0'`−0.181,
  `'  '`−0.179, `' is'`−0.175, `'Agent'`−0.167, `' $'`−0.166, `'ect'`−0.158,
  `' Def'`−0.152. A superposition of all three answer vocabularies — dollar
  digits, fish words, Defect wordpieces — plus whitespace.
* **non_dollar_unit** — altruistic `' fish'`+0.179, `' risk'`+0.153,
  `' catch'`+0.148, `' unsustainable'`+0.124, `'="<?='`+0.121, `' tăng'`+0.121,
  `'利好'`+0.120, `'太过'`+0.120, `'壓力'`+0.117, `' cybersecurity'`+0.113;
  self-interested `' Agent'`−0.285, `' the'`−0.267, `' and'`−0.259, `'0'`−0.256,
  `' a'`−0.231, `'2'`−0.215, `'ect'`−0.204, `' Def'`−0.194, `' The'`−0.190,
  `' or'`−0.188. Overfishing's answer vocabulary on one pole and PD's on the
  other — a union of two surfaces, not something shared between them.

Note what those lists also contain: **whitespace, punctuation and high-frequency
function words load heavily on the self-interested side of every scheme** (`' '`,
`','`, `' the'`, `' and'`, `' a'`). That is a second surface artefact and not a
decision signal — self-interested responses differ from altruistic ones in length
and formatting as well as in the answer token — and it grows as a share of the
direction exactly where the digit signature shrinks. It is **stated as an
observation from the decode, not measured**; no number here quantifies it.

**One correction to the framing.** "Non-dollar" is not "non-numeric": overfishing's
answers are fish counts, so they are digits too. Two of the three answer-format
families are numeric, which means format balancing puts two thirds of the weight
on numeric-answer games, not one third. That is why the aggregate digit share
cannot reach the control band by this route — the only non-numeric game in the set
is the Prisoner's Dilemma, and it has 18 altruistic rows.

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

**There is cross-surface transfer at depth.** A direction that never saw the
Prisoner's Dilemma still separates Cooperate from Defect at 0.969 at layer 20
while managing only 0.595 on the same rows at layer 0. That gap cannot be the
answer tokens, because layer 0 *is* the answer tokens and it is close to the floor
there.

It coexists with, and is smaller than, the surface component, and **it does not
show up as vector alignment** (§1). But §5 locates it: under `non_dollar_unit`,
where the direction scoring PD is fit on overfishing alone, PD falls to **0.509**
— chance. What transferred to PD was the four dollar games, not something the six
share. §6 then shows it does not survive removing the Dictator direction either.

**The PD row is 18 cooperating responses**, 14 of them from a single wording. It
is the most cited number in this file and the thinnest. §8b.

## 5. Nine weightings, out to 66 degrees, and the answer is still no

The objection to §2 is that the pool was weighted by usable cells, and 85 of 111
cells are dollar cells. So the pool was rebuilt under nine explicit weightings
(`METHOD.md` §6), of which two balance the axis the confound actually lives on:

| what is balanced | scheme | cos vs Dictator-only @20 | angle | layer-0 digit share | pooled split-half L0 | worst held-out L20 |
|---|---|---|---|---|---|---|
| nothing; inverse-variance | `game_precision_raw` | **+0.961** | 16° | 0.427 | 0.878 | 0.678 (fish) |
| nothing; usable cells | `cell_balanced` | +0.959 | 16° | 0.419 | 0.855 | 0.666 (fish) |
| the **game** axis | `game_equal_unit` | +0.901 | 26° | 0.426 | 0.831 | 0.651 (fish) |
| the **answer-format** axis | `family_balanced_unit` | **+0.714** | 44° | 0.368 | 0.791 | 0.609 (fish) |
| the dollar games dropped | `non_dollar_unit` | **+0.412** | 66° | 0.334 | 0.630 | 0.505 (ultimatum) |

Full table, layer 20, both policies:

| scheme | cos vs Dictator-only, strict | relaxed | cos vs cell-balanced pool | cos vs shipped `altruism` | norm @20 |
|---|---|---|---|---|---|
| `game_precision_raw` | +0.961 | +0.909 | +0.995 | +0.117 | 5.63 |
| `cell_balanced` | +0.959 | +0.923 | 1.000 | +0.110 | 5.07 |
| `game_precision_unit` | +0.957 | +0.885 | +0.994 | +0.090 | 0.80 |
| `game_equal_raw` | +0.916 | +0.888 | +0.980 | +0.112 | 4.49 |
| `game_equal_unit` | +0.901 | +0.876 | +0.966 | +0.067 | 0.69 |
| **`family_balanced_raw`** | **+0.732** | +0.705 | +0.817 | +0.113 | 4.04 |
| **`family_balanced_unit`** | **+0.714** | +0.680 | +0.795 | +0.033 | 0.65 |
| **`non_dollar_unit`** | **+0.412** | +0.385 | +0.493 | −0.005 | 0.73 |
| **`non_dollar_raw`** | **+0.403** | +0.390 | +0.493 | +0.093 | 4.46 |

At layer 0, strict: precision_raw +0.905, precision_unit +0.904, cell_balanced
+0.890, game_equal_unit +0.819, game_equal_raw +0.769, family_balanced_unit
+0.566, family_balanced_raw +0.446, non_dollar_unit +0.205, non_dollar_raw +0.143.
The four *families* keep their order, but two of them — `game_equal` and
`family_balanced` — reverse internally between the layers: `raw` is the closer of
the pair to the Dictator direction at layer 20 and the further at layer 0. The
other two do not reverse; `precision`'s pair sits within 0.001 of itself at layer
0, and `non_dollar`'s `unit` is the closer member at both layers.

**Balancing the game axis is not balancing the format axis.** Four of the six
games answer in dollars, so one sixth each still puts 66.7% of the weight on
dollar-format answers. The first pass measured only game-axis weightings and saw
them land between 16° and 26° of the Dictator direction — under ten degrees —
and said in as many words that it made no claim about weightings balancing a
different axis. It did not claim the family was exhausted, and nothing here is a
retraction of it. What the format axis adds is a far end: `family_balanced_unit`
at 44° and `non_dollar_raw` at **66°**, so the nine now cover 16° to 66°.

**The conclusion survives, and more strongly: every column moves in the same
direction at once.** Getting further from the Dictator vector costs surface
separation and deep separation in lockstep, and the digit share never reaches its
control band. There is no point on this curve where the vector is both
surface-free and a good decision direction.

Against the shipped `altruism` vector every scheme is still at nothing (+0.117
down to −0.005, against nulls near 0.16). Moving away from the Dictator vector
does not move anything toward the trait vector.

### Two of the six games never align with the pool, at any weighting

`cos(pool, that game's own vector)` at layer 20, against the **within-cell** label
null. The pool *contains* the game, so both the cosine and its null are inflated
by the game's own share — which makes a failure below stronger, not weaker.

| game | cell_balanced | game_equal_unit | family_balanced_unit | non_dollar_unit |
|---|---|---|---|---|
| dictator | +0.960 / 0.727 | +0.893 / 0.608 | +0.693 / 0.456 | +0.381 / 0.331 |
| trust | +0.851 / 0.683 | +0.851 / 0.651 | +0.687 / 0.485 | +0.411 / 0.344 |
| apology | +0.898 / 0.671 | +0.796 / 0.601 | +0.578 / 0.407 | +0.269 / 0.255 |
| ultimatum | +0.569 / 0.624 **within** | +0.647 / 0.633 | +0.448 / 0.450 **within** | +0.180 / 0.318 **within** |
| **prisoners_dilemma** | +0.446 / 0.670 **within** | +0.523 / 0.623 **within** | +0.693 / 0.792 **within** | +0.730 / 0.821 **within** |
| **overfishing** | +0.274 / 0.657 **within** | +0.430 / 0.616 **within** | +0.643 / 0.802 **within** | +0.730 / 0.821 **within** |

Overfishing and PD are inside their null under every weighting, **including the
two that hand them a third and a half of the vector.** Their raw cosines rise
steadily (overfishing +0.274 → +0.430 → +0.643 → +0.730) but their nulls rise
faster, because a game that is a third of the pool dominates the pooled vector
under permutation too. Relaxed gives the identical pattern.

That null inflation is why the leave-one-out version is the one to read for the
format-balanced schemes — no shared rows, and the null drops back to zero-centred:

| held out | cell_balanced | game_equal_unit | family_balanced_unit | non_dollar_unit |
|---|---|---|---|---|
| dictator | +0.895 / 0.329 | +0.824 / 0.308 | +0.665 / 0.340 | +0.381 / 0.361 |
| trust | +0.782 / 0.403 | +0.757 / 0.382 | +0.648 / 0.416 | +0.411 / 0.411 **within** |
| apology | +0.776 / 0.332 | +0.676 / 0.320 | +0.526 / 0.287 | +0.269 / 0.267 |
| ultimatum | +0.499 / 0.301 | +0.470 / 0.266 | +0.368 / 0.260 | +0.180 / 0.261 **within** |
| **prisoners_dilemma** | +0.327 / 0.341 **within** | +0.314 / 0.315 **within** | +0.237 / 0.336 **within** | +0.066 / 0.343 **within** |
| **overfishing** | +0.209 / 0.382 **within** | +0.204 / 0.348 **within** | +0.162 / 0.381 **within** | +0.066 / 0.343 **within** |

The last column is the sharpest measurement here. Under `non_dollar_unit`, holding
out overfishing leaves the pool equal to PD's own vector and vice versa, so those
two cells are literally `cos(overfishing, PD)` = **+0.066** — the lowest cell in
§1's matrix, reproduced from an independent construction. **The two games that
share no answer surface with the dollar games share nothing with each other
either.**

Corrected for attenuation (`cos / sqrt(rel_pool * rel_game)`, split-half
reliability measured within cell), overfishing↔PD rises only to **+0.113**. Under
`family_balanced_unit` the corrected figures are overfishing +0.261 and PD +0.327,
against dictator +0.779 and trust +0.773. Even on the most generous noise
correction available the non-dollar games are a third as aligned as the dollar
games. (Magnitude comparison, not a significance test.)

### And the transfer falls with it, on every game

Leave-one-game-out AUC, strict. The direction never saw the game it scores.

| held out | cell_balanced L0/L20 | game_equal_unit | family_balanced_unit | non_dollar_unit |
|---|---|---|---|---|
| trust | 0.931 / 0.968 | 0.929 / 0.967 | 0.851 / 0.959 | 0.568 / 0.935 |
| dictator | 0.870 / 0.940 | 0.834 / 0.922 | 0.759 / 0.908 | 0.583 / 0.865 |
| apology | 0.824 / 0.937 | 0.763 / 0.929 | 0.658 / 0.899 | 0.489 / 0.802 |
| prisoners_dilemma | 0.595 / 0.969 | 0.659 / 0.957 | 0.428 / 0.770 | 0.251 / **0.509** |
| ultimatum | 0.618 / 0.713 | 0.626 / 0.696 | 0.634 / 0.632 | 0.587 / **0.505** |
| overfishing | 0.599 / 0.666 | 0.556 / 0.651 | 0.530 / 0.609 | 0.491 / **0.530** |

**Layer-0 separation falls exactly as intended, and layer-20 separation falls with
it, on every single game.** There is no weighting at which the surface component
drops and the deep component holds.

The single most informative cell: **PD at 0.509 under `non_dollar_unit`** — a
direction fit on overfishing alone separates Cooperate from Defect at chance. §4's
"cross-surface transfer at depth" was the four dollar games transferring to PD.
Remove them and the transfer is gone.

**That cell is policy-unstable and should not be leaned on.** Under RELAXED, where
overfishing's self pole grows from 50 rows to 128 (effective n 28.9 → 73.6), the
same cell reads **0.817**, not 0.509. Overfishing→PD is unresolved: 0.509 or 0.817
depending on where the overfishing pole is drawn. The reverse direction,
PD→overfishing, is near chance under both (0.530 strict, 0.595 relaxed) and is the
robust half of the statement.

Split-half within cell (fit on half of every cell's rows, score the other half),
strict: pooled layer 0 goes 0.855 → 0.831 → 0.791 → 0.630 across cell_balanced,
game_equal_unit, family_balanced_unit and non_dollar_unit, and layer 20 goes 0.908
→ 0.900 → 0.891 → 0.858 with the macro average over games 0.850 → 0.852 → 0.849 →
0.824. Layer 0 finally does fall substantially, which is the surface component
being removed — and layer 20 and the macro average fall with it.

An independent split at a different seed, run by `scripts/verify_committed.py`,
gives the same ordering and a steeper layer-20 decline (0.926 → 0.921 → 0.870 →
0.739). The direction of the effect is stable; its size depends on the draw.

### Where the format-balanced pool actually sits

| scheme | cos(pool, dollar-4 sub-pool) | cos(pool, non-dollar-2 sub-pool) | cos(dollar-4, non-dollar-2) |
|---|---|---|---|
| cell_balanced | 0.990 | 0.508 | 0.379 |
| game_equal_unit | 0.945 | 0.652 | 0.368 |
| **family_balanced_unit** | **0.713** | **0.914** | 0.368 |

Format balancing genuinely moves the pool across to the non-dollar side — it is no
longer the dollar vector. But the last column barely moves: the two halves agree
at +0.379 under cell balancing and +0.368 under either game-level scheme, so how
they are mixed does not bring them any closer to each other. **The format-balanced
vector is not a shared direction, it is the midpoint of a disagreement**, and a
midpoint inherits neither half's ability to separate poles.

### Two caveats that bound the two new schemes

* **The `non_dollar_raw`/`_unit` schemes are thin.** They rest on overfishing (effective n 28.9) and
  the Prisoner's Dilemma (effective n 10.2) — 39.1 of the 244.3 total effective
  rows under strict, **16%**. Every positive from them is provisional, and one of
  their headline numbers swings from 0.509 to 0.817 between pole policies.
* **Family balancing is post-hoc.** It was chosen after the §1 agreement matrix
  showed the dollar / non-dollar split — that split is the reason to equalise the
  format axis, and it came from this same data. It is not pre-registered, and the
  nulls quoted here do not price in the search that produced it.

Both bear on the same unresolvable point: moving weight off the dollar games is
also moving weight onto the thinnest evidence. "The transfer falls as the dollar
weight falls" therefore has two readings — the dollar games were carrying the
signal, or they were carrying the *precision* — and this dataset cannot separate
them. Fixing PD's row count (§8c) is the experiment that would.

## 6. The decisive test: project the Dictator direction out

Take the leave-one-game-out pool, remove the Dictator-only direction per layer,
and re-run the held-out separation. Strict, layer 20, before → after (residual
norm fraction in brackets):

| held out | game_equal_unit | family_balanced_unit | non_dollar_unit |
|---|---|---|---|
| apology | 0.929 → **0.313** (0.53) | 0.899 → 0.280 (0.72) | 0.802 → 0.274 (0.91) |
| dictator | 0.922 → **0.500** (0.55) | 0.908 → 0.498 (0.72) | 0.865 → 0.493 (0.91) |
| trust | 0.967 → **0.811** (0.48) | 0.959 → 0.814 (0.70) | 0.935 → 0.773 (0.91) |
| prisoners_dilemma | 0.957 → **0.314** (0.42) | 0.770 → 0.356 (0.70) | 0.509 → 0.361 (0.97) |
| ultimatum | 0.696 → **0.329** (0.43) | 0.632 → 0.308 (0.67) | 0.505 → 0.306 (0.91) |
| overfishing | 0.651 → **0.484** (0.41) | 0.609 → 0.463 (0.65) | 0.530 → 0.450 (0.93) |

**About 41–55% of the game-balanced vector's length survives and almost none of
its separating power does.** Several games land below 0.5, meaning the residual
orders the poles backwards. The pooled vector adds nothing over the Dictator
vector.

The last column is the striking one. The `non_dollar_unit` vector is **already
91% orthogonal** to the Dictator direction — removing it takes away only 9% of its
length — and yet that 9% carries almost all of its ability to separate any game's
poles: apology 0.802 → 0.274, dictator 0.865 → 0.493, trust 0.935 → 0.773.
Whatever separates poles in these activations lies along the Dictator axis, and a
vector built with no dollar games in it borrows that axis rather than finding
another one.

The Dictator-only vector was fit on a **separate, earlier Dictator-only generation
run** — no row of this grid entered it — so this removes a direction estimated
from independent data. That makes the collapse more meaningful, not less: it is
not the pool being made orthogonal to a piece of itself.

Two honest qualifications:

* **Under RELAXED the collapse is much smaller** (game_equal_unit: PD 0.924 →
  0.821, overfishing 0.656 → 0.513; non_dollar_unit: PD 0.817 → 0.794). The
  qualitative claim — the residual separates far worse than the original — holds
  under both, but the size does not, and the strict figures should not be quoted
  alone. The PD swing is particularly large for a pole set that is *identical*
  under both policies, which is 18 rows talking.
* Trust survives best (0.811). Whatever residual structure exists is not uniformly
  zero.

## 7. Against the vectors that already exist

Layer 20, cell-balanced pooled vector, game-wide label null from 1000 draws.

| against | cos | label null p97.5 | verdict |
|---|---|---|---|
| Dictator-only | **+0.959** | 0.424 | above; the pooled vector ≈ the Dictator vector |
| the repo's shipped `altruism` | **+0.110** | 0.184 | **within null — indistinguishable from nothing** |

Per game, against the Dictator-only vector / the shipped altruism vector at layer
20: dictator +0.983 / +0.109, apology +0.842 / +0.163, trust +0.802 / −0.013,
ultimatum +0.501 / +0.026, PD +0.359 / +0.136, overfishing +0.242 / −0.143.

**Nothing built here aligns with the shipped altruism vector.** For reference,
`cos(Dictator-only, shipped altruism) = +0.088` at layer 20 — the same
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

| game | usable cells | alt / self rows | rows inside usable cells | **effective n** | equal weight | cell weight | precision weight | format weight | norm @20 |
|---|---|---|---|---|---|---|---|---|---|
| apology | 25 | 422 / 217 | 306 / 217 | **75.8** | .167 | .225 | .310 | .083 | 8.00 |
| dictator | 30 | 167 / 395 | 167 / 395 | **75.2** | .167 | .270 | .308 | .083 | 7.50 |
| trust | 20 | 300 / 79 | 180 / 79 | **39.2** | .167 | .180 | .161 | .083 | 5.37 |
| overfishing | 17 | 1297 / 50 | 735 / 50 | **28.9** | .167 | .153 | .118 | **.333** | 2.27 |
| ultimatum | 10 | 252 / 29 | 90 / 27 | **15.1** | .167 | .090 | .062 | .083 | 5.37 |
| **prisoners_dilemma** | 9 | **18** / 213 | **18** / 52 | **10.2** | .167 | .081 | .042 | **.333** | 8.47 |

Equal weighting gives the Prisoner's Dilemma one sixth of the vector on an
effective 10.2 rows; **format balancing gives it one third.** Three of the six
games carry their share on thin evidence — PD 10.2, ultimatum 15.1, overfishing
28.9 — and **two of those three, PD and overfishing, are exactly the two that fail
every alignment test in §1 and §5**. Those two facts are not independent and this
data cannot separate them: PD's and overfishing's non-alignment could be about
their answer surface or about their thin poles. That is inference either way, and
it is the single largest unresolved ambiguity in this directory.

Effective n by format group, strict: dollar 205.3 (four games), fish 28.9, word
10.2. Under RELAXED the Ultimatum's self pole widens from 29 rows across 10 cells
to 235 across 24, so its effective n goes 15.1 → 92.3 and it becomes the heaviest
game under precision weighting; overfishing goes 28.9 → 73.6. PD is unchanged by
the policy — its answer space has two points and no middle — and stays at 10.2
either way.

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
(`evidence/pd_probe_6matrices_16samples.csv`).

**That probe is weaker evidence than it looks, and the difference matters.** It
holds the wording fixed to isolate the matrix, but the wording it holds fixed is a
**sixth** one, written for the probe and absent from the grid
(`scripts/prompting/pd_probe.py`), so its own cooperation rate was never measured
against the payoff ladder. A wording that cooperates 0% regardless — which is what
`upstream` does in the grid, 0 of 46 — would produce exactly this result whatever
the matrix did. So the probe bounds the matrix's effect *within one untested
wording* and no further.

What is measured, then, is the **spread across wordings**: 14 of 47 against 0-to-2
of 45-to-47, on one grid, with the payoff ladder varying underneath both. Whether
the matrix moves PD cooperation at all is **not established here**, in either
direction. Stating it as "wording drives cooperation, the matrix does not" would
be a causal claim this design cannot support.

Every PD figure in this file — the +0.066 matrix cell, the 0.969 leave-one-out
transfer, the 0.957 → 0.314 collapse, the 0.509 under `non_dollar_unit` — rests on
those 18 rows across 9 cells. It is the weakest row and column of every table here,
and format balancing makes it a third of the vector.

### c. PD was generated at 240 rows against 720 and 1440

That part is a **design** imbalance rather than a behavioural one: the run was
planned when PD looked like a corner rather than a vector. The committed
calibration pass (`evidence/calibration_upstream_wording_8samples.csv`, the
`upstream` wording, 8 samples in each of a game's 6 stakes) has PD at **0
cooperations in 48**, and it is the reason for the whole sample allocation —
overfishing got 48 per cell because its self-interested pole ran at **2 of 48**,
the amount games got 24 because both their poles filled (dictator 13 alt / 23
self, apology 17 / 24, trust 16 / 9, ultimatum 13 / 1), and PD got 8 as a census.
It is the cheapest thing in this study to fix, and it is also the experiment that
would separate the two readings in §5's last paragraph.

### d. Both poles are focal points, and no weighting can remove it

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
cannot remove this — and reweighting preserves it perfectly, since it changes only
how the games are mixed. Any shared direction found in this data could be the
focal-point contrast rather than a decision.

(The overfishing self-interested cell needs care. Under STRICT that pole is "caught
the maximum M", so it is 100% at the maximum *by construction*; the 94.0% is the
share sitting on one single number, `K`, across the six fisheries — 47 of 50, with
the other 3 at `2K` where `M = 2K`. The largest single *raw* catch value holds 23
of the 50.)

### e. Overfishing is the weakest-separating game by a distance

Own layer-20 AUC 0.724 against 0.87–0.98 for the rest, and its self pole is 50 rows
of 1425. Its non-alignment is measured, not an artefact of a failed extraction, but
it rests on a small pole and it has the worst within-cell split-half reliability of
the six (0.436 at layer 20, against apology 0.969 and dictator 0.965).

### f. Both pole policies were run end to end and they agree

Relaxed pooled layer 0 = 0.877 against layer 20 = 0.874; the same
dollar-games-agree / non-dollar-games-do-not split (dictator–apology 0.841,
overfishing–PD 0.149, all overfishing pairs within null). Where the policies differ
it is the Ultimatum, whose self pole widens from 29 rows to 235, lifting its
leave-one-out separation from 0.696 to 0.887, and the two figures flagged in §5 and
§6. Nothing about the digit signature, the two failing games, the weighting span or
the Dictator collapse changes qualitatively. Full numbers for both are in
`analysis/crossgame_analysis.json` under `by_policy` and in
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

`scripts/measurement/selftest_analysis.py` runs the six-game battery — §1–§4, §7
and §8, i.e. `analyze_crossgame.py`, not the pooling stage — twice on synthetic
activations: it recovers a planted shared direction (pooled AUC 1.000, agreement
+0.837 against a null p97.5 of +0.032, leave-one-out 1.000 on both games),
returns chance on pure noise (pooled 0.513, agreement +0.012, leave-one-out 0.539
and 0.537), and confirms a single-poled game is excluded from every pooled
structure and gets no vector. **Re-run from the committed copy, from where it
sits: all 7 checks pass**, with the numbers above. It takes about forty seconds
and no test command runs it — it is a manual check, deliberately kept out of
`python -m pytest -q`.

## 9. What this does not establish

* **Nothing about steering.** No intervention was run. "The pooled vector is
  0.96 to the Dictator vector" is a statement about two directions, not about what
  either does to behaviour.
* **Not that PD and overfishing fail *because* of their answer surface.** They are
  also the two thinnest games (§8a). The two explanations are confounded in this
  data and cannot be separated with it.
* **Not that the digit component survives *because* it is the shared component.**
  The per-digit decomposition in §3 shows the dollar signature is dismantled and a
  fish/Defect surface replaces it; that reading is consistent with the decode but
  was not tested separately.
* **Not that a decision direction does not exist.** What is measured is that
  difference-of-means over these six games, under these nine weightings and both
  pole policies, does not find one, and that what it does find is dominated by a
  direction the Dictator game alone already gives.
* **Not that PD cooperation is or is not driven by the payoff matrix.** §8b's
  probe holds a wording fixed that the grid never tested, so it cannot separate
  "the matrix does not move this" from "this wording does not cooperate". The
  wording spread is measured; the matrix's contribution is not.
* **Nothing is corrected for multiplicity.** §1 reads 15 pairwise cosines against
  15 separate p97.5 bars and §5 reads 48 more the same way. Each bar is a nominal
  2.5% one-sided test, so if every one of those 63 cells were truly null about 1.6
  of them would still read "above". No correction is applied and none of the
  verdicts is adjusted for the number of tests, so the marginal ones — trust–PD
  +0.396 against 0.332, dictator–PD +0.339 against 0.263 — are the first to
  distrust. Five of the six dollar–dollar pairs are not in that class — they clear
  their bars by 0.16 to 0.49. The sixth, ultimatum–apology, clears by 0.081, which
  is no better than dictator–PD's 0.076 and should be read the same way.
* **No AUC here carries an interval.** Every AUC is a point estimate from one
  split at one seed. The nulls price the *cosines*; nothing prices an AUC, and
  several rest on a pole of 18 (PD), 29 (ultimatum) or 50 (overfishing) rows,
  where the sampling error is large. The independent-seed split-half in §5 is the
  only evidence offered about how much an AUC moves with the draw, and it moves a
  lot (layer-20 pooled 0.908 against 0.926 at the other seed).
* **Not that the focal-point reading is wrong.** §8d is unresolved and is the
  cheapest thing to test next: build a contrast whose altruistic answer is not the
  halfway focal point and see whether the layer-0 signature follows the focal point
  or the decision.

## 10. What is in this directory

```
README.md                  this file — the result
METHOD.md                  the construction: grid, poles, balancings, nulls
vectors/                   30 vectors, (29, 3584) float32
extraction/                THE GENERATIONS THE VECTORS WERE BUILT FROM — 4560 rows, 4503 scored
evidence/                  the PD payoff-matrix probe and the calibration pass
analysis/                  every number above, both policies, all layers, all nulls
provenance/                the manifest and the execution logs of every stage
scripts/                   prompting/, measurement/, pooling/ and the two drivers
```

There is **no `rows/` directory and no steering output**, because no steering was
run. `extraction/` is the analogue of the Dictator directory's `extraction/`: the
unsteered answers whose outcomes became the pole labels. It holds the extracted
rows, not an extractor — the extractor is `lab/extract.py`, one level up and shared
with the Dictator study.

### `vectors/`

30 files, all `(29, 3584)` float32, all cell-balanced, both pole policies — every
vector any claim in this file rests on, and nothing else:

* `decision_<game>_response_avg_diff_cellbalanced_<policy>.pt` — the six per-game
  vectors (12 files)
* `decision_pooled_<scheme>_response_avg_diff_<policy>.pt` — the nine weightings
  (18 files): `cell_balanced`, `game_equal_raw`/`_unit`,
  `game_precision_raw`/`_unit`, `family_balanced_raw`/`_unit`,
  `non_dollar_raw`/`_unit`

The Dictator-only vector §6 projects out is not duplicated here: it is
`results/dictator-decision-vector/vectors/decision_response_avg_diff_cellbalanced.pt`,
and `scripts/pooling/dictator_vector.py` rebuilds it from the re-extracted
activations and checks it against that file (`analysis/dictator_vector.json`).

The unbalanced (non-cell-balanced) per-game and pooled vectors exist in the run
and their layer-20 norms are quoted in the analysis JSON, but no claim here rests
on them and they are 14 more files at 417 KB each, so they are not committed.

### `extraction/`

One CSV per game, 4560 rows total, 35 columns, one row per generation, scored by
`audit.parse` with the resolution path named in its `tag`. These make the pole
counts, the focal-point shares and the PD wording breakdown checkable rather than
merely stated, and they are the input the re-extraction ran from.

### `evidence/`

* `pd_probe_6matrices_16samples.csv` — 96 generations over six additional PD payoff
  matrices, all resolving to Defect. The evidence for §8b.
* `calibration_upstream_wording_8samples.csv` — the calibration pass that set the
  per-game sample counts.

**What is *not* here: the corrupted PD CSV.** During the generation run, one PD
generation completed and reported success, and the extraction that followed it
died with `_csv.Error: line contains NUL` (`provenance/finish.log`, 15:03:27Z). PD
was regenerated with the same parameters and the second extraction succeeded
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
| `crossgame_layer0_tokens.json` | the §3 token decode of the cell-balanced pool, overfishing, PD, the Dictator-only vector and the shipped `altruism` vector |
| `pooled_build.json` | the §8a census, effective n, all nine weightings, and the rebuild check against the committed vectors |
| `pooled_analysis.json` | §5: cosines by layer for all nine schemes, in-pool and leave-one-out, both label nulls, leave-one-game-out AUC by layer, split-half within cell, both policies |
| `pooled_addendum.json` | §5's leave-one-out nulls, §6's Dictator-orthogonalised transfer, and the dollar / non-dollar decomposition |
| `pooled_reliability.json` | within-cell split-half reliability, the attenuation-corrected cosines, and the §8d focal-point census |
| `pooled_layer0_tokens.json` | the §3 token decode of all nine schemes under both policies, the six per-game vectors, and the two external vectors |
| `pooled_digit_share.json` | the §3 scalar table: digit-span share, both control bands, the per-digit cosines and the `'0'`-only share |
| `dictator_vector.json` | the Dictator-only vector rebuilt from the re-extracted activations, against the committed one |
| `activation_equivalence.json` | §0: the re-extraction against the archived activations, row by row, all seven corpora |
| `rebuild_deltas.json` | §12: this revision against the previous one — every vector on either side, and the seven analysis JSONs, leaf by leaf |
| `verification.json`, `verification_relaxed.json` | the independent recomputation, one file per pole policy — see §11 |

### `provenance/`

`manifest.json` is the per-game record: rows CSV path, activation directory,
generated-row count, tag counts, the activation meta (model id and revision, dtype,
attention implementation, torch and transformers versions, hidden size, layer
count, chat-template hash, device, batch size, padding, shard count, dropped rows,
capture time), and the per-policy pole census, usable cells, per-layer norms and
split-half for both policies.

The logs are the execution trace. `main_run.log`, `resume_run.log`, `finish.log`
and `finish_pd.log` are the **generation** run — the first pass (overfishing, then
the four amount games, ending in the CUDA OOM that killed `trust`), trust
regenerated from scratch, ultimatum, and the PD generation whose extraction hit
the NUL error, then PD regenerated. `extraction.log` is this revision's
re-extraction of the whole corpus, and `analysis.log` and the rest are the CPU
stages it fed.

`generation_run_notes.md` is a **temporary** note beside those four logs: how that
run was launched, and what the machine was doing while it ran. It is a record of
one past run, not part of the pipeline, and it is meant to be deleted once it
stops being worth remembering. Everything in it that is a property of the
committed data — and not of that machine — is stated in this file instead.

### `scripts/`

| path | what it is |
|---|---|
| `run_extraction.sh` | drives `lab/extract.py` over the whole corpus, one game at a time, resumable |
| `run_analysis.sh` | everything downstream of the activations, in order, CPU only |
| `prompting/crossgame_grid.py`, `poles.py` | the grid and pole **definitions** — declarations, not computation |
| `prompting/gen_crossgame.py` | the generation stage; not re-run by this revision |
| `prompting/calib_report.py`, `pd_probe.py` | the calibration read-out and the §8b probe |
| `measurement/analyze_crossgame.py`, `land_family.py` | §1–§4, §7, §8 |
| `measurement/decode_layer0.py` | the §3 token decode |
| `measurement/summarize.py` | a printer for the analysis JSON; computes nothing |
| `measurement/selftest_analysis.py` | the six-game battery on synthetic data with a known answer; **manual**, no test command runs it |
| `pooling/` | the nine weightings and everything measured on them |
| `verify_committed.py` | §11, the independent recomputation |
| `compare_activations.py`, `compare_published.py` | §0 and §12, the two diffs |
| `tests/` | the reproduction gate and the rebuild comparator, offline; part of `python -m pytest -q` from the repository root |

`prompting/` is the front half — what was asked, in which wording, at which stake
— and `measurement/` is what was measured off the activations that came back.
Neither holds an extractor: the teacher-forced capture is `lab/extract.py`, a
top-level package that holds this project's own experimental pipeline (`audit/` is
the clean-room reimplementation of the upstream paper, and `lab/` borrows its
prompt rendering and game declarations rather than the other way round). One
extractor serves this study and the Dictator-only one, which is why it sits beside
them rather than inside either.

## 11. How to check these numbers

Two independent things are true, and they are not the same claim. **Both of them
start at the activations, which are not committed** — 5.3 GB for the six games,
2.1 GB for the Dictator grid. So "checkable from what is here" means checkable
given a GPU, the model weights and a re-extraction (stage 3 of §12); it is the one
step that cannot be replayed from the repository alone.

**One: every published number is produced by the code in this directory.**
`scripts/run_extraction.sh` then `scripts/run_analysis.sh` regenerate every vector
and every analysis file from the committed rows CSVs. `scripts/compare_published.py`
diffs the result against what is committed; §12 reports what it found.

**Two: the load-bearing numbers are also re-derived by code that shares nothing
with the analysis.** `scripts/verify_committed.py` reads the activation shards
directly, imports only `crossgame_grid` and `poles` (the grid and pole
*definitions*, not the computation), does its own linear algebra, and rebuilds:

The two vector sections are a **gate**, not a report: if any committed vector came
back at a worst-layer cosine below `--min-cosine` (default 1 − 1e-9), or at a norm
on **any** layer further than `--max-norm-drift` (default 1e-6) from the committed
one, the script writes its report and exits non-zero. The cosine alone would not
be enough — it is scale-invariant, so a vector twice as long still reads 1.000000
— and the norm is checked on every layer, not only on layer 20: §1, §3 and §5 are
read at layer 0, and a scale error there is just as much a failure to reproduce.
The two tolerances are different sizes because the committed vectors are stored
**float32**: that rounding leaves the cosine at 1 − 1e-15 but moves the norm by up
to ~1e-8, so the norm tolerance has to clear the storage floor. At 1e-6 it still
fails on a scale error of one part in 100,000. Every other row below is a
recomputation with no committed artifact to disagree with, and is reported rather
than gated.

The gate itself is tested, offline and with no activations, in
`scripts/tests/test_verify_committed.py`: a scale error on any layer fails, a
turned vector fails, a NaN fails, and a gate that compared nothing does not pass.
The storage floor is not taken on the docstring's word either — one test exhibits
two float64 vectors 3.0e-8 apart in norm that save to exactly the same committed
`.pt` bytes, and another reads the drifts the two committed reports actually
recorded and shows a 1e-9 tolerance failing on them.

| check | result |
|---|---|
| the six per-game cell-balanced vectors, against the committed `.pt` files | **gated, passes**: worst-layer cosine 1 − 1.1e-15, every layer's norm within 8.8e-09, all six, both policies |
| the nine pooled vectors, against the committed `.pt` files | **gated, passes**: worst-layer cosine 1 − 1.8e-15, every layer's norm within 7.8e-09, all nine, both policies |
| the 6x6 agreement matrix at layers 20 and 0 | matches to 4 decimal places, all 15 pairs |
| leave-one-game-out AUC, all nine weightings | matches L0, L20, best and best-layer for all six games |
| `cos(pool, Dictator-only)` at layers 0 and 20, and `cos(pool, cell-balanced pool)` at layer 20 | matches all nine schemes |
| the leave-one-game-out cosine against the held-out game's own vector | matches all nine schemes, including `cos(overfishing, PD) = +0.066` |
| the layer-0 digit-span share, the signed alignment, the per-digit decomposition, the `'0'`-only share and both control bands | matches for all 17 vectors, to 4 decimal places |
| the layer-0 token decode | same tokens, same order, same values |
| the Dictator-projected-out collapse, all six games | matches AUC before, AUC after, best-after layer and residual norm fraction |
| the pole census, tag counts, parse coverage, focal shares, effective n | matches all six games |
| the PD wording breakdown and the payoff-matrix probe | 18/231; 14/47 `rules_first`; 0/96 in the probe |
| split-half within cell, at a seed the run did not use | same ordering across schemes; magnitudes differ with the draw (§5) |

It runs **one pole policy per invocation**, and both are committed:
`analysis/verification.json` is strict, `analysis/verification_relaxed.json` is
relaxed. Each file is literally the output of one run of the script — nothing is
merged or edited after the fact.

```
for policy in strict relaxed; do
  python results/crossgame-decision-vectors/scripts/verify_committed.py \
      --acts <activation root> --policy $policy \
      --dictator results/dictator-decision-vector/vectors/decision_response_avg_diff_cellbalanced.pt \
      --snapshot <local Qwen2.5-7B-Instruct snapshot> \
      --out results/crossgame-decision-vectors/analysis/verification$([ $policy = relaxed ] && echo _relaxed).json
done
```

Without `--snapshot` the digit-share and token-decode section is skipped and
everything else still runs.

To check things by hand from what is committed:

* **Pole counts and focal shares** — from `extraction/*.csv` alone. Drop rows whose
  `tag` is `unparsed`/`refusal`/`empty` and rows whose tag is a *derived* path
  (`complement`, `keep`, `mixed`), then apply the §4 rules of `METHOD.md` using
  each row's `game_id` to look up its `pole_scale`.
* **The PD wording breakdown** — group `extraction/prisoners_dilemma.csv` by the
  wording segment of `game_id` and count `value == 1.0`.
* **The vectors' identity** — `torch.load(path)[20].double().norm()` reproduces the
  §8a norm column, and the pairwise cosines reproduce §1.
* **Everything downstream of the activations** — needs `acts/`, which is not
  committed. See §12.

## 12. How to rebuild it, and what changed when it was

| stage | what it does | code | needs |
|---|---|---|---|
| 1. the grid | the 180 prompt cells, each fingerprint checked against its own text | `scripts/prompting/crossgame_grid.py` | nothing |
| 2. generation | 4560 generations, `neutral` preset, seed 0, batch 32. Output is `extraction/*.csv`. **Not re-run by this revision.** | `scripts/prompting/gen_crossgame.py`, one game per invocation | GPU + model weights |
| 3. activations | teacher-forced forward pass over prompt+response, batch 1, no padding, three poolings x 29 layers. **5.3 GB for the six games, 2.1 GB for the Dictator grid, not committed.** | `lab/extract.py`, driven by `scripts/run_extraction.sh` | GPU + model weights |
| 4. everything CPU-side | per-game landing, the six-game battery, the nine weightings, both token decodes, the digit share | `scripts/run_analysis.sh` | the activations |
| 5. the two diffs | the re-extraction against the archived activations; the rebuild against what was published | `scripts/compare_activations.py`, `scripts/compare_published.py` | the activations |
| 6. the verification | recomputes §1–§8 from the activations without either stage's analysis code | `scripts/verify_committed.py` | the activations, and the model weights for §3 |
| the self-test | the whole battery on synthetic activations with a known answer | `scripts/measurement/selftest_analysis.py` | nothing |
| the PD probe | §8b's six extra payoff matrices | `scripts/prompting/pd_probe.py` | GPU + model weights |
| the calibration | the pass that set the per-game sample counts | `scripts/prompting/calib_report.py` | `evidence/calibration_*.csv` |

```
ACTS=<8 GB somewhere> PY=<python with torch+transformers> DEVICE=0 \
    bash results/crossgame-decision-vectors/scripts/run_extraction.sh
ACTS=<same> WORK=<scratch> PY=<same> \
    bash results/crossgame-decision-vectors/scripts/run_analysis.sh
```

### The two big directories, and how to regenerate them

**`acts/` — 7.4 GB, not committed.** Seven directories of activation shards plus a
`meta.json` each. Stage 3 above regenerates them from the committed CSVs. It
re-renders each row's prompt and verifies it byte-for-byte against the
`prompt_sha256` the row already carries, so a mismatch is caught rather than
silently producing activations for a different prompt. On the hardware this ran on
it was 0.4 to 5.1 minutes per game (240 to 1785 rows), plus a model load each.

**`cache/response_avg.pt` — 1.8 GB, not committed.** Purely derived: the
`response_avg` slice of all six games' shards concatenated into one
`(4503, 29, 3584)` float32 tensor plus `labels.json`, so the pooling stage does
not re-read the shards on every invocation. `scripts/pooling/common.py` writes it
on first use if it is absent, in about 9 seconds. Do not commit it; delete it
freely.

### What the rebuild moved

`analysis/rebuild_deltas.json` is the leaf-by-leaf diff of this revision against
the previous one, produced by `scripts/compare_published.py`. **No published
statistic moved.**

The comparator itself was corrected in this revision, and the table below is its
output, not the older one's. It used to abandon a subtree whose two lists differed
in length, which meant it compared **nothing** in the token files while still
reporting them as unmoved; it walked only the old tree, which hid every vector the
rebuild added; and it counted a NaN leaf as identical, because `nan > 0` is False,
so a NaN that appeared or vanished came out the far side as agreement. A NaN on one
side only is now a difference, and a NaN on both is counted apart from the leaves
that were actually compared.

| | |
|---|---|
| committed vectors rebuilt | **22 of 22 to bit-identical tensors**; worst per-layer cosine 0.9999999999999986. (The `.pt` files differ as bytes — `torch.save` writes a zip with timestamps — but every element of every tensor is equal.) |
| the other 8 committed vectors | the four new schemes, ×2 policies. The previous revision has no counterpart to diff them against, so this table cannot cover them and does not claim to: they are gated in §11 instead, and `n_missing: 9` in the report names them (the ninth is the Dictator-only vector, which moved directory). |
| `crossgame_analysis.json` (§1–§4, §7, §8) | 8936 numeric leaves, **all 8936 identical**. Four more leaves are NaN on both sides — the layer-0 `prompt_last` split-half, where within-cell variance is zero (§8h) — and are reported as unverifiable in either direction rather than as agreement. |
| `pooled_build.json` (§8a census, effective n, weights) | no numeric movement |
| `crossgame_layer0_tokens.json`, `pooled_layer0_tokens.json` (§3) | 96 and 240 numeric leaves, **all identical**; every shared token and cosine equal |
| `pooled_reliability.json` (§5) | largest delta 3.3e-16 — float64 noise |
| `pooled_analysis.json`, `pooled_addendum.json` (§5, §6) | largest delta **7.6e-07**, and every moved leaf is a label-null distribution summary |

That last row is a deliberate change, not drift. The null blocks were being
accumulated in float32 against a float64 statistic; they are now float64, which
moves a null's `mean`/`sd`/`min`/`max`/`p97.5` in the seventh decimal and moves
no cosine, AUC, census figure or vector at all. Every verdict in this file is
read against a null's p97.5, and none of them is within 7.6e-07 of its bar.

The 221 structural differences are all **additions**: the four weightings the
previous revision could not commit (`family_balanced_raw`/`_unit`,
`non_dollar_raw`/`_unit`) and everything measured on them, the per-digit
decomposition and the `'0'`-only share, the cross-pipeline check in
`pooled_build.json`, two more decoded vectors, and the extra provenance fields the
shared extractor writes into each `meta.json`. One key is renamed: the
Dictator-only vector is decoded under
`decision_dictator_only_response_avg_diff_cellbalanced.pt` rather than
`decision_response_avg_diff_cellbalanced.pt`; its tokens and cosines are
unchanged.

**Every committed artifact names the extractor `lab.extract`.** The `activation_meta`
embedded in `analysis/crossgame_analysis.json` and in `provenance/manifest.json` is
the `meta.json` the extractor wrote beside the shards those numbers were computed
from — same module, same revision, same kernel, and a `rows_csv_sha256` that pins
each rows CSV by content. There is no chain to follow and none to take on trust:
one corpus, re-extracted through `lab.extract` and reported bit-identical to the
archive in §0 (1,960,648,704 of 1,960,648,704 elements exactly equal), consumed by
`scripts/run_analysis.sh` for everything in this file, and read again by
`scripts/verify_committed.py`, which reproduces all 15 committed vectors from it
under both pole policies.

### What runs from where it sits

Everything in `scripts/` does, with no `PYTHONPATH` and no copying, provided the
repository root is its ancestor. Stage 2 is the one stage with no driver here:
`prompting/gen_crossgame.py` generates one game per invocation, and the five shell
wrappers that chained those invocations are gone. They hardcoded a worktree, a
`.venv` inside it, an output directory outside this repository, and an
`extract_crossgame.py` that was never committed — a record of five particular
launches on one machine, not pipeline code.
`provenance/generation_run_notes.md` records what they did, and `METHOD.md` §3
carries the part of it that is method rather than circumstance. The capture step
they called is now `lab/extract.py`, driven by `run_extraction.sh`, and every
activation behind every number in this file came off it.

**Records of past runs still name paths that no longer exist.** The four
generation logs in `provenance/` quote the `scratch/` worktree those runs executed
from, the deleted wrappers by name, and the tracebacks they raised. They are frozen
history and are not rewritten to match the current layout.

## 13. Provenance

Model `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bfloat16, `sdpa` (both pinned,
resolved from the loaded model, and recorded on every row and in every
`meta.json`), transformers 4.52.3, torch 2.6.0+cu124, chat template sha256 prefix
`cd8e9439f0570856`, stop tokens 151645 and 151643. `neutral` preset: temperature
1.0, top_p 1.0, top_k 0, min_p 0.0, repetition_penalty 1.0, max_new_tokens 1000,
min_new_tokens 1. Seed 0 for generation, batch size 32, device `cuda:0`. Analysis
seed 20260820, 1000 shuffles for the pooled nulls and 300 per pair for the matrix
and the reweighting nulls. 4560 generations, 4503 scored and captured, 0 dropped.
Hidden size 3584, 29 hidden states (embedding + 28 layers).

**The recorded commit does not pin the code that generated the rows.** Every row
records `repo_commit = 4d10f19dfe7436845456f0f8b67d2adef25776a4` with
`repo_dirty = true`, across all 4560 rows. The working tree had uncommitted
changes — the scripts were an untracked `scratch/` directory at the time, which is
why they are committed here — so the commit identifies the baseline the generation
run started from, not the code it executed. That is a statement about **stage 2**
only: stages 3 onward were re-run from committed code for this revision, and the
activations they produced are bit-identical to the ones the original scripts
produced (§0), which is the strongest available evidence that the generation-time
extractor was the same computation too.

**The generation run was interrupted twice and one game was generated twice.**
`main_run.log` records the model load OOMing twice and retrying before the first
pass started, and then a CUDA OOM inside a forward pass that killed `trust`;
`resume_run.log` records `trust` regenerated from scratch, and a second OOM on its
first retry. `finish.log` and `finish_pd.log`
record the PD generation that completed and reported success but whose CSV could
not be read (`_csv.Error: line contains NUL`), and the regeneration that followed.
**Each game is internally seed-consistent** — one run, seed 0, batch 32 — but the
set was not produced by a single uninterrupted invocation, and two games were
produced by their second attempt rather than their first.
`provenance/generation_run_notes.md` is the temporary record of what the machine
was doing at the time.

All six CSVs were verified NUL-free with row counts and SHA-256 recorded at landing
time, re-verified during packaging, and each `extraction/<game>.csv` is
byte-identical to the `rows.csv` beside the activations it was extracted from.

**The re-extraction used one device, pinned, at the same dtype, attention kernel,
batch size and revision.** Nothing was relaxed to make a run fit; there was no OOM
and no eviction, and 0 rows were dropped across all 6288.

**No GPU was used for any analysis**, and no model was loaded: everything from §1
onward is CPU-only linear algebra over the activations, plus reading the embedding
matrix off the safetensors shard for the §3 decode.

Nothing under `persona_vectors/` or `results/dictator-decision-vector/` was
modified by this work. A new top-level package, `lab/`, holds `extract.py` and its
tests.
