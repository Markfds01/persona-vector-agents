# Literature

Papers this work builds on and is audited against. One entry per paper: what
it is, its licence, and why it's here.

The paper this repo is the code release for lives at
[`docs/persona-vectors-in-games-2603.21398.pdf`](../persona-vectors-in-games-2603.21398.pdf).

## Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits

Jinyi Ye, Lei Cao, Ding Chen, Emilio Ferrara. arXiv:2605.18890.
https://arxiv.org/abs/2605.18890

Licence: CC BY 4.0. PDF: local, [`stop-drawing-scientific-claims-2605.18890.pdf`](stop-drawing-scientific-claims-2605.18890.pdf).

Robustness-audit paper. Its Prisoner's Dilemma case study tests three persona
formats, meaning held fixed: a paragraph of plain prose (PLAIN), a descriptive
bullet list (DESCRIPTIVE), and a structured key-value table (TABULAR).
DESCRIPTIVE cooperates 76 percentage points less often than PLAIN and 73pp
less than TABULAR — prose vs. table itself is the near-null cell, roughly a
3pp gap. The effect is also model-dependent: on that same PLAIN-vs-DESCRIPTIVE
perturbation, gpt-5.2 shows a 76pp gap, claude-haiku-4-5 a separately measured
~77pp (close to gpt-5.2's by coincidence, not a repeat of it), gemini-2.5-flash
~36pp, and deepseek-v3 ~1pp.

It bears directly on this repo: our own unpublished elicitation check on the
Dictator game (not part of this repo) found the logit-derived expected value
of the donation moving from roughly $40 to roughly $16 depending only on how
the answer was elicited (a terse vs. an explicit answer stub) — the same
class of artifact. Reason to audit our elicitation format before trusting a
number from it, not a refutation of the method.

## Persona Vectors: Monitoring and Controlling Character Traits in Language Models

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey.
arXiv:2507.21509. https://arxiv.org/abs/2507.21509

Licence: arXiv.org perpetual, non-exclusive license (the arXiv default) — not
an open licence, so no local copy; link-only.

Source of the persona-vector method: a trait direction extracted as the
difference in mean activations between contrastive positive/negative prompt
pairs, then steered into generation by adding that vector at a chosen
coefficient. This repo's own paper is the game-theoretic application of that
method.
