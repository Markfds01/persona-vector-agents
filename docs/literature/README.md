# Literature

Papers this work builds on and is audited against. One entry per paper: what
it is, its licence, and why it's here.

The paper this repo is the code release for lives at
`docs/persona-vectors-in-games-2603.21398.pdf` (not moved here).

## Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits

Jinyi Ye, Lei Cao, Ding Chen, Emilio Ferrara. arXiv:2605.18890.
https://arxiv.org/abs/2605.18890

Licence: CC BY 4.0. PDF: local, `stop-drawing-scientific-claims-2605.18890.pdf`.

Robustness-audit paper. Central finding: two persona prompts that differ only
in surface format (prose vs. table), same content, produce a 76-percentage-point
swing in 10-round Prisoner's Dilemma cooperation rate on one frontier model,
and the effect is model-dependent (~77pp on one model, ~1pp on another). It
bears directly on this repo: our own Dictator-game measurements show the same
class of artifact — the expected-value read off the logits moves from about
$46 to about $16 on a reword of the answer stub alone, nothing else changed.
Reason to audit our elicitation format before trusting a number from it, not
a refutation of the method.

## Persona Vectors: Monitoring and Controlling Character Traits in Language Models

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey.
arXiv:2507.21509. https://arxiv.org/abs/2507.21509

Licence: arXiv.org perpetual, non-exclusive license (the arXiv default,
checked on the abstract page and in the PDF's own `dc:rights` metadata across
v1-v3 — not CC BY). That licence doesn't clearly permit third-party
redistribution, so no local copy; link-only.

Source of the persona-vector method this repo applies: contrastive activation
addition (mean-difference vectors between positive/negative trait prompts),
steered at inference by a coefficient. This repo's upstream is the
game-theoretic application of that method.
