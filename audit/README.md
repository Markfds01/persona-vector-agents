# `audit/` — free, offline scoring of the game responses

A self-contained package for re-scoring the committed generations without the
paid judge. It imports with no API keys, no network and no GPU, and it never
imports `eval/`, `judge.py`, `config.py` or `generate_vec*.py` (`config.setup_credentials()`
runs at import time and raises without `OPENAI_API_KEY` and `HF_TOKEN`). Upstream
data files are read directly and never modified.

This first slice is scorers, answer spaces and the regression suite. Generation,
logits and steering are not here.

## Layout

| file | what it owns |
| --- | --- |
| `parse.py` | the deterministic scorers |
| `games.py` | per-game declaration: question source, answer space, which scorer |
| `paths.py` | the only place that knows where the upstream data files live |
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

## Tests

```bash
python -m pytest audit/tests -q
```

Runs in about five seconds. No keys, no network, no GPU — verified by running it
with the keys unset and outbound connections blocked.

The repo ships 2,200 generations with the paid judge's value beside each one, so
the suite measures agreement rather than asserting it. Held out means every
steering condition except `coef0.0`, the one the scorers were tuned on.

| set | n | measured |
| --- | --- | --- |
| `altruism_v2`, 10 held-out steering conditions | 2,000 | 97.6% resolve; **95.6%** of those match the judge exactly |
| — condition means | 10 | r = 0.9992, largest gap $2.46 |
| Overfishing, per-question judge | 210 | 99.5% resolve; 92.8% exact |
| Prisoner's Dilemma, per-question judge | 210 | 100% resolve; 98.1% exact |
| naive last-number regex, same rows | 2,000 | 20.8% exact (the scorers: 93.3%) |

"Exact" is `round(judge) == ours`; the judge is a soft expectation over its own
token distribution and is never more than $0.48 from an integer here.

Assertion floors sit below the measured values so the suite is not brittle, and
each carries its measurement in a comment. If a change moves one of these
numbers, re-measure and say so — the agreement figure is the only reason this
code is trusted.

**Known weak spot.** The fish scorer resolves a trailing sentence in place of the
decision ("I will catch 50 fish… so the total is 100"), which is most of the gap
between 92.8% and the amount scorer's 95.6%. It is not fixed here: this slice
ports the validated behavior unchanged.
