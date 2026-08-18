# `audit/` — free, offline scoring of the game responses

A self-contained package for re-scoring the committed generations without the
paid judge. It imports with no API keys, no network and no GPU, and it never
imports `eval/`, `judge.py`, `config.py` or `generate_vec*.py`. Upstream data
files are read directly and never modified.

Why nothing of theirs is imported: `judge.py:12`, `eval/eval_persona.py:34`,
`eval/eval_persona_batch.py:50`, `eval/run_judges.py:37` and `training.py:20` all
call `setup_credentials()` at module import, which raises without both
`OPENAI_API_KEY` and `HF_TOKEN`. Importing `config.py` itself does not raise —
credential validation is lazy — but line 2 sets `os.environ["HF_HOME"] = 'hf_cache'`
as an import side effect, repointing the HuggingFace cache at a repo-relative
path for the rest of the process.

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

The repo ships generations with the paid judge's value beside each one, so the
suite measures agreement rather than asserting it. It consumes 2,420 labelled
rows: 2,200 `altruism_v2` rows (11 steering conditions x 200) plus 220 Table-1
rows carrying per-question extraction judges. Held out means every steering
condition except `coef0.0`, the one the scorers were tuned on.

| set | n | measured | tuning condition |
| --- | --- | --- | --- |
| `altruism_v2`, 10 held-out steering conditions | 2,000 | 97.6% resolve; **95.6%** of those match the judge exactly | excluded |
| — condition means | 10 | r = 0.9992, largest gap $2.46 | excluded |
| Overfishing, per-question judge | 210 | 99.5% resolve; 91.9% exact | included (excluded: 91.1%) |
| Prisoner's Dilemma, per-question judge | 210 | 100% resolve; 98.1% exact | included (excluded: 97.8%) |
| naive last-number regex, same rows | 2,000 | 20.8% exact (the scorers: 93.3%) | excluded |

The two per-question rows come from a different question set than the tuning
rows, but they are not tuning-free, so both figures are given.

"Exact" is `round(judge) == ours`; the judge is a soft expectation over its own
token distribution and is never more than $0.48 from an integer here.

Assertion floors sit below the measured values so the suite is not brittle, and
each carries its measurement in a comment. If a change moves one of these
numbers, re-measure and say so — the agreement figure is the only reason this
code is trusted.

## Known limits

None of these are fixed here. This slice ports the validated behavior unchanged,
because the measured agreement above is the reason the code is trusted and a
behavior change in the same commit would invalidate it.

**The fish scorer is the weak one** — 91.9% against the amount scorer's 95.6%.
Its 17 disagreements are three distinct failures, not one:

| n | failure |
| --- | --- |
| 8 | hedged-range midpoint deltas — "around 45 or 47", "40-50"; gaps of $0.3 to $20 |
| 6 | trailing-sentence contamination — decides on 50, then explains "the total is 100", resolves to 100 |
| 3 | reads the digit in an agent label as a count — "I will catch 50 fish alongside Agent 2" resolves to 2 |

The third is the worst of them and the least obvious. A fix should start there,
and should raise the test floor with it.

**Two inherited quirks that a run with different stakes would hit:**

- `extract_cooperate` tags "I'd cooperate" as `mixed`, because the bare-letter
  branch of the defect pattern matches the `d` in `I'd`. The value is still right
  (the later mention wins) and 9 of 210 Prisoner's Dilemma rows land on that tag,
  so `mixed` is not evidence of a genuinely ambiguous response.
- The amount and fish patterns match `\d{1,3}` with no right-hand boundary, so a
  four-digit figure truncates: `"$1000 to Agent 2"` resolves to 100.0 with the
  highest-confidence tag, `a2_anchor`. Harmless at the $0-100 stakes every
  committed question uses; not harmless at larger ones.
