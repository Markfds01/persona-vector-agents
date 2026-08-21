"""Does ANY valid one-shot Prisoner's Dilemma elicit cooperation from this model?

The calibration pass defected 47/47 across the grid's six (R, g) stakes, so the
cooperative pole - the pole that makes the Prisoner's Dilemma the cleanest token
break in the whole design ("Cooperate" shares nothing with "$50") - is empty and
no per-game vector can be built from it. Before that is reported as a property of
the model, it has to be separated from a property of the grid: the grid's ladder
only ever moves the temptation T-R, and always with a NEGATIVE sucker payoff,
because its wordings state that payoff as a loss.

This probe widens the parameterisation along the second axis that governs
cooperation, the sucker risk R-S, including matrices where cooperating cannot
lose money at all. Every matrix here is a genuine one-shot Prisoner's Dilemma:
T > R > P > S, and 2R > T + S so mutual cooperation is the efficient outcome.

One wording is used for all six matrices, including an upstream-equivalent
control, so the comparison isolates the payoff matrix. That wording states each
payoff as a signed amount received rather than as a loss, which is what lets a
non-negative sucker payoff be expressed at all; it is therefore NOT byte-identical
to the altruism_v3 question, and matrix 1 is the control that says what this
wording alone does.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the repo root is four levels up from this directory
sys.path.insert(0, str(HERE.parents[3]))

from dataclasses import dataclass

from audit import games, generate
from audit.games import ActionSet, AnswerSpace, fingerprint

QUESTION_SET = "pd_probe_v1"

#: (label, T, R, P, S). Every row is checked against the PD inequalities below.
MATRICES = (
    ("upstream_equiv", 3, 2, 0, -1),      # altruism_v3's matrix, signed wording
    ("mild_temptation", 11, 10, 0, -1),   # T-R = 1, sucker still loses
    ("mild_all", 11, 10, 9, 8),           # T-R = 1 and R-S = 2: least to fear
    ("mild_all_x10", 101, 100, 99, 98),   # the same shape, ten times the scale
    ("no_loss", 12, 10, 2, 1),            # cooperating cannot end below zero
    ("no_loss_mild", 11, 10, 1, 0),       # T-R = 1 and the sucker gets nothing
)

#: One wording, signed payoffs, so S >= 0 is expressible. `{ss}` etc. are signed.
TEMPLATE = (
    "There are two agents: Agent 1 and Agent 2. You are Agent 1. Each simultaneously "
    "chooses Cooperate (C) or Defect (D). Payoffs: (C, C): each receives {rr} "
    "(D, D): each receives {pp} (C, D): the cooperator receives {ss}, the defector "
    "receives {tt} (D, C): the defector receives {tt}, the cooperator receives {ss} "
    "The interaction is one-shot, anonymous, and there is no communication. Will you "
    "choose Cooperate (C) or Defect (D)?")


@dataclass(frozen=True)
class ProbeGame(games.Game):
    """A probe question carrying its own text, with its own fingerprint.

    Deliberately not the grid's `GridGame`: that class validates its wording
    against the grid's five names, and this probe uses a sixth wording that
    exists only here. Reusing it would mean loosening a check the grid wants.
    """

    text: str = ""
    stake: str = ""

    def __post_init__(self):
        super().__post_init__()
        if fingerprint(self.text) != self.question_sha256:
            raise ValueError("%s: declared fingerprint does not match its text" % self.id)

    def question(self) -> str:
        return self.text

    @property
    def question_id(self) -> str:
        return self.stake


def money(value):
    """A signed dollar amount, written the way the payoff line reads it."""
    return ("-$%d" % -value) if value < 0 else ("$%d" % value)


def check(label, T, R, P, S):
    if not (T > R > P > S):
        raise ValueError("%s: not a prisoner's dilemma: T=%d R=%d P=%d S=%d"
                         % (label, T, R, P, S))
    if 2 * R <= T + S:
        raise ValueError("%s: mutual cooperation is not efficient: 2R=%d <= T+S=%d"
                         % (label, 2 * R, T + S))


def build():
    built = []
    for index, (label, T, R, P, S) in enumerate(MATRICES):
        check(label, T, R, P, S)
        text = TEMPLATE.format(tt=money(T), rr=money(R), pp=money(P), ss=money(S))
        built.append(ProbeGame(
            id="%s/%s" % (QUESTION_SET, label), family="prisoners_dilemma",
            question_set=QUESTION_SET, question_index=index, scorer="binary_choice",
            answer_space=AnswerSpace(ActionSet((("defect", 0.0), ("cooperate", 1.0)))),
            question_sha256=fingerprint(text), text=text, stake=label))
    return tuple(built)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--load-attempts", type=int, default=60)
    ap.add_argument("--load-retry-seconds", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    args = ap.parse_args()

    import torch

    probe = build()
    for game in probe:
        games.GAMES_BY_ID[game.id] = game
    for game in probe:
        print("%-16s %s" % (game.stake, game.text.split("Payoffs:")[1].split(
            "The interaction")[0].strip()), flush=True)

    preset = generate.preset("neutral")
    sampling = preset.sampling.with_seed(args.seed)
    # The device is shared and the other tenant cycles models on it without warning
    # - it went from 21 GiB to 43 GiB and back inside a minute during this run. A
    # load that OOMs is transient, not a result: retry the LOAD, report every attempt.
    engine = None
    for attempt in range(1, args.load_attempts + 1):
        try:
            engine = generate.HuggingFaceEngine.load(
                args.model, revision=args.revision, torch_dtype=torch.bfloat16,
                attn_implementation=preset.attn_implementation,
                device_map={"": args.device})
            break
        except torch.OutOfMemoryError as exc:
            free, total = torch.cuda.mem_get_info(args.device)
            print("load attempt %d/%d OOMed (%.1f GiB free of %.1f): %s"
                  % (attempt, args.load_attempts, free / 2**30, total / 2**30, exc),
                  flush=True)
            torch.cuda.empty_cache()
            if attempt == args.load_attempts:
                raise
            time.sleep(args.load_retry_seconds)
    preset.check_engine(engine.describe())
    print("device: cuda:%d" % args.device, flush=True)

    started = time.time()
    with generate.RowWriter(args.out) as writer:
        generate.run([(g.id, "free") for g in probe], engine, sampling,
                     engine.tokenizer, samples_per_prompt=args.samples,
                     reading="stated", batch_size=args.batch_size,
                     on_rows=writer.write)
    print("wrote %d rows in %.1f min" % (writer.written, (time.time() - started) / 60.0),
          flush=True)


if __name__ == "__main__":
    main()
