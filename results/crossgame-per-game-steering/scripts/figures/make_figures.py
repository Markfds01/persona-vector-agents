"""The six games on one intervention axis, each drawn against its OWN null.

`results/dictator-decision-vector/scripts/make_plot.py` is the precedent: the
decision measure and the pole share the vector was built from, the responsive
range marked, and the asymmetry drawn AS asymmetric rather than smoothed. This is
that figure for six games, and it adds the thing that run did not need — every
game's own shuffled-label null on the same axes as its real arm. Four of six
nulls move significantly and two move the WRONG way, so a figure showing only the
real arm would misrepresent the result: the bar is beating the null, so the null
is drawn.

Three things are decided from the data rather than written into the caption, each
by a pure function below so a reader can check the rule rather than trust the
picture:

* `no_room` — a game whose baseline already sits against a bound cannot move that
  way, and a flat arm there is a ceiling, not a failed intervention. Overfishing
  starts at 0.959 and the Prisoner's Dilemma at 0.000. The bound is called out of
  reach when the distance to it is inside the baseline's own Wilson HALF-width.
  That selects those two arms and nothing else, but Overfishing's ceiling clears
  by 0.0013 — read `no_room` before quoting it, the margin is stated there.
* `point_flags` — two DIFFERENT kinds of degraded point, kept apart because they
  mean different things. `degenerate` is "the model stopped answering in English"
  (non-Latin share >= 0.5); it is true only at the Prisoner's Dilemma's k=+4 and
  k=+5, where EVERY answer is non-Latin, and those points are struck through and
  excluded from the supported range: they are a degeneration, not a result.
  `low_coverage` is "the scorer could not read enough of it" (parse coverage <
  0.90); it is a caveat drawn on the marker, not a disqualification.
* `null_verdict` — beating the null needs the real arm to move FURTHER than the
  null in the steered direction, so the sign is checked, not just the interval.
  An end where either arm stopped producing a distribution is `undetermined`
  whichever way its interval fell, because a contrast computed across a point
  this figure strikes out cannot be the evidence for a win either: the
  Ultimatum's k=-5 null parsed 8 answers of 100 and the Prisoner's Dilemma's
  k=+5 decision arm answered entirely in Chinese, and both ends say nothing.

Reads `analysis/steering.json` only. No torch, no GPU, no re-analysis — every
number drawn here is one `analyze_game.py` wrote.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurement"))

#: The two kinds of degraded point and the thresholds that separate them are
#: shared with `crossgame_tables.py`: the figure and the table must call the same
#: point unreadable. Re-exported so this module reads as its own vocabulary.
from degradation import (COLLAPSED_COVERAGE, DEGENERATE_NON_LATIN,  # noqa: E402
                         LOW_COVERAGE, point_flags, produced_no_distribution)

#: The ladder's axis: the layer-20 norm of the repo's shipped altruism trait
#: vector. METHOD.md section 3 says why it is this constant and not another.
REFERENCE_NORM = 10.508308410644531

#: Canonical order. A run that covered fewer games draws fewer panels, in this
#: order; nothing here assumes all six are present.
GAMES = ("dictator", "trust", "ultimatum", "apology", "overfishing",
         "prisoners_dilemma")
LABELS = {"dictator": "Dictator", "trust": "Trust", "ultimatum": "Ultimatum",
          "apology": "Apology", "overfishing": "Overfishing",
          "prisoners_dilemma": "Prisoner's Dilemma"}

DECISION = "#1b6ca8"
NULL = "#5c8001"
BAD = "#b3202c"
GOOD = "#1a6b3c"
UNDET = "#c77700"
GREY = "#666666"

#: A supported band wears the verdict of the end it sits on.
BAND_COLOUR = {"beats": DECISION, "null": BAD, "undetermined": UNDET}


# --- the rules, as functions rather than as caption text ----------------------

def wilson_half_width(pole):
    return (pole["hi"] - pole["lo"]) / 2.0


def no_room(baseline_pole, side):
    """Is the bound on `side` already inside the baseline's own noise?

    `side` is +1 for the altruistic bound at 1.0, -1 for the self-interested one
    at 0.0. An arm with no room cannot be read as a failed intervention, so the
    figure marks the ceiling or the floor instead of letting a flat line lie.

    HOW CLOSE THIS IS. Overfishing's strict ceiling is 94 of 98: the distance to
    1.0 is 0.040816 against a half-width of 0.042151, so it clears by 0.001335 —
    less than one observation, and it flips at 93/98. The published figures are
    drawn on this rule. The equally natural reading, "is the bound inside the
    baseline's own Wilson INTERVAL", would drop it: that interval is
    [0.8997, 0.9840] and 1.0 is outside it. The Prisoner's Dilemma's floor is not
    close under either reading. Which rule to keep is an open question and not
    one this module has settled; what it must not do is claim a clean selection.
    """
    bound = 1.0 if side > 0 else 0.0
    return abs(bound - baseline_pole["p"]) <= wilson_half_width(baseline_pole)


def supported_range(arm, side):
    """The ks on one side of 0 where the decision arm really moved.

    Significant against the arm's OWN k=0 (95% Newcombe on P(altruistic)) and not
    degenerate. Returned as `(k_nearest, k_farthest)` by absolute value, or None
    when the side carries nothing - which is how an asymmetric arm draws
    asymmetric instead of being smoothed into one band.
    """
    moves = arm["move_from_zero"]
    keep = []
    for key, point in arm["points"].items():
        k = int(key)
        if k == 0 or (k > 0) != (side > 0):
            continue
        degenerate, _low = point_flags(point)
        if degenerate:
            continue
        move = moves.get(key)
        if move and move["altruistic"]["excludes_zero"]:
            keep.append(k)
    if not keep:
        return None
    return (min(keep, key=abs), max(keep, key=abs))


def ladder_ends(game):
    """`{-1: key, +1: key}` - the outermost rung each side that BOTH arms ran.

    `run_sweep.py` takes `--ks` and `--null-ks`, so the ladder a run covers is
    not a constant and the extremes are read off the analysis rather than
    written in. A full run gives the -5 and +5 this package reports.
    """
    keys = [int(key) for key in game.get("decision_vs_null", {})]
    negative = [k for k in keys if k < 0]
    positive = [k for k in keys if k > 0]
    if not negative or not positive:
        raise SystemExit(
            "%s: the two arms share rungs %s, which do not reach both sides of "
            "zero; there is no end-to-end verdict to draw"
            % (game.get("family", "?"), sorted(keys)))
    return {-1: str(min(negative)), +1: str(max(positive))}


def comparison_degraded(game, side):
    """Did either arm fail to produce readable decisions at this end?

    An arm that answered in Chinese, or that stated an offer in 7 answers of
    100, has not produced a distribution, and an interval computed across it -
    whether or not it clears zero - is not evidence about the direction. This is
    the gate `null_verdict` uses, and the threshold is COLLAPSED_COVERAGE
    deliberately: a merely thin arm is a caveat, not a disqualification.
    """
    key = ladder_ends(game)[side]
    for arm in ("decision", "shuffled-null"):
        point = game["arms"][arm]["points"].get(key)
        if point is None:
            continue
        if produced_no_distribution(point):
            return True
    return False


def degraded_ends(game):
    """The ends whose comparison rests on an arm that did not answer."""
    return [side for side in (-1, +1) if comparison_degraded(game, side)]


def null_verdict(game):
    """Per end: does the real arm beat its own null, fail to, or say nothing?

    Degradation is checked FIRST, and it settles the end on its own. An end
    where an arm did not produce a distribution is `undetermined` whichever way
    its interval fell: a contrast computed across a point the figure strikes out
    as "not a result" cannot be the evidence for a win any more than for a loss.
    Otherwise `beats` needs the difference to exclude zero AND to point the way
    the arm is being steered - a significant difference the wrong way is not a
    win.
    """
    out = {}
    ends = ladder_ends(game)
    for side in (-1, +1):
        if comparison_degraded(game, side):
            out[side] = "undetermined"
            continue
        contrast = game["decision_vs_null"][ends[side]][
            "altruistic_decision_minus_null"]
        right_way = ((contrast["diff"] > 0) if side > 0 else (contrast["diff"] < 0))
        out[side] = "beats" if (contrast["excludes_zero"] and right_way) else "null"
    return out


def verdict_line(verdict):
    """One phrase naming both ends, for the panel to wear above its own axes."""
    words = {"beats": "beats its null", "null": "does not beat its null",
             "undetermined": "undetermined"}
    negative, positive = verdict[-1], verdict[+1]
    if negative == positive == "beats":
        return "beats its own null at both ends", GOOD
    if negative == positive == "null":
        return "does NOT beat its own null at either end", BAD
    if "beats" not in (negative, positive):
        return ("negative end %s; positive end %s"
                % (words[negative], words[positive]), BAD)
    # only a WIN is softened by an undetermined other end. A failure on the end
    # that was measured is a finding, and README section 5 reports it as one
    end = "negative" if negative == "beats" else "positive"
    other = positive if negative == "beats" else negative
    colour = GOOD if other == "null" else UNDET
    return "beats its own null on the %s end only (other end: %s)" % (
        end, words[other]), colour


def degenerate_runs(arm):
    """Contiguous spans of degenerate ks, so the figure can strike them out."""
    ks = sorted(int(key) for key, point in arm["points"].items()
                if point_flags(point)[0])
    runs, run = [], []
    for k in ks:
        if run and k != run[-1] + 1:
            runs.append(run)
            run = []
        run.append(k)
    if run:
        runs.append(run)
    return runs


# --- drawing ------------------------------------------------------------------

def _series(arm, getter):
    points = sorted(arm["points"].items(), key=lambda item: int(item[0]))
    out = []
    for key, point in points:
        value = getter(point)
        if value is None:
            continue
        low, high, centre = value
        degenerate, low_coverage = point_flags(point)
        # a Wilson bound at p=0 or p=1 lands within float noise of the centre and
        # can cross it by ~1e-18; an interval arm is a length, never negative
        out.append({"k": int(key), "x": int(key) * REFERENCE_NORM, "y": centre,
                    "lo": max(0.0, centre - low), "hi": max(0.0, high - centre),
                    "degenerate": degenerate, "low_coverage": low_coverage,
                    "n_parsed": point["n_parsed"]})
    return out


def _pole_getter(policy):
    """Read `P(altruistic)` under the policy the ANALYSIS was computed under.

    The contrasts in that file - move-from-zero, decision-versus-null - are all
    under one policy, so a panel that read a different one would draw a curve its
    own verdict did not describe.
    """
    def getter(point):
        pole = point["poles"][policy]["altruistic"]
        # nothing parsed is no share and no Wilson bounds; the point is dropped
        # from the line the way an own measure with no mean already is
        if pole["p"] is None:
            return None
        return (pole["lo"], pole["hi"], pole["p"])
    return getter


def _measure(point):
    measure = point["measure"]
    if measure["mean"] is None:
        return None
    return (measure["ci_low"], measure["ci_high"], measure["mean"])


def _draw_arm(ax, series, colour, marker, linestyle, linewidth, label, zorder):
    """The line, its intervals, and the two kinds of degraded marker on top."""
    if not series:
        return
    xs = [p["x"] for p in series]
    ys = [p["y"] for p in series]
    ax.errorbar(xs, ys, yerr=[[p["lo"] for p in series], [p["hi"] for p in series]],
                color=colour, marker=marker, linestyle=linestyle, capsize=2.5,
                markersize=5.5, linewidth=linewidth, label=label, zorder=zorder)
    stagger = 0
    for point in series:
        if point["degenerate"]:
            ax.plot(point["x"], point["y"], marker="x", color=BAD, markersize=11,
                    markeredgewidth=2.2, linestyle="none", zorder=zorder + 2)
        elif point["low_coverage"]:
            ax.plot(point["x"], point["y"], marker=marker, color=colour,
                    markerfacecolor="white", markersize=5.5, markeredgewidth=1.4,
                    linestyle="none", zorder=zorder + 1)
            # adjacent degraded rungs would print their counts on top of each other
            edge = point["k"] <= -5
            drop = -13 - 9 * (stagger % 2)
            stagger += 1
            ax.annotate("n=%d" % point["n_parsed"], (point["x"], point["y"]),
                        textcoords="offset points",
                        xytext=(10, drop + 2) if edge else (0, drop),
                        ha="left" if edge else "center",
                        fontsize=6.8, color=colour)


def _axis_furniture(ax):
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_xlim(-59, 59)
    ticks = [k * REFERENCE_NORM for k in range(-5, 6)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%+d" % k if k else "0" for k in range(-5, 6)], fontsize=8)
    ax.set_xlabel("k   (unit beta = k x %.4f, the length of the layer-20 edit)"
                  % REFERENCE_NORM, fontsize=8.5)


def _mark_bands(ax, game, arm, policy, ymin, ymax, on_pole_axis):
    """Supported range, ceiling/floor, and the struck-out degenerate points.

    The supported band is coloured by whether that END beat its null, because
    moving away from baseline is not the finding and a blue band on an arm that
    never cleared its own null would say it was.
    """
    verdict = null_verdict(game)
    for side in (-1, +1):
        span = supported_range(arm, side)
        if span is None:
            continue
        colour = BAND_COLOUR[verdict[side]]
        near, far = (abs(span[0]) * side, abs(span[1]) * side)
        lo, hi = sorted((near * REFERENCE_NORM, far * REFERENCE_NORM))
        if hi > lo:
            ax.axvspan(lo, hi, color=colour, alpha=0.075, zorder=0)
        for edge in (lo, hi):
            ax.axvline(edge, color=colour, linewidth=1.0, alpha=0.7,
                       linestyle=(0, (4, 3)), zorder=1)

    baseline = arm["points"]["0"]["poles"][policy]["altruistic"]
    for side, name, bound in ((+1, "ceiling", 1.0), (-1, "floor", 0.0)):
        if not no_room(baseline, side):
            continue
        lo, hi = (0, 59) if side > 0 else (-59, 0)
        ax.axvspan(lo, hi, facecolor=GREY, alpha=0.10, zorder=0)
        ax.axvspan(lo, hi, facecolor="none", edgecolor=GREY, hatch="\\\\",
                   linewidth=0.0, alpha=0.22, zorder=0)
        if not on_pole_axis:
            continue
        ax.axhline(bound, color=GREY, linewidth=1.4, zorder=2)
        ax.text((lo + hi) / 2.0, bound + (0.030 if side > 0 else 0.115),
                "no room: the %s. Baseline is already %.3f."
                % (name, baseline["p"]),
                ha="center", va="bottom", fontsize=8.0, color="#444444",
                weight="bold")

    for run in degenerate_runs(arm):
        lo = (min(run) - 0.5) * REFERENCE_NORM
        hi = (max(run) + 0.5) * REFERENCE_NORM
        ax.axvspan(lo, hi, facecolor=BAD, alpha=0.075, zorder=0)
        ax.axvspan(lo, hi, facecolor="none", edgecolor=BAD, hatch="xx",
                   linewidth=0.0, alpha=0.45, zorder=0)
        ax.text((lo + hi) / 2.0, ymax - 0.03 * (ymax - ymin),
                "not English:\nnot a result", ha="center", va="top", fontsize=7.6,
                color=BAD, weight="bold")


def _panel(ax, game_name, game, policy, getter, ylabel, ylim, on_pole_axis):
    decision = game["arms"]["decision"]
    null = game["arms"]["shuffled-null"]
    baseline_point = decision["points"]["0"]
    baseline_value = getter(baseline_point)
    if baseline_value is None:
        raise SystemExit("%s: the k=0 no-op point resolved no answers, and every "
                         "baseline and band on this panel is measured against it"
                         % game_name)
    baseline = baseline_value[2]

    ymin, ymax = ylim
    _mark_bands(ax, game, decision, policy, ymin, ymax, on_pole_axis)
    ax.axhline(baseline, color="#999999", linewidth=0.9, linestyle=(0, (2, 3)),
               zorder=1)
    _draw_arm(ax, _series(null, getter), NULL, "^", "--", 1.6,
              "shuffled-label null", 3)
    _draw_arm(ax, _series(decision, getter), DECISION, "o", "-", 2.1,
              "decision vector", 5)

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=8.5)
    norm = baseline_point["vector_layer_norm"]
    ax.text(0.0, 1.085, "%s      ||v||@20 = %.4f" % (LABELS[game_name], norm),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5)
    line, colour = verdict_line(null_verdict(game))
    ends = degraded_ends(game)
    if ends:
        line += "   [%s end: an arm there did not answer]" % (
            " and ".join("negative" if side < 0 else "positive" for side in ends))
        colour = UNDET if colour == GOOD else colour
    ax.text(0.0, 1.015, line, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.0, color=colour, weight="bold")
    _axis_furniture(ax)


def _legend_handles():
    return [
        Line2D([], [], color=DECISION, marker="o", linewidth=2.1,
               label="decision vector (that game's own, 11 rungs)"),
        Line2D([], [], color=NULL, marker="^", linestyle="--", linewidth=1.6,
               label="shuffled-label null (matched, run at k = -5, 0, +5 only)"),
        Line2D([], [], color=BAD, marker="x", linestyle="none", markersize=10,
               markeredgewidth=2.2,
               label="every answer non-Latin: a degeneration, not a result"),
        Line2D([], [], color=GREY, marker="o", markerfacecolor="white",
               linestyle="none", markeredgewidth=1.4,
               label="parse coverage < %.2f (parsed n on the marker)" % LOW_COVERAGE),
        Patch(facecolor=DECISION, alpha=0.22,
              label="moved significantly from its own k=0 AND beat its null there"),
        Patch(facecolor=BAD, alpha=0.22,
              label="moved from its own k=0 but did NOT beat its null there"),
        Patch(facecolor=UNDET, alpha=0.22,
              label="moved from its own k=0; the null comparison is undetermined"),
        Patch(facecolor=GREY, alpha=0.20,
              label="no room: the baseline is already against that bound"),
    ]


def footer(policy):
    return (
        "Qwen2.5-7B-Instruct rev a09a3545, bfloat16, sdpa, altruism_v3, mode free, layer 20, positions=all, norm=unit, neutral preset, seed 0, n = 100 per point, %s poles.\n"
        "Error bars are 95%% intervals - Wilson on a share, Student's t on a mean. One k is the SAME sized activation edit in every game (unit beta = k x %.4f), which is what makes the panels\n"
        "comparable; the per-game equivalent raw beta is unit beta / ||v||@20. k=0 is one shared no-op run: both arms generated identical text, verified per row. The null is the same construction on the same\n"
        "activations over the same cells with the pole labels permuted within each cell - it is not inert, and where it moves further than the real arm the real arm has shown nothing."
    ) % (policy, REFERENCE_NORM)


def _grid(count):
    """Panels laid out two rows deep, which is what six games want."""
    columns = max(1, (count + 1) // 2)
    rows = 1 if count <= 3 else 2
    return rows, columns


def _headline(analysis, games):
    """The count of games clearing the bar, said rather than asserted.

    Every game is counted by how many of its two ends the real arm beat its own
    null on, so the three totals partition the games drawn and a reader can add
    them up. An end that could not be compared is not a clearance - a game
    carrying one can still clear at neither end, which is the negative result and
    not a gap - so how many carry one is said separately rather than folded in.
    """
    verdicts = [null_verdict(analysis["games"][g]) for g in games]
    beaten = [sum(1 for side in (-1, +1) if verdict[side] == "beats")
              for verdict in verdicts]
    unusable = sum(1 for v in verdicts if "undetermined" in (v[-1], v[+1]))
    line = ("%d of %d clear that bar at both ends, %d at one end, %d at neither"
            % (beaten.count(2), len(games), beaten.count(1), beaten.count(0)))
    if unusable:
        line += "; %d with an end that could not be compared" % unusable
    return line


def _annotate_a_clean_null(axes, analysis, games, policy):
    """Name the game whose null simply arrives where its real arm arrives.

    Only where the arm had room to move: a flat arm against a ceiling is already
    marked as a ceiling, and pointing at it here would say something else.
    """
    for ax, name in zip(axes, games):
        game = analysis["games"][name]
        if null_verdict(game)[+1] != "null":
            continue
        decision = game["arms"]["decision"]
        baseline = decision["points"]["0"]["poles"][policy]["altruistic"]
        if no_room(baseline, +1):
            continue
        key = ladder_ends(game)[+1]
        real = decision["points"][key]["poles"][policy]["altruistic"]["p"]
        null = (game["arms"]["shuffled-null"]["points"][key]["poles"][policy]
                ["altruistic"]["p"])
        ax.annotate("the null gets to %.3f where the real arm gets to %.3f"
                    % (null, real),
                    xy=(int(key) * REFERENCE_NORM, null), xytext=(2, 0.80),
                    fontsize=8.2, color=BAD, ha="left",
                    arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))


def _comparison_notes(analysis, games):
    """Name every end whose comparison rests on an arm that did not answer.

    Both the interval that fails to clear zero and the one that clears it wide
    belong here: what disqualifies them is the same missing distribution.
    """
    lines = []
    for name in games:
        game = analysis["games"][name]
        for side in (-1, +1):
            if not comparison_degraded(game, side):
                continue
            key = ladder_ends(game)[side]
            # the arm that TRIPPED the gate, not whichever parsed fewer: a healthy
            # arm can be the thinner one and would then be named for the other's fault
            failed = [(name, game["arms"][arm]["points"][key])
                      for name, arm in (("decision", "decision"),
                                        ("null", "shuffled-null"))
                      if produced_no_distribution(game["arms"][arm]["points"][key])]
            arm, worst = min(failed, key=lambda pair: pair[1]["n_parsed"])
            lines.append(
                "%s at k=%+d: its %s arm parsed %d answers of %d (%.0f%% non-Latin), "
                "so that comparison is not evidence either way."
                % (LABELS[name], int(key), arm, worst["n_parsed"], worst["n_rows"],
                   100 * worst["degeneracy"]["share_with_non_latin_script"]))
    return "\n".join(lines)


def _finish(fig, suptitle, policy):
    fig.suptitle(suptitle, fontsize=14.5, y=0.988)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3, fontsize=8.8,
               frameon=False, bbox_to_anchor=(0.5, 0.098))
    fig.text(0.5, 0.006, footer(policy), ha="center", fontsize=7.7, color="#444444")
    fig.tight_layout(rect=(0, 0.170, 1, 0.955), h_pad=4.2)


def figure_pole_shares(analysis, games, policy, label, out_path):
    rows, columns = _grid(len(games))
    fig, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 6.0 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, games):
        _panel(ax, name, analysis["games"][name], policy, _pole_getter(policy),
               "P(altruistic pole)", (-0.05, 1.08), True)
    for ax in flat[len(games):]:
        ax.axis("off")
    _annotate_a_clean_null(flat, analysis, games, policy)
    _finish(fig, "Each game steered with its OWN %s decision vector, against its "
                 "OWN null: %s" % (label, _headline(analysis, games)), policy)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def figure_own_measure(analysis, games, policy, label, out_path):
    rows, columns = _grid(len(games))
    fig, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 6.0 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, games):
        game = analysis["games"][name]
        values = [point["measure"]["ci_high"]
                  for arm in game["arms"].values()
                  for point in arm["points"].values()
                  if point["measure"]["ci_high"] is not None]
        top = max(values) * 1.20 if values else 1.0
        units = game["measure_units"]
        if not game["altruistic_is_high_on_own_measure"]:
            units += "   (HIGHER = LESS altruistic)"
        _panel(ax, name, game, policy, _measure, units, (-0.04 * top, top), False)
    for ax in flat[len(games):]:
        ax.axis("off")
    _finish(fig, "The same %s arms on each game's own measure - not comparable "
                 "across games, which is why every shared claim is made on the "
                 "pole share" % label, policy)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def figure_vs_null(analysis, games, policy, label, out_path):
    """The bar itself: decision minus its own null at each extreme, per game."""
    fig, ax = plt.subplots(figsize=(11.6, 1.24 * len(games) + 0.9))
    colours = {"beats": GOOD, "null": BAD, "undetermined": UNDET}
    rows, y = [], 0.0
    for name in games:
        game = analysis["games"][name]
        verdict = null_verdict(game)
        ends = ladder_ends(game)
        for side in (+1, -1):
            key = ends[side]
            contrast = game["decision_vs_null"][key]["altruistic_decision_minus_null"]
            rows.append((y, int(key), contrast, verdict[side],
                         comparison_degraded(game, side)))
            y -= 1.0
        y -= 0.6

    for pos, k, contrast, state, degraded in rows:
        side = 1 if k > 0 else -1
        colour = colours[state]
        ax.errorbar([contrast["diff"]], [pos],
                    xerr=[[contrast["diff"] - contrast["lo"]],
                          [contrast["hi"] - contrast["diff"]]],
                    color=colour, marker="o" if side > 0 else "s", capsize=3.5,
                    markersize=7, linewidth=2.0, linestyle="none",
                    markerfacecolor="white" if degraded else colour, zorder=4)
        ax.text(contrast["hi"] + 0.02, pos,
                "k=%+d%s" % (k, "  (an arm here did not answer)"
                             if degraded else ""),
                va="center", fontsize=8.2, color=colour)

    ax.axvline(0, color="#333333", linewidth=1.2, zorder=2)
    ax.set_yticks([rows[2 * i][0] - 0.5 for i in range(len(games))])
    ax.set_yticklabels([LABELS[g] for g in games], fontsize=10.5)
    ax.set_xlabel("P(altruistic), decision arm minus its own shuffled-label null "
                  "   (95% Newcombe interval)", fontsize=9.5)
    ax.set_xlim(-1.12, 1.28)
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.6)
    # the pad clears the two-line strapline below it, which is anchored va=bottom
    ax.set_title("Beating the null is the bar (%s vectors). An interval touching "
                 "the line is not a result." % label, fontsize=12.5, loc="left",
                 pad=40)
    ax.text(0.0, 1.012,
            "A point LEFT of the line at k=-5 and RIGHT of it at k=+5 is the real "
            "arm moving further than the null in the steered direction.\n"
            "An end where an arm did not answer is undetermined wherever its "
            "interval fell: it is not the evidence for a win any more than for a "
            "loss.",
            transform=ax.transAxes, va="bottom", fontsize=8.6, color=GREY)
    fig.legend(handles=[
        Line2D([], [], color=GOOD, marker="o", linewidth=2,
               label="beats its null at that end"),
        Line2D([], [], color=BAD, marker="o", linewidth=2,
               label="does not: the interval contains zero on healthy points"),
        Line2D([], [], color=UNDET, marker="o", linewidth=2,
               label="undetermined: an arm at that end did not produce a "
                     "distribution, wherever its interval fell"),
        Line2D([], [], color=GREY, marker="o", markerfacecolor="white",
               linestyle="none", markeredgewidth=1.6,
               label="hollow: an arm at that end did not produce readable answers"),
    ], loc="lower center", ncol=2, fontsize=8.6, frameon=False,
        bbox_to_anchor=(0.5, 0.055))
    note = _comparison_notes(analysis, games)
    if note:
        fig.text(0.5, 0.012, note, ha="center", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=(0, 0.075 + (0.03 if note else 0.0), 1, 1))
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


#: Each figure and the stem its filename is built from.
FIGURES = ((figure_pole_shares, "steering_pole_shares"),
           (figure_own_measure, "steering_own_measure"),
           (figure_vs_null, "steering_vs_null"))


def figure_files(policy):
    """`(builder, filename)` for one policy's three figures.

    Both policies write into ONE `figures/` directory, so the policy is part of
    every name - the convention `vectors/` already uses for its strict and
    relaxed pairs. The suffix comes from the analysis's own policy; there is no
    unsuffixed name to fall back to.
    """
    return [(build, "%s_%s.png" % (stem, policy)) for build, stem in FIGURES]


def _require_drawable(name, game):
    """Refuse an analysis this cannot draw, by name, rather than deep in a panel.

    `run_sweep.py` takes `--ks` and `--null-ks`, so a legitimate subset run can
    land here missing a rung this needs. Every panel measures its baseline and
    its bands against the shared k=0 no-op, and the whole figure is each arm
    against its own null, so neither is optional.
    """
    for arm in ("decision", "shuffled-null"):
        if arm not in game.get("arms", {}):
            raise SystemExit("%s: the analysis has no %s arm; this figure draws "
                             "each real arm against its own null" % (name, arm))
        if "0" not in game["arms"][arm]["points"]:
            raise SystemExit("%s: the %s arm has no k=0 point; every baseline and "
                             "every band here is measured against it" % (name, arm))
    ladder_ends(game)


def main():
    here = Path(__file__).resolve()
    package = here.parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(package / "analysis" / "steering.json"))
    ap.add_argument("--out-dir", default=str(package / "figures"),
                    help="where the three land; both policies write to the same "
                         "directory and are told apart by the name")
    ap.add_argument("--label", default=None,
                    help="which vectors these are, for the headings; defaults to "
                         "the analysis's own pole policy")
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis).read_text())
    policy = analysis.get("policy")
    if policy not in ("strict", "relaxed"):
        raise SystemExit("%s carries pole policy %r, which is not one this draws"
                         % (args.analysis, policy))
    games = [game for game in GAMES if game in analysis["games"]]
    unknown = sorted(set(analysis["games"]) - set(GAMES))
    if unknown:
        raise SystemExit("%s holds games this does not know how to label: %s"
                         % (args.analysis, unknown))
    if not games:
        raise SystemExit("%s holds no games" % args.analysis)
    for name in games:
        _require_drawable(name, analysis["games"][name])
    label = args.label or policy

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for build, name in figure_files(policy):
        print("wrote", build(analysis, games, policy, label, out_dir / name))


if __name__ == "__main__":
    main()
