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
  reach when the distance to it is inside the baseline's own Wilson half-width,
  which selects exactly those two arms and nothing else.
* `point_flags` — two DIFFERENT kinds of degraded point, kept apart because they
  mean different things. `degenerate` is "the model stopped answering in English"
  (non-Latin share >= 0.5); it is true only at the Prisoner's Dilemma's k=+4 and
  k=+5, where EVERY answer is non-Latin, and those points are struck through and
  excluded from the supported range: they are a degeneration, not a result.
  `low_coverage` is "the scorer could not read enough of it" (parse coverage <
  0.90); it is a caveat drawn on the marker, not a disqualification.
* `null_verdict` — beating the null needs the real arm to move FURTHER than the
  null in the steered direction, so the sign is checked, not just the interval.
  A comparison that fails to exclude zero on a degraded point is reported
  `undetermined`, not `null`: the Ultimatum's k=-5 null parsed 8 answers of 100,
  and a wide interval there is missing evidence rather than evidence of absence.

Reads `analysis/steering.json` only. No torch, no GPU, no re-analysis — every
number drawn here is one `analyze_game.py` wrote.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

#: The ladder's axis: the layer-20 norm of the repo's shipped altruism trait
#: vector. METHOD.md section 3 says why it is this constant and not another.
REFERENCE_NORM = 10.508308410644531

#: Every answer non-Latin means the measurement is no longer about the game. Half
#: is far past any rate the healthy points show (they sit at 0.00-0.13).
DEGENERATE_NON_LATIN = 0.5
#: Below this the point is drawn hollow with its parsed count: still measured,
#: measured on less.
LOW_COVERAGE = 0.90

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

def point_flags(point):
    """`(degenerate, low_coverage)` for one beta point.

    Degenerate is the hard one - the answers are not in English any more and the
    point is not a result. Low coverage is a caveat: fewer rows resolved.
    """
    degeneracy = point["degeneracy"]
    return (degeneracy["share_with_non_latin_script"] >= DEGENERATE_NON_LATIN,
            point["parse_coverage"] < LOW_COVERAGE)


def wilson_half_width(pole):
    return (pole["hi"] - pole["lo"]) / 2.0


def no_room(baseline_pole, side):
    """Is the bound on `side` already inside the baseline's own noise?

    `side` is +1 for the altruistic bound at 1.0, -1 for the self-interested one
    at 0.0. An arm with no room cannot be read as a failed intervention, so the
    figure marks the ceiling or the floor instead of letting a flat line lie.
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


def null_verdict(game):
    """Per end: does the real arm beat its own null, fail to, or say nothing?

    `beats` needs the difference to exclude zero AND to point the way the arm is
    being steered - a significant difference the wrong way is not a win. A
    comparison that fails to exclude zero on a degraded point is `undetermined`,
    because a collapsed arm widens the interval and proves nothing either way.
    """
    out = {}
    for key, side in (("-5", -1), ("5", +1)):
        contrast = game["decision_vs_null"][key]["altruistic_decision_minus_null"]
        excludes = contrast["excludes_zero"]
        right_way = (contrast["diff"] > 0) if side > 0 else (contrast["diff"] < 0)
        if excludes and right_way:
            out[side] = "beats"
            continue
        degraded = False
        for arm in ("decision", "shuffled-null"):
            point = game["arms"][arm]["points"].get(key)
            if point and any(point_flags(point)):
                degraded = True
        out[side] = "undetermined" if (not excludes and degraded) else "null"
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


def _pole(point):
    pole = point["poles"]["strict"]["altruistic"]
    return (pole["lo"], pole["hi"], pole["p"])


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


def _axis_furniture(ax, xlabel=True):
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_xlim(-59, 59)
    ticks = [k * REFERENCE_NORM for k in range(-5, 6)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%+d" % k if k else "0" for k in range(-5, 6)], fontsize=8)
    if xlabel:
        ax.set_xlabel("k   (unit beta = k x %.4f, the length of the layer-20 edit)"
                      % REFERENCE_NORM, fontsize=8.5)


def _mark_bands(ax, game, arm, ymin, ymax, on_pole_axis):
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

    baseline = arm["points"]["0"]["poles"]["strict"]["altruistic"]
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


def _panel(ax, game_name, game, getter, ylabel, ylim, on_pole_axis):
    decision = game["arms"]["decision"]
    null = game["arms"]["shuffled-null"]
    baseline_point = decision["points"]["0"]
    baseline = getter(baseline_point)[2]

    ymin, ymax = ylim
    _mark_bands(ax, game, decision, ymin, ymax, on_pole_axis)
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


FOOTER = (
    "Qwen2.5-7B-Instruct rev a09a3545, bfloat16, sdpa, altruism_v3, mode free, layer 20, positions=all, norm=unit, neutral preset, seed 0, n = 100 per point, strict poles.\n"
    "Error bars are 95%% intervals - Wilson on a share, Student's t on a mean. One k is the SAME sized activation edit in every game (unit beta = k x %.4f), which is what makes the six panels\n"
    "comparable; the per-game equivalent raw beta is unit beta / ||v||@20. k=0 is one shared no-op run, verified byte-identical across the two arms. The null is the same construction on the same\n"
    "activations over the same cells with the pole labels permuted within each cell - it is not inert, and where it moves further than the real arm the real arm has shown nothing."
) % REFERENCE_NORM


def figure_pole_shares(analysis, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.0))
    for ax, name in zip(axes.ravel(), GAMES):
        _panel(ax, name, analysis["games"][name], _pole,
               "P(altruistic pole)", (-0.05, 1.08), True)
    axes.ravel()[2].annotate(
        "the null gets to 0.495 where the real arm gets to 0.515",
        xy=(5 * REFERENCE_NORM, 0.495), xytext=(2, 0.80), fontsize=8.2,
        color=BAD, ha="left",
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))
    fig.suptitle("Each game steered with its OWN decision vector, against its OWN null:"
                 " three of six clear that bar at both ends, one clears it at neither",
                 fontsize=14.5, y=0.988)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3, fontsize=8.8,
               frameon=False, bbox_to_anchor=(0.5, 0.098))
    fig.text(0.5, 0.006, FOOTER, ha="center", fontsize=7.7, color="#444444")
    fig.tight_layout(rect=(0, 0.170, 1, 0.955), h_pad=4.2)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def figure_own_measure(analysis, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.0))
    for ax, name in zip(axes.ravel(), GAMES):
        game = analysis["games"][name]
        values = [point["measure"]["ci_high"]
                  for arm in game["arms"].values()
                  for point in arm["points"].values()
                  if point["measure"]["ci_high"] is not None]
        top = max(values) * 1.20 if values else 1.0
        units = game["measure_units"]
        if not game["altruistic_is_high_on_own_measure"]:
            units += "   (HIGHER = LESS altruistic)"
        _panel(ax, name, game, _measure, units, (-0.04 * top, top), False)
    fig.suptitle("The same six arms on each game's own measure - not comparable across "
                 "games, which is why every shared claim is made on the pole share",
                 fontsize=14.5, y=0.988)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3, fontsize=8.8,
               frameon=False, bbox_to_anchor=(0.5, 0.098))
    fig.text(0.5, 0.006, FOOTER, ha="center", fontsize=7.7, color="#444444")
    fig.tight_layout(rect=(0, 0.170, 1, 0.955), h_pad=4.2)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def figure_vs_null(analysis, out_path):
    """The bar itself: decision minus null at each extreme, for all six games."""
    fig, ax = plt.subplots(figsize=(11.6, 7.4))
    colours = {"beats": GOOD, "null": BAD, "undetermined": UNDET}
    rows, y = [], 0.0
    for name in GAMES:
        game = analysis["games"][name]
        verdict = null_verdict(game)
        for key, side in (("5", +1), ("-5", -1)):
            contrast = game["decision_vs_null"][key]["altruistic_decision_minus_null"]
            rows.append((y, side, contrast, verdict[side], name))
            y -= 1.0
        y -= 0.6

    for pos, side, contrast, state, _name in rows:
        colour = colours[state]
        ax.errorbar([contrast["diff"]], [pos],
                    xerr=[[contrast["diff"] - contrast["lo"]],
                          [contrast["hi"] - contrast["diff"]]],
                    color=colour, marker="o" if side > 0 else "s", capsize=3.5,
                    markersize=7, linewidth=2.0, zorder=4)
        ax.text(contrast["hi"] + 0.02, pos, "k=%+d" % (5 * side), va="center",
                fontsize=8.2, color=colour)

    ax.axvline(0, color="#333333", linewidth=1.2, zorder=2)
    ax.set_yticks([rows[2 * i][0] - 0.5 for i in range(len(GAMES))])
    ax.set_yticklabels([LABELS[g] for g in GAMES], fontsize=10.5)
    ax.set_xlabel("P(altruistic), decision arm minus its own shuffled-label null "
                  "   (95% Newcombe interval)", fontsize=9.5)
    ax.set_xlim(-1.12, 1.28)
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.6)
    ax.set_title("Beating the null is the bar. An interval touching the line is not "
                 "a result.", fontsize=12.5, loc="left", pad=22)
    ax.text(0.0, 1.012,
            "A point LEFT of the line at k=-5 and RIGHT of it at k=+5 is the real "
            "arm moving further than the null in the steered direction.",
            transform=ax.transAxes, fontsize=8.6, color=GREY)
    fig.legend(handles=[
        Line2D([], [], color=GOOD, marker="o", linewidth=2,
               label="beats its null at that end"),
        Line2D([], [], color=BAD, marker="o", linewidth=2,
               label="does not: the interval contains zero on healthy points"),
        Line2D([], [], color=UNDET, marker="o", linewidth=2,
               label="undetermined: the interval contains zero, but a point there "
                     "is degraded"),
    ], loc="lower center", ncol=3, fontsize=8.6, frameon=False,
        bbox_to_anchor=(0.5, 0.088))
    fig.text(0.5, 0.012,
             "Ultimatum is the negative of the run: its k=+5 comparison is a clean "
             "null on healthy arms (+0.021 [-0.117, +0.157]), and its k=-5 "
             "comparison settles nothing because the NULL arm there\nparsed 8 "
             "answers of 100. Overfishing's k=+5 and the Prisoner's Dilemma's k=-5 "
             "are a ceiling and a floor - those arms had nowhere to go.",
             ha="center", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=(0, 0.135, 1, 1))
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def main():
    here = Path(__file__).resolve()
    package = here.parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(package / "analysis" / "steering.json"))
    ap.add_argument("--out-dir", default=str(package / "figures"))
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis).read_text())
    if analysis.get("policy") != "strict":
        raise SystemExit("these figures report the strict poles; %s carries policy %r"
                         % (args.analysis, analysis.get("policy")))
    missing = sorted(set(GAMES) - set(analysis["games"]))
    if missing:
        raise SystemExit("%s has no rows for %s" % (args.analysis, missing))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for build, name in ((figure_pole_shares, "steering_pole_shares.png"),
                        (figure_own_measure, "steering_own_measure.png"),
                        (figure_vs_null, "steering_vs_null.png")):
        print("wrote", build(analysis, out_dir / name))


if __name__ == "__main__":
    main()
