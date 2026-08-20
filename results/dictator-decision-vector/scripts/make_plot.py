"""The steering arms on one matched intervention axis, with the decision arm's
responsive range marked. That range is ASYMMETRIC and the band is drawn that way.

Left: mean dollars given. Right: P(gives exactly $0) - one of the two poles the
decision vector was constructed from, so the contrast the test is really about.

POSITIVE side: the decision arm keeps rising past 10.51 to a peak at +21.02
(+21.02 vs +10.51 is +5.39 [+2.78, +8.01], p = 6.2e-05) and declines significantly
beyond it (+52.54 vs +21.02: -5.44 [-7.19, -3.70], p = 2.2e-09).

NEGATIVE side: the arm IS saturated from the first step. It delivers 97% of its
movement by -10.51 and nothing after that is distinguishable from it (-21.02 p=0.25,
-31.52 p=0.40, -42.03 p=0.16, -52.54 p=0.75).

So the band runs -10.51 to +21.02. Either way the extreme comparison is
uninformative - on the right because the arm is going backwards, on the left because
it stopped moving at the first step - and the measurement that separates mechanism
from magnitude is the one at +-10.51, not the one at the edge.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/marco/dockmaster/data/steer-decision")
UNIT = 10.508308410644531
OUR_NORM = 6.872108459472656
PEAK = 2 * UNIT  # positive side: the decision arm's measured peak, +21.02
SAT = UNIT       # negative side: saturated from -10.51 out (97% of movement)
TEST = UNIT      # +-10.51: the coefficient the mechanism test is read at

table = json.loads((OUT / "comparison.json").read_text())["table"]
poles = json.loads((OUT / "pole_curves.json").read_text())


def carry_zero(rows, arm):
    """beta=0 is delta=0 for every vector - one shared no-op run, verified
    byte-identical across arms. Arms not run at 0 carry it so their line is not
    drawn straight through a point that was never measured."""
    if any(r["arm"] == arm and r["their_raw_beta"] == 0 for r in rows):
        return rows
    zero = dict(next(r for r in rows if r["arm"] == "decision"
                     and r["their_raw_beta"] == 0))
    zero["arm"] = arm
    return rows + [zero]


table = carry_zero(table, "orthogonal-null")
poles = carry_zero(poles, "orthogonal-null")

STYLE = {
    "decision":        ("#1b6ca8", "o", "-",  2.2, "decision vector (ours)"),
    "theirs":          ("#c1440e", "s", "-",  1.7, "altruism trait vector (theirs)"),
    "shuffled-null":   ("#5c8001", "^", "--", 1.7, "shuffled-label null (cos +0.242 to ours)"),
    "orthogonal-null": ("#7a3b9e", "D", ":",  2.2, "orthogonal null (cos ~0 to ours)"),
}
ORDER = ["decision", "theirs", "shuffled-null", "orthogonal-null"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))

panels = [
    (axes[0], table, "mean", "ci_low", "ci_high", 12.96,
     "mean given, dollars of a $100 endowment", "Mean amount given",
     "unsteered baseline $12.96", (-4, 88), 13.9),
    (axes[1], poles, "p_give_zero", "p_give_zero_lo", "p_give_zero_hi", 0.610,
     "P(gives exactly $0)", "Probability of the self-interested pole",
     "unsteered baseline 0.610", (-0.05, 1.05), 0.635),
]

for ax, data, ykey, lokey, hikey, base, ylab, title, basetxt, ylim, texty in panels:
    for arm in ORDER:
        pts = sorted([r for r in data if r["arm"] == arm], key=lambda r: r["unit_beta"])
        if not pts:
            continue
        colour, marker, line, lw, label = STYLE[arm]
        x = [r["unit_beta"] for r in pts]
        y = [r[ykey] for r in pts]
        lo = [r[ykey] - r[lokey] for r in pts]
        hi = [r[hikey] - r[ykey] for r in pts]
        ax.errorbar(x, y, yerr=[lo, hi], color=colour, marker=marker, linestyle=line,
                    capsize=3, markersize=6, linewidth=lw, label=label, zorder=4)
    ax.axhline(base, color="#999999", linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
    ax.text(-56, texty, basetxt, fontsize=8.5, color="#666666")
    # the decision arm's responsive range - asymmetric: it saturates at -10.51 on
    # the negative side but keeps rising to +21.02 on the positive one
    ax.axvspan(-SAT, PEAK, color="#1b6ca8", alpha=0.055, zorder=0)
    for s in (-SAT, PEAK):
        ax.axvline(s, color="#1b6ca8", linewidth=1.1, linestyle=(0, (4, 3)),
                   alpha=0.75, zorder=2)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_ylim(ylim)

axes[0].annotate("decision arm peaks at +21.02\nand declines beyond;\n"
                 "flat from $-$10.51 out",
                 xy=(PEAK, 45.5), xytext=(-9, 62), fontsize=8.5, color="#1b6ca8",
                 ha="left", arrowprops=dict(arrowstyle="->", color="#1b6ca8", lw=1))
axes[1].annotate("at $\\pm$10.51 the orthogonal null\nis indistinguishable from baseline",
                 xy=(TEST, 0.548), xytext=(15, 0.80), fontsize=8.5, color="#7a3b9e",
                 ha="left", arrowprops=dict(arrowstyle="->", color="#7a3b9e", lw=1))

for ax in axes:
    ax.set_xlabel("unit beta  =  length of the activation edit added at layer 20")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlim(-58, 58)
    ticks = [k * UNIT for k in range(-5, 6)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%.1f" % t for t in ticks], fontsize=8)
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels([("%+d" % k if k else "0") for k in range(-5, 6)], fontsize=8)
    top.set_xlabel("their raw beta  (unit beta / 10.5083, their vector's layer-20 norm)",
                   fontsize=9)

axes[0].legend(loc="upper left", fontsize=8.8, framealpha=0.95)

fig.suptitle("Steering the Dictator game at matched intervention size: inside the decision "
             "arm's responsive range it does what its orthogonal complement cannot",
             fontsize=13, y=0.995)
fig.text(0.5, 0.008,
         "Qwen2.5-7B-Instruct, altruism_v3 Dictator, mode free, layer 20, positions=all, "
         "neutral preset, seed 0, n=200 per point. Error bars 95%% CI (t for means, Wilson "
         "for shares).\n"
         "Every arm is steered at unit norm, so a given x is the same edit magnitude for "
         "each. Our raw beta = unit beta / %.4f; theirs = unit beta / %.4f. "
         "beta=0 is one shared no-op run (verified byte-identical across arms).\n"
         "Shaded band: the decision arm's responsive range, -10.51 to +21.02, asymmetric - it "
         "saturates at the first negative step (97%% of its movement) but rises to a peak at\n"
         "+21.02. The orthogonal null was run at 4 coefficients; +-31.52 was not run." % (OUR_NORM, UNIT),
         ha="center", fontsize=8.4, color="#444444")

fig.tight_layout(rect=(0, 0.10, 1, 0.962))
fig.savefig(OUT / "steering_comparison.png", dpi=200)
print("wrote", OUT / "steering_comparison.png")
