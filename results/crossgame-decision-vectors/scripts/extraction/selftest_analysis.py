"""Exercise the six-game battery on synthetic activations. No GPU, no real data.

The real run costs hours on a shared GPU, so the analysis must not be first
executed on its output. This builds fake rows and fake activations for three
families - one of them deliberately single-poled, like the Prisoner's Dilemma -
and runs `analyze_crossgame.py` over them twice, once with a planted direction and
once with none. That is the battery of README sections 1-4 and 8; the pooling
stage (`scripts/pooling/`) is NOT covered here.

MANUAL: about forty seconds, and no test command runs it. `python -m pytest -q`
from the repository root is the suite; this is the check you run by hand before
trusting the analysis on real data. It checks that:

  * a planted direction is recovered with a high out-of-sample AUC
  * pure noise gives an AUC near 0.5, so the measurement is not self-fulfilling
  * a single-poled family is excluded from every pooled structure rather than
    silently contributing one side of the contrast
  * the agreement matrix, the label nulls and leave-one-game-out all complete
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# this directory holds the grid definition; the repo root is four up
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import crossgame_grid  # noqa: E402

LAYERS, HIDDEN = 29, 3584


def build(tmp, planted_strength):
    """Three families. dictator+overfishing get both poles, PD only self."""
    crossgame_grid.register()
    torch.manual_seed(7)
    planted = torch.randn(LAYERS, HIDDEN)
    planted /= planted.norm(dim=1, keepdim=True)

    manifest = {"families": []}
    for family, samples, single_pole in (("dictator", 6, False),
                                         ("overfishing", 6, False),
                                         ("prisoners_dilemma", 6, True)):
        cells = [g for g in crossgame_grid.GRID if g.family == family]
        rows, acts = [], []
        for game in cells:
            for k in range(samples):
                if family == "prisoners_dilemma":
                    value, tag = 0.0, "defect_decl"
                elif family == "dictator":
                    alt = k % 2 == 0
                    value = (game.pole_scale if alt else 0.0)
                    tag = "a2_anchor"
                else:
                    alt = k % 2 == 0
                    # overfishing: altruistic is the SMALL catch, self is the max
                    value = ((game.pole_scale / 2.0) if alt
                             else float(game.answer_space.stated.high))
                    tag = "fish"
                rows.append({"game_id": game.id, "value": value, "tag": tag})
                is_alt = (family != "prisoners_dilemma") and (k % 2 == 0)
                noise = torch.randn(LAYERS, HIDDEN)
                signal = planted * (planted_strength if is_alt else -planted_strength)
                acts.append(noise + signal)

        acts_dir = tmp / ("acts_%s" % family)
        acts_dir.mkdir(parents=True)
        rows_csv = acts_dir / "rows.csv"
        with rows_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["game_id", "value", "tag"])
            writer.writeheader()
            writer.writerows(rows)
        stacked = torch.stack(acts)
        torch.save({"row_index": torch.arange(len(rows)),
                    "prompt_len": torch.full((len(rows),), 10),
                    "total_len": torch.full((len(rows),), 20),
                    "prompt_avg": stacked, "prompt_last": stacked,
                    "response_avg": stacked}, acts_dir / "shard_0000.pt")
        (acts_dir / "meta.json").write_text(json.dumps({"synthetic": True}))
        manifest["families"].append({"family": family, "rows_csv": str(rows_csv),
                                     "acts_dir": str(acts_dir)})
    path = tmp / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def run(tmp, manifest, label):
    out = tmp / ("out_" + label)
    cmd = [sys.executable, str(HERE / "analyze_crossgame.py"),
           "--manifest", str(manifest), "--out", str(out),
           "--their-vectors", str(ROOT / "persona_vectors" / "Qwen2.5-7B-Instruct"),
           "--shuffles", "40", "--pair-shuffles", "20"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-3000:]); print(done.stderr[-3000:])
        raise SystemExit("analysis failed for %s" % label)
    return json.loads((out / "analysis.json").read_text())


def main():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)

        signal = run(tmp, build(tmp / "signal", 10.0), "signal")
        noise = run(tmp, build(tmp / "noise", 0.0), "noise")

        for label, report in (("planted signal", signal), ("pure noise", noise)):
            R = report["by_policy"]["strict"]
            auc = R["pooled"]["split_half"]["by_layer"][20]["auc"]
            print("%-15s pooled layer-20 AUC = %.3f | pooled families %s | excluded %s"
                  % (label, auc, R["pooled_families"], R["families_excluded_from_pool"]))
            pairs = R["agreement_matrix_cellbalanced"]["pairs"]
            for k, v in pairs.items():
                print("    agreement %-42s cos %+.3f  null p97.5 %+.3f"
                      % (k, v["cosine"], v["label_null"]["p97.5"]))
            for fam, e in R["leave_one_game_out"].items():
                if e.get("usable"):
                    print("    LOGO held-out %-20s layer-20 AUC %.3f"
                          % (fam, e["by_layer"][20]["auc"]))

        checks = []
        s = signal["by_policy"]["strict"]
        n = noise["by_policy"]["strict"]
        checks.append(("planted signal recovered (pooled AUC > 0.9)",
                       s["pooled"]["split_half"]["by_layer"][20]["auc"] > 0.9))
        checks.append(("noise gives chance AUC (0.35 < AUC < 0.65)",
                       0.35 < n["pooled"]["split_half"]["by_layer"][20]["auc"] < 0.65))
        checks.append(("single-poled family excluded from the pool",
                       s["families_excluded_from_pool"] == ["prisoners_dilemma"]))
        checks.append(("single-poled family has no vector",
                       not s["per_game"]["prisoners_dilemma"]["usable"]))
        checks.append(("agreement is high between planted families",
                       all(v["cosine"] > 0.8 for v in
                           s["agreement_matrix_cellbalanced"]["pairs"].values())))
        checks.append(("agreement is not high under noise",
                       all(abs(v["cosine"]) < 0.6 for v in
                           n["agreement_matrix_cellbalanced"]["pairs"].values())))
        checks.append(("leave-one-out transfers under planted signal",
                       all(e["by_layer"][20]["auc"] > 0.9
                           for e in s["leave_one_game_out"].values() if e.get("usable"))))
        print()
        ok = True
        for name, passed in checks:
            print("  %-52s %s" % (name, "PASS" if passed else "FAIL"))
            ok = ok and passed
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
