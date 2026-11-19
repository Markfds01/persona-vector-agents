"""Integrity checks of the full generation run (PREREG D1 `div8_main`), from the rows on disk.

    python scripts/validation/full_generation_check.py            # from the package directory

Reads `rows_full/` and `provenance/generation_full/` (DEVIATIONS 2026-11-03) and writes
`provenance/generation_full/integrity.json` and `analysis/full_generation_report.md`. Nothing is
generated or judged. Checks, each a pass/fail:

* every planned point exists and is whole (120 rows, one per (question, sample), the frozen
  question file's sha256, the arm table's vector sha256, the full-run interpreter);
* the hook fired exactly as often as the batches' lengths say, at every hooked point;
* k = 0 byte-identity: the hooked baseline against the unhooked run, with its positive
  controls (the unhooked run reproduces itself; k != 0 changes ids) — the pilot's rule
  (`pilot_validation.control_verdict`);
* every full-run row names the full-run interpreter, and no pilot row does. (A full-run row
  can carry the same output ids as a pilot row - the k = 0 baseline does when both interpreters
  generate the same tokens - so the overlap of ids is reported, not checked.)

Reported, not checked: counts per arm / set / k, truncation, a regex refusal heuristic (the
per-game study's `audit.parse._REFUSAL_RX`; the judges' REFUSAL status comes with judging),
bridge parse statuses (not the choices), generated-token lengths, run time. Blinding: no answer
text is written, only counts.
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
REPO = PACKAGE.parents[1]
for sub in ("judging", "measurement", "validation"):
    sys.path.insert(0, str(PACKAGE / "scripts" / sub))
sys.path.insert(0, str(REPO))

import bridge_parse  # noqa: E402
import pilot_validation as pv  # noqa: E402
from audit import parse as audit_parse  # noqa: E402
from lab import general_questions as gq  # noqa: E402

ROWS = PACKAGE / "rows_full"
PROV = PACKAGE / "provenance" / "generation_full"
TORCH, TRANSFORMERS = "2.6.0+cu124", "4.52.3"
SAMPLES, N_QUESTIONS = 3, 40

DECISION = ["decision/%s" % g for g in ("dictator", "trust", "ultimatum", "apology", "overfishing",
                                         "prisoners_dilemma")]
SECONDARY = ["relaxed/ultimatum", "relaxed/overfishing", "dictator_only"]
NULLS = ["%s/%s" % (n, g) for n in ("shuffled_null", "orthogonal_null")
         for g in ("dictator", "trust", "ultimatum", "apology", "overfishing", "prisoners_dilemma")]
TRAIT = "trait/altruism"
ALL22 = DECISION + SECONDARY + [TRAIT] + NULLS


def planned_points():
    """(arm, set, k) of every planned point, PREREG section 4's table plus the unhooked controls."""
    plan = [("baseline", s, 0.0) for s in ("heldout", "in_distribution", "bridge")]
    plan += [("unhooked", "heldout", 0.0), ("unhooked_repeat", "heldout", 0.0)]
    plan += [(a, "heldout", k) for a in ALL22 for k in (-3.0, 3.0)]
    plan += [(TRAIT, "heldout", k) for k in (-5.0, -1.0, 1.0, 5.0)]
    plan += [(a, "in_distribution", k) for a in DECISION + [TRAIT] for k in (-3.0, 3.0)]
    plan += [(a, "bridge", k) for a in ALL22 for k in (-3.0, 3.0)]
    return plan


def arm_shas():
    table = json.loads((PACKAGE / "provenance" / "arm_vectors.json").read_text())
    shas = {a["arm"]: a["sha256"] for a in table["arms"]}
    shas["baseline"] = shas[TRAIT]
    shas["unhooked"] = shas["unhooked_repeat"] = None
    return shas


def check_point(rows, arm, qset, k, question_sha, vector_sha):
    problems = []
    if len(rows) != N_QUESTIONS * SAMPLES:
        problems.append("rows %d" % len(rows))
    keys = {(r["question_id"], r["sample_index"]) for r in rows}
    if len(keys) != len(rows) or {s for _, s in keys} != set(range(SAMPLES)) or len({q for q, _ in keys}) != N_QUESTIONS:
        problems.append("(question, sample) pairs not one each")
    for field, want in (("arm", arm), ("question_set", qset), ("k", k), ("question_file_sha256", question_sha),
                        ("torch_version", TORCH), ("engine_version", TRANSFORMERS),
                        ("max_new_tokens", 1000), ("seed", 0), ("use_model_defaults", False)):
        got = {r.get(field) for r in rows}
        if got != {want}:
            problems.append("%s %s" % (field, sorted(map(str, got))))
    got = {r.get("vector_sha256") for r in rows}
    if got != {vector_sha}:
        problems.append("vector_sha256 %s" % sorted(map(str, got)))
    return problems


def main():
    shas = arm_shas()
    question_shas = {s: gq.load_questions(PACKAGE, s)[1] for s in gq.QUESTION_SETS}
    hook_points = {}
    for path in sorted(PROV.glob("generation_points_*.json")):
        for p in json.loads(path.read_text())["points"].values():
            hook_points["%s k%+g|%s" % (p["arm"], p["k"], p["question_set"])] = p

    points, missing, problems = {}, [], {}
    for arm, qset, k in planned_points():
        path = ROWS / (gq.point_slug(arm, qset, k) + ".jsonl")
        name = "%s k%+g|%s" % (arm, k, qset)
        if not path.exists():
            missing.append(name)
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        points[name] = rows
        bad = check_point(rows, arm, qset, k, question_shas[qset], shas[arm])
        if bad:
            problems[name] = bad
    planned = {"%s k%+g|%s" % (a, k, s) for a, s, k in planned_points()}
    planned_paths = {ROWS / (gq.point_slug(a, s, k) + ".jsonl") for a, s, k in planned_points()}
    extra = sorted(str(p.relative_to(ROWS)) for p in ROWS.rglob("*")
                   if p.is_file() and p not in planned_paths and p.name != ".gitignore")

    hooked = {n: p for n, p in hook_points.items() if p.get("hooked")}
    hook_ok = {n: p["hook_fires"] == p["hook_fires_expected"] and p["hook_fires"] > 0 for n, p in hooked.items()}
    hook_missing = sorted(n for n in points if not n.startswith("unhooked") and n not in hooked)
    identity = pv.identity_checks(points, None)
    controls = pv.control_verdict(identity, hooked)

    pilot_ids = {json.loads(line)["output_ids_sha256"] + "|" + json.loads(line)["row_id"]
                 for p in (PACKAGE / "rows").rglob("*.jsonl") for line in p.read_text().splitlines() if line.strip()}
    pilot_torch = {json.loads(line)["torch_version"]
                   for p in (PACKAGE / "rows").rglob("*.jsonl") for line in p.read_text().splitlines()[:1]}
    full_rows = [r for rows in points.values() for r in rows]
    overlap = sum((r["output_ids_sha256"] + "|" + r["row_id"]) in pilot_ids for r in full_rows)
    # descriptive: the pilot's k = 0 baseline (other interpreter) against the full run's
    pilot_vs_full_k0 = {}
    for qset in ("heldout", "bridge"):
        pilot_path = PACKAGE / "rows" / "baseline" / ("%s_k+0.jsonl" % qset)
        name = "baseline k+0|%s" % qset
        if pilot_path.exists() and name in points:
            a = {(r["question_id"], r["sample_index"]): r["output_ids_sha256"]
                 for r in map(json.loads, pilot_path.read_text().splitlines())}
            b = {(r["question_id"], r["sample_index"]): r["output_ids_sha256"] for r in points[name]}
            pilot_vs_full_k0[qset] = [sum(a[x] == b.get(x) for x in a), len(a)]

    refusal = lambda text: bool(audit_parse._REFUSAL_RX.search(text or ""))  # noqa: E731
    per_point = {}
    for name, rows in sorted(points.items()):
        arm_k, qset = name.split("|")
        entry = {"rows": len(rows),
                 "truncated": sum(r["generation_status"] == "truncated" for r in rows),
                 "refusal_regex": sum(refusal(r.get("answer")) for r in rows),
                 "mean_tokens": statistics.mean(r["n_generated_tokens"] for r in rows),
                 "max_tokens": max(r["n_generated_tokens"] for r in rows),
                 "minutes": hook_points.get(name, {}).get("minutes"),
                 "device": hook_points.get(name, {}).get("run", {}).get("device")}
        if qset == "bridge":
            entry["bridge_parse"] = dict(Counter(bridge_parse.parse_choice(r["answer"]).status for r in rows))
        per_point[name] = entry

    checks = {
        "all_points_present": not missing,
        "all_points_whole_and_consistent": not problems,
        "no_unplanned_files": not extra,
        "hook_fired_exactly_as_expected": bool(hook_ok) and all(hook_ok.values()) and not hook_missing,
        "k0_identity_verdict": controls["verdict"],
        "every_row_names_the_full_run_interpreter": bool(full_rows) and all(
            r["torch_version"] == TORCH and r["engine_version"] == TRANSFORMERS for r in full_rows)
        and TORCH not in pilot_torch,
    }
    checks["pass"] = all(v is True for k, v in checks.items() if k != "k0_identity_verdict") \
        and controls["verdict"] == "pass"
    out = {"what": "full generation run integrity (scripts/validation/full_generation_check.py)",
           "planned_points": len(planned), "present_points": len(points), "missing_points": missing,
           "point_problems": problems, "unplanned_files": extra, "hook_points_missing": hook_missing,
           "hook_mismatch": sorted(n for n, ok in hook_ok.items() if not ok),
           "identity": identity, "controls": controls, "pilot_torch_versions": sorted(pilot_torch),
           "rows_also_in_pilot": overlap, "pilot_vs_full_k0_identical_rows": pilot_vs_full_k0,
           "per_point": per_point, "checks": checks}
    (PROV / "integrity.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    write_report(out, points, hook_points)
    print(json.dumps(checks))
    return 0 if checks["pass"] else 1


def write_report(out, points, hook_points):
    rows = [r for rs in points.values() for r in rs]
    by_set = Counter(r["question_set"] for r in rows)
    lines = ["# Full generation run — report", "",
             "Written by `scripts/validation/full_generation_check.py` from `rows_full/` and "
             "`provenance/generation_full/` (numbers also in `provenance/generation_full/integrity.json`). "
             "Generation only: nothing has been judged. Blinding: counts only, no answer text.", "",
             "## Checks", "", "| check | result |", "|---|---|"]
    for k, v in out["checks"].items():
        lines.append("| %s | %s |" % (k, v))
    c = out["controls"]
    idn = out["identity"]
    lines += ["", "k = 0 identity: hooked baseline vs unhooked identical rows %s; unhooked vs repeat %s; "
              "k != 0 points identical to k = 0 (max over points) %s; verdict **%s**."
              % (idn["k0_hooked_vs_unhooked_identical_rows"], idn["unhooked_vs_unhooked_repeat_identical_rows"],
                 max((v for v in idn["k_nonzero_vs_k0_identical_rows"].values() if v), default=None),
                 c["verdict"]),
              "",
              "Descriptive (not a check): the pilot's k = 0 baseline (torch 2.7.1 / transformers 4.54.0) "
              "against the full run's (torch 2.6.0 / transformers 4.52.3), identical output ids "
              "[identical, rows]: %s. Full-run rows whose (row id, output ids) also occur in the pilot: %d."
              % (out["pilot_vs_full_k0_identical_rows"], out["rows_also_in_pilot"]),
              "", "## Counts", "",
              "Points: %d planned, %d present. Rows: %d (%s)."
              % (out["planned_points"], out["present_points"], len(rows),
                 ", ".join("%s %d" % (s, by_set[s]) for s in ("heldout", "in_distribution", "bridge"))),
              "The study's 13,200 rows are these minus the `unhooked_repeat` control (120 rows).", "",
              "| set | arm | k | rows | truncated | refusal (regex) | mean tokens | max tokens | minutes | device |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in out["per_point"].items():
        arm_k, qset = name.split("|")
        arm, k = arm_k.rsplit(" k", 1)
        lines.append("| %s | %s | %s | %d | %d | %d | %.0f | %d | %s | %s |"
                     % (qset, arm, k, e["rows"], e["truncated"], e["refusal_regex"], e["mean_tokens"],
                        e["max_tokens"], "%.1f" % e["minutes"] if e["minutes"] is not None else "—",
                        e["device"] or "—"))
    lines += ["", "## Truncation and refusal by set", "", "| set | rows | truncated | share | refusal (regex) | share |",
              "|---|---|---|---|---|---|"]
    for s in ("heldout", "in_distribution", "bridge"):
        es = [e for n, e in out["per_point"].items() if n.endswith("|" + s)]
        n = sum(e["rows"] for e in es)
        t = sum(e["truncated"] for e in es)
        f = sum(e["refusal_regex"] for e in es)
        if n:
            lines.append("| %s | %d | %d | %.2f%% | %d | %.2f%% |" % (s, n, t, 100 * t / n, f, 100 * f / n))
    bp = Counter()
    for n, e in out["per_point"].items():
        bp.update(e.get("bridge_parse", {}))
    lines += ["", "Bridge parse statuses over all bridge rows (`scripts/measurement/bridge_parse.py`; the "
              "choices themselves are not reported here): %s." % dict(sorted(bp.items())),
              "", "The refusal column is the per-game study's regex (`audit.parse._REFUSAL_RX`), a heuristic "
              "built for game answers; the judges' REFUSAL status is the measure the analysis uses.", ""]
    minutes = [p["minutes"] for p in hook_points.values() if p.get("minutes") is not None]
    lines += ["## Run time", "", "Sum of per-point generation minutes over all processes: %.0f (%.1f GPU-process "
              "hours); the wall-clock window is in the part logs (`provenance/generation_full/generation_part*.log`)."
              % (sum(minutes), sum(minutes) / 60), ""]
    stamps = sorted(re.findall(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", "".join(
        p.read_text() for p in PROV.glob("generation_part*.log"))))
    if stamps:
        lines += ["Wall clock, first part start to last part end: %s to %s." % (stamps[0], stamps[-1]), ""]
    lines += ["## How the run went (2026-11-03)", "", NOTES, ""]
    (PACKAGE / "analysis" / "full_generation_report.md").write_text("\n".join(lines) + "\n")


#: facts of the 2026-11-03 run that the rows and logs record but do not explain
NOTES = """\
* Interpreter: torch 2.6.0+cu124 / transformers 4.52.3 (`provenance/interpreter_full_run.md`); every row
  names it (checked above).
* Processes: parts 1-4 (`scripts/run_full_generation.sh`) started 11:12:30 UTC, two per A40. When part 4
  finished (11:37:48), parts 1 and 2 were stopped and their remaining held-out points relaunched as parts
  5-7 (DEVIATIONS 2026-11-03, "Full-run processes rebalanced"); the two points in progress were
  regenerated whole. Part 3's driver printed a shell syntax error after its last point (12:54:29): the
  driver file had been edited (parts 5-7 added) while that bash was still reading it. Every point of
  part 3 had already been written and checked; only its closing `PART 3 COMPLETE` line is missing.
* The full run reproduces the pilot byte for byte where they overlap: all 2,400 pilot rows (same arm,
  set, k, question and sample) have identical output ids in the full run, although the pilot ran under
  torch 2.7.1+cu126 / transformers 4.54.0. So, for this model and these points, the interpreter change
  did not change a single generated token.
* Nothing was judged; the counts above contain no score."""


if __name__ == "__main__":
    sys.exit(main())
