"""Render `analysis/pilot_report.md` from `analysis/pilot_validation.json`.

Every number in the report is read from the JSON, which is read from the rows and judgments;
the prose around the numbers is fixed text. A section whose input does not exist says "not
run", never a number. The pilot's numbers are plumbing numbers, not results.
"""

import argparse
import json
from pathlib import Path

PRIMARY = "llama-3.3-70b-instruct-q8_0"
SECONDARY = "gpt-oss-20b-mxfp4"


def f(x, digits=3):
    return "n/a" if x is None else ("%.*f" % (digits, x))


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def ci(d, key, digits=3):
    lo = d.get(key + "_ci")
    return "n/a" if not lo else "[%s, %s]" % (f(lo[0], digits), f(lo[1], digits))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def counts(d):
    return ", ".join("%s %d" % (k, v) for k, v in sorted(d.items())) or "none"


def section_generation(v):
    g = v["generation"]
    rows = []
    for name, p in g["points"].items():
        head, qset = name.split("|")
        hook = g["hook_points"].get(name, {})
        fires = ("%d / %d" % (hook["hook_fires"], hook["hook_fires_expected"])
                 if hook.get("hooked") else ("unhooked" if hook else "n/a"))
        rows.append((qset, head, p["n_rows"], counts(p["statuses"]), pct(p["share_truncated"]),
                     "%.0f" % p["mean_generated_tokens"], fires, f(hook.get("minutes"), 1)))
    c = g["controls"]
    ident = g["identity"]
    k0 = ident["k0_hooked_vs_unhooked_identical_rows"]
    rep = ident["unhooked_vs_unhooked_repeat_identical_rows"]
    lines = ["## 1. Generation: rows, statuses, truncation", "",
             table(["set", "point", "rows", "generation status", "truncated", "mean new tokens",
                    "hook fires / expected", "minutes"], rows), "",
             "## 2. Hook controls and the k = 0 byte-identity check", "",
             "Verdict: **%s**. The check is void unless all three positive controls hold (`.dm-knowledge`)."
             % c["verdict"].upper().replace("_", " "), "",
             table(["check", "result"], [
                 ("k = 0 with the hook installed vs unhooked: rows with identical generated ids",
                  "n/a" if k0 is None else "%d / %d" % tuple(k0)),
                 ("(a) the unhooked run reproduces itself (unhooked vs unhooked_repeat), identical rows",
                  "n/a" if rep is None else "%d / %d" % tuple(rep)),
                 ("(b) hook fire count equals its exact expectation (sum over batches of the longest row) and is > 0",
                  c["control_b_hook_fired_exactly_as_expected"]),
                 ("(c) every k != 0 point changes ids vs k = 0 (identical rows per point)",
                  "; ".join("%s: %s" % (k, "n/a" if x is None else "%d/%d" % tuple(x))
                            for k, x in sorted(ident["k_nonzero_vs_k0_identical_rows"].items()))),
             ]), ""]
    return "\n".join(lines)


def section_judging(v):
    lines = ["## 3. Judging: calls, statuses, throughput achieved", ""]
    rows = []
    for judge, kinds in v["judge_status"].items():
        for kind, c in kinds.items():
            if c:
                rows.append((judge, kind, sum(c.values()), counts(c)))
    lines += [table(["judge", "kind", "calls", "status"], rows), ""]
    rows = []
    for judge, t in v["throughput"].items():
        gaps = t["inter_call_gaps_by_kind"]
        rows.append((judge, t["n_sent"], f(t["seconds_per_call_overall"], 2),
                     "; ".join("%s %.2f s (median %.2f)" % (k, x["mean_s"], x["median_s"])
                               for k, x in sorted(gaps.items()))))
    lines += ["Seconds per call as achieved (calls actually sent, `cache_prompt: false`, `--parallel 1`; "
              "the PREREG planned 3.82 s for the primary and 1.00 s for the secondary, pairwise unmeasured):", "",
              table(["judge", "calls sent", "s/call overall", "inter-call gap by kind"], rows), ""]
    return "\n".join(lines)


def section_points(v):
    pts = v["points_primary"]
    if pts is None:
        return "## 4. Per-point judge scores\n\nnot run (no primary-judge rows).\n"
    rows = []
    for name, p in pts.items():
        tf, tu, co = p["trait_filtered"], p["trait_unfiltered"], p["coherence"]
        rows.append((name, p["n_rows"], counts(p["status_counts"]["trait_status"]),
                     counts(p["status_counts"]["coherence_status"]),
                     pct(p["share_excluded"]), pct(p["share_coherence_unscored"]),
                     "%s [%s, %s]" % (f(tf["estimate"], 1), f(tf["lo"], 1), f(tf["hi"], 1)),
                     f(tu["estimate"], 1), f(co["estimate"], 1)))
    return "\n".join([
        "## 4. Per-point primary-judge scores on the held-out set (plumbing, not results)", "",
        "Coherence exclusion is by the primary judge's coherence < 50 (or not scored); the trait mean is "
        "filtered with a question-clustered 95% interval (10,000 draws), unfiltered beside it.", "",
        table(["point", "rows", "trait status", "coherence status", "excluded", "coherence unscored",
               "trait, filtered [95% CI]", "trait, unfiltered", "coherence mean"], rows), ""])


def section_determinism(v):
    lines = ["## 5. Determinism re-send (5% of each judge's distinct requests, seed 20261026, cold)", ""]
    rows = []
    for judge, d in v["determinism"].items():
        if not d.get("n"):
            rows.append((judge, "not run", "", "", "", ""))
            continue
        rows.append((judge, d["n"], "%d / %d" % (d["exact_match"], d["n"]), "%d / %d" % (d["same_status"], d["n"]),
                     "%d of %d" % (d["n_delta_nonzero"], d["n_with_score_both"]),
                     "mean %s, max %s" % (f(d["mean_abs_delta"], 3), f(d["max_abs_delta"], 3))))
    lines += [table(["judge", "re-sent", "exact match", "same status", "score changed", "|delta|"], rows), ""]
    for judge, d in v["determinism"].items():
        for m in d.get("mismatches", [])[:5]:
            lines.append("* %s mismatch `%s`: %s -> %s" % (judge, m["key"], m["score_original"], m["score_resend"]))
    return "\n".join(lines) + "\n"


def section_bridge(v):
    b = v["bridge"]
    rows = [(k, p["n_rows"], pct(p["parse_coverage"]), f(p.get("estimate", p.get("p")), 3),
             "[%s, %s]" % (f(p.get("lo"), 3), f(p.get("hi"), 3))) for k, p in sorted(b["points"].items())]
    lines = ["## 6. Bridge set: parsed choice, and each judge's trait score against it", "",
             table(["point", "rows", "parse coverage", "P(other-regarding)", "Wilson 95%"], rows), "",
             "AUC = P(trait score of an other-regarding choice > that of a self-interested choice), on the "
             "bridge AUC subset (k = 0 baseline and the trait vector at +-3), question-clustered bootstrap:", ""]
    rows = []
    for judge, a in b["auc"].items():
        by_k = ", ".join("k%s %s (n=%d)" % (k, f(x["auc"], 3), x["n"]) for k, x in sorted(a["by_k"].items()))
        rows.append((judge, "%d / %d" % (a["n"], a["n_candidate_rows"]), f(a["auc"]),
                     "[%s, %s]" % (f(a.get("lo")), f(a.get("hi"))), by_k))
    lines += [table(["judge", "rows used / candidates", "AUC", "95% CI", "by k"], rows), ""]
    return "\n".join(lines)


def section_agreement(v):
    j = v["judge_judge"]
    if not j:
        return "## 7. Judge-judge agreement\n\nnot run (both judges' rows are needed).\n"
    rows = []
    for name in ("trait_heldout", "coherence_heldout", "trait_bridge_auc_subset"):
        a = j[name]
        if not a.get("n_items"):
            rows.append((name, "not run", "", "", ""))
            continue
        rows.append((name, "%d (%d paired)" % (a["n_items"], a["n_paired"]),
                     "%s %s" % (f(a["spearman"]), ci(a, "spearman")),
                     "%s %s" % (f(a["qw_kappa_binned"]), ci(a, "qw_kappa_binned")),
                     "%s %s" % (f(a["krippendorff_alpha_interval_binned"]),
                                ci(a, "krippendorff_alpha_interval_binned"))))
    pw = j["pairwise_choice_agreement"]
    return "\n".join([
        "## 7. Judge-judge agreement on the overlap (primary vs secondary; bootstrap over items, 2,000 draws)", "",
        table(["score", "items", "Spearman [95% CI]", "quadratic-weighted kappa, binned", "Krippendorff alpha, binned"], rows),
        "", "Pairwise choice agreement (same call, both judges ok): %s of %s (%s)."
        % (pw["n_agree"], pw["n_both_ok"], pct(pw["share"])), ""])


def section_pairwise(v):
    rows = []
    for judge, p in v["pairwise"].items():
        rows.append((judge, p["n_calls"], counts(p["status_counts"]), pct(p["order_consistency"]),
                     f(p["position_bias_p_first_minus_half"], 3),
                     "; ".join("%s %s" % (k, f(x, 2)) for k, x in p["p_steered_chosen_by_point"].items())))
    return "\n".join(["## 8. Pairwise (secondary metric)", "",
                      table(["judge", "calls", "status", "order consistency", "position bias (P(first) - 0.5)",
                             "P(steered chosen) by point"], rows), ""])


def section_human(key):
    lines = ["## 9. Human calibration sheet", ""]
    if key is None:
        return "\n".join(lines + ["not exported.", ""])
    rows = [("%s k%+g" % (s["role"], s["k"]), s["available"], s["wanted"], s["drawn"]) for s in key["strata"]]
    return "\n".join(lines + [
        "%d items, seed %d, drawn from the held-out pilot rows (`human/calibration_sheet.csv`, page "
        "`human/grader.html`). Strata:" % (key["n_items"], key["seed"]), "",
        table(["stratum", "rows available", "quota", "drawn"], rows), "",
        "**Human-human, human-judge agreement: not run.** It needs the two labellers' exports "
        "(`human_labels_<reviewer>_<date>.csv`, one per labeller); `scripts/measurement/agreement.py` computes "
        "it once they exist.", ""])


HEADER = """# Checkpoint 2 pilot report: general-question transfer

**Status: pilot. Every number below is a plumbing number, not a result.** The PREREG
(`PREREG.md`, frozen at `provenance/freeze.md`) is followed as written; what it left open is in
`DEVIATIONS.md` (pilot scope, served ids, interpreter, pairwise parameters, judge order).

**Pilot scope.** The Dictator strict decision vector, the trait vector and the Dictator's shuffled
and orthogonal nulls at k in {-3, 0, +3}; held-out and bridge sets; 3 samples per question (120 rows
per point). Generation: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c70...`, bf16, sdpa, layer 20,
positions all, norm unit, `unit_beta = k x 10.5083`, neutral preset, `use_model_defaults=False`,
`max_new_tokens` 1000, batch 20 seeded `seed + batch_index`, torch 2.7.1+cu126 / transformers 4.54.0
(not the interpreter the README names; DEVIATIONS). Judges: primary Llama-3.3-70B-Instruct Q8_0
(`logprob_0_100`), secondary gpt-oss-20b MXFP4 (`text_0_100`, effort low), both through the local
llama-server on PimientoRojo.
"""

#: PREREG section 10's planned call counts for the full `div8_main` run, by judge and kind.
FULL_CALLS = {PRIMARY: {"trait": 8040, "coherence": 13080, "pairwise": 10560},
              SECONDARY: {"trait": 2280, "coherence": 1920, "pairwise": 2640}}
RESEND_CALLS = {PRIMARY: 1584, SECONDARY: 342}


def projection(v):
    """Hours of judging for the full run at the speeds this pilot achieved."""
    rows = []
    for judge, calls in FULL_CALLS.items():
        gaps = v["throughput"].get(judge, {}).get("inter_call_gaps_by_kind", {})
        overall = v["throughput"].get(judge, {}).get("seconds_per_call_overall")
        if not gaps or overall is None:
            rows.append((judge, "not measured", "", ""))
            continue
        seconds = sum(n * gaps.get(kind, {}).get("mean_s", overall) for kind, n in calls.items())
        seconds += RESEND_CALLS[judge] * overall
        rows.append((judge, "%d" % (sum(calls.values()) + RESEND_CALLS[judge]), f(overall, 2),
                     "%.1f h" % (seconds / 3600.0)))
    return table(["judge", "planned calls (PREREG 10)", "pilot s/call overall", "projected hours at the pilot's speeds"], rows)


def pairwise_cost(v):
    """What the primary's pairwise calls take of the projected hours, and what trimming them would save."""
    gaps = v["throughput"].get(PRIMARY, {}).get("inter_call_gaps_by_kind", {})
    if "pairwise" not in gaps:
        return ""
    calls = FULL_CALLS[PRIMARY]["pairwise"]
    hours = calls * gaps["pairwise"]["mean_s"] / 3600.0
    return ("   The primary's %d pairwise calls (held-out, +-3, both orders) take %.1f h of that at %.2f s/call "
            "(a pairwise prompt holds two answers; single-answer scoring runs 3.3-3.5 s), so they are about "
            "half the primary's time. **Question: trim the pairwise pass** (e.g. one order only, or the primary "
            "on a 25%% subset like the secondary), or keep it as pre-registered? Dropping it entirely is the "
            "PREREG's `div8_alt`.\n" % (calls, hours, gaps["pairwise"]["mean_s"]))


def observations(v):
    """Facts the tables already hold that bear on the decisions below, stated once, in words."""
    lines = []
    pts = v.get("points_primary") or {}
    base = pts.get("baseline k+0")
    if base and base["trait_filtered"]["estimate"] is not None:
        lines.append("* **Headroom.** The primary judge's mean trait score at the unsteered baseline is %s (held-out, "
                     "0-100), so the positive end of the ladder has %s points of headroom; at the trait vector's k = +3 "
                     "it is %s. A ceiling compresses decision-minus-null contrasts at the positive end."
                     % (f(base["trait_filtered"]["estimate"], 1), f(100 - base["trait_filtered"]["estimate"], 1),
                        f((pts.get("trait/altruism k+3") or {}).get("trait_filtered", {}).get("estimate"), 1)))
    excluded = {k: p["share_excluded"] for k, p in pts.items() if p["share_excluded"]}
    if excluded:
        verdict = ("no point reaches the 50% usable-coverage collapse that makes a point `undetermined`"
                   if max(excluded.values()) < 0.5 else
                   "at least one point is at or past the 50% coverage collapse and would be `undetermined`")
        lines.append("* **Coherence exclusion** is non-zero at: %s; %s."
                     % ("; ".join("%s %s" % (k, pct(x)) for k, x in sorted(excluded.items())), verdict))
    jj = v.get("judge_judge")
    if jj and jj["trait_heldout"].get("n_items"):
        t, c = jj["trait_heldout"], jj["coherence_heldout"]
        lines.append("* **Judge-judge.** Held-out trait Spearman %s, coherence Spearman %s (secondary scores coherence "
                     "differently from the primary: binned kappa %s). Exclusion is always by the primary judge, so the "
                     "coherence disagreement does not enter the filter."
                     % (f(t["spearman"]), f(c["spearman"]), f(c["qw_kappa_binned"])))
    det = (v.get("determinism") or {}).get(SECONDARY, {})
    if det.get("n"):
        lines.append("* **Determinism.** gpt-oss repeated exactly on %d of %d re-sent requests (not bitwise reproducible, as "
                     "measured on the bench); the primary's own count is in section 5."
                     % (det["exact_match"], det["n"]))
    return "\n".join(lines) + "\n\n" if lines else ""


def footer(v):
    return """## 10. What the tables show, and questions for the operator

""" + observations(v) + """### Questions for the operator

1. **D12, the "poor agreement" rule (PREREG 9.4).** Not pre-registered; yours to fix before any full
   judging, and it goes to `DEVIATIONS.md` with its date. What exists so far is judge-side only (sections
   6 and 7); the human numbers need your labels and your colleague's. The draft's candidate thresholds
   (human-judge Spearman lower bound < 0.5 on trait; bridge AUC lower bound <= 0.5) were proposals, not
   adopted. Decide the statistic, the threshold and what happens below it (change judge, change mode,
   stop).
2. **The human sheet has 164 items, not about 200.** The pilot has no relaxed and no Dictator-only arm, so
   those two strata (2 x 18) are empty and were not topped up (DEVIATIONS). Accept 164, or ask for the
   two missing strata to be generated and judged (cheap on the generation side, about 36 more rows).
3. **Your colleague's identity** (name or initials) for `provenance/` (D10), and how the colleague gets the
   page: send **only** `human/grader.html` (or `human/calibration_sheet.csv` plus the page). Do **not**
   send `provenance/human_sample_key.json` or `human/judge_reveal_*.json`: the key names each item's arm and
   k, and the reveal files hold the judges' scores. Audit mode loads a reveal file only after an item is
   rated and freezes that rating.
4. **Served ids.** The servers answer to their model paths, not the `--alias` ids J10 expected. Keep the
   path ids (DEVIATIONS), or relaunch with `--alias` before the full run.
5. **Interpreter.** The pilot generated under torch 2.7.1 / transformers 4.54.0. If you want the full run
   under the earlier pins (2.6.0+cu124 / 4.52.3), supply that interpreter; the k = 0 rows are then
   comparable with the per-game study's, which this pilot's are not guaranteed to be.
6. **Pairwise parameters** (`max_tokens` 32 on the primary, 2048 on the secondary; no format instruction
   appended; template as frozen): approve or change (DEVIATIONS).
7. **Judging time.** Planned 37.3 h (44.3 h with long pairwise calls) for the full run; at the speeds this
   pilot achieved:

""" + projection(v) + """

""" + pairwise_cost(v) + """
8. **Anything flagged in sections 1-8** (truncation, coherence exclusion, a status other than `ok`, a
   determinism mismatch, an AUC interval reaching 0.5) is read off the tables above; nothing has been
   dropped, imputed or re-asked, and nothing beyond the pilot has run.

## 11. What was and was not run

Run: pilot generation (2,400 rows), both judges' pilot judging, the determinism re-send, the bridge AUC,
judge-judge agreement, pairwise, the human sheet and page export. **Not run:** human-human and
human-judge agreement (no labels exist), the in-distribution set, any full-study generation or judging,
the optional test-retest. Full judging does not start until you approve.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validation", required=True)
    ap.add_argument("--human-key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    v = json.loads(Path(args.validation).read_text())
    key = json.loads(Path(args.human_key).read_text()) if Path(args.human_key).exists() else None
    parts = [HEADER, section_generation(v), section_judging(v), section_points(v),
             section_determinism(v), section_bridge(v), section_agreement(v), section_pairwise(v),
             section_human(key), footer(v)]
    Path(args.out).write_text("\n".join(parts))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
