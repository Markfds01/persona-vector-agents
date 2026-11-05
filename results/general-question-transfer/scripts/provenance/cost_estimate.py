"""Checkpoint-1 cost estimate: generations, tokens, GPU hours, judge calls.

Every number that is MEASURED comes from committed files, and every number that
is ASSUMED is a named parameter printed beside its result:

* generation throughput - measured: the per-game study's sweep
  (`results/crossgame-per-game-steering`, strict, 8,400 generations on one A40,
  batch 20, `max_new_tokens` 1000) gives wall-clock minutes per game in
  `provenance/sweep_provenance_strict.json`; its committed rows give the number of
  generated tokens (Qwen tokenizer, offline).
* answer length on GENERAL questions - measured on the closest committed proxy:
  Qwen2.5-7B-Instruct's own answers to upstream's general altruism questions,
  `eval_persona_extract/Qwen2.5-7B-Instruct/altruism_{pos,neg}_instruct.csv`
  (4,000 answers, generated upstream with `max_tokens=1000`). They were generated
  under a persona system prompt and unsteered, so they bound the length of a
  steered general answer only roughly; that is stated, not hidden.
* judge throughput on the judge server - MEASURED on 2026-10-23 on PimientoRojo
  (`provenance/judge_measurement_2026-10-23.md`, 12 bench items, no study data):
  the constants in `MEASURED_JUDGE` below are copied from that report, not assumed.
  The one assumed quantity is the pairwise call's length, priced beside the flat
  measured mean as a sensitivity row.

    python scripts/provenance/cost_estimate.py      # writes analysis/cost.json and cost.md
"""

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parents[1]
QWEN = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

#: The default design of PREREG.md section 4.
K_NONZERO = (-5, -3, -1, 1, 3, 5)
N_QUESTIONS = 40
N_SAMPLES = 5
QUESTION_SETS = ("heldout", "in_distribution", "bridge")
#: arms steered at every k != 0 (PREREG section 3)
ARMS = (["decision/%s" % g for g in ("dictator", "trust", "ultimatum", "apology",
                                     "overfishing", "prisoners_dilemma")]
        + ["relaxed/ultimatum", "relaxed/overfishing", "dictator_only", "trait/altruism"]
        + ["shuffled_null/%s" % g for g in ("dictator", "trust", "ultimatum", "apology",
                                            "overfishing", "prisoners_dilemma")]
        + ["orthogonal_null/%s" % g for g in ("dictator", "trust", "ultimatum", "apology",
                                              "overfishing", "prisoners_dilemma")])
#: the unhooked k=0 run for the byte-identity check (one extra point)
EXTRA_POINTS = 1
#: pairwise: secondary-judge share (PREREG section 8), sets it runs on
PAIRWISE_SECONDARY_SHARE = 0.20
PAIRWISE_SETS = ("heldout", "in_distribution")
DETERMINISM_SHARE = 0.05
#: judge call shape. trait and coherence template lengths are MEASURED below from
#: the verbatim upstream templates (Qwen tokenizer as a proxy for the judges');
#: the pairwise template's length is taken as the trait template's.
#: MEASURED judge speeds (provenance/judge_measurement_2026-10-23.md section 2): cold,
#: `cache_prompt=false`, one request at a time, mean seconds per call over 24 distinct
#: prompts (12 trait + 12 coherence, ~570-630 prompt tokens).
MEASURED_JUDGE = {
    "primary": {"judge": "llama-3.3-70b-instruct-q8_0", "mode": "logprob_0_100",
                "mean_s": 3.82, "median_s": 3.71, "calls_per_hour": 942,
                "bench_prompt_tokens": 590},
    "primary_text_mode": {"judge": "llama-3.3-70b-instruct-q8_0", "mode": "text_0_100",
                          "mean_s": 6.09, "median_s": 5.97, "calls_per_hour": 591,
                          "bench_prompt_tokens": 610},
    "secondary": {"judge": "gpt-oss-20b-mxfp4", "mode": "text_0_100 (reasoning low)",
                  "mean_s": 1.00, "median_s": 0.94, "calls_per_hour": 3600,
                  "bench_prompt_tokens": 630},
}
#: gpt-oss prefill speed measured 1,800-2,000 tok/s: the extra prompt a pairwise call
#: carries (two answers) costs this much more per call
GPTOSS_PREFILL_TPS = 1900.0


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION, local_files_only=True)


def n_tokens(tok, texts):
    return [len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]]


def measured_throughput(tok):
    prov = json.loads((REPO / "results/crossgame-per-game-steering/provenance/"
                       "sweep_provenance_strict.json").read_text())
    rows_root = REPO / "results/crossgame-per-game-steering/rows"
    per_game = {}
    csv.field_size_limit(1 << 30)
    for game, entry in prov["games"].items():
        texts = []
        for path in sorted(glob.glob(str(rows_root / game / "*" / "*.csv"))):
            with open(path, newline="", encoding="utf-8") as handle:
                texts += [r["continuation"] for r in csv.DictReader(handle)]
        counts = n_tokens(tok, texts)
        seconds = entry["minutes"] * 60.0
        per_game[game] = {"generations": len(texts), "minutes": entry["minutes"],
                          "generated_tokens": sum(counts),
                          "mean_tokens": sum(counts) / len(counts),
                          "share_at_1000": sum(c >= 1000 for c in counts) / len(counts),
                          "tokens_per_s": sum(counts) / seconds,
                          "seconds_per_generation": seconds / len(texts)}
    total_tokens = sum(g["generated_tokens"] for g in per_game.values())
    total_seconds = sum(g["minutes"] for g in per_game.values()) * 60
    total_gens = sum(g["generations"] for g in per_game.values())
    return {"per_game": per_game,
            "pooled": {"generations": total_gens, "hours": total_seconds / 3600,
                       "generated_tokens": total_tokens,
                       "tokens_per_s": total_tokens / total_seconds,
                       "seconds_per_generation": total_seconds / total_gens}}


def general_answer_lengths(tok):
    csv.field_size_limit(1 << 30)
    answers, questions = [], []
    for name in ("altruism_pos_instruct.csv", "altruism_neg_instruct.csv"):
        with open(REPO / "eval_persona_extract/Qwen2.5-7B-Instruct" / name, newline="",
                  encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                answers.append(r["answer"])
                questions.append(r["question"])
    counts = sorted(n_tokens(tok, answers))
    q_counts = n_tokens(tok, sorted(set(questions)))

    def pct(p):
        return counts[min(len(counts) - 1, int(p * len(counts)))]
    return {"source": "eval_persona_extract/Qwen2.5-7B-Instruct/altruism_{pos,neg}_instruct.csv",
            "n": len(counts), "mean": sum(counts) / len(counts), "median": pct(0.5),
            "p90": pct(0.9), "p99": pct(0.99), "max": counts[-1],
            "share_ge": {str(t): sum(c >= t for c in counts) / len(counts)
                         for t in (400, 600, 800, 1000)},
            "mean_question_tokens": statistics.mean(q_counts)}


def template_tokens(tok):
    sys.path.insert(0, str(REPO))
    trait = json.loads((REPO / "data_generation/trait_data_eval/altruistic.json").read_text())
    text = (REPO / "eval/prompts.py").read_text()
    namespace = {}
    exec(compile(text, "eval/prompts.py", "exec"), namespace)   # a dict literal, no imports
    coherence = namespace["Prompts"]["coherence_0_100"]
    t, c = n_tokens(tok, [trait["eval_prompt"], coherence])
    return {"trait": t, "coherence": c, "pairwise": t}


#: Trims the operator may choose from (PREREG section 10). Each changes ONE thing
#: from the default; none is adopted here.
SCENARIOS = {
    "default": {},
    "ladder_4k": {"k_nonzero": (-5, -3, 3, 5)},
    "ladder_2k": {"k_nonzero": (-3, 3)},
    "samples_3": {"n_samples": 3},
    "pairwise_heldout_k3_only": {"pairwise_sets": ("heldout",), "pairwise_k": (-3, 3)},
    "no_secondary_pairwise": {"secondary_pairwise_share": 0.0},
    "secondary_on_20pct": {"secondary_share": 0.20},
    "in_distribution_core_arms_only": {"in_distribution_arms": 10},
}


def design(k_nonzero=K_NONZERO, n_samples=N_SAMPLES, pairwise_sets=PAIRWISE_SETS,
           pairwise_k=None, in_distribution_arms=None, **_):
    per_point = N_QUESTIONS * n_samples
    arms = {qs: len(ARMS) for qs in QUESTION_SETS}
    if in_distribution_arms is not None:
        arms["in_distribution"] = in_distribution_arms
    gens = sum((arms[qs] * len(k_nonzero) + 1 + EXTRA_POINTS) * per_point for qs in QUESTION_SETS)
    pk = len(pairwise_k) if pairwise_k is not None else len(k_nonzero)
    steered_rows_pairwise = sum(arms[qs] * pk * per_point for qs in pairwise_sets)
    points_per_set = len(ARMS) * len(k_nonzero) + 1 + EXTRA_POINTS
    return {"arms_steered": len(ARMS), "k_nonzero": list(k_nonzero), "n_samples": n_samples,
            "points_per_question_set": points_per_set,
            "rows_per_point": per_point, "question_sets": list(QUESTION_SETS),
            "generations": gens,
            "judged_rows": gens - EXTRA_POINTS * per_point * len(QUESTION_SETS),
            "pairwise_steered_rows": steered_rows_pairwise}


def judge_calls(d, secondary_pairwise_share=PAIRWISE_SECONDARY_SHARE, secondary_share=1.0, **_):
    judged = d["judged_rows"]
    primary_pairwise = 2 * d["pairwise_steered_rows"]            # both orders
    secondary_pairwise = int(round(secondary_pairwise_share * primary_pairwise))
    out = {}
    for judge, pairwise, share in (("primary", primary_pairwise, 1.0),
                                   ("secondary", secondary_pairwise, secondary_share)):
        judged_here = int(round(share * judged))
        base = 2 * judged_here + pairwise                         # trait + coherence + pairwise
        out[judge] = {"trait": judged_here, "coherence": judged_here, "pairwise": pairwise,
                      "determinism_resend": int(round(DETERMINISM_SHARE * base)),
                      "total": base + int(round(DETERMINISM_SHARE * base))}
    return out


def seconds_per_call(role, kind, templates, answer_tokens, question_tokens):
    """Measured mean seconds per call; a pairwise call is priced by prompt length.

    Trait and coherence calls are the measured bench workload. A pairwise call carries
    two answers; its prompt is longer by one answer, priced for the primary judge in
    proportion to prompt tokens (its call is prefill-bound: one generated token) and
    for the secondary at its measured prefill speed. That length is ASSUMED.
    """
    m = MEASURED_JUDGE[role]
    if kind != "pairwise":
        return m["mean_s"]
    pair_tokens = templates["pairwise"] + question_tokens + 2 * answer_tokens
    if role == "primary":
        return m["mean_s"] * pair_tokens / m["bench_prompt_tokens"]
    return m["mean_s"] + (pair_tokens - m["bench_prompt_tokens"]) / GPTOSS_PREFILL_TPS


def hours(calls, role, templates, answer_tokens, question_tokens, flat):
    """Serial hours at `--parallel 1` for one judge's calls (`flat`: pairwise at the mean)."""
    seconds = 0.0
    for kind in ("trait", "coherence", "pairwise", "determinism_resend"):
        per = (MEASURED_JUDGE[role]["mean_s"] if flat else
               seconds_per_call(role, "pairwise" if kind == "pairwise" else kind, templates,
                                answer_tokens, question_tokens))
        seconds += calls[kind] * per
    return seconds / 3600


#: The two /8 designs (PREREG sections 4, 10). `sets[name]` = (arms, k list, samples, whether
#: the trait score is asked of every row of the set); `extra_k` = more rungs for one arm.
ALL_ARMS = tuple(ARMS)
DECISION_AND_TRAIT = tuple(a for a in ARMS if a.startswith("decision/")) + ("trait/altruism",)
DIV8_DESIGNS = {
    "div8_main": {
        "label": "main: {-3,0,+3} for every arm, 3 samples, pairwise on held-out at +-3",
        "sets": {"heldout": {"arms": ALL_ARMS, "k": (-3, 3), "samples": 3, "trait": "all",
                             "extra_k": {"trait/altruism": (-5, -1, 1, 5)}},
                 "in_distribution": {"arms": DECISION_AND_TRAIT, "k": (-3, 3), "samples": 3,
                                     "trait": "all", "extra_k": {}},
                 "bridge": {"arms": ALL_ARMS, "k": (-3, 3), "samples": 3, "trait": "auc",
                            "extra_k": {}}},
        "pairwise": True, "secondary_share": 0.25},
    "div8_alt": {
        "label": "alternative: as main but no pairwise anywhere and 5 samples on the held-out set",
        "sets": {"heldout": {"arms": ALL_ARMS, "k": (-3, 3), "samples": 5, "trait": "all",
                             "extra_k": {"trait/altruism": (-5, -1, 1, 5)}},
                 "in_distribution": {"arms": DECISION_AND_TRAIT, "k": (-3, 3), "samples": 3,
                                     "trait": "all", "extra_k": {}},
                 "bridge": {"arms": ALL_ARMS, "k": (-3, 3), "samples": 3, "trait": "auc",
                            "extra_k": {}}},
        "pairwise": False, "secondary_share": 0.25},
}
#: the bridge set's trait score exists only where the bridge-AUC check (PREREG 9.3) needs it:
#: the k=0 baseline and the trait vector at +-3
BRIDGE_AUC_POINTS = 3


def div8_plan(spec, secondary_share=None, secondary_heldout_full=False):
    """Exact generation and judge-call counts of a /8 design."""
    share = spec["secondary_share"] if secondary_share is None else secondary_share
    gens, judged = 0, 0
    prim = {"trait": 0, "coherence": 0, "pairwise": 0}
    sec = {"trait": 0, "coherence": 0, "pairwise": 0}
    detail = {}
    for name, cfg in spec["sets"].items():
        per = N_QUESTIONS * cfg["samples"]
        points = 1 + sum(len(cfg["k"]) for _ in cfg["arms"]) \
            + sum(len(v) for v in cfg["extra_k"].values())
        unhooked = 1 if name == "heldout" else 0           # the byte-identity check, once
        rows = points * per
        gens += rows + unhooked * per
        judged += rows
        prim["coherence"] += rows
        prim["trait"] += rows if cfg["trait"] == "all" else BRIDGE_AUC_POINTS * per
        s_share = 1.0 if (secondary_heldout_full and name == "heldout") else share
        if cfg["trait"] == "all":
            sec["trait"] += int(round(s_share * rows))
            sec["coherence"] += int(round(s_share * rows)) if name != "bridge" else 0
        else:
            sec["trait"] += BRIDGE_AUC_POINTS * per          # AUC subset is judged by both
        detail[name] = {"points": points, "rows_per_point": per, "generations": rows
                        + unhooked * per}
        if name == "heldout" and spec["pairwise"]:
            steered = len(cfg["arms"]) * len(cfg["k"]) * per
            prim["pairwise"] = 2 * steered                       # both orders
            sec["pairwise"] = int(round(share * prim["pairwise"]))
    out = {}
    for role, c in (("primary", prim), ("secondary", sec)):
        base = sum(c.values())
        out[role] = {**c, "determinism_resend": int(round(DETERMINISM_SHARE * base)),
                     "total": base + int(round(DETERMINISM_SHARE * base))}
    return {"generations": gens, "judged_rows": judged, "per_set": detail, "judge_calls": out}


def flat_hours(calls):
    return {role: calls[role]["total"] * MEASURED_JUDGE[role]["mean_s"] / 3600
            for role in ("primary", "secondary")}


def main():
    tok = tokenizer()
    through = measured_throughput(tok)
    lengths = general_answer_lengths(tok)
    templates = template_tokens(tok)
    d = design()
    gen_tokens = d["generations"] * lengths["mean"]
    gpu_hours_tokens = gen_tokens / through["pooled"]["tokens_per_s"] / 3600
    gpu_hours_gens = d["generations"] * through["pooled"]["seconds_per_generation"] / 3600
    calls = judge_calls(d)
    default_hours = flat_hours(calls)
    scenarios = {}
    for name, change in SCENARIOS.items():
        sd = design(**change)
        sc = judge_calls(sd, **change)
        scenarios[name] = {"change": {k: list(v) if isinstance(v, tuple) else v
                                      for k, v in change.items()},
                           "generations": sd["generations"],
                           "gpu_hours": sd["generations"] * lengths["mean"]
                           / through["pooled"]["tokens_per_s"] / 3600,
                           "primary_calls": sc["primary"]["total"],
                           "secondary_calls": sc["secondary"]["total"],
                           "judge_hours_flat": flat_hours(sc)}
    div8 = {}
    variants = [("div8_main", {}, ""), ("div8_alt", {}, ""),
                ("div8_main", {"secondary_heldout_full": True}, "_secondary_100pct_heldout")]
    for name, kw, suffix in variants:
        spec = DIV8_DESIGNS[name]
        plan = div8_plan(spec, **kw)
        c = plan["judge_calls"]
        ql = lengths["mean_question_tokens"]
        plan.update({
            "label": spec["label"] + (" (secondary judge on ALL held-out rows)" if suffix else ""),
            "gpu_hours_tokens": plan["generations"] * lengths["mean"]
            / through["pooled"]["tokens_per_s"] / 3600,
            "gpu_hours_gens": plan["generations"] * through["pooled"]["seconds_per_generation"] / 3600,
            "judge_hours_flat": flat_hours(c),
            "judge_hours_pairwise_by_length": {
                role: hours(c[role], role, templates, lengths["mean"], ql, flat=False)
                for role in ("primary", "secondary")},
            "pairwise_seconds_per_call": {
                role: seconds_per_call(role, "pairwise", templates, lengths["mean"], ql)
                for role in ("primary", "secondary")}})
        div8[name + suffix] = plan
    result = {"measured_generation_throughput": through, "general_answer_length_proxy": lengths,
              "design": d, "generated_tokens_estimate": gen_tokens,
              "gpu_hours_estimate": {"from_tokens_per_s": gpu_hours_tokens,
                                     "from_seconds_per_generation": gpu_hours_gens,
                                     "note": "one A40, batch 20, exclusive; the per-game sweep's "
                                             "pooled rate. Two A40s in parallel halve the wall clock."},
              "judge_template_tokens": templates,
              "judge_calls": calls, "measured_judge": MEASURED_JUDGE,
              "default_judge_hours_flat": default_hours,
              "trim_scenarios": scenarios, "div8": div8}
    out = PACKAGE / "analysis" / "cost.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    (PACKAGE / "analysis" / "cost.md").write_text(markdown(result))
    print(json.dumps({"generations": d["generations"], "gpu_hours": [gpu_hours_tokens, gpu_hours_gens],
                      "calls": calls, "default_hours": default_hours,
                      "div8": {k: {"generations": v["generations"],
                                   "calls": {r: v["judge_calls"][r]["total"] for r in v["judge_calls"]},
                                   "hours": v["judge_hours_flat"]} for k, v in div8.items()}},
                     indent=1))


def markdown(r):
    t = r["measured_generation_throughput"]
    L = r["general_answer_length_proxy"]
    d = r["design"]
    g = r["gpu_hours_estimate"]
    lines = [
        "# Cost estimate (checkpoint 1)",
        "",
        "Generated by `scripts/provenance/cost_estimate.py` into `cost.json`; every number "
        "below is in that file. **Measured** = read from committed files. **Assumed** = a "
        "parameter, printed with its result. The judge speeds are measured on the judge host "
        "(bench items only); no study data has been judged.",
        "",
        "## Generation (Qwen2.5-7B-Instruct, one A40)",
        "",
        "**Measured throughput** - the per-game study's strict sweep, %d generations in %.2f h "
        "on one A40 at batch 20, `max_new_tokens` 1000: %d generated tokens (Qwen tokenizer), "
        "**%.1f tokens/s**, %.2f s per generation." % (
            t["pooled"]["generations"], t["pooled"]["hours"], t["pooled"]["generated_tokens"],
            t["pooled"]["tokens_per_s"], t["pooled"]["seconds_per_generation"]),
        "",
        "| game | generations | mean tokens | share at 1000 | tokens/s | s/generation |",
        "|---|---|---|---|---|---|",
    ]
    for game, v in t["per_game"].items():
        lines.append("| %s | %d | %.0f | %.4f | %.1f | %.2f |" % (
            game, v["generations"], v["mean_tokens"], v["share_at_1000"], v["tokens_per_s"],
            v["seconds_per_generation"]))
    lines += [
        "",
        "**Measured answer length on general questions (proxy)** - `%s`: n = %d unsteered "
        "answers to upstream's general altruism questions (under a persona system prompt): "
        "mean %.0f tokens, median %d, p90 %d, p99 %d, max %d; share >= 600: %.4f, >= 800: %.4f, "
        ">= 1000: %.4f. Mean question length %.1f tokens." % (
            L["source"], L["n"], L["mean"], L["median"], L["p90"], L["p99"], L["max"],
            L["share_ge"]["600"], L["share_ge"]["800"], L["share_ge"]["1000"],
            L["mean_question_tokens"]),
        "",
        "**Default design** (PREREG section 4): %d steered arms x k in %s, plus the shared k=0 "
        "point and one unhooked k=0 point for the byte-identity check, = %d points per question "
        "set; %d rows per point (40 questions x %d samples); %d question sets (%s)." % (
            d["arms_steered"], d["k_nonzero"], d["points_per_question_set"], d["rows_per_point"],
            d["n_samples"], len(d["question_sets"]), ", ".join(d["question_sets"])),
        "",
        "| quantity | value |",
        "|---|---|",
        "| generations | %d |" % d["generations"],
        "| generated tokens (mean proxy length x generations) | %.2e |" % r["generated_tokens_estimate"],
        "| GPU hours, one A40 (tokens / measured tokens/s) | %.1f |" % g["from_tokens_per_s"],
        "| GPU hours, one A40 (generations x measured s/generation) | %.1f |" % g["from_seconds_per_generation"],
        "| wall clock on both A40s of this machine, one process each | about %.0f-%.0f h |" % (
            g["from_seconds_per_generation"] / 2, g["from_tokens_per_s"] / 2),
        "",
        "The two GPU-hour rows bracket the estimate: the first scales by answer length (general "
        "answers run longer than game answers), the second assumes game-length answers. Steered "
        "answers at the ladder's ends can run to `max_new_tokens`, which this does not price.",
        "",
        "## Judge calls",
        "",
        "Template lengths measured (Qwen tokenizer as proxy): trait %d tokens, coherence %d; "
        "pairwise template assumed equal to trait. Judged rows = every generation except the unhooked "
        "check point = %d." % (r["judge_template_tokens"]["trait"],
                               r["judge_template_tokens"]["coherence"], d["judged_rows"]),
        "",
        "| judge | trait | coherence | pairwise (both orders) | 5% determinism re-send | total |",
        "|---|---|---|---|---|---|",
    ]
    for role, c in r["judge_calls"].items():
        lines.append("| %s | %d | %d | %d | %d | **%d** |" % (
            role, c["trait"], c["coherence"], c["pairwise"], c["determinism_resend"], c["total"]))
    lines += [
        "",
        "Calls per judge are the same in either scoring mode; the mode changes the cost per "
        "call (one output token for `logprob_0_100`, a short generation for `text_0_100`, a "
        "reasoning trace plus answer for gpt-oss).",
        "",
        "## Judge throughput (MEASURED, 2026-10-23)",
        "",
        "Measured on PimientoRojo by `provenance/judge_measurement_2026-10-23.md` (llama.cpp "
        "b11836, Vulkan, `--parallel 1`, `cache_prompt=false`, 24 distinct prompts of ~600 "
        "tokens, one request at a time; 12 bench items, no study data). Seconds per call are "
        "means; calls/hour = 3600 / mean.",
        "",
        "| judge / mode | mean s/call | median s | calls/hour | usable? |",
        "|---|---|---|---|---|",
    ]
    usable = {"primary": "yes: digit mass >= 0.999 on 48/48, 20 top_logprobs returned",
              "primary_text_mode": "yes (not the plan; 1.6x slower, coarse 4-value scores)",
              "secondary": "yes (its logprob mode is NOT usable: first token `<|channel|>`, digit "
                           "mass 0 on 48/48)"}
    for role, m in r["measured_judge"].items():
        lines.append("| %s `%s` | %.2f | %.2f | %d | %s |" % (
            m["judge"], m["mode"], m["mean_s"], m["median_s"], m["calls_per_hour"], usable[role]))
    dh = r["default_judge_hours_flat"]
    lines += [
        "",
        "Parallelism gives at most 1.3x (gpt-oss, `--parallel 4`; the 70B logprob workload does "
        "not scale at all), so the figures are `--parallel 1`, serial. The two judges share one "
        "GPU: treat their hours as additive.",
        "",
        "**Default design, measured speeds**: primary %d calls x %.2f s = **%.1f h**; secondary "
        "%d calls x %.2f s = **%.1f h**; together **%.1f h (%.1f days)**. It fits no schedule "
        "short of two weeks, which is why the /8 design below exists." % (
            r["judge_calls"]["primary"]["total"], MEASURED_JUDGE["primary"]["mean_s"], dh["primary"],
            r["judge_calls"]["secondary"]["total"], MEASURED_JUDGE["secondary"]["mean_s"],
            dh["secondary"], dh["primary"] + dh["secondary"], (dh["primary"] + dh["secondary"]) / 24),
        "",
        "## The /8 design (PREREG sections 4 and 10)",
        "",
        "Target: total judge work about 1/8 of the default's %.1f h = %.1f h. Every row below is "
        "computed by `div8_plan`; counts include the 5%% determinism re-send." % (
            dh["primary"] + dh["secondary"], (dh["primary"] + dh["secondary"]) / 8),
        "",
        "| design | generations | GPU h (1 A40, tokens / s-per-gen) | primary calls | primary h | "
        "secondary calls | secondary h | total judge h | vs default |",
        "|---|---|---|---|---|---|---|---|---|",
        "| default | %d | %.1f / %.1f | %d | %.1f | %d | %.1f | %.1f | 1 |" % (
            d["generations"], g["from_tokens_per_s"], g["from_seconds_per_generation"],
            r["judge_calls"]["primary"]["total"], dh["primary"],
            r["judge_calls"]["secondary"]["total"], dh["secondary"], dh["primary"] + dh["secondary"]),
    ]
    for name, v in r["div8"].items():
        h = v["judge_hours_flat"]
        total = h["primary"] + h["secondary"]
        lines.append("| %s | %d | %.1f / %.1f | %d | %.1f | %d | %.1f | %.1f | 1/%.1f |" % (
            name, v["generations"], v["gpu_hours_tokens"], v["gpu_hours_gens"],
            v["judge_calls"]["primary"]["total"], h["primary"],
            v["judge_calls"]["secondary"]["total"], h["secondary"], total,
            (dh["primary"] + dh["secondary"]) / total))
    lines += ["", "What each design is:", ""]
    for name, v in r["div8"].items():
        lines.append("* `%s` - %s." % (name, v["label"]))
    main_plan = r["div8"]["div8_main"]
    lines += ["", "Per question set (`div8_main`; points = steered arm x k points + the shared k=0 "
              "baseline; the unhooked byte-identity check is one extra generation-only point on the "
              "held-out set):", "",
              "| set | points | rows per point | generations |", "|---|---|---|---|"]
    for name, v in main_plan["per_set"].items():
        lines.append("| %s | %d | %d | %d |" % (name, v["points"], v["rows_per_point"],
                                                  v["generations"]))
    lines += ["", "Judge calls of `div8_main` by kind:", "",
              "| judge | trait | coherence | pairwise (both orders) | 5% determinism re-send | total |",
              "|---|---|---|---|---|---|"]
    for role, c in main_plan["judge_calls"].items():
        lines.append("| %s | %d | %d | %d | %d | **%d** |" % (
            role, c["trait"], c["coherence"], c["pairwise"], c["determinism_resend"], c["total"]))
    pw = main_plan["pairwise_seconds_per_call"]
    hp = main_plan["judge_hours_pairwise_by_length"]
    lines += ["",
              "Sensitivity (ASSUMED pairwise prompt length): a pairwise prompt carries two answers, "
              "so priced by length it costs %.1f s on the primary judge (not the measured %.2f) and "
              "%.2f s on the secondary; `div8_main` then takes %.1f h primary + %.1f h secondary = "
              "**%.1f h** (1/%.1f of the default). No pairwise call was measured." % (
                  pw["primary"], MEASURED_JUDGE["primary"]["mean_s"], pw["secondary"],
                  hp["primary"], hp["secondary"], hp["primary"] + hp["secondary"],
                  (dh["primary"] + dh["secondary"]) / (hp["primary"] + hp["secondary"])),
              "",
              "## Other trims, priced at the measured speeds (for reference; none adopted)",
              "",
              "Each row changes one thing from the default.",
              "",
              "| option | change | generations | GPU h (1 A40) | primary calls | primary h | "
              "secondary calls | secondary h |",
              "|---|---|---|---|---|---|---|---|"]
    for name, sc in r["trim_scenarios"].items():
        h = sc["judge_hours_flat"]
        lines.append("| %s | %s | %d | %.1f | %d | %.1f | %d | %.1f |" % (
            name, json.dumps(sc["change"]) if sc["change"] else "-", sc["generations"],
            sc["gpu_hours"], sc["primary_calls"], h["primary"], sc["secondary_calls"],
            h["secondary"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
