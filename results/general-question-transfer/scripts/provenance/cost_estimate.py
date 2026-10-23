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
* judge throughput on the judge server - NOT measured (no endpoint call in this
  item). Shown as a table over assumed prompt-processing and generation speeds.

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
TEXT_MODE_OUTPUT_TOKENS = {"llama-3.3-70b-instruct-q8_0": 12, "gpt-oss-20b-mxfp4": 250}
#: assumed judge speeds (tokens/s), NOT measured; one row each in the table
ASSUMED_SPEEDS = {
    "llama-3.3-70b-instruct-q8_0": [{"prompt_tps": 60, "gen_tps": 3},
                                    {"prompt_tps": 120, "gen_tps": 4},
                                    {"prompt_tps": 250, "gen_tps": 5}],
    "gpt-oss-20b-mxfp4": [{"prompt_tps": 500, "gen_tps": 30},
                          {"prompt_tps": 1000, "gen_tps": 50},
                          {"prompt_tps": 2000, "gen_tps": 70}],
}


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


def judge_hours(calls, judge, mode, answer_tokens, question_tokens, template_tokens):
    rows = []
    out_tokens = 1 if mode == "logprob_0_100" else TEXT_MODE_OUTPUT_TOKENS[judge]
    for speed in ASSUMED_SPEEDS[judge]:
        seconds = 0.0
        for kind in ("trait", "coherence", "pairwise", "determinism_resend"):
            n = calls[kind]
            body = answer_tokens * (2 if kind == "pairwise" else 1) + question_tokens
            template = template_tokens.get(kind, template_tokens["trait"])
            seconds += n * ((template + body) / speed["prompt_tps"] + out_tokens / speed["gen_tps"])
        total = calls["total"]
        rows.append({**speed, "mode": mode, "output_tokens_per_call": out_tokens,
                     "seconds_per_call": seconds / total, "calls_per_hour": total / (seconds / 3600),
                     "hours_single_stream": seconds / 3600})
    return rows


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
    hours = {}
    for role, judge, modes in (("primary", "llama-3.3-70b-instruct-q8_0", ("logprob_0_100", "text_0_100")),
                               ("secondary", "gpt-oss-20b-mxfp4", ("text_0_100",))):
        for mode in modes:
            hours["%s|%s|%s" % (role, judge, mode)] = judge_hours(
                calls[role], judge, mode, lengths["mean"], lengths["mean_question_tokens"],
                templates)
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
                           "secondary_calls": sc["secondary"]["total"]}
    result = {"measured_generation_throughput": through, "general_answer_length_proxy": lengths,
              "design": d, "generated_tokens_estimate": gen_tokens,
              "gpu_hours_estimate": {"from_tokens_per_s": gpu_hours_tokens,
                                     "from_seconds_per_generation": gpu_hours_gens,
                                     "note": "one A40, batch 20, exclusive; the per-game sweep's "
                                             "pooled rate. Two A40s in parallel halve the wall clock."},
              "judge_template_tokens": templates,
              "judge_calls": calls, "judge_hours_assumed_speeds": hours,
              "trim_scenarios": scenarios}
    out = PACKAGE / "analysis" / "cost.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    (PACKAGE / "analysis" / "cost.md").write_text(markdown(result))
    print(json.dumps({"generations": d["generations"], "gpu_hours": [gpu_hours_tokens, gpu_hours_gens],
                      "calls": calls, "len": lengths}, indent=1))


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
        "parameter, printed with its result. Nothing here was run against a judge.",
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
        "pairwise assumed equal to trait. Judged rows = every generation except the unhooked "
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
        "## Judge throughput (ASSUMED speeds - not measured)",
        "",
        "The judge server (PimientoRojo: AMD Strix Halo, Radeon 8060S iGPU, ~128 GB unified "
        "memory, llama.cpp `llama-server` b11836 on Vulkan) has not been benchmarked by this "
        "item. Speeds below are assumptions spanning a plausible range for a dense 70B at Q8_0 "
        "and a 20B MoE (3.6B active) at MXFP4 on that hardware; replace them with a measured "
        "`llama-bench` / pilot number at checkpoint 2. Single stream; `--parallel` slots help "
        "generation but little with prompt processing, which dominates a one-token logprob call.",
        "",
        "| judge / mode | prompt tok/s | gen tok/s | out tokens/call | s/call | calls/hour | hours for all calls |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, rows in r["judge_hours_assumed_speeds"].items():
        role, judge, mode = key.split("|")
        for row in rows:
            lines.append("| %s %s `%s` | %d | %d | %d | %.1f | %.0f | %.0f |" % (
                role, judge, mode, row["prompt_tps"], row["gen_tps"],
                row["output_tokens_per_call"], row["seconds_per_call"], row["calls_per_hour"],
                row["hours_single_stream"]))
    primary_total = r["judge_calls"]["primary"]["total"]
    lines += [
        "",
        "Calls/hour NEEDED for the primary judge's %d calls: %.0f/h to finish in one week "
        "(168 h), %.0f/h in two weeks, %.0f/h in 72 h. Against the assumed range above "
        "(roughly 200-1,100 calls/hour) the default design does **not** fit in a week on one "
        "judge server. **Trims are therefore needed or the timeline is weeks** - the operator's "
        "choice, below." % (primary_total, primary_total / 168, primary_total / 336,
                            primary_total / 72),
        "",
        "## Trim options (for the operator; none adopted)",
        "",
        "Each row changes one thing from the default.",
        "",
        "| option | change | generations | GPU h (1 A40) | primary calls | secondary calls |",
        "|---|---|---|---|---|---|",
    ]
    for name, sc in r["trim_scenarios"].items():
        lines.append("| %s | %s | %d | %.1f | %d | %d |" % (
            name, json.dumps(sc["change"]) if sc["change"] else "-", sc["generations"],
            sc["gpu_hours"], sc["primary_calls"], sc["secondary_calls"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
