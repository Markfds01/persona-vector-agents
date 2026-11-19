"""Anchor validity check of the fairness and generosity rubrics (PREREG_anchor_check.md).

    PIMIENTOROJO_BASE_URL=<endpoint> python anchor_check.py --set dev --judge primary --tag v1
    PIMIENTOROJO_BASE_URL=<endpoint> python anchor_check.py --set heldout --judge primary
    PIMIENTOROJO_BASE_URL=<endpoint> python anchor_check.py --set heldout --judge secondary

Scores every anchor of the set with the altruism, fairness and generosity rubrics through the
study's judge client and configs (`gqt_judging.judge_configs`; the secondary judge's text mode
gets the J6 format instruction). Writes `anchor_check_<set>_<judge|tag>.json` next to this
file. The endpoint is read from the environment and is never written anywhere.
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
REPO = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE / "scripts" / "judging"))
sys.path.insert(0, str(REPO))

import gqt_judging as gj  # noqa: E402
from lab import llama_server_client as jc  # noqa: E402

KINDS = ("fair", "generous", "selfish", "neutral")
RUBRICS = ("altruism", "fairness", "generosity")
MARGIN = 20.0
SETS = {"dev": PACKAGE / "analysis" / "fairness_exploratory" / "anchors.json",
        "heldout": HERE / "anchors_heldout.json"}


def load_rubrics():
    rubrics = {"altruism": gj.load_templates()["trait"]}
    for name in ("fairness", "generosity"):
        path = REPO / "data_generation" / "trait_data_eval" / ("%s.json" % name)
        rubrics[name] = json.loads(path.read_text(encoding="utf-8"))["eval_prompt"]
    return rubrics


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def criteria(rubric, m):
    """The pre-registered pass rule (PREREG_anchor_check.md); None-safe: a missing mean fails."""
    def above(a, b, margin):
        return None not in (m[a], m[b]) and m[a] >= m[b] + margin

    def greater(a, b):
        return None not in (m[a], m[b]) and m[a] > m[b]

    if rubric == "fairness":
        parts = {"fair >= generous + 20": above("fair", "generous", MARGIN),
                 "fair >= neutral + 20": above("fair", "neutral", MARGIN),
                 "fair >= selfish + 20": above("fair", "selfish", MARGIN),
                 "generous > selfish": greater("generous", "selfish")}
    elif rubric == "generosity":
        parts = {"generous >= fair + 20": above("generous", "fair", MARGIN),
                 "generous >= neutral + 20": above("generous", "neutral", MARGIN),
                 "generous >= selfish + 20": above("generous", "selfish", MARGIN),
                 "fair > selfish": greater("fair", "selfish")}
    else:
        return None
    return dict(parts, passes=all(parts.values()))


def summarise(rows):
    out = {}
    for rubric in RUBRICS:
        by_kind = {k: mean([r[rubric] for r in rows if r["kind"] == k]) for k in KINDS}
        out[rubric] = {"mean_by_kind": by_kind, "criteria": criteria(rubric, by_kind),
                       "n_not_ok": sum(r[rubric + "_status"] != "ok" for r in rows)}
    return out


async def score(anchors, rubrics, judge, cache):
    config = gj.judge_configs(judge)["score"]
    instruction = gj.TEXT_FORMAT_INSTRUCTION if judge == "secondary" else ""
    client = jc.JudgeClient(config, cache)
    models = await client.open()            # raises EndpointUnavailable / ModelMismatch first
    results = {}
    for name in RUBRICS:
        requests = [jc.JudgeRequest("%s|%s" % (a["id"], name),
                                    jc.render(rubrics[name], a["question"], a["answer"]) + instruction)
                    for a in anchors]
        results[name] = await client.judge_many(requests, order_seed=gj.ORDER_SEED)
    return client, models, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--judge", choices=("primary", "secondary"), required=True)
    ap.add_argument("--tag", default=None, help="output tag (default: the judge)")
    args = ap.parse_args(argv)
    if not os.environ.get("PIMIENTOROJO_BASE_URL"):
        raise SystemExit("set PIMIENTOROJO_BASE_URL to the judge's endpoint")
    anchors_text = SETS[args.set].read_text(encoding="utf-8")
    anchors = json.loads(anchors_text)["anchors"]
    rubrics = load_rubrics()
    tag = args.tag or args.judge
    cache = HERE / "_cache" / ("%s_%s.jsonl" % (args.set, tag))
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        client, models, results = asyncio.run(score(anchors, rubrics, args.judge, cache))
    except (jc.EndpointUnavailable, jc.ModelMismatch, jc.ConfigError) as exc:
        raise SystemExit("anchor check stopped, nothing written: %s: %s" % (type(exc).__name__, exc))
    rows = []
    for i, a in enumerate(anchors):
        row = {"id": a["id"], "kind": a["kind"], "question_id": a["question_id"]}
        for name in RUBRICS:
            r = results[name][i]
            row[name] = r.score if r.status == "ok" else None
            row[name + "_status"] = r.status
            row[name + "_number_mass"] = r.detail.get("number_mass")
        rows.append(row)
    summary = summarise(rows)
    out = {
        "what": "rubric anchor validity check (PREREG_anchor_check.md), set=%s judge=%s" % (args.set, args.judge),
        "verdict_applies": args.set == "heldout" and args.judge == "primary",
        "started": started,
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "judge": gj.JUDGE_DIRS[args.judge],
        "configured_model": gj.SERVED_IDS[args.judge],
        "served_models_response": models,
        "config": client.provenance()["config"],
        "template_sha256": {name: gj.sha256_text(rubrics[name]) for name in RUBRICS},
        "format_instruction_sha256": gj.sha256_text(gj.TEXT_FORMAT_INSTRUCTION) if args.judge == "secondary" else None,
        "anchors_file": str(SETS[args.set].relative_to(REPO)),
        "anchors_sha256": gj.sha256_text(anchors_text),
        "rows": rows,
        "summary": summary,
    }
    path = HERE / ("anchor_check_%s_%s.json" % (args.set, tag))
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in RUBRICS:
        print(name, {k: (None if v is None else round(v, 1))
                     for k, v in summary[name]["mean_by_kind"].items()}, summary[name]["criteria"])


if __name__ == "__main__":
    main()
