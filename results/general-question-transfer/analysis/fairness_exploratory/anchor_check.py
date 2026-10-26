"""EXPLORATORY validity check of the fairness rubric on the hand-written anchors.

Run BEFORE the real fairness run (docs/fairness_exploratory_prereg.md section 4). Scores the
12 anchors (anchors.json) with the fairness template and, for comparison, the existing
altruism template; both through the same primary judge and client as the pilot's trait scores.

    PIMIENTOROJO_BASE_URL=<the primary judge's endpoint> python anchor_check.py

Writes anchor_check.json next to this file (the scores, the per-kind means, the pass rule).
The endpoint is read from the environment and is never written anywhere.
"""

import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE / "scripts" / "judging"))
sys.path.insert(0, str(PACKAGE.parents[1]))

import gqt_judging as gj  # noqa: E402
from lab import llama_server_client as jc  # noqa: E402

KINDS = ("fair", "generous", "neutral", "selfish")
#: expected order and the tie margin, fixed in prereg section 4
TIE_MARGIN = 10.0


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def order_check(means):
    """prereg section 4: fair > max(generous, neutral); min(generous, neutral) > selfish;
    |generous - neutral| < 10. Every term is None-safe: a missing mean fails the check."""
    f, g, n, s = (means[k] for k in ("fair", "generous", "neutral", "selfish"))
    if None in (f, g, n, s):
        return {"fair_above_generous_and_neutral": None, "generous_and_neutral_above_selfish": None,
                "generous_approx_neutral": None, "pass": False}
    parts = {
        "fair_above_generous_and_neutral": f > max(g, n),
        "generous_and_neutral_above_selfish": min(g, n) > s,
        "generous_approx_neutral": abs(g - n) < TIE_MARGIN,
    }
    parts["pass"] = all(parts.values())
    return parts


async def score_all(anchors, templates):
    config = gj.judge_configs("primary")["score"]
    cache = HERE / "_cache" / "anchor_score.jsonl"
    client = jc.JudgeClient(config, cache)
    models = await client.open()              # raises EndpointUnavailable / ModelMismatch first
    results = {}
    for name in ("fairness", "trait"):
        requests = [jc.JudgeRequest("%s|%s" % (a["id"], name),
                                    jc.render(templates[name], a["question"], a["answer"]))
                    for a in anchors]
        results[name] = await client.judge_many(requests, order_seed=gj.ORDER_SEED)
    return client, models, results


def main():
    if not os.environ.get("PIMIENTOROJO_BASE_URL"):
        raise SystemExit("set PIMIENTOROJO_BASE_URL to the primary judge's endpoint")
    doc = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))
    anchors = doc["anchors"]
    templates = gj.load_templates()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        client, models, results = asyncio.run(score_all(anchors, templates))
    except (jc.EndpointUnavailable, jc.ModelMismatch, jc.ConfigError) as exc:
        raise SystemExit("anchor check stopped, nothing written: %s: %s" % (type(exc).__name__, exc))

    rows = []
    for i, a in enumerate(anchors):
        f, t = results["fairness"][i], results["trait"][i]
        rows.append({"id": a["id"], "kind": a["kind"], "question_id": a["question_id"],
                     "fairness": f.score, "fairness_status": f.status,
                     "altruism": t.score, "altruism_status": t.status,
                     "answer": a["answer"]})

    summary = {}
    for name in ("fairness", "altruism"):
        field = name
        per_kind = {k: mean([r[field] for r in rows if r["kind"] == k]) for k in KINDS}
        summary[name] = {"mean_by_kind": per_kind, "order": order_check(per_kind)}
    out = {
        "status": "EXPLORATORY validity check (prereg section 4)",
        "started": started,
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "judge": gj.JUDGE_DIRS["primary"],
        "configured_model": gj.SERVED_IDS["primary"],
        "served_models_response": models,
        "config": client.provenance()["config"],
        "fairness_template_sha256": gj.sha256_text(templates["fairness"]),
        "altruism_template_sha256": gj.sha256_text(templates["trait"]),
        "anchors_sha256": gj.sha256_text((HERE / "anchors.json").read_text(encoding="utf-8")),
        "rows": rows,
        "summary": summary,
        "fairness_check_pass": summary["fairness"]["order"]["pass"],
    }
    (HERE / "anchor_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    for name in ("fairness", "altruism"):
        print(name, {k: (None if v is None else round(v, 1)) for k, v in summary[name]["mean_by_kind"].items()},
              summary[name]["order"])


if __name__ == "__main__":
    main()
