"""What the judges are asked, for which rows, and how a judgment becomes a row.

The library behind `run_judging.py` (the CLI) and the validation stage. Pure Python: it
imports no model and talks to no server; `lab.llama_server_client` does that.

Which judge scores what (PREREG section 6, `div8_main`), restricted to the rows that exist:

* **primary** (Llama-3.3-70B-Instruct Q8_0, `logprob_0_100`): trait and coherence on every
  held-out row; coherence on every bridge row; trait on the bridge AUC subset (the k = 0
  baseline and the trait vector at +-3); pairwise on every steered held-out row, both orders.
* **secondary** (gpt-oss-20b MXFP4, `text_0_100`): trait and coherence on the pre-registered
  25% subset of the held-out rows (30 of the 120 (question, sample) pairs of a point, one
  fixed subset for every point, seed 20261024); trait on the bridge AUC subset; pairwise on
  the same 25% subset.

The control points (`unhooked`, `unhooked_repeat`) are never judged.

The judge sees a rendered prompt and nothing else: the verbatim template with the question
and the answer substituted in (and, for the secondary judge's text mode, PREREG J6's format
instruction appended). A judgment row carries the generation `row_id` so the analysis can
join it back; that id never reaches the server.
"""

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence

PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parents[1]
sys.path.insert(0, str(REPO))

from lab import llama_server_client as jc  # noqa: E402

LLAMA_DATE_STRING = "23 Oct 2026"
#: PREREG J6, appended after a verbatim template in text mode (sha256 in provenance/freeze.md)
TEXT_FORMAT_INSTRUCTION = ("\n\nWrite the number inside <score></score> tags, for example <score>73</score>, "
                           "and write nothing else. If the model clearly refused to answer, write REFUSAL instead.")

#: the ids the pilot's servers report at /v1/models. PREREG J10 expected `--alias` ids
#: (`llama-3.3-70b-instruct-q8_0`, `gpt-oss-20b-mxfp4`); the servers reached through the tunnel
#: were launched without an alias, so the served id is the model path (DEVIATIONS.md).
SERVED_IDS = {
    "primary": "models/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
    "secondary": "models/gpt-oss-20b-MXFP4.gguf",
}
JUDGE_DIRS = {"primary": "llama-3.3-70b-instruct-q8_0", "secondary": "gpt-oss-20b-mxfp4"}

SUBSET_SEED = 20261024
SUBSET_SIZE = 30
RESEND_SEED = 20261026
RESEND_SHARE = 0.05
ORDER_SEED = 20261027
JUDGE_SEED = 0

CONTROL_ARMS = ("unhooked", "unhooked_repeat")
KINDS = ("trait", "coherence", "pairwise")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_templates() -> Dict[str, str]:
    """The verbatim templates: trait and coherence from the repo, pairwise from PREREG, and the
    exploratory fairness rubric (docs/fairness_exploratory_prereg.md) next to the trait one."""
    trait = json.loads((REPO / "data_generation" / "trait_data_eval" / "altruistic.json")
                       .read_text(encoding="utf-8"))["eval_prompt"]
    fairness = json.loads((REPO / "data_generation" / "trait_data_eval" / "fairness_exploratory.json")
                          .read_text(encoding="utf-8"))["eval_prompt"]
    from eval.prompts import Prompts
    prereg = (PACKAGE / "PREREG.md").read_text(encoding="utf-8")
    start = prereg.index("The template, written once")
    a = prereg.index("```text\n", start) + len("```text\n")
    b = prereg.index("\n```", a)
    return {"trait": trait, "coherence": Prompts["coherence_0_100"], "pairwise": prereg[a:b],
            "fairness": fairness}


def judge_configs(judge: str) -> Dict[str, "jc.JudgeConfig"]:
    """`score` (trait, coherence) and `pairwise` configs of one judge; both fixed by PREREG 11."""
    if judge == "primary":
        extra = {"chat_template_kwargs": {"date_string": LLAMA_DATE_STRING}}
        return {
            "score": jc.JudgeConfig(model=SERVED_IDS[judge], mode="logprob_0_100", max_tokens=1,
                                    seed=JUDGE_SEED, concurrency=1, top_logprobs=20,
                                    extra_payload=extra),
            "pairwise": jc.JudgeConfig(model=SERVED_IDS[judge], mode="choice_1_2", max_tokens=32,
                                       seed=JUDGE_SEED, concurrency=1, extra_payload=extra),
        }
    if judge == "secondary":
        extra = {"chat_template_kwargs": {"reasoning_effort": "low"}}
        return {
            "score": jc.JudgeConfig(model=SERVED_IDS[judge], mode="text_0_100", max_tokens=2048,
                                    seed=JUDGE_SEED, concurrency=1, extra_payload=extra),
            "pairwise": jc.JudgeConfig(model=SERVED_IDS[judge], mode="choice_1_2", max_tokens=2048,
                                       seed=JUDGE_SEED, concurrency=1, extra_payload=extra),
        }
    raise ValueError("judge must be primary or secondary, got %r" % judge)


_PAIRWISE_FIELD = re.compile(r"\{(question|answer_1|answer_2)\}")


def render_pairwise(template: str, question: str, answer_1: str, answer_2: str) -> str:
    """One pass over the template, so a field name inside an answer is never substituted again."""
    values = {"question": question, "answer_1": answer_1, "answer_2": answer_2}
    return _PAIRWISE_FIELD.sub(lambda match: values[match.group(1)], template)


def secondary_subset(n_questions_samples: Sequence["tuple[str, int]"]) -> "set[tuple[str, int]]":
    """The pre-registered 25%: `random.Random(20261024).sample(sorted pairs, 30)` of a point's 120."""
    pairs = sorted(n_questions_samples)
    if len(pairs) < SUBSET_SIZE:
        raise ValueError("a point has %d (question, sample) pairs, need at least %d"
                         % (len(pairs), SUBSET_SIZE))
    return set(random.Random(SUBSET_SEED).sample(pairs, SUBSET_SIZE))


def read_jsonl(path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_generation_rows(rows_dir) -> List[Dict]:
    """Every row of every judged point (controls excluded), in a stable order."""
    rows = []
    for path in sorted(Path(rows_dir).rglob("*.jsonl")):
        for row in read_jsonl(path):
            if row["arm"] in CONTROL_ARMS:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["question_set"], r["arm"], r["k"], r["question_id"], r["sample_index"]))
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row_id among the generation rows")
    return rows


def bridge_auc_row(row: Mapping) -> bool:
    """The AUC subset: the k = 0 baseline and the trait vector at +-3, on the bridge set."""
    return row["question_set"] == "bridge" and (
        row["arm"] == "baseline" or (row["arm"] == "trait/altruism" and row["k"] != 0))


class Task(NamedTuple):
    """One judge call to make: the row it is about and the prompt that goes out."""

    key: str
    kind: str
    row_id: str
    order: Optional[str]        # pairwise only: "baseline_first" | "steered_first"
    steered_position: Optional[int]
    prompt: str
    template_sha256: str
    instruction_sha256: Optional[str]
    config_name: str            # "score" | "pairwise"


def build_plan(judge: str, rows: Sequence[Mapping], templates: Mapping[str, str],
               with_fairness: bool = False) -> List[Task]:
    """Every call this judge makes for these rows, in a stable (unshuffled) order.

    `with_fairness` adds the exploratory fairness rubric on the held-out rows the trait rubric
    scores (the same rows, the same subset for the secondary judge). It is off by default, so
    the confirmatory plan, its determinism re-send and the existing judgments are unchanged."""
    if judge not in SERVED_IDS:
        raise ValueError(judge)
    text_mode = judge == "secondary"
    instruction = TEXT_FORMAT_INSTRUCTION if text_mode else ""
    instruction_sha = sha256_text(TEXT_FORMAT_INSTRUCTION) if text_mode else None
    shas = {kind: sha256_text(templates[kind]) for kind in templates}
    unknown = sorted({r["question_set"] for r in rows} - {"heldout", "bridge"})
    if unknown:
        raise ValueError("no judging plan for question sets %s (the pilot judges heldout and bridge; "
                         "the in-distribution set is added with the full run)" % unknown)
    heldout = [r for r in rows if r["question_set"] == "heldout"]
    by_point_pairs: Dict["tuple[str, float]", List["tuple[str, int]"]] = {}
    for r in heldout:
        by_point_pairs.setdefault((r["arm"], r["k"]), []).append((r["question_id"], r["sample_index"]))
    subset = None
    if judge == "secondary":
        subset = secondary_subset(next(iter(by_point_pairs.values()))) if by_point_pairs else set()

    def in_scope(r):
        return subset is None or (r["question_id"], r["sample_index"]) in subset

    plan: List[Task] = []

    def scored(r, kind):
        prompt = jc.render(templates[kind], r["question"], r["answer"]) + instruction
        plan.append(Task("%s|%s" % (r["row_id"], kind), kind, r["row_id"], None, None, prompt,
                         shas[kind], instruction_sha, "score"))

    for r in rows:
        if r["question_set"] == "heldout" and in_scope(r):
            scored(r, "trait")
            scored(r, "coherence")
            if with_fairness:
                scored(r, "fairness")
        elif r["question_set"] == "bridge":
            if judge == "primary":
                scored(r, "coherence")
            if bridge_auc_row(r):
                scored(r, "trait")
    baseline = {(r["question_id"], r["sample_index"]): r for r in heldout if r["arm"] == "baseline"}
    for r in heldout:
        if r["k"] == 0 or abs(r["k"]) != 3 or not in_scope(r):
            continue
        base = baseline.get((r["question_id"], r["sample_index"]))
        if base is None:
            raise ValueError("steered row %s has no baseline row for the same question and sample: "
                             "the pairwise pair cannot be formed" % r["row_id"])
        for order, steered_position in (("baseline_first", 2), ("steered_first", 1)):
            first, second = (base, r) if order == "baseline_first" else (r, base)
            prompt = render_pairwise(templates["pairwise"], r["question"], first["answer"],
                                     second["answer"])
            plan.append(Task("%s|pairwise|%s" % (r["row_id"], order), "pairwise", r["row_id"],
                             order, steered_position, prompt, shas["pairwise"], None, "pairwise"))
    keys = [t.key for t in plan]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate judge request keys in the plan")
    return plan


def shuffled(plan: Sequence[Task], seed: int = ORDER_SEED) -> List[Task]:
    order = list(range(len(plan)))
    random.Random(seed).shuffle(order)
    return [plan[i] for i in order]


def decoding(config: "jc.JudgeConfig") -> Dict:
    return {"mode": config.mode, "temperature": config.temperature, "seed": config.seed,
            "max_tokens": config.max_tokens,
            "top_logprobs": config.top_logprobs if config.mode == "logprob_0_100" else None,
            "cache_prompt": False, "extra_payload": dict(config.extra_payload)}


def judgment_row(judge: str, task: Task, result: "jc.JudgeResult", config: "jc.JudgeConfig",
                 run_id: str) -> Dict:
    return {
        "key": task.key, "row_id": task.row_id, "kind": task.kind, "order": task.order,
        "steered_position": task.steered_position, "judge": JUDGE_DIRS[judge],
        "status": result.status, "score": result.score,
        "payload_sha256": result.payload_sha256, "served_model": result.served_model,
        "prompt_template_sha256": task.template_sha256,
        "format_instruction_sha256": task.instruction_sha256,
        "decoding": decoding(config), "timestamp": result.timestamp,
        "from_cache": result.from_cache, "attempts": result.attempts,
        "detail": result.detail, "raw_response": result.raw_response, "run_id": run_id,
    }


def latest_by_key(rows: Iterable[Mapping]) -> Dict[str, Mapping]:
    """The last row per key: a re-run's row supersedes an earlier `transport_error`."""
    out: Dict[str, Mapping] = {}
    for row in rows:
        out[row["key"]] = row
    return out


def done_keys(rows: Iterable[Mapping]) -> "set[str]":
    return {k for k, r in latest_by_key(rows).items() if r["status"] != "transport_error"}


def resend_selection(payload_shas: Iterable[str], seed: int = RESEND_SEED,
                     share: float = RESEND_SHARE) -> List[str]:
    """5% of the sorted distinct request-payload sha256s, selected by seed (PREREG 9.2)."""
    ordered = sorted(set(payload_shas))
    if not ordered:
        return []
    return sorted(random.Random(seed).sample(ordered, max(1, math.ceil(share * len(ordered)))))


def comparable(response: Optional[Mapping], mode: str):
    """What 'exactly the same answer' means for a response: the content, or the top logprobs."""
    if not response:
        return None
    choice = (response.get("choices") or [{}])[0]
    if mode == "logprob_0_100":
        content = (choice.get("logprobs") or {}).get("content") or []
        top = content[0].get("top_logprobs") if content else None
        return [[e.get("token"), e.get("logprob")] for e in (top or [])]
    return (choice.get("message") or {}).get("content")
