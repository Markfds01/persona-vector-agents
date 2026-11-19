"""PREREG section 11: does the Llama judge server render the system header freeze.md pins?

    PIMIENTOROJO_BASE_URL=<the primary judge's endpoint> python llama_header_check.py

Asks the running llama-server to render one user turn through its own chat template
(`/apply-template`, the same `chat_template_kwargs` as every study request), cuts the system
header off the rendered prompt, and compares its sha256 (and the sha256 of header + each
judge template) with `provenance/freeze.md`. Also renders once with no `date_string`, to
show the override is what sets the date. Writes provenance/llama_header_check.json; the
endpoint is never written.
"""

import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PACKAGE / "scripts" / "judging"))

import freeze  # noqa: E402
import gqt_judging as gj  # noqa: E402


def post(base, path, payload):
    req = urllib.request.Request(base.rstrip("/") + path, json.dumps(payload).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base, path):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def header_of(prompt):
    end = prompt.index("<|eot_id|>") + len("<|eot_id|>")
    return prompt[:end]


def main():
    base = os.environ.get("PIMIENTOROJO_BASE_URL")
    if not base:
        raise SystemExit("set PIMIENTOROJO_BASE_URL to the primary judge's endpoint")
    kwargs = gj.judge_configs("primary")["score"].extra_payload["chat_template_kwargs"]
    messages = [{"role": "user", "content": "X"}]
    served = post(base, "/apply-template", {"messages": messages, "chat_template_kwargs": kwargs})["prompt"]
    default = post(base, "/apply-template", {"messages": messages})["prompt"]
    props = get(base, "/props")
    header = header_of(served)
    recorded = freeze.recorded()
    frozen_header = freeze.llama_header()
    templates = gj.load_templates()
    for name in ("fairness", "generosity"):
        templates[name] = json.loads((freeze.REPO / "data_generation" / "trait_data_eval" / ("%s.json" % name))
                                     .read_text(encoding="utf-8"))["eval_prompt"]
    header_label = "rendered Llama system header, `date_string` = \"%s\"" % freeze.LLAMA_DATE_STRING
    as_sent = {}
    for label, sha in recorded.items():
        if label.startswith("Llama header + "):
            name = label[len("Llama header + "):].split(" template")[0]
            if name in templates:
                as_sent[name] = {"recorded": sha, "served": freeze.sha(header + templates[name]),
                                 "matches": freeze.sha(header + templates[name]) == sha}
    out = {
        "what": "PREREG section 11 Llama header check: the header as the PimientoRojo llama-server renders it",
        "checked": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "server_build_info": props.get("build_info"),
        "server_model_path": props.get("model_path"),
        "server_chat_template_sha256": freeze.sha(props.get("chat_template", "")),
        "chat_template_kwargs": kwargs,
        "served_header": header,
        "served_header_sha256": freeze.sha(header),
        "freeze_md_header_sha256": recorded[header_label],
        "header_matches_freeze_md": freeze.sha(header) == recorded[header_label] and header == frozen_header,
        "header_plus_template": as_sent,
        "default_header_without_date_string": header_of(default),
        "user_turn_after_header": served[len(header):],
    }
    out["all_match"] = out["header_matches_freeze_md"] and all(v["matches"] for v in as_sent.values())
    (PACKAGE / "provenance" / "llama_header_check.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("header matches freeze.md: %s; header + templates: %s"
          % (out["header_matches_freeze_md"], {k: v["matches"] for k, v in as_sent.items()}))
    return 0 if out["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
