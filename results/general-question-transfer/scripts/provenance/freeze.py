"""The freeze record: sha256 of everything PREREG.md pins, written once at the freezing commit.

Writes `provenance/freeze.md` with the sha256 of PREREG.md, both question files, the
pairwise template, the judge prompt templates (with the Llama system header rendered
for the fixed `date_string`), the arm-vectors table, and the commit the freeze was made
on top of. A commit's own hash cannot be inside itself, so the PARENT is recorded.

    python scripts/provenance/freeze.py --parent <sha>     # write provenance/freeze.md
    python scripts/provenance/freeze.py --check            # recompute, compare with the file

`--check` compares every recorded sha256 except the parent line. After the freeze,
PREREG.md is never edited; a changed hash is a bug or an undeclared edit.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parents[1]
OUT = PACKAGE / "provenance" / "freeze.md"

FREEZE_DATE = "2026-10-23"
#: The explicit `date_string` passed to the Llama judge (PREREG D6 / J12), in the
#: Llama-3.x template's own `D Mon YYYY` format.
LLAMA_DATE_STRING = "23 Oct 2026"
LLAMA_TEMPLATE_DEFAULT_DATE = "26 Jul 2024"
#: PREREG J6, appended after the verbatim template in text mode.
TEXT_FORMAT_INSTRUCTION = ("\n\nWrite the number inside <score></score> tags, for example <score>73</score>, "
                           "and write nothing else. If the model clearly refused to answer, write REFUSAL instead.")


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def pairwise_template(prereg_text):
    """The text between the fences after 'The template, written once', no trailing newline."""
    start = prereg_text.index("The template, written once")
    a = prereg_text.index("```text\n", start) + len("```text\n")
    b = prereg_text.index("\n```", a)
    return prereg_text[a:b]


def llama_header():
    """The rendered Llama system header for `LLAMA_DATE_STRING`, from the committed render."""
    check = json.loads((PACKAGE / "provenance" / "tokenizer_check.json").read_text(encoding="utf-8"))
    prefix = check["judges"]["llama-3.3-70b-instruct-q8_0"]["gguf"]["chat_prefix"]
    end = prefix.index("<|eot_id|>") + len("<|eot_id|>")
    header = prefix[:end]
    assert "Today Date: %s\n" % LLAMA_TEMPLATE_DEFAULT_DATE in header, header
    return header.replace(LLAMA_TEMPLATE_DEFAULT_DATE, LLAMA_DATE_STRING)


def entries():
    """Ordered (section, label, sha256 or text) triples."""
    prereg = (PACKAGE / "PREREG.md").read_bytes()
    trait = json.loads((REPO / "data_generation" / "trait_data_eval" / "altruistic.json")
                       .read_text(encoding="utf-8"))["eval_prompt"]
    sys.path.insert(0, str(REPO))
    from eval.prompts import Prompts
    coherence = Prompts["coherence_0_100"]
    pairwise = pairwise_template(prereg.decode("utf-8"))
    header = llama_header()
    rows = [
        ("files", "results/general-question-transfer/PREREG.md", sha(prereg)),
        ("files", "results/general-question-transfer/questions/heldout.json",
         sha((PACKAGE / "questions" / "heldout.json").read_bytes())),
        ("files", "results/general-question-transfer/questions/bridge.json",
         sha((PACKAGE / "questions" / "bridge.json").read_bytes())),
        ("files", "results/general-question-transfer/provenance/arm_vectors.json (the arm-vectors table)",
         sha((PACKAGE / "provenance" / "arm_vectors.json").read_bytes())),
        ("templates", "pairwise template (PREREG § 6, text between the fences)", sha(pairwise)),
        ("templates", "trait template (`eval_prompt` of data_generation/trait_data_eval/altruistic.json)", sha(trait)),
        ("templates", "coherence template (`Prompts[\"coherence_0_100\"]` of eval/prompts.py)", sha(coherence)),
        ("templates", "text-mode format instruction appended after a template (PREREG J6)", sha(TEXT_FORMAT_INSTRUCTION)),
        ("llama", "rendered Llama system header, `date_string` = \"%s\"" % LLAMA_DATE_STRING, sha(header)),
        ("llama", "Llama header + trait template, as sent", sha(header + trait)),
        ("llama", "Llama header + coherence template, as sent", sha(header + coherence)),
        ("llama", "Llama header + pairwise template, as sent", sha(header + pairwise)),
    ]
    return rows, header


ROW = re.compile(r"^\| (?P<label>.+?) \| `(?P<sha>[0-9a-f]{64})` \|$")


def render(parent):
    rows, header = entries()
    lines = [
        "# Freeze record — general-question transfer",
        "",
        "Frozen %s. Written by `scripts/provenance/freeze.py`; `--check` recomputes every hash below." % FREEZE_DATE,
        "",
        "**From this commit on `PREREG.md` is never edited.** Later changes go to `DEVIATIONS.md`,",
        "dated, with a reason. `scripts/tests/test_gqt_freeze.py` fails if the sha256 of `PREREG.md`",
        "differs from the one recorded here.",
        "",
        "**Commit.** A commit's own hash cannot be recorded inside itself. The freezing commit's",
        "**parent** is `%s`; the freezing commit is the commit made on top of it." % parent,
        "",
        "**Authorship of the question sets.** Drafted by an LLM agent (Claude) on 2026-10-23,",
        "accepted unedited by the operator on 2026-10-23.",
        "",
    ]
    titles = {"files": "Files", "templates": "Templates", "llama": "Llama judge prompt as sent"}
    for section in ("files", "templates", "llama"):
        lines += ["## " + titles[section], "", "| what | sha256 |", "|---|---|"]
        lines += ["| %s | `%s` |" % (label, digest) for s, label, digest in rows if s == section]
        lines.append("")
    lines += [
        "The Llama system header, rendered (the GGUF-embedded Llama-3.x template with the explicit",
        "`date_string` the study passes via `chat_template_kwargs`, PREREG D6/J12):",
        "",
        "```text",
        header,
        "```",
        "",
        "\"Header + template, as sent\" is the sha256 of the header string followed immediately by the",
        "template text (the user turn then holds the template with its `{question}`/`{answer}` slots",
        "filled). The gpt-oss header carries the call date and cannot be pinned (PREREG D7).",
        "",
    ]
    return "\n".join(lines)


def recorded():
    return {m["label"]: m["sha"] for m in map(ROW.match, OUT.read_text(encoding="utf-8").split("\n")) if m}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parent")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        now = {label: digest for _, label, digest in entries()[0]}
        if now != recorded():
            bad = sorted(k for k in set(now) | set(recorded()) if now.get(k) != recorded().get(k))
            raise SystemExit("freeze record differs: %s" % bad)
        print("ok: freeze record reproduces")
        return
    if not args.parent or not re.fullmatch(r"[0-9a-f]{40}", args.parent):
        raise SystemExit("--parent <40-hex commit> is required")
    OUT.write_text(render(args.parent), encoding="utf-8")
    print("wrote %s" % OUT.relative_to(PACKAGE))


if __name__ == "__main__":
    sys.exit(main())
