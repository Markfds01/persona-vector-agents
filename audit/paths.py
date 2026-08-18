"""Filesystem anchors for the audit package.

One owner for every path into the upstream repo, so a moved data file breaks in
exactly one place.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Upstream question sets (`altruism_v3.json` and friends). Read, never written.
QUESTION_DIR = REPO_ROOT / "data_generation" / "trait_data_eval"

#: Committed generations + paid-judge labels for Qwen2.5-7B-Instruct.
EVAL_DIR = REPO_ROOT / "eval_persona_eval" / "Qwen2.5-7B-Instruct"
