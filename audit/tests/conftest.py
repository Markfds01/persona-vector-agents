"""Labelled fixtures built from the generations committed in this repo.

The upstream CSVs pair a raw generation with the paid gpt-4.1-mini judge's
extracted value, which makes them a free labelled test set for the scorers. No
network, no keys, no GPU: everything here is a file read.
"""

import csv
import re
from typing import NamedTuple, Optional

import pytest

from audit.paths import EVAL_DIR

#: `altruism_v2` runs: one file per steering coefficient, `altruism` column is a
#: dollar amount because v2's eval_prompt is the only amount-extraction judge.
V2_DIR = EVAL_DIR / "v2"

#: The paper's Table 1 suite (`altruism_v3` questions), despite the directory name.
#: Its `*_compare_with_expectation.csv` files carry per-question extraction judges.
TABLE1_DIR = EVAL_DIR / "v1"

#: The scorers were tuned on the `altruism_v2` Dictator rows of this condition.
#: `v2_held_out` drops it; `per_question_labels` does NOT (it returns all seven
#: conditions). Tests that include it say so.
TUNING_COEFFICIENT = 0.0

_COEFFICIENT_RX = re.compile(r"coef(-?\d+(?:\.\d+)?)")


class LabelledRow(NamedTuple):
    question_id: str
    answer: str
    judge: float


def steering_coefficient(path) -> float:
    match = _COEFFICIENT_RX.search(path.name)
    if match is None:
        raise ValueError("no steering coefficient in %s" % path.name)
    return float(match.group(1))


def _labelled_rows(path, judge_column, keep_question_ids=None):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = row["question_id"]
            if keep_question_ids is not None and question_id not in keep_question_ids:
                continue
            judge = _as_float(row.get(judge_column))
            if judge is None:
                continue
            rows.append(LabelledRow(question_id, row["answer"], judge))
    return rows


def _as_float(raw) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@pytest.fixture(scope="session")
def v2_by_coefficient():
    """{steering coefficient: [LabelledRow]} over every committed altruism_v2 run."""
    files = sorted(V2_DIR.glob("altruism_steer_*_game_v2.csv"))
    assert files, "no altruism_v2 result files under %s" % V2_DIR
    return {steering_coefficient(path): _labelled_rows(path, "altruism") for path in files}


@pytest.fixture(scope="session")
def v2_held_out(v2_by_coefficient):
    """Every altruism_v2 condition the scorers were not tuned on."""
    held_out = {coefficient: rows for coefficient, rows in v2_by_coefficient.items()
                if coefficient != TUNING_COEFFICIENT}
    assert len(held_out) == len(v2_by_coefficient) - 1
    return held_out


@pytest.fixture(scope="session")
def per_question_labels():
    """{judge column: [LabelledRow]} from the Table-1 per-question extraction judges.

    All seven steering conditions, `TUNING_COEFFICIENT` included.
    `*_rejudged.csv` is skipped: same generations, judged a second time.
    """
    files = sorted(TABLE1_DIR.glob("altruism_steer_*_compare_with_expectation.csv"))
    assert files, "no compare_with_expectation files under %s" % TABLE1_DIR
    wanted = {"q3_fish_caught": "altruism_3", "q4_cooperated": "altruism_4"}
    labels = {}
    for column, question_id in wanted.items():
        rows = []
        for path in files:
            rows.extend(_labelled_rows(path, column, keep_question_ids={question_id}))
        assert rows, "no labelled rows for %s" % column
        labels[column] = rows
    return labels
