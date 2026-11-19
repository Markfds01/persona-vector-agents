# Human labellers (PREREG § 0 D10, § 9.1)

Recorded 2026-11-03 (DEVIATIONS 2026-11-03, "Second labeller is AAC").

| labeller | who | sent |
|---|---|---|
| 1 | the operator | `human/grader.html` |
| 2 | **AAC** (initials, the operator's choice) | `human/grader.html` **only** |

Neither labeller is sent `provenance/human_sample_key.json` (it names each item's arm and k)
or `human/judge_reveal_*.json` (the judges' scores). Each labeller labels independently and
blind; their exports are `human_labels_<reviewer>_<date>.csv`.
