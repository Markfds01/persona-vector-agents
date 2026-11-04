# What code the full generation run executed (`repo_commit`, `repo_dirty`)

The 13,320 rows in `rows_full/` record one of three `repo_commit` values, all with `repo_dirty = True`:

| `repo_commit` | what it is | rows (question set: count) | generation processes |
|---|---|---|---|
| `425629d584d3` | "Add the four-process driver for the full generation run" (2026-11-03 11:12 UTC) | in-distribution 1,800; bridge 3,120; held-out 960 | parts 1–4 (launched 11:12:30 UTC) |
| `663122a79934` | "Rebalance the full run's remaining held-out points over three slots" (11:38 UTC) | held-out 4,680 | parts 5–7 (launched 11:38:24 UTC) |
| `e417d87668d9` | "Full run milestone: 59 of 111 generation points" (12:12 UTC) | bridge 2,280; held-out 480 | processes still running at that commit (parts 3 and 7 and their relaunches) |

(Counts are of rows by the commit each row records; the part logs in this directory are the per-process record.)

**The generation code is identical across the three.** `git diff 425629d e417d87` is empty for
`scripts/generation/`, `lab/`, `eval/`, `persona_vectors/`, `questions/`, `vectors/`, `config.py` and
`utils.py`. What changed between the commits is the driver `scripts/run_full_generation.sh`
(the 11:38 rebalance, DEVIATIONS 2026-11-03), the rubric-validation files, the validation script,
the provenance logs and the documentation: nothing a generation process imports. Each process
recorded the commit that was `HEAD` of the checkout when it was *launched* (parts 5–7) or when
each point was *written* (the long-running parts), so the commit tracks the checkout, not the code.

**Why `repo_dirty = True` on every row.** The generation processes wrote their output into the
checkout they ran from (`rows_full/`, `provenance/generation_full/` logs and point files, the
interpreter record), so the working tree always held uncommitted or untracked files from the very
first point, and the flag is set for that reason. I could not recover the exact `git status` of
the generation machine at each launch: it was not saved. What I can state is limited to the
repository history: no tracked file under the generation code paths listed above differs between
the three commits, and a diff of the committed state against any of them in those paths is empty.
It is not proven that no uncommitted edit to those files existed on the generation machine; the
evidence against it is that the rows' own hashes (`question_file_sha256`, `vector_sha256`,
`chat_template_sha256`, `prompt_sha256`, `output_ids_sha256`) and the integrity checks
(`integrity.json`, `analysis/full_generation_report.md`) agree with the committed inputs.

`*.lock` files that the points' atomic writes create next to the point files are git-ignored
(`.gitignore` here); none was committed on this branch (`git ls-files` finds none).
