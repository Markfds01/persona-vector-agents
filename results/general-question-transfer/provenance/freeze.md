# Freeze record — general-question transfer

Frozen 2026-10-23. Written by `scripts/provenance/freeze.py`; `--check` recomputes every hash below.

**From this commit on `PREREG.md` is never edited.** Later changes go to `DEVIATIONS.md`,
dated, with a reason. `scripts/tests/test_gqt_freeze.py` fails if the sha256 of `PREREG.md`
differs from the one recorded here.

**Commit.** A commit's own hash cannot be recorded inside itself. The freezing commit's
**parent** is `9b24e7e24020562b04af2290c46ab2c6e61081d5`; the freezing commit is the commit made on top of it.

**Authorship of the question sets.** Drafted by an LLM agent (Claude) on 2026-10-23,
accepted unedited by the operator on 2026-10-23.

## Files

| what | sha256 |
|---|---|
| results/general-question-transfer/PREREG.md | `72a9cb77d5be934984e0dced42dd5e437f35f67d55eeb2f140393ca869633302` |
| results/general-question-transfer/questions/heldout.json | `d82c3347705e232fb2b08f72063bf8b7caa99a08bf0978d22722cdec2892bdae` |
| results/general-question-transfer/questions/bridge.json | `17d8d745d16acabaefdecef9ed0e1229b629ad63dd8b7618e2aa130a8d94b6a9` |
| results/general-question-transfer/provenance/arm_vectors.json (the arm-vectors table) | `4551c58f839f5ace55034467db540a71581cd7444afdf76a725262305a9fcba1` |

## Templates

| what | sha256 |
|---|---|
| pairwise template (PREREG § 6, text between the fences) | `6ecc680ffa1bc3877b9be65ea094a295557e790f014b995d33a84137228f875b` |
| trait template (`eval_prompt` of data_generation/trait_data_eval/altruistic.json) | `07fe12f75c7a50dd8cd398d41f6b33b52c5bb0be9bd6c039877042fb6cb518dd` |
| coherence template (`Prompts["coherence_0_100"]` of eval/prompts.py) | `78945e05210836da5c492e01cead92cdc8f8aa40f1b0c22d162b4dc2a1344a97` |
| text-mode format instruction appended after a template (PREREG J6) | `06d2d93aa31e94207739b50ff6c7feb70191144649dafc1bfc310d832df9b81c` |

## Llama judge prompt as sent

| what | sha256 |
|---|---|
| rendered Llama system header, `date_string` = "23 Oct 2026" | `5923dabeef9e437608744c3e7efd883d59859a80a135cf7962d4b814fa229a54` |
| Llama header + trait template, as sent | `0f6058e93f3274e51a9e42fa46544c0f0b430ee984d6a7707b0e79560fd4a099` |
| Llama header + coherence template, as sent | `ff586643b7bb7415f8e05e33f8d3abae44c3d89230d2ce809bade01b1a517f0e` |
| Llama header + pairwise template, as sent | `20a2415e52b623c17d6376532b2dfbe4a023225161b40b8658b89fb3b8276bf0` |

The Llama system header, rendered (the GGUF-embedded Llama-3.x template with the explicit
`date_string` the study passes via `chat_template_kwargs`, PREREG D6/J12):

```text
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 Oct 2026

<|eot_id|>
```

"Header + template, as sent" is the sha256 of the header string followed immediately by the
template text (the user turn then holds the template with its `{question}`/`{answer}` slots
filled). The gpt-oss header carries the call date and cannot be pinned (PREREG D7).
