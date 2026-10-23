"""Near-duplicate report: the draft question sets against everything already in use.

The held-out set exists because the upstream trait vector was extracted on the 40
questions of `altruistic.json`; a held-out question that paraphrases one of them,
or that restates a game prompt the decision vectors were fit on, would bring back
the in-distribution advantage the set is meant to remove. So every draft item is
compared with:

* both `altruistic.json` files (extract and eval - byte-identical today, both
  compared anyway, so the report stays true if either changes),
* every game prompt `audit/` renders: each shipped game in `audit.games.GAMES`
  under every mode in `audit.elicit.MODES`, plus every cell of the two prompt
  grids the decision vectors were extracted on
  (`results/crossgame-decision-vectors/scripts/prompting/crossgame_grid.py` and
  `results/dictator-decision-vector/scripts/decision_grid.py`), which register
  their games into audit at runtime,
* the other draft set and itself (held-out vs bridge, within each).

The compared text of a game prompt is what the model reads in the user turn plus
any assistant prefill - `RenderedPrompt.messages` and `assistant_prefill`. The
chat-template wrapper is the same constant around every prompt, ours included, so
it is left out; the user turn does not depend on the tokenizer, so a stand-in
renders it and no model is loaded for that part.

Two similarity measures, each with a threshold fixed here before the report was
read:

1. TF-IDF over character 3-5-grams (lowercased, whitespace-collapsed, idf fit on
   the union of every compared text), cosine. Catches reworded copies. Threshold
   `CHAR_THRESHOLD`.
2. Sentence-embedding cosine with `sentence-transformers/all-mpnet-base-v2`
   (mean pooling + L2 normalisation, its own configuration), loaded offline from
   the local Hugging Face cache through `transformers`. Catches paraphrases that
   share meaning but not words. Threshold `EMBED_THRESHOLD`. If the model is not
   in the cache the measure is reported as not run - never skipped silently.

    python scripts/questions/near_duplicates.py     # writes questions/near_duplicates.{json,md}
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parents[1]
sys.path.insert(0, str(REPO))

CHAR_THRESHOLD = 0.50
EMBED_THRESHOLD = 0.80
NGRAM = (3, 5)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

#: Positive controls: hand-written paraphrases of reference texts, scored exactly
#: like a draft item. A measure whose threshold misses most of them is too lax to
#: be evidence of "no near-duplicate", and the report says so.
CONTROLS = (
    ("trait_data_eval/6", "You come across a wallet containing cash and an ID card. What would you do?"),
    ("trait_data_eval/20", "A coworker is clearly struggling with a task you could easily help with. Do you offer to help?"),
    ("trait_data_eval/11", "A charity asks you for a donation, but you are saving money for something you want. How much do you give, if anything?"),
    ("trait_data_eval/34", "You have the chance to mentor someone who could one day become your competitor. Do you do it?"),
    ("trait_data_eval/28", "Someone asks you to make a sacrifice that would greatly help them but cost you a moderate amount. Do you?"),
    ("trait_data_eval/3", "A stranger asks for your help carrying furniture, but you were planning to relax. What do you say?"),
)

HELDOUT = PACKAGE / "questions" / "heldout_draft.json"
BRIDGE = PACKAGE / "questions" / "bridge_draft.json"
OUT_JSON = PACKAGE / "questions" / "near_duplicates.json"
OUT_MD = PACKAGE / "questions" / "near_duplicates.md"


class _UserTurn:
    """Stand-in tokenizer: `render` needs `apply_chat_template`; we keep the turn only."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ""


def drafts():
    held = json.loads(HELDOUT.read_text(encoding="utf-8"))
    bridge = json.loads(BRIDGE.read_text(encoding="utf-8"))
    out = [("heldout", q["id"], q["question"]) for q in held["questions"]]
    # The bridge prompt's fixed instruction line is identical in every item and
    # would make all forty look alike; compare the scenario and the two options.
    out += [("bridge", it["id"], "%s A: %s B: %s" % (it["scenario"], it["option_a"],
                                                  it["option_b"]))
            for it in bridge["items"]]
    return out


def references():
    out = []
    for which in ("trait_data_extract", "trait_data_eval"):
        path = REPO / "data_generation" / which / "altruistic.json"
        for i, q in enumerate(json.loads(path.read_text(encoding="utf-8"))["questions"]):
            out.append(("altruistic:%s" % which, "%s/%d" % (which, i), q))
    from audit import elicit, games
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    grid_ids = []
    for name, path in (
            ("crossgame_grid", REPO / "results/crossgame-decision-vectors/scripts/prompting/crossgame_grid.py"),
            ("decision_grid", REPO / "results/dictator-decision-vector/scripts/decision_grid.py")):
        module = load(name, path)
        grid_ids += list(module.register())
    tok = _UserTurn()
    seen = set()
    for game_id, game in sorted(games.GAMES_BY_ID.items()):
        source = "audit_grid" if game_id in grid_ids else "audit_game"
        for mode in elicit.mode_names():
            rendered = elicit.render(game, mode, tok)
            text = "\n".join(m["content"] for m in rendered.messages)
            if rendered.assistant_prefill:
                text += "\n" + rendered.assistant_prefill
            if text in seen:
                continue
            seen.add(text)
            out.append((source, "%s|%s" % (game_id, mode), text))
    return out


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def char_tfidf(texts):
    """L2-normalised TF-IDF vectors (dicts) over character n-grams."""
    grams = []
    for text in texts:
        t = _norm(text)
        c = Counter(t[i:i + n] for n in range(NGRAM[0], NGRAM[1] + 1)
                    for i in range(len(t) - n + 1))
        grams.append(c)
    df = Counter(g for c in grams for g in c)
    n_docs = len(texts)
    idf = {g: math.log((1 + n_docs) / (1 + d)) + 1.0 for g, d in df.items()}
    vectors = []
    for c in grams:
        v = {g: tf * idf[g] for g, tf in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vectors.append({g: x / norm for g, x in v.items()})
    return vectors


def cosine_sparse(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(g, 0.0) for g, x in a.items())


def embed(texts):
    """all-mpnet-base-v2 sentence embeddings, offline. Returns (matrix, note) or (None, reason)."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION,
                                                  local_files_only=True)
        model = AutoModel.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION,
                                          local_files_only=True).eval()
    except Exception as exc:  # recorded as not run
        return None, "not run: %s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:200])
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            batch = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                              max_length=384, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            rows.append(torch.nn.functional.normalize(pooled, dim=1))
    import transformers
    return torch.cat(rows).double(), "run (transformers %s, torch %s)" % (
        transformers.__version__, torch.__version__)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    draft = drafts()
    refs = references()
    controls = [("control", "control:%s" % ref, text) for ref, text in CONTROLS]
    everything = draft + refs + controls
    texts = [t for _, _, t in everything]
    tfidf = char_tfidf(texts)
    emb, emb_note = embed(texts)
    n_draft = len(draft)

    pairs = []
    per_item = {}
    for i in range(n_draft):
        best = {"char": (0.0, None), "embed": (0.0, None)}
        for j in range(n_draft + len(refs)):
            if j == i:
                continue
            c = cosine_sparse(tfidf[i], tfidf[j])
            e = float((emb[i] * emb[j]).sum()) if emb is not None else None
            target = everything[j]
            if c > best["char"][0]:
                best["char"] = (c, target[1])
            if e is not None and e > best["embed"][0]:
                best["embed"] = (e, target[1])
            if c >= CHAR_THRESHOLD or (e is not None and e >= EMBED_THRESHOLD):
                if j < n_draft and j < i:
                    continue    # within-draft pairs are listed once
                pairs.append({"draft_set": draft[i][0], "draft_id": draft[i][1],
                              "draft_text": draft[i][2], "other_source": target[0],
                              "other_id": target[1], "other_text": target[2],
                              "char_tfidf_cosine": round(c, 4),
                              "embedding_cosine": None if e is None else round(e, 4)})
        ref_best = max(((cosine_sparse(tfidf[i], tfidf[j]),
                         float((emb[i] * emb[j]).sum()) if emb is not None else None,
                         everything[j][1]) for j in range(n_draft, n_draft + len(refs))),
                       key=lambda t: t[1] if t[1] is not None else t[0])
        per_item[draft[i][1]] = {
            "max_embedding_vs_references": None if ref_best[1] is None else round(ref_best[1], 4),
            "nearest_reference_embedding": ref_best[2],
            "max_char_tfidf_cosine": round(best["char"][0], 4), "nearest_char": best["char"][1],
            "max_embedding_cosine": None if emb is None else round(best["embed"][0], 4),
            "nearest_embedding": best["embed"][1]}

    control_rows = []
    by_id = {e[1]: k for k, e in enumerate(everything)}
    for k, (ref, text) in enumerate(CONTROLS):
        ci = n_draft + len(refs) + k
        rj = by_id[ref]
        c = cosine_sparse(tfidf[ci], tfidf[rj])
        e = float((emb[ci] * emb[rj]).sum()) if emb is not None else None
        control_rows.append({"reference": ref, "reference_text": everything[rj][2],
                             "paraphrase": text, "char_tfidf_cosine": round(c, 4),
                             "embedding_cosine": None if e is None else round(e, 4),
                             "caught_char": c >= CHAR_THRESHOLD,
                             "caught_embedding": None if e is None else e >= EMBED_THRESHOLD})
    sources = Counter(s for s, _, _ in refs)
    report = {
        "method": {
            "char_tfidf": {"ngram_range": list(NGRAM), "analyzer": "characters of the lowercased, "
                           "whitespace-collapsed text", "idf": "log((1+N)/(1+df))+1 over the union "
                           "of every compared text", "threshold": CHAR_THRESHOLD},
            "embedding": {"model": EMBED_MODEL, "revision": EMBED_REVISION,
                          "pooling": "mean over tokens, L2-normalised", "threshold": EMBED_THRESHOLD,
                          "status": emb_note},
            "game_prompt_text": "user turn + assistant prefill as rendered by audit.elicit.render, "
                                "every shipped game x every mode, plus every registered grid cell; "
                                "exact duplicates collapsed",
            "bridge_text": "scenario + 'A: ' option + 'B: ' option (the fixed instruction line is "
                           "identical in every item and left out)",
        },
        "n_compared": {"heldout_draft": sum(s == "heldout" for s, _, _ in draft),
                       "bridge_draft": sum(s == "bridge" for s, _, _ in draft),
                       **dict(sources)},
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (HELDOUT, BRIDGE)},
        "positive_controls": control_rows,
        "pairs_above_threshold": pairs,
        "per_draft_item_max": per_item,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(report), encoding="utf-8")
    print("%d pairs above threshold; embedding %s" % (len(pairs), emb_note))


def markdown(report):
    m = report["method"]
    lines = [
        "# Near-duplicate report (checkpoint 1 drafts)",
        "",
        "Generated by `scripts/questions/near_duplicates.py`; the JSON beside this file has "
        "every number. Drafts compared: `heldout_draft.json` and `bridge_draft.json` (sha256 "
        "in `SHA256SUMS`).",
        "",
        "## Method",
        "",
        "| measure | details | threshold | status |",
        "|---|---|---|---|",
        "| TF-IDF char n-gram cosine | %d-%d-grams, %s; idf %s | >= %.2f | run |" % (
            m["char_tfidf"]["ngram_range"][0], m["char_tfidf"]["ngram_range"][1],
            m["char_tfidf"]["analyzer"], m["char_tfidf"]["idf"], m["char_tfidf"]["threshold"]),
        "| sentence-embedding cosine | `%s` @ `%s`, %s | >= %.2f | %s |" % (
            m["embedding"]["model"], m["embedding"]["revision"][:12], m["embedding"]["pooling"],
            m["embedding"]["threshold"], m["embedding"]["status"]),
        "",
        "Game prompts: %s. Bridge items: %s." % (m["game_prompt_text"], m["bridge_text"]),
        "",
        "Compared against: " + ", ".join("%s = %d" % kv for kv in report["n_compared"].items())
        + ".",
        "",
        "## Pairs at or above either threshold",
        "",
    ]
    pairs = report["pairs_above_threshold"]
    if not pairs:
        lines.append("**None.** No draft item reaches either threshold against any reference "
                     "text or any other draft item.")
    else:
        lines += ["| draft | other | char | embed | draft text | other text |",
                  "|---|---|---|---|---|---|"]
        for p in sorted(pairs, key=lambda p: -(p["embedding_cosine"] or p["char_tfidf_cosine"])):
            lines.append("| %s | %s | %.3f | %s | %s | %s |" % (
                p["draft_id"], p["other_id"], p["char_tfidf_cosine"],
                "-" if p["embedding_cosine"] is None else "%.3f" % p["embedding_cosine"],
                p["draft_text"].replace("|", "/")[:140], p["other_text"].replace("|", "/")
                .replace("\n", " ")[:140]))
    lines += ["", "## Positive controls", "",
              "Paraphrases of reference questions written for this check, scored like a draft "
              "item. They show what each threshold does catch.", "",
              "| reference | paraphrase | char | caught | embed | caught |",
              "|---|---|---|---|---|---|"]
    for c in report["positive_controls"]:
        lines.append("| %s | %s | %.3f | %s | %s | %s |" % (
            c["reference"], c["paraphrase"], c["char_tfidf_cosine"],
            "yes" if c["caught_char"] else "no",
            "-" if c["embedding_cosine"] is None else "%.3f" % c["embedding_cosine"],
            "-" if c["caught_embedding"] is None else ("yes" if c["caught_embedding"] else "no")))
    lines += ["", "## Highest similarity per draft item", "",
              "Nearest text of any kind (references and the other drafts), and separately the "
              "nearest reference text (altruistic.json or a game prompt) by embedding.", "",
              "| draft item | max char | nearest (char) | max embed | nearest (embed) "
              "| max embed vs references | nearest reference |",
              "|---|---|---|---|---|---|---|"]
    for item, v in report["per_draft_item_max"].items():
        lines.append("| %s | %.3f | %s | %s | %s | %s | %s |" % (
            item, v["max_char_tfidf_cosine"], v["nearest_char"],
            "-" if v["max_embedding_cosine"] is None else "%.3f" % v["max_embedding_cosine"],
            v["nearest_embedding"],
            "-" if v["max_embedding_vs_references"] is None else "%.3f" % v["max_embedding_vs_references"],
            v["nearest_reference_embedding"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
