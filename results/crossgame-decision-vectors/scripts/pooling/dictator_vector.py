"""Rebuild the Dictator-only vector from activations, and check it against the
committed one. CPU only.

Section 6 of `README.md` projects this direction out of the pooled vector, and
section 7 cosines against it, so it has to come off the same extractor as
everything it is compared with. It was fit on a separate, earlier Dictator-only
generation run - no row of the cross-game grid entered it - which is what makes
projecting it out meaningful rather than circular.

The construction is the one `results/dictator-decision-vector/scripts/analyze.py`
used, restated here rather than imported because that directory is a published
artefact and is not modified by this work:

  cell        one (wording, endowment); usable when it holds rows in both poles
  poles       self = transferred exactly 0, altruistic = at least half the
              endowment, labelled by FRACTION because the endowment varies 50x
  tags        direct-read paths only. `complement` and `keep` subtract a read
              amount from the pot, which lands a row exactly on the pole boundary
              in the WRONG pole; they are counted and excluded, never silently used
  vector      unweighted mean over usable cells of alt-mean minus self-mean,
              float64, saved float32

Writes `dictator_vector.json` with the per-layer cosine against the committed
vector. Agreement there is what says the whole corpus is comparable; a
disagreement is a finding, not something to absorb.
"""

import csv
import json
from pathlib import Path

import torch

import common

#: value read straight off the response; see the module docstring on the rest
DIRECT_TAGS = ("a2_anchor", "a2_near", "verb_obj", "answer_is", "bare", "bare_int")
FAMILY = "dictator_only"


def load(acts_dir):
    """(N, 29, 3584) response_avg and one label per captured row, in shard order."""
    game_dir = Path(acts_dir)
    with open(game_dir / "rows.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    shards = sorted(game_dir.glob("shard_*.pt"))
    if not shards:
        raise SystemExit("no activation shards in %s" % game_dir)
    index, chunks = [], []
    for path in shards:
        payload = torch.load(path, map_location="cpu")
        index.extend(payload["row_index"].tolist())
        chunks.append(payload["response_avg"].clone())
        del payload
    acts = torch.cat(chunks, dim=0)
    del chunks
    if len(index) != acts.shape[0]:
        raise SystemExit("%s: shard row count disagrees with activations" % game_dir)

    labels = []
    for position, row_index in enumerate(index):
        row = rows[row_index]
        wording, endowment = row["game_id"].split("/")[1], row["game_id"].split("/e")[1]
        labels.append({"position": position, "tag": row["tag"],
                       "cell": "%s/e%s" % (wording, endowment),
                       "fraction": float(row["value"]) / int(endowment)})
    return acts, labels


def usable_cells(labels, known_tags):
    """({cell: (alt, self)} for cells holding BOTH poles, census).

    A row that resolved along a derived path is EXCLUDED and counted, never
    silently used: `complement` and `keep` subtract a read amount from the pot,
    which lands a row on the pole boundary in the wrong pole. A tag this script
    does not recognise at all is an error rather than a quiet exclusion - a
    renamed tag would otherwise shrink the fit set without saying so.
    """
    seen, census = {}, {"n_tag_excluded": 0, "n_middle_discarded": 0}
    unknown = sorted({item["tag"] for item in labels} - known_tags)
    if unknown:
        raise SystemExit("unrecognised parser tags %s; the pole rules do not cover them"
                         % unknown)
    for item in labels:
        if item["tag"] not in DIRECT_TAGS:
            census["n_tag_excluded"] += 1
            continue
        fraction = item["fraction"]
        if fraction >= 0.5:
            seen.setdefault(item["cell"], ([], []))[0].append(item["position"])
        elif fraction == 0.0:
            seen.setdefault(item["cell"], ([], []))[1].append(item["position"])
        else:
            census["n_middle_discarded"] += 1
    census["n_cells_seen"] = len(seen)
    return ({cell: pair for cell, pair in sorted(seen.items()) if pair[0] and pair[1]},
            census)


def main():
    acts, labels = load(common.ACTS / FAMILY)
    known = set(DIRECT_TAGS) | set(common.poles.DERIVED_TAGS["amount"]) \
        | set(common.poles.UNRESOLVED_TAGS)
    cells, census = usable_cells(labels, known)
    if not cells:
        raise SystemExit("no cell has both poles")
    vector = torch.stack([common.mean_of(acts, alt) - common.mean_of(acts, self_)
                          for _c, (alt, self_) in sorted(cells.items())]).mean(dim=0)

    committed = torch.load(common.DICTATOR_VECTOR, map_location="cpu")
    cos = common.cosines(vector, committed)
    identical = bool(torch.equal(vector.float(), committed.float()))
    n_alt = sum(len(a) for a, _s in cells.values())
    n_self = sum(len(s) for _a, s in cells.values())
    report = {
        "acts_dir": str((common.ACTS / FAMILY).resolve()),
        "committed_vector": str(common.DICTATOR_VECTOR),
        "n_rows_captured": acts.shape[0],
        "n_rows_in_the_vector": n_alt + n_self,
        "n_tag_excluded": census["n_tag_excluded"],
        "n_middle_discarded": census["n_middle_discarded"],
        "n_cells_seen": census["n_cells_seen"],
        "usable_cells": len(cells),
        "n_altruistic": n_alt,
        "n_self_interested": n_self,
        "bit_identical_to_committed": identical,
        "cos_vs_committed_layer0": cos[0].item(),
        "cos_vs_committed_layer20": cos[20].item(),
        "cos_vs_committed_min_over_layers": cos.min().item(),
        "max_abs_elementwise_diff": float((vector.float() - committed.float())
                                          .abs().max()),
    }
    common.OUT.mkdir(parents=True, exist_ok=True)
    (common.OUT / "vectors").mkdir(parents=True, exist_ok=True)
    torch.save(vector.float(), common.OUT / "vectors"
               / "decision_dictator_only_response_avg_diff_cellbalanced.pt")
    (common.OUT / "dictator_vector.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    print("rebuilt from %d pole rows (%d alt / %d self) over %d of %d cells; %d rows "
          "tag-excluded, %d middle. Bit-identical to the committed vector: %s "
          "(min cos %.9f)"
          % (report["n_rows_in_the_vector"], n_alt, n_self, report["usable_cells"],
             census["n_cells_seen"], census["n_tag_excluded"],
             census["n_middle_discarded"], identical,
             report["cos_vs_committed_min_over_layers"]), flush=True)


if __name__ == "__main__":
    main()
