"""Run-quality numbers of a judging output directory, and nothing else.

    python run_quality.py judgments/full/llama-3.3-70b-instruct-q8_0

Prints calls done, status counts, the `low_mass` share, the `number_mass` distribution (all
calls pooled), the determinism re-send match rate and the time span. It never prints a score,
by any grouping: judge scores stay blind until the human labels exist (PREREG 9.1)."""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(path):
    path = Path(path)
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def main(directory):
    d = Path(directory)
    out = {}
    for kind in ("trait",):
        rows = {r["key"]: r for r in load(d / ("%s.jsonl" % kind))}.values()
        rows = list(rows)
        masses = sorted(r["detail"]["number_mass"] for r in rows if r.get("detail") and "number_mass" in r["detail"])
        n = len(rows)
        q = (lambda p: masses[min(len(masses) - 1, int(p * len(masses)))]) if masses else (lambda p: None)
        out[kind] = {
            "calls": n, "status": dict(Counter(r["status"] for r in rows)),
            "low_mass_share": (sum(r["status"] == "low_mass" for r in rows) / n) if n else None,
            "number_mass": {"n": len(masses), "min": masses[0] if masses else None, "p05": q(.05), "median": q(.5),
                            "mean": statistics.fmean(masses) if masses else None,
                            "share_below_0.9": (sum(m < .9 for m in masses) / len(masses)) if masses else None,
                            "share_below_0.5": (sum(m < .5 for m in masses) / len(masses)) if masses else None},
            "first": min((r["timestamp"] for r in rows), default=None),
            "last": max((r["timestamp"] for r in rows), default=None),
        }
    rs = load(d / "determinism_resend.jsonl")
    out["resend"] = {"n": len(rs), "exact_match": sum(r["exact_match"] for r in rs),
                     "same_status": sum(r["same_status"] for r in rs),
                     "exact_match_rate": (sum(r["exact_match"] for r in rs) / len(rs)) if rs else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
