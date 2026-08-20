"""The shuffled-label null with the real decision direction projected OUT.

Why this control is needed. The within-cell shuffled-label vector is not
orthogonal to the real one: cos = 0.2423 at layer 20, which is exactly the
anisotropy the extraction task measured (any difference-of-means over these rows
lands partly in the same ~8-dimensional subspace). Steered at unit beta 52.54 the
null therefore delivers 0.2423 * 52.54 = 12.73 unit-beta ALONG the real direction
- and the real arm reaches its full effect by unit beta 10.51. So the plain null
carries a supra-saturating dose of the real signal and cannot separate "the
decision direction did it" from "any direction did it".

Removing the real component fixes that: this vector is the shuffled-label
direction's residual after projecting out the real one, so it is exactly
orthogonal to the real direction at the steered layer and carries none of it.
"""
import argparse, hashlib, json
from pathlib import Path
import torch


def sha16(path):
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()[:16]


ap = argparse.ArgumentParser()
ap.add_argument("--null-vector", required=True)
ap.add_argument("--real-vector", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--layer", type=int, default=20)
args = ap.parse_args()

null = torch.load(args.null_vector, map_location="cpu").double()
real = torch.load(args.real_vector, map_location="cpu").double()
if null.shape != real.shape:
    raise SystemExit("shape mismatch %s vs %s" % (tuple(null.shape), tuple(real.shape)))

# per-layer Gram-Schmidt: residual of null after removing its real component
unit_real = real / real.norm(dim=1, keepdim=True).clamp_min(1e-12)
projection = (null * unit_real).sum(dim=1, keepdim=True) * unit_real
residual = null - projection

L = args.layer
def cos(a, b):
    return float((a[L] * b[L]).sum() / (a[L].norm() * b[L].norm()))

report = {
    "layer": L,
    "cos_null_vs_real": cos(null, real),
    "cos_residual_vs_real": cos(residual, real),
    "cos_residual_vs_null": cos(residual, null),
    "norm_null": float(null[L].norm()),
    "norm_real": float(real[L].norm()),
    "norm_residual": float(residual[L].norm()),
    "real_component_of_null_at_unit_beta_52_5415": cos(null, real) * 52.5415,
    "source_null": str(Path(args.null_vector).resolve()),
    "source_null_sha256_16": sha16(args.null_vector),
    "source_real_sha256_16": sha16(args.real_vector),
}
torch.save(residual.float(), args.out)
report["out"] = str(Path(args.out).resolve())
report["out_sha256_16"] = sha16(args.out)
# re-check after the float32 round-trip, since that is the file that gets steered
saved = torch.load(args.out, map_location="cpu").double()
report["cos_residual_vs_real_after_save"] = cos(saved, real)
Path(args.report).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
