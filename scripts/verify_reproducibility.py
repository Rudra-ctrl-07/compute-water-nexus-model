"""
Verify that the headline numbers in the worked example are
bit-identical to a known reference. This is the CI check that the
paper's claims are reproducible from the public data alone
(Claim C4).

Reference values are stored in data/REFERENCE_NUMBERS.json. Any
mismatch within numerical tolerance triggers a hard failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swpd_wf import make_snapshot, compute_swpd_wf  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    snap = make_snapshot()
    res = compute_swpd_wf(
        snap, region="us-central1", cooling_class="water-cooled",
        accelerator_class="H100", trace_name="public-llm-serving-2024",
    )
    actual = {
        "scalar": res.scalar,
        "W_T_D": float(res.attribution[0]),
        "W_T_G": float(res.attribution[1]),
        "W_E":   float(res.attribution[2]),
        "W_I_D": float(res.attribution[3]),
        "W_I_G": float(res.attribution[4]),
        "W_I_E": float(res.attribution[5]),
    }
    with open(DATA_DIR / "REFERENCE_NUMBERS.json", encoding="utf-8") as f:
        ref = json.load(f)
    tol = 1e-9
    for k, v_ref in ref.items():
        v_actual = actual[k]
        if abs(v_actual - v_ref) > tol:
            print(f"MISMATCH on {k}: actual={v_actual} ref={v_ref}", file=sys.stderr)
            sys.exit(1)
    print("All headline numbers match the reference within tolerance.")


if __name__ == "__main__":
    main()
