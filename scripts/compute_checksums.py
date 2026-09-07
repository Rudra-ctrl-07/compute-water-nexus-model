"""
Compute SHA-256 of the frozen snapshot and write to data/CHECKSUMS.txt.

The CI workflow re-runs this script and compares the output to
data/CHECKSUMS.txt; any mismatch is a hard fail. This is the
operational form of "reproducible from public data" (Claim C4).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swpd_wf import make_snapshot, write_checksums  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    snap = make_snapshot()
    write_checksums(snap, DATA_DIR / "CHECKSUMS.txt")
    print(f"Wrote {DATA_DIR / 'CHECKSUMS.txt'}")
    # Also dump the CSVs for downstream tooling
    snap.wue.to_csv(DATA_DIR / "wue.csv", index=False)
    snap.egrid.to_csv(DATA_DIR / "egrid.csv", index=False)
    snap.aqueduct.to_csv(DATA_DIR / "aqueduct.csv", index=False)
    snap.embodied.to_csv(DATA_DIR / "embodied.csv", index=False)
    snap.workloads.to_csv(DATA_DIR / "workloads.csv", index=False)
    print("Wrote CSVs to data/")


if __name__ == "__main__":
    main()
