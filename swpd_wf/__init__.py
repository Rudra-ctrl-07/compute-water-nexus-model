"""
SWPD-WF: Stress-Weighted, Phase-Disaggregated Water-Footprint model
for hyperscale AI compute.

Public API:

  from swpd_wf import make_snapshot, compute_swpd_wf, monte_carlo, reconcile
"""

from .data import (
    Snapshot,
    make_snapshot,
    write_checksums,
    load_snapshot,
    WUEEntry,
    EGRIDEntry,
    AqueductEntry,
    EmbodiedEntry,
    WorkloadEntry,
)
from .model import (
    SWPDResult,
    Percentiles,
    compute_swpd_wf,
    monte_carlo,
    reconcile,
    PHASES,
    FLOWS,
    ATTR_LEN,
)

__all__ = [
    "Snapshot",
    "make_snapshot",
    "write_checksums",
    "load_snapshot",
    "WUEEntry",
    "EGRIDEntry",
    "AqueductEntry",
    "EmbodiedEntry",
    "WorkloadEntry",
    "SWPDResult",
    "Percentiles",
    "compute_swpd_wf",
    "monte_carlo",
    "reconcile",
    "PHASES",
    "FLOWS",
    "ATTR_LEN",
]

__version__ = "0.1.0"
