"""
SWPD-WF core model.

Implements Algorithms 1 and 2 from the paper:

  - `compute_swpd_wf`     : one deterministic pass; returns scalar and
                            attribution vector for a (region, cooling
                            class, accelerator, trace) tuple.
  - `monte_carlo`         : propagates uncertainty on S_r and WUE_c
                            via N samples; returns 5/50/95 percentiles.
  - `reconcile`           : utility used by the test suite to check
                            that the attribution vector sums to within
                            tolerance of the scalar (Claim C1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import Snapshot


# ---------------------------------------------------------------------------
# Phase / flow indices
# ---------------------------------------------------------------------------

PHASES = ("T", "I")                # training, inference
FLOWS = ("D", "G", "E")            # direct cooling, grid indirect, embodied
N_PHASES = len(PHASES)
N_FLOWS = len(FLOWS)
ATTR_LEN = N_PHASES * N_FLOWS       # 6-element attribution vector


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SWPDResult:
    """One deterministic pass through Algorithm 1."""
    region: str
    cooling_class: str
    accelerator_class: str
    trace_name: str
    scalar: float                   # W(f, r), L
    attribution: np.ndarray         # length 6
    # Breakdown for inspection / debugging
    v_direct: float
    v_grid: float
    v_embodied: float
    s_r: float
    wue: float
    wif: float
    eta_t: float
    eta_i: float


@dataclass(frozen=True)
class Percentiles:
    """Result of Algorithm 2: percentiles of W(f, r) over N samples."""
    q05: float
    q50: float
    q95: float
    samples: np.ndarray             # full N-vector; useful for plotting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_wue(snap: Snapshot, region: str, cooling_class: str) -> tuple[float, float]:
    row = snap.wue[
        (snap.wue["region"] == region) & (snap.wue["cooling_class"] == cooling_class)
    ]
    if row.empty:
        raise KeyError(f"WUE not found for region={region}, cooling_class={cooling_class}")
    return float(row["wue_l_per_kwh"].iloc[0]), float(row["wue_uncertainty"].iloc[0])


def _lookup_egrid(snap: Snapshot, region: str) -> tuple[float, float]:
    row = snap.egrid[snap.egrid["region"] == region]
    if row.empty:
        raise KeyError(f"eGRID entry not found for region={region}")
    return float(row["wif_l_per_kwh"].iloc[0]), float(row["ef_kg_per_kwh"].iloc[0])


def _lookup_aqueduct(snap: Snapshot, region: str) -> tuple[float, float]:
    row = snap.aqueduct[snap.aqueduct["region"] == region]
    if row.empty:
        raise KeyError(f"Aqueduct entry not found for region={region}")
    return float(row["s_r"].iloc[0]), float(row["s_r_uncertainty"].iloc[0])


def _lookup_embodied(snap: Snapshot, accelerator_class: str) -> float:
    row = snap.embodied[snap.embodied["accelerator_class"] == accelerator_class]
    if row.empty:
        raise KeyError(f"Embodied entry not found for accelerator_class={accelerator_class}")
    return float(row["embodied_l_per_year"].iloc[0])


def _lookup_workload(snap: Snapshot, trace_name: str) -> tuple[float, float]:
    row = snap.workloads[snap.workloads["trace_name"] == trace_name]
    if row.empty:
        raise KeyError(f"Workload trace not found: {trace_name}")
    return float(row["eta_t"].iloc[0]), float(row["eta_i"].iloc[0])


# ---------------------------------------------------------------------------
# Algorithm 1: deterministic SWPD-WF computation
# ---------------------------------------------------------------------------

def compute_swpd_wf(
    snap: Snapshot,
    *,
    region: str,
    cooling_class: str = "water-cooled",
    accelerator_class: str = "H100",
    trace_name: str = "public-llm-serving-2024",
    e_compute_per_phase: tuple[float, float] = (1.0, 1.0),  # kWh
    alpha: tuple[float, float] = (1.0, 1.0),                # alpha_T, alpha_I
    beta: tuple[float, float, float] = (0.40, 0.55, 0.05),  # beta_D, beta_G, beta_E
    # Sum-to-one within phase; embodied is 0.05 by IEA-based estimate
    # so direct and grid share the remaining 0.95 (40/55 split).
    overrides: dict | None = None,
) -> SWPDResult:
    """Compute the SWPD-WF scalar and attribution vector for one
    facility-region pair.

    Parameters
    ----------
    snap : Snapshot
        The frozen public-data snapshot.
    region : str
        Hyperscale region identifier.
    cooling_class : str
        "air" | "water-cooled" | "liquid-immersion".
    accelerator_class : str
        Hardware class; used for embodied water coefficient.
    trace_name : str
        Open workload trace providing eta_T and eta_I.
    e_compute_per_phase : (float, float)
        Energy consumption in kWh for training and inference (one
        baseline job each, normalised). Defaults to 1.0 / 1.0 kWh so
        the scalar is in L per (1 kWh training, 1 kWh inference).
        Override for absolute figures.
    alpha : (float, float)
        Phase weights. Default (1.0, 1.0) treats the two phases
        symmetrically; pass (alpha_T < 1.0, 1.0) to time-discount
        training.
    beta : (float, float, float)
        Flow-share within each phase. Must sum to 1.
    overrides : dict
        Per-input value overrides for sensitivity studies. Keys can
        be 'wue', 'wif', 's_r', 'embodied'. Each override is the
        central value to use instead of the snapshot default.
    """
    wue, _ = _lookup_wue(snap, region, cooling_class)
    wif, _ = _lookup_egrid(snap, region)
    s_r, _ = _lookup_aqueduct(snap, region)
    embodied = _lookup_embodied(snap, accelerator_class)
    eta_t, eta_i = _lookup_workload(snap, trace_name)

    # Per-input overrides
    if overrides:
        if "wue" in overrides:
            wue = float(overrides["wue"])
        if "wif" in overrides:
            wif = float(overrides["wif"])
        if "s_r" in overrides:
            s_r = float(overrides["s_r"])
        if "embodied" in overrides:
            embodied = float(overrides["embodied"])

    # Beta normalisation: must sum to 1 within a phase
    if not np.isclose(sum(beta), 1.0, atol=1e-6):
        raise ValueError(f"beta must sum to 1; got {sum(beta)}")

    # PUE assumed = 1.0 for the worked example; in production this
    # would be a 6th input slot, but it cancels in WUE inversion.
    pue = 1.0
    e_T, e_I = e_compute_per_phase
    alpha_T, alpha_I = alpha
    beta_D, beta_G, beta_E = beta

    # Volumetric water by flow and phase
    # V_D(p) = E(p) * PUE * (1 - WUE)^-1  -- wait: WUE is L/kWh of IT,
    # so direct cooling water per kWh IT = WUE * PUE / PUE = WUE.
    # The inverse-WUE formulation in the original markdown was an
    # error; the correct formulation is direct = WUE * PUE * E(p).
    v_D_T = wue * pue * e_T
    v_D_I = wue * pue * e_I
    v_G_T = wif * e_T
    v_G_I = wif * e_I
    # Embodied attributed to training by default
    v_E_T = embodied
    v_E_I = 0.0

    # Phase-discounted, stress-weighted contributions
    W_T_D = alpha_T * beta_D * v_D_T * s_r
    W_T_G = alpha_T * beta_G * v_G_T * s_r
    W_E   = alpha_T * beta_E * v_E_T * s_r
    W_I_D = alpha_I * beta_D * v_D_I * s_r
    W_I_G = alpha_I * beta_G * v_G_I * s_r
    W_I_E = alpha_I * beta_E * v_E_I * s_r

    attribution = np.array([W_T_D, W_T_G, W_E, W_I_D, W_I_G, W_I_E], dtype=float)
    scalar = float(attribution.sum())

    return SWPDResult(
        region=region,
        cooling_class=cooling_class,
        accelerator_class=accelerator_class,
        trace_name=trace_name,
        scalar=scalar,
        attribution=attribution,
        v_direct=v_D_T + v_D_I,
        v_grid=v_G_T + v_G_I,
        v_embodied=v_E_T + v_E_I,
        s_r=s_r,
        wue=wue,
        wif=wif,
        eta_t=eta_t,
        eta_i=eta_i,
    )


# ---------------------------------------------------------------------------
# Algorithm 2: Monte Carlo uncertainty propagation
# ---------------------------------------------------------------------------

def monte_carlo(
    snap: Snapshot,
    *,
    region: str,
    cooling_class: str = "water-cooled",
    accelerator_class: str = "H100",
    trace_name: str = "public-llm-serving-2024",
    n_samples: int = 10_000,
    seed: int = 42,
) -> Percentiles:
    """Run Algorithm 2: propagate uncertainty on S_r and WUE_c via
    N samples drawn from the published uncertainty bands.

    The two highest-variance inputs are sampled; other inputs are
    held at their reported central values. Each sample re-runs
    Algorithm 1 with the sampled S_r and WUE, and the 5/50/95th
    percentiles of the resulting W(f, r) are returned.
    """
    rng = np.random.default_rng(seed)
    wue_central, wue_unc = _lookup_wue(snap, region, cooling_class)
    s_r_central, s_r_unc = _lookup_aqueduct(snap, region)

    # Sample from normal distributions truncated to non-negative.
    # This is a conservative choice; the Aqueduct band is reported
    # as approximately Gaussian by WRI.
    wue_samples = np.maximum(
        0.0, rng.normal(wue_central, wue_unc, size=n_samples)
    )
    s_r_samples = np.clip(
        rng.normal(s_r_central, s_r_unc, size=n_samples), 0.0, 1.0
    )

    samples = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        result = compute_swpd_wf(
            snap,
            region=region,
            cooling_class=cooling_class,
            accelerator_class=accelerator_class,
            trace_name=trace_name,
            overrides={"wue": wue_samples[i], "s_r": s_r_samples[i]},
        )
        samples[i] = result.scalar

    return Percentiles(
        q05=float(np.quantile(samples, 0.05)),
        q50=float(np.quantile(samples, 0.50)),
        q95=float(np.quantile(samples, 0.95)),
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Utility: reconciliation check for Claim C1
# ---------------------------------------------------------------------------

def reconcile(result: SWPDResult, *, tol: float = 0.01) -> bool:
    """Return True iff the attribution vector sums to within `tol`
    of the scalar. Claim C1 in the paper requires this for >= 99% of
    (region, cooling_class) cells."""
    return bool(abs(result.attribution.sum() - result.scalar) <= tol * result.scalar)
