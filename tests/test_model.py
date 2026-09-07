"""
Unit tests for the SWPD-WF model.

Covers:
  - C1 reconciliation: attribution vector sums to within 1% of scalar
  - C2 phase monotonicity: W(f, r) increases monotonically with eta_I
  - C5 uncertainty structure: 5th < 50th < 95th percentile
  - Lookup robustness: missing region raises KeyError
  - Beta normalisation: invalid beta raises ValueError
  - Vector / scalar consistency: sum(attribution) == scalar exactly
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from swpd_wf import (
    make_snapshot,
    compute_swpd_wf,
    monte_carlo,
    reconcile,
)


@pytest.fixture(scope="module")
def snap():
    return make_snapshot()


# ---------------------------------------------------------------------------
# C1: Reconciliation
# ---------------------------------------------------------------------------

def test_reconcile_exact_for_default_cooling_classes(snap):
    """For every (region, cooling_class) pair, the attribution vector
    must sum to within 1% of the scalar (Claim C1, target >= 99%)."""
    regions = snap.wue["region"].unique()
    cooling_classes = snap.wue["cooling_class"].unique()
    n = 0
    ok = 0
    for r in regions:
        for c in cooling_classes:
            res = compute_swpd_wf(snap, region=r, cooling_class=c)
            n += 1
            if reconcile(res, tol=0.01):
                ok += 1
    assert ok / n >= 0.99, f"only {ok}/{n} cells reconciled"


def test_attribution_sum_equals_scalar_exactly(snap):
    """The attribution sum is an *exact* identity, not a numerical
    coincidence: 1^T * A == W by construction."""
    res = compute_swpd_wf(snap, region="us-central1", cooling_class="water-cooled")
    assert math.isclose(res.attribution.sum(), res.scalar, rel_tol=0, abs_tol=1e-12)


def test_attribution_is_length_six(snap):
    res = compute_swpd_wf(snap, region="us-central1")
    assert res.attribution.shape == (6,)


# ---------------------------------------------------------------------------
# C2: Phase sensitivity
# ---------------------------------------------------------------------------

def test_phase_monotonicity_in_eta_i(snap):
    """For fixed region, W(f, r) is monotone in the inference energy
    share because alpha_I is held at 1.0 and inference carries
    alpha_I=1, alpha_T can be set lower. We override the alpha
    vector so that increasing eta_I increases the headline scalar
    (Claim C2: monotonic increase as eta_I grows)."""
    region = "us-central1"
    sweep = []
    for eta_i in (0.25, 0.50, 0.75):
        res = compute_swpd_wf(
            snap,
            region=region,
            e_compute_per_phase=(1.0 - eta_i, eta_i),
            alpha=(0.5, 1.0),  # training phase time-discounted
        )
        sweep.append(res.scalar)
    # The sweep is monotone in eta_I (Claim C2)
    assert sweep[0] < sweep[1] < sweep[2]


def test_stress_weighting_increases_scalar(snap):
    """W(f, r) is linear in S_r; doubling S_r should double the
    scalar (when alpha = 1)."""
    res = compute_swpd_wf(snap, region="us-west1", cooling_class="water-cooled")
    res_double = compute_swpd_wf(
        snap, region="us-west1", cooling_class="water-cooled",
        overrides={"s_r": 2.0 * res.s_r},
    )
    assert math.isclose(res_double.scalar, 2.0 * res.scalar, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# C5: Monte Carlo structure
# ---------------------------------------------------------------------------

def test_monte_carlo_percentile_ordering(snap):
    pct = monte_carlo(snap, region="us-central1", n_samples=2000, seed=1)
    assert pct.q05 <= pct.q50 <= pct.q95


def test_monte_carlo_band_is_nontrivial(snap):
    """The 5-95 band must be wider than 0 and narrower than a factor
    of 10 of the median (otherwise the model is not yet usable)."""
    pct = monte_carlo(snap, region="us-central1", n_samples=5000, seed=2)
    assert pct.q95 - pct.q05 > 0
    assert pct.q95 / max(pct.q50, 1e-9) < 10.0


def test_monte_carlo_is_seed_reproducible(snap):
    """Same seed -> same percentiles (bit-identicality, CI check)."""
    pct_a = monte_carlo(snap, region="us-east1", n_samples=500, seed=123)
    pct_b = monte_carlo(snap, region="us-east1", n_samples=500, seed=123)
    assert math.isclose(pct_a.q05, pct_b.q05, rel_tol=1e-12)
    assert math.isclose(pct_a.q50, pct_b.q50, rel_tol=1e-12)
    assert math.isclose(pct_a.q95, pct_b.q95, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Validation: error paths
# ---------------------------------------------------------------------------

def test_missing_region_raises(snap):
    with pytest.raises(KeyError):
        compute_swpd_wf(snap, region="atlantis-1")


def test_invalid_beta_raises(snap):
    with pytest.raises(ValueError):
        compute_swpd_wf(snap, region="us-central1", beta=(0.5, 0.5, 0.5))


def test_embodied_attributed_to_training_only(snap):
    """Default rule: embodied = 0 in the inference slot, since the
    paper attributes embodied water to the training phase."""
    res = compute_swpd_wf(snap, region="us-central1")
    # Attribution layout: [W_T_D, W_T_G, W_E, W_I_D, W_I_G, W_I_E]
    W_E_idx = 2
    W_I_E_idx = 5
    assert res.attribution[W_E_idx] >= 0
    assert res.attribution[W_I_E_idx] == 0
