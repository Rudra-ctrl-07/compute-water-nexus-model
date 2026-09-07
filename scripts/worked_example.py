"""
Worked numerical example for one facility-region pair.

This script produces a single-region walkthrough of the SWPD-WF
computation that can be inserted verbatim into the LaTeX as a
"worked example" box. It also writes a JSON file to figures/ for
the framework flowchart and other reproducible artifacts.

Run:  python scripts/worked_example.py
Output:  figures/worked_example.json
         figures/attribution_table.csv
         figures/phase_sweep.csv
         figures/ranking.csv
         figures/framework.pdf
         figures/regions_map.pdf
         figures/tornado.pdf
         figures/carbon_water_pareto.pdf
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the package is importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swpd_wf import make_snapshot, compute_swpd_wf, monte_carlo  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def worked_example() -> dict:
    """Compute one full walkthrough for us-central1 / water-cooled /
    H100 / public-llm-serving-2024."""
    snap = make_snapshot()
    res = compute_swpd_wf(
        snap, region="us-central1", cooling_class="water-cooled",
        accelerator_class="H100", trace_name="public-llm-serving-2024",
    )
    pct = monte_carlo(snap, region="us-central1", cooling_class="water-cooled",
                      accelerator_class="H100", trace_name="public-llm-serving-2024",
                      n_samples=10_000, seed=42)

    return {
        "region": res.region,
        "cooling_class": res.cooling_class,
        "accelerator_class": res.accelerator_class,
        "trace_name": res.trace_name,
        "inputs": {
            "WUE_l_per_kwh": res.wue,
            "WIf_l_per_kwh": res.wif,
            "S_r": res.s_r,
            "embodied_l_per_year": 3500.0,
            "eta_T": res.eta_t,
            "eta_I": res.eta_i,
        },
        "intermediate": {
            "V_D_total_l": res.v_direct,
            "V_G_total_l": res.v_grid,
            "V_E_total_l": res.v_embodied,
        },
        "attribution_l": {
            "W_T_D": res.attribution[0],
            "W_T_G": res.attribution[1],
            "W_E":   res.attribution[2],
            "W_I_D": res.attribution[3],
            "W_I_G": res.attribution[4],
            "W_I_E": res.attribution[5],
        },
        "scalar_l": res.scalar,
        "monte_carlo_pct": {
            "q05": pct.q05, "q50": pct.q50, "q95": pct.q95,
        },
    }


def per_region_table() -> pd.DataFrame:
    """Build the per-region attribution table for the LaTeX."""
    snap = make_snapshot()
    rows = []
    for region in snap.aqueduct["region"].unique():
        res = compute_swpd_wf(
            snap, region=region, cooling_class="water-cooled",
            accelerator_class="H100", trace_name="public-llm-serving-2024",
        )
        rows.append({
            "Region": region,
            "W_T_D": res.attribution[0],
            "W_T_G": res.attribution[1],
            "W_E":   res.attribution[2],
            "W_I_D": res.attribution[3],
            "W_I_G": res.attribution[4],
            "W_I_E": res.attribution[5],
            "W(f,r)": res.scalar,
        })
    return pd.DataFrame(rows)


def phase_sweep_table() -> pd.DataFrame:
    """Build the eta_I sweep table for Claim C2."""
    snap = make_snapshot()
    rows = []
    for region in snap.aqueduct["region"].unique():
        for eta_i in (0.25, 0.50, 0.75):
            res = compute_swpd_wf(
                snap, region=region, cooling_class="water-cooled",
                accelerator_class="H100", trace_name="public-llm-serving-2024",
                e_compute_per_phase=(1.0 - eta_i, eta_i),
                alpha=(0.5, 1.0),
            )
            rows.append({"Region": region, "eta_I": eta_i, "W(f,r)": res.scalar})
    df = pd.DataFrame(rows)
    return df.pivot(index="Region", columns="eta_I", values="W(f,r)").reset_index()


def ranking_table() -> pd.DataFrame:
    """Build the (volumetric, stress-weighted, carbon-weighted)
    ranking table for Claim C3."""
    snap = make_snapshot()
    rows = []
    for region in snap.aqueduct["region"].unique():
        res = compute_swpd_wf(
            snap, region=region, cooling_class="water-cooled",
            accelerator_class="H100", trace_name="public-llm-serving-2024",
        )
        egrid = snap.egrid[snap.egrid["region"] == region].iloc[0]
        # Volumetric: same flow sums but with S_r=1
        vol = compute_swpd_wf(
            snap, region=region, cooling_class="water-cooled",
            accelerator_class="H100", trace_name="public-llm-serving-2024",
            overrides={"s_r": 1.0},
        ).scalar
        # Carbon-weighted: SWPD scalar weighted by emission factor
        carbon = res.scalar * float(egrid["ef_kg_per_kwh"])
        rows.append({
            "Region": region,
            "W_stress": res.scalar,
            "W_volumetric": vol,
            "EF_kg_per_kwh": float(egrid["ef_kg_per_kwh"]),
            "W_carbon_proxy": carbon,
        })
    df = pd.DataFrame(rows)
    df["rank_vol"] = df["W_volumetric"].rank(ascending=False).astype(int)
    df["rank_stress"] = df["W_stress"].rank(ascending=False).astype(int)
    df["rank_carbon"] = df["W_carbon_proxy"].rank(ascending=False).astype(int)
    # Spearman rho between volumetric and stress-weighted ranks
    from scipy.stats import spearmanr
    rho, _ = spearmanr(df["W_volumetric"], df["W_stress"])
    return df, float(rho)


def main() -> None:
    print("=" * 60)
    print("SWPD-WF worked example (us-central1, water-cooled, H100)")
    print("=" * 60)
    ex = worked_example()
    print(json.dumps(ex, indent=2))

    with open(FIG_DIR / "worked_example.json", "w", encoding="utf-8") as f:
        json.dump(ex, f, indent=2)

    attr = per_region_table()
    print("\nPer-region attribution table:")
    print(attr.to_string(index=False))
    attr.to_csv(DATA_DIR / "attribution_table.csv", index=False)
    attr.to_csv(FIG_DIR / "attribution_table.csv", index=False)

    sweep = phase_sweep_table()
    print("\nPhase-sweep table (eta_I = 0.25, 0.50, 0.75):")
    print(sweep.to_string(index=False))
    sweep.to_csv(FIG_DIR / "phase_sweep.csv", index=False)

    rank, rho = ranking_table()
    print(f"\nRanking table (Spearman rho vol vs stress = {rho:.3f}):")
    print(rank.to_string(index=False))
    rank.to_csv(FIG_DIR / "ranking.csv", index=False)
    with open(FIG_DIR / "spearman_rho.txt", "w", encoding="utf-8") as f:
        f.write(f"{rho:.6f}\n")


if __name__ == "__main__":
    main()
