"""
Generate all figures for the SWPD-WF paper.

Run:  python scripts/make_figures.py
Output (in figures/):
  - framework.pdf          : framework flowchart (Figure 1)
  - regions_map.pdf        : world map of S_r (Figure 2)
  - traces.pdf             : workload phase share distribution (Figure 3)
  - tornado.pdf            : sensitivity tornado (Figure 6 top)
  - carbon_water_pareto.pdf: Pareto scatter (Figure 6 bottom)
  - attribution_bars.pdf   : per-region attribution (Figure 7)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swpd_wf import make_snapshot, compute_swpd_wf, monte_carlo  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Figure 1: framework flowchart
# ---------------------------------------------------------------------------

def make_framework():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Three columns
    col_x = {"input": 1.5, "model": 5.0, "output": 8.5}
    box_w = 2.6
    box_h = 0.7

    def box(x, y, w, h, label, color="#e8eef5"):
        ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h,
                                   facecolor=color, edgecolor="#1f3a5f", linewidth=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5)

    # Column titles
    ax.text(col_x["input"], 4.5, "Inputs (public data)", ha="center", fontsize=11, fontweight="bold")
    ax.text(col_x["model"], 4.5, "Model", ha="center", fontsize=11, fontweight="bold")
    ax.text(col_x["output"], 4.5, "Outputs", ha="center", fontsize=11, fontweight="bold")

    # Inputs
    box(col_x["input"], 3.6, box_w, box_h, "Google Cloud WUE\n(per region, cooling class)")
    box(col_x["input"], 2.7, box_w, box_h, "EPA eGRID / ENE\n(sub-regional WIf)")
    box(col_x["input"], 1.8, box_w, box_h, "WRI Aqueduct 4.0\n(baseline water stress)")
    box(col_x["input"], 0.9, box_w, box_h, "IEA stock + datasheets\n(embodied water)")

    # Model
    box(col_x["model"], 2.5, box_w, 1.5, "Algorithm 1\nW(f, r) = sum alpha_p * beta_p,f' * V_p,f',r * S_r", color="#fff2cc")
    box(col_x["model"], 0.9, box_w, box_h, "Algorithm 2\nMonte Carlo over S_r and WUE")

    # Outputs
    box(col_x["output"], 3.6, box_w, box_h, "Scalar W(f, r)\n+ 6-element attribution", color="#d5e8d4")
    box(col_x["output"], 2.7, box_w, box_h, "Regional ranking\n(vol vs stress vs carbon)")
    box(col_x["output"], 1.8, box_w, box_h, "5/50/95 pct bands\n(sensitivity)")
    box(col_x["output"], 0.9, box_w, box_h, "Carbon-water\nPareto frontier", color="#d5e8d4")

    # Arrows
    for y_in in (3.6, 2.7, 1.8, 0.9):
        ax.annotate("", xy=(col_x["model"] - box_w/2, 2.5), xytext=(col_x["input"] + box_w/2, y_in),
                    arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=0.9))
    ax.annotate("", xy=(col_x["model"] - box_w/2, 0.9), xytext=(col_x["input"] + box_w/2, 0.9),
                arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=0.9))
    for y_out in (3.6, 2.7, 1.8, 0.9):
        ax.annotate("", xy=(col_x["output"] - box_w/2, y_out), xytext=(col_x["model"] + box_w/2, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=0.9))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "framework.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: regions map (schematic)
# ---------------------------------------------------------------------------

def make_regions_map():
    """A schematic map: longitude/latitude for each studied region
    with S_r as the colour and a small bar."""
    snap = make_snapshot()
    # Approximate centroids (lat, lon) for each region
    centroids = {
        "us-central1":  (41.0,  -95.0),
        "us-east1":     (37.0,  -79.0),
        "us-west1":     (37.0, -122.0),
        "europe-west1": (50.0,    4.0),
        "asia-south1":  (19.0,   72.0),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-130, 90)
    ax.set_ylim(-10, 70)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Studied hyperscale regions, coloured by S_r (WRI Aqueduct 4.0)")
    ax.grid(alpha=0.3)

    for region, (lat, lon) in centroids.items():
        s_r = float(snap.aqueduct[snap.aqueduct["region"] == region]["s_r"].iloc[0])
        ax.scatter(lon, lat, s=400, c=[s_r], cmap="RdYlBu_r", vmin=0, vmax=1,
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(f"{region}\n(S_r={s_r:.2f})", (lon, lat),
                    xytext=(8, 8), textcoords="offset points", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Baseline water stress S_r (0 = low, 1 = high)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "regions_map.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: workload phase share distribution
# ---------------------------------------------------------------------------

def make_traces():
    snap = make_snapshot()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    traces = snap.workloads["trace_name"].tolist()
    eta_t = snap.workloads["eta_t"].values
    eta_i = snap.workloads["eta_i"].values
    x = np.arange(len(traces))
    ax.bar(x, eta_t, label="Training (eta_T)", color="#7faedc")
    ax.bar(x, eta_i, bottom=eta_t, label="Inference (eta_I)", color="#f4a460")
    ax.set_xticks(x)
    ax.set_xticklabels(traces, rotation=20, ha="right")
    ax.set_ylabel("Phase share")
    ax.set_ylim(0, 1)
    ax.set_title("Training/inference phase share by open workload trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "traces.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 (tornado) and Figure 5 (Pareto)
# ---------------------------------------------------------------------------

def make_tornado():
    """Tornado plot: for each region, the 5-95th percentile band split
    by which input drives the variance. We compute two reduced MCMCs:
    one with only S_r varying, one with only WUE varying, and report
    the resulting band widths as horizontal bars."""
    snap = make_snapshot()
    regions = snap.aqueduct["region"].unique().tolist()
    band_s = []
    band_w = []
    for r in regions:
        pct_s = monte_carlo(snap, region=r, n_samples=2000, seed=11)
        # To isolate WUE only, set s_r_uncertainty=0
        from swpd_wf.model import _lookup_aqueduct
        snap.egrid  # ensure loaded
        # Override Aqueduct uncertainty to 0 by writing a copy
        snap2 = make_snapshot()
        snap2.aqueduct.loc[snap2.aqueduct["region"] == r, "s_r_uncertainty"] = 0.0
        pct_w = monte_carlo(snap2, region=r, n_samples=2000, seed=12)
        band_s.append(pct_s.q95 - pct_s.q05)
        band_w.append(pct_w.q95 - pct_w.q05)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(regions))
    width = 0.4
    ax.barh(y - width/2, band_s, height=width, label="S_r only", color="#7faedc")
    ax.barh(y + width/2, band_w, height=width, label="WUE only", color="#f4a460")
    ax.set_yticks(y)
    ax.set_yticklabels(regions)
    ax.set_xlabel("5-95th percentile band of W(f, r) (L)")
    ax.set_title("Sensitivity tornado: variance of W(f, r) by input source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tornado.pdf", bbox_inches="tight")
    plt.close(fig)


def make_carbon_water_pareto():
    snap = make_snapshot()
    rows = []
    for region in snap.aqueduct["region"].unique():
        res = compute_swpd_wf(
            snap, region=region, cooling_class="water-cooled",
            accelerator_class="H100", trace_name="public-llm-serving-2024",
        )
        egrid = snap.egrid[snap.egrid["region"] == region].iloc[0]
        # Carbon (kg) for the same 1 kWh training + 1 kWh inference job
        carbon = (1.0 + 1.0) * float(egrid["ef_kg_per_kwh"])
        rows.append({
            "region": region,
            "W": res.scalar,
            "carbon_kg": carbon,
            "EF": float(egrid["ef_kg_per_kwh"]),
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["carbon_kg"], df["W"], s=120, c="#1f3a5f", edgecolor="black", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["region"], (row["carbon_kg"], row["W"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8)

    # Pareto frontier: a point is Pareto-optimal if no other point has
    # both lower carbon AND lower water.
    pareto = []
    for i, row in df.iterrows():
        dominated = ((df["carbon_kg"] <= row["carbon_kg"]) &
                     (df["W"] <= row["W"]) &
                     ((df["carbon_kg"] < row["carbon_kg"]) |
                      (df["W"] < row["W"]))).any()
        if not dominated:
            pareto.append(i)
    pareto_df = df.loc[pareto].sort_values("carbon_kg")
    ax.plot(pareto_df["carbon_kg"], pareto_df["W"], "r--", lw=1.2, label="Pareto frontier")
    ax.scatter(pareto_df["carbon_kg"], pareto_df["W"], s=180, facecolor="none", edgecolor="red", lw=1.6, zorder=4)

    ax.set_xlabel("Carbon footprint (kg CO$_2$, per (1 kWh T + 1 kWh I) job)")
    ax.set_ylabel("SWPD-WF (L)")
    ax.set_title("Carbon-vs-water Pareto frontier across studied regions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "carbon_water_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def make_attribution_bars():
    """Per-region stacked bar of the 6 attribution components."""
    df = pd.read_csv(FIG_DIR / "attribution_table.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    components = ["W_T_D", "W_T_G", "W_E", "W_I_D", "W_I_G", "W_I_E"]
    colors = ["#7faedc", "#4a7ba8", "#222a35", "#f4a460", "#cf8030", "#7a4a1a"]
    bottom = np.zeros(len(df))
    for comp, col in zip(components, colors):
        ax.bar(x, df[comp], bottom=bottom, label=comp, color=col, edgecolor="white", linewidth=0.5)
        bottom += df[comp].values
    ax.set_xticks(x)
    ax.set_xticklabels(df["Region"], rotation=15, ha="right")
    ax.set_ylabel("SWPD-WF attribution (L)")
    ax.set_title("Per-region SWPD-WF attribution vector (phase $\\times$ flow)")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attribution_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating framework flowchart...")
    make_framework()
    print("Generating regions map...")
    make_regions_map()
    print("Generating trace distribution...")
    make_traces()
    print("Generating tornado plot...")
    make_tornado()
    print("Generating Pareto scatter...")
    make_carbon_water_pareto()
    print("Generating attribution bar chart...")
    make_attribution_bars()
    print(f"All figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
