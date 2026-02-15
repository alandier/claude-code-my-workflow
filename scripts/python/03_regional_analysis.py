"""
03_regional_analysis.py -- Regional M/B time-series: US vs Europe vs Asia.

Purpose: Compute and plot regional valuation trends from the panel data,
         showing how M/B evolves over time for three broad regions.

Inputs:
  - output/regressions/panel_data.parquet
  - output/regressions/country_effects_mb.parquet

Outputs:
  - output/regressions/regional_mb_timeseries.pdf/png
  - output/regressions/regional_mb_country_effects.pdf/png
  - output/regressions/regional_roa_timeseries.pdf/png
  - output/regressions/regional_summary.txt

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

OUT_DIR = Path("output/regressions")
PARQUET_PATH = OUT_DIR / "panel_data.parquet"

PALETTE = {
    "primary": "#2c3e50",
    "secondary": "#7f8c8d",
    "accent": "#2980b9",
    "highlight": "#8e44ad",
    "alert": "#c0392b",
}

# Region definitions
EUROPE = [
    "GBR", "DEU", "FRA", "ITA", "ESP", "NLD", "CHE", "SWE", "NOR", "DNK",
    "FIN", "BEL", "AUT", "IRL", "PRT", "GRC", "POL", "CZE", "HUN", "ROU",
    "HRV", "SVN", "SVK", "LTU", "LVA", "EST", "BGR", "SRB", "UKR", "RUS",
    "TUR", "LUX", "ISL", "CYP", "MLT",
]
ASIA = [
    "JPN", "CHN", "IND", "KOR", "TWN", "HKG", "SGP", "MYS", "THA", "IDN",
    "PHL", "VNM", "PAK", "BGD", "LKA", "KAZ", "ISR",
]

REGION_COLORS = {
    "US": PALETTE["accent"],
    "Europe": PALETTE["primary"],
    "Asia": PALETTE["alert"],
}

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure in dual format (PDF + PNG) at 300 DPI."""
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {name}.pdf/png")


def assign_region(fic: str) -> str:
    """Map country code to region."""
    if fic == "USA":
        return "US"
    elif fic in EUROPE:
        return "Europe"
    elif fic in ASIA:
        return "Asia"
    else:
        return "Other"


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

print("Loading panel data...")
df = pd.read_parquet(PARQUET_PATH)
df["region"] = df["fic"].astype(str).map(assign_region)

print(f"  Panel: {len(df):,} obs")
print(f"  Region breakdown:")
for region, cnt in df["region"].value_counts().items():
    print(f"    {region:<8s} {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

# Also load country effects
effects_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")
effects_mb["region"] = effects_mb["fic"].map(assign_region)

# ---------------------------------------------------------------------------
# 2. Regional M/B Time-Series (median log M/B by region-year)
# ---------------------------------------------------------------------------

print("\nComputing regional M/B time-series...")

# Use median (robust to outliers) and also compute equal-weighted mean
regional_mb = (
    df.groupby(["region", "fyear"])["log_mb"]
    .agg(["median", "mean", "count"])
    .reset_index()
)

# Filter to regions with enough data and years >= 1990 for cleaner plot
regional_mb = regional_mb[
    regional_mb["region"].isin(["US", "Europe", "Asia"])
    & (regional_mb["count"] >= 50)
]

fig, ax = plt.subplots(figsize=(8, 5))
for region in ["US", "Europe", "Asia"]:
    rdata = regional_mb[regional_mb["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["median"],
            color=REGION_COLORS[region], label=region,
            linewidth=2)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Median log(M/B)")
ax.set_title("Valuation Levels by Region Over Time")
ax.legend()
save_figure(fig, "regional_mb_timeseries")

# ---------------------------------------------------------------------------
# 3. Regional ROA Time-Series
# ---------------------------------------------------------------------------

print("Computing regional ROA time-series...")

regional_roa = (
    df.groupby(["region", "fyear"])["roa"]
    .agg(["median", "mean", "count"])
    .reset_index()
)
regional_roa = regional_roa[
    regional_roa["region"].isin(["US", "Europe", "Asia"])
    & (regional_roa["count"] >= 50)
]

fig, ax = plt.subplots(figsize=(8, 5))
for region in ["US", "Europe", "Asia"]:
    rdata = regional_roa[regional_roa["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["median"],
            color=REGION_COLORS[region], label=region,
            linewidth=2)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Median ROA")
ax.set_title("Profitability by Region Over Time")
ax.legend()
save_figure(fig, "regional_roa_timeseries")

# ---------------------------------------------------------------------------
# 4. Country Effects by Region (from year-by-year regressions)
# ---------------------------------------------------------------------------

print("Computing regional country effects...")

# Average country effect by region-year (excluding USA = reference)
regional_effects = (
    effects_mb[effects_mb["region"].isin(["Europe", "Asia"])]
    .groupby(["region", "fyear"])["country_effect"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
for region in ["Europe", "Asia"]:
    rdata = regional_effects[regional_effects["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["country_effect"],
            color=REGION_COLORS[region], label=region,
            linewidth=2)

ax.axhline(0, color=REGION_COLORS["US"], linewidth=2, linestyle="-",
           alpha=0.7, label="US (baseline)")
ax.set_xlabel("Year")
ax.set_ylabel("Avg Country Effect (vs USA)")
ax.set_title("Regional Valuation Premium/Discount vs USA")
ax.legend()
save_figure(fig, "regional_mb_country_effects")

# ---------------------------------------------------------------------------
# 5. Summary Statistics by Region
# ---------------------------------------------------------------------------

print("\nRegional Summary Statistics:")
print("=" * 70)

summary_lines = []
summary_lines.append("Regional M/B and ROA Summary")
summary_lines.append("=" * 70)

for region in ["US", "Europe", "Asia"]:
    rdf = df[df["region"] == region]
    summary_lines.append(f"\n  {region}:")
    summary_lines.append(f"    Obs: {len(rdf):,}")
    summary_lines.append(f"    Countries: {rdf['fic'].nunique()}")
    summary_lines.append(f"    Firms: {rdf['gvkey'].nunique():,}")
    summary_lines.append(f"    Years: {rdf['fyear'].min():.0f}–{rdf['fyear'].max():.0f}")
    summary_lines.append(f"    Median log(M/B): {rdf['log_mb'].median():.3f}")
    summary_lines.append(f"    Mean log(M/B):   {rdf['log_mb'].mean():.3f}")
    summary_lines.append(f"    Median ROA:      {rdf['roa'].median():.4f}")
    summary_lines.append(f"    Mean ROA:        {rdf['roa'].mean():.4f}")

    # Time trend: compare early vs late
    early = rdf[rdf["fyear"] <= rdf["fyear"].quantile(0.25)]["log_mb"].median()
    late = rdf[rdf["fyear"] >= rdf["fyear"].quantile(0.75)]["log_mb"].median()
    summary_lines.append(f"    Early M/B median: {early:.3f}")
    summary_lines.append(f"    Late M/B median:  {late:.3f}")
    summary_lines.append(f"    Trend: {'UP' if late > early else 'DOWN'} "
                         f"({late - early:+.3f})")

# M/B gap analysis
summary_lines.append("\n" + "=" * 70)
summary_lines.append("M/B Gap: US minus Region (recent 5 years)")
summary_lines.append("=" * 70)

recent = df[df["fyear"] >= df["fyear"].max() - 4]
us_recent = recent[recent["region"] == "US"]["log_mb"].median()
for region in ["Europe", "Asia"]:
    r_recent = recent[recent["region"] == region]["log_mb"].median()
    gap = us_recent - r_recent
    summary_lines.append(f"  US - {region}: {gap:+.3f} log points "
                         f"(~{(np.exp(gap) - 1)*100:+.0f}% in levels)")

summary_text = "\n".join(summary_lines)
print(summary_text)

(OUT_DIR / "regional_summary.txt").write_text(summary_text, encoding="utf-8")
print(f"\n  Summary saved to {OUT_DIR / 'regional_summary.txt'}")
print("  Done.")
