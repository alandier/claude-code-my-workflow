"""
03_regional_analysis.py -- Regional M/B time-series with expanded regions.

Purpose: Compute and plot regional valuation trends from the panel data,
         showing how M/B evolves over time across regions.
         Includes both equal-weighted (median) and value-weighted aggregates.
         Separates Japan and China from rest of Asia.

Inputs:
  - output/regressions/panel_data.parquet
  - output/regressions/country_effects_mb.parquet

Outputs:
  - output/regressions/regional_mb_timeseries.pdf/png
  - output/regressions/regional_mb_vw_timeseries.pdf/png
  - output/regressions/regional_roa_timeseries.pdf/png
  - output/regressions/regional_roa_vw_timeseries.pdf/png
  - output/regressions/regional_mb_country_effects.pdf/png
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

# Region definitions (expanded: Japan and China separated from Asia)
EUROPE = [
    "GBR", "DEU", "FRA", "ITA", "ESP", "NLD", "CHE", "SWE", "NOR", "DNK",
    "FIN", "BEL", "AUT", "IRL", "PRT", "GRC", "POL", "CZE", "HUN", "ROU",
    "HRV", "SVN", "SVK", "LTU", "LVA", "EST", "BGR", "SRB", "UKR", "RUS",
    "TUR", "LUX", "ISL", "CYP", "MLT",
]
ASIA_EX = [
    "IND", "KOR", "TWN", "HKG", "SGP", "MYS", "THA", "IDN",
    "PHL", "VNM", "PAK", "BGD", "LKA", "KAZ", "ISR",
]

REGION_COLORS = {
    "US": PALETTE["accent"],
    "Europe": PALETTE["primary"],
    "Japan": "#e67e22",
    "China": PALETTE["alert"],
    "Other Asia": PALETTE["highlight"],
}

# For 3-region plots (backward compatible)
REGION3_COLORS = {
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
    """Map country code to 5-region classification."""
    if fic == "USA":
        return "US"
    elif fic in EUROPE:
        return "Europe"
    elif fic == "JPN":
        return "Japan"
    elif fic == "CHN":
        return "China"
    elif fic in ASIA_EX:
        return "Other Asia"
    else:
        return "Other"


def assign_region3(fic: str) -> str:
    """Map country code to 3-region classification (for summary)."""
    if fic == "USA":
        return "US"
    elif fic in EUROPE:
        return "Europe"
    elif fic == "JPN" or fic == "CHN" or fic in ASIA_EX:
        return "Asia"
    else:
        return "Other"


def vw_mean_within_country(group: pd.DataFrame, var: str) -> float:
    """Compute VW mean within each country, then EW across countries.

    Market caps are in local currency, so cross-country VW is meaningless.
    We first compute VW mean within each country (same currency), then
    take an equal-weighted average across countries in the group.
    """
    results = []
    for _fic, cdf in group.groupby("fic", observed=True):
        weights = cdf["market_cap"]
        valid = weights.notna() & cdf[var].notna() & (weights > 0)
        if valid.sum() < 5:
            continue
        results.append(np.average(cdf.loc[valid, var], weights=weights[valid]))
    if not results:
        return np.nan
    return np.mean(results)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

print("Loading panel data...")
df = pd.read_parquet(PARQUET_PATH)
df["region"] = df["fic"].astype(str).map(assign_region)
df["region3"] = df["fic"].astype(str).map(assign_region3)

print(f"  Panel: {len(df):,} obs")
print(f"  Region breakdown (5 regions):")
for region, cnt in df["region"].value_counts().items():
    print(f"    {region:<12s} {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

# Also load country effects
effects_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")
effects_mb["region"] = effects_mb["fic"].map(assign_region)

# ---------------------------------------------------------------------------
# 2. Regional M/B Time-Series (EW median)
# ---------------------------------------------------------------------------

REGIONS_5 = ["US", "Europe", "Japan", "China", "Other Asia"]

print("\nComputing regional M/B time-series...")

regional_mb = (
    df.groupby(["region", "fyear"])["log_mb"]
    .agg(["median", "mean", "count"])
    .reset_index()
)
regional_mb = regional_mb[
    regional_mb["region"].isin(REGIONS_5)
    & (regional_mb["count"] >= 50)
]

fig, ax = plt.subplots(figsize=(8, 5))
for region in REGIONS_5:
    rdata = regional_mb[regional_mb["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["median"],
            color=REGION_COLORS[region], label=region, linewidth=2)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Median log(M/B)")
ax.set_title("Valuation Levels by Region Over Time")
ax.legend()
save_figure(fig, "regional_mb_timeseries")

# ---------------------------------------------------------------------------
# 2b. Regional M/B Time-Series (Value-Weighted)
# ---------------------------------------------------------------------------

print("Computing value-weighted regional M/B time-series...")

regional_mb_vw = (
    df[df["region"].isin(REGIONS_5)]
    .groupby(["region", "fyear"])
    .apply(lambda g: pd.Series({
        "vw_mean": vw_mean_within_country(g, "log_mb"),
        "count": len(g),
    }), include_groups=False)
    .reset_index()
)
regional_mb_vw = regional_mb_vw[regional_mb_vw["count"] >= 50]

fig, ax = plt.subplots(figsize=(8, 5))
for region in REGIONS_5:
    rdata = regional_mb_vw[regional_mb_vw["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["vw_mean"],
            color=REGION_COLORS[region], label=region, linewidth=2)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Value-Weighted Mean log(M/B)")
ax.set_title("Valuation Levels by Region (Value-Weighted)")
ax.legend()
save_figure(fig, "regional_mb_vw_timeseries")

# ---------------------------------------------------------------------------
# 3. Regional ROA Time-Series (EW median)
# ---------------------------------------------------------------------------

print("Computing regional ROA time-series...")

regional_roa = (
    df.groupby(["region", "fyear"])["roa"]
    .agg(["median", "mean", "count"])
    .reset_index()
)
regional_roa = regional_roa[
    regional_roa["region"].isin(REGIONS_5)
    & (regional_roa["count"] >= 50)
]

fig, ax = plt.subplots(figsize=(8, 5))
for region in REGIONS_5:
    rdata = regional_roa[regional_roa["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["median"],
            color=REGION_COLORS[region], label=region, linewidth=2)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Median ROA")
ax.set_title("Profitability by Region Over Time")
ax.legend()
save_figure(fig, "regional_roa_timeseries")

# ---------------------------------------------------------------------------
# 3b. Regional ROA Time-Series (Value-Weighted)
# ---------------------------------------------------------------------------

print("Computing value-weighted regional ROA time-series...")

regional_roa_vw = (
    df[df["region"].isin(REGIONS_5)]
    .groupby(["region", "fyear"])
    .apply(lambda g: pd.Series({
        "vw_mean": vw_mean_within_country(g, "roa"),
        "count": len(g),
    }), include_groups=False)
    .reset_index()
)
regional_roa_vw = regional_roa_vw[regional_roa_vw["count"] >= 50]

fig, ax = plt.subplots(figsize=(8, 5))
for region in REGIONS_5:
    rdata = regional_roa_vw[regional_roa_vw["region"] == region].sort_values("fyear")
    ax.plot(rdata["fyear"], rdata["vw_mean"],
            color=REGION_COLORS[region], label=region, linewidth=2)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Value-Weighted Mean ROA")
ax.set_title("Profitability by Region (Value-Weighted)")
ax.legend()
save_figure(fig, "regional_roa_vw_timeseries")

# ---------------------------------------------------------------------------
# 4. Country Effects by Region (from year-by-year regressions)
# ---------------------------------------------------------------------------

print("Computing regional country effects...")

# Average country effect by region-year (excluding USA = reference)
non_us_regions = ["Europe", "Japan", "China", "Other Asia"]
regional_effects = (
    effects_mb[effects_mb["region"].isin(non_us_regions)]
    .groupby(["region", "fyear"])["country_effect"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
for region in non_us_regions:
    rdata = regional_effects[regional_effects["region"] == region].sort_values("fyear")
    if len(rdata) < 3:
        continue
    ax.plot(rdata["fyear"], rdata["country_effect"],
            color=REGION_COLORS[region], label=region, linewidth=2)

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

for region in REGIONS_5:
    rdf = df[df["region"] == region]
    if len(rdf) == 0:
        continue
    summary_lines.append(f"\n  {region}:")
    summary_lines.append(f"    Obs: {len(rdf):,}")
    summary_lines.append(f"    Countries: {rdf['fic'].nunique()}")
    summary_lines.append(f"    Firms: {rdf['gvkey'].nunique():,}")
    summary_lines.append(f"    Years: {rdf['fyear'].min():.0f}–{rdf['fyear'].max():.0f}")
    summary_lines.append(f"    Median log(M/B): {rdf['log_mb'].median():.3f}")
    summary_lines.append(f"    Mean log(M/B):   {rdf['log_mb'].mean():.3f}")

    # Value-weighted mean (VW within country, then EW across countries)
    vw = vw_mean_within_country(rdf, "log_mb")
    if not np.isnan(vw):
        summary_lines.append(f"    VW Mean log(M/B): {vw:.3f}")

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
us_recent_ew = recent[recent["region"] == "US"]["log_mb"].median()
us_recent_vw = vw_mean_within_country(recent[recent["region"] == "US"], "log_mb")

for region in ["Europe", "Japan", "China", "Other Asia"]:
    r_recent = recent[recent["region"] == region]
    if len(r_recent) < 50:
        continue
    ew_gap = us_recent_ew - r_recent["log_mb"].median()
    summary_lines.append(f"  US - {region:<12s}: {ew_gap:+.3f} log points "
                         f"(~{(np.exp(ew_gap) - 1)*100:+.0f}% in levels, EW median)")

    # VW gap (VW within country, EW across countries)
    r_vw = vw_mean_within_country(r_recent, "log_mb")
    if not np.isnan(r_vw) and not np.isnan(us_recent_vw):
        vw_gap = us_recent_vw - r_vw
        summary_lines.append(f"  US - {region:<12s}: {vw_gap:+.3f} log points "
                             f"(~{(np.exp(vw_gap) - 1)*100:+.0f}% in levels, VW mean)")

summary_text = "\n".join(summary_lines)
print(summary_text)

(OUT_DIR / "regional_summary.txt").write_text(summary_text, encoding="utf-8")
print(f"\n  Summary saved to {OUT_DIR / 'regional_summary.txt'}")
print("  Done.")
