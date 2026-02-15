"""
03_regional_analysis.py -- Regional valuation time-series with expanded regions.

Purpose: Compute and plot regional valuation trends from the panel data,
         showing how P/E and M/B evolve over time across regions.
         Includes both equal-weighted (median) and value-weighted aggregates.
         Separates Japan and China from rest of Asia.

Inputs:
  - output/regressions/panel_data.parquet
  - output/regressions/country_effects_pe.parquet
  - output/regressions/country_effects_mb.parquet

Outputs:
  - output/regressions/regional_pe_timeseries.pdf/png
  - output/regressions/regional_pe_vw_timeseries.pdf/png
  - output/regressions/regional_mb_timeseries.pdf/png
  - output/regressions/regional_mb_vw_timeseries.pdf/png
  - output/regressions/regional_roa_timeseries.pdf/png
  - output/regressions/regional_roa_vw_timeseries.pdf/png
  - output/regressions/regional_pe_country_effects.pdf/png
  - output/regressions/regional_mb_country_effects.pdf/png
  - output/regressions/regional_leverage_timeseries.pdf/png
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

REGIONS_5 = ["US", "Europe", "Japan", "China", "Other Asia"]

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


def plot_regional_ew(df, var, ylabel, title, fig_name):
    """Plot EW median of a variable by region over time."""
    regional = (
        df.groupby(["region", "fyear"])[var]
        .agg(["median", "count"])
        .reset_index()
    )
    regional = regional[
        regional["region"].isin(REGIONS_5) & (regional["count"] >= 50)
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for region in REGIONS_5:
        rdata = regional[regional["region"] == region].sort_values("fyear")
        ax.plot(rdata["fyear"], rdata["median"],
                color=REGION_COLORS[region], label=region, linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, fig_name)


def plot_regional_vw(df, var, ylabel, title, fig_name):
    """Plot VW mean (within-country weighted, EW across countries) by region."""
    regional_vw = (
        df[df["region"].isin(REGIONS_5)]
        .groupby(["region", "fyear"])
        .apply(lambda g: pd.Series({
            "vw_mean": vw_mean_within_country(g, var),
            "count": len(g),
        }), include_groups=False)
        .reset_index()
    )
    regional_vw = regional_vw[regional_vw["count"] >= 50]

    fig, ax = plt.subplots(figsize=(8, 5))
    for region in REGIONS_5:
        rdata = regional_vw[regional_vw["region"] == region].sort_values("fyear")
        ax.plot(rdata["fyear"], rdata["vw_mean"],
                color=REGION_COLORS[region], label=region, linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, fig_name)


def plot_country_effects_by_region(effects_df, title, fig_name):
    """Plot average country effect by region (vs USA baseline)."""
    non_us_regions = ["Europe", "Japan", "China", "Other Asia"]
    regional_effects = (
        effects_df[effects_df["region"].isin(non_us_regions)]
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
    ax.set_title(title)
    ax.legend()
    save_figure(fig, fig_name)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

print("Loading panel data...")
df = pd.read_parquet(PARQUET_PATH)
df["region"] = df["fic"].astype(str).map(assign_region)

print(f"  Panel: {len(df):,} obs")
print(f"  Region breakdown (5 regions):")
for region, cnt in df["region"].value_counts().items():
    print(f"    {region:<12s} {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

# Load country effects
effects_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
effects_pe["region"] = effects_pe["fic"].map(assign_region)

effects_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")
effects_mb["region"] = effects_mb["fic"].map(assign_region)

# ---------------------------------------------------------------------------
# 2. P/E Time-Series
# ---------------------------------------------------------------------------

print("\nComputing regional P/E time-series...")
plot_regional_ew(df, "log_pe", "Median log(P/E)",
                 "P/E Levels by Region Over Time", "regional_pe_timeseries")

print("Computing value-weighted regional P/E time-series...")
plot_regional_vw(df, "log_pe", "Value-Weighted Mean log(P/E)",
                 "P/E Levels by Region (Value-Weighted)", "regional_pe_vw_timeseries")

# ---------------------------------------------------------------------------
# 3. M/B Time-Series
# ---------------------------------------------------------------------------

print("\nComputing regional M/B time-series...")
plot_regional_ew(df, "log_mb", "Median log(M/B)",
                 "M/B Levels by Region Over Time", "regional_mb_timeseries")

print("Computing value-weighted regional M/B time-series...")
plot_regional_vw(df, "log_mb", "Value-Weighted Mean log(M/B)",
                 "M/B Levels by Region (Value-Weighted)", "regional_mb_vw_timeseries")

# ---------------------------------------------------------------------------
# 4. ROA Time-Series
# ---------------------------------------------------------------------------

print("\nComputing regional ROA time-series...")
plot_regional_ew(df, "roa", "Median ROA",
                 "Profitability by Region Over Time", "regional_roa_timeseries")

print("Computing value-weighted regional ROA time-series...")
plot_regional_vw(df, "roa", "Value-Weighted Mean ROA",
                 "Profitability by Region (Value-Weighted)", "regional_roa_vw_timeseries")

# ---------------------------------------------------------------------------
# 5. Country Effects by Region
# ---------------------------------------------------------------------------

print("\nComputing regional country effects...")
plot_country_effects_by_region(
    effects_pe, "Regional P/E Premium/Discount vs USA",
    "regional_pe_country_effects",
)
plot_country_effects_by_region(
    effects_mb, "Regional M/B Premium/Discount vs USA",
    "regional_mb_country_effects",
)

# ---------------------------------------------------------------------------
# 6. Regional Leverage Time-Series
# ---------------------------------------------------------------------------

print("Computing regional leverage time-series...")
plot_regional_ew(df, "leverage", "Median Leverage",
                 "Leverage by Region Over Time", "regional_leverage_timeseries")

# ---------------------------------------------------------------------------
# 7. Summary Statistics by Region
# ---------------------------------------------------------------------------

print("\nRegional Summary Statistics:")
print("=" * 70)

summary_lines = []
summary_lines.append("Regional Valuation and ROA Summary (P/E + M/B)")
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

    # P/E
    pe_valid = rdf["log_pe"].notna()
    summary_lines.append(f"    Median log(P/E): {rdf.loc[pe_valid, 'log_pe'].median():.3f} "
                         f"(N={pe_valid.sum():,})")
    vw_pe = vw_mean_within_country(rdf, "log_pe")
    if not np.isnan(vw_pe):
        summary_lines.append(f"    VW Mean log(P/E): {vw_pe:.3f}")

    # M/B
    summary_lines.append(f"    Median log(M/B): {rdf['log_mb'].median():.3f} "
                         f"(N={rdf['log_mb'].notna().sum():,})")
    vw_mb = vw_mean_within_country(rdf, "log_mb")
    if not np.isnan(vw_mb):
        summary_lines.append(f"    VW Mean log(M/B): {vw_mb:.3f}")

    summary_lines.append(f"    Median leverage: {rdf['leverage'].median():.3f}")
    summary_lines.append(f"    Median ROA:      {rdf['roa'].median():.4f}")
    summary_lines.append(f"    Mean ROA:        {rdf['roa'].mean():.4f}")

    # Time trend: compare early vs late (P/E)
    early_pe = rdf[rdf["fyear"] <= rdf["fyear"].quantile(0.25)]["log_pe"].median()
    late_pe = rdf[rdf["fyear"] >= rdf["fyear"].quantile(0.75)]["log_pe"].median()
    summary_lines.append(f"    P/E trend: early={early_pe:.3f}, late={late_pe:.3f} "
                         f"({'UP' if late_pe > early_pe else 'DOWN'} {late_pe - early_pe:+.3f})")

    early_mb = rdf[rdf["fyear"] <= rdf["fyear"].quantile(0.25)]["log_mb"].median()
    late_mb = rdf[rdf["fyear"] >= rdf["fyear"].quantile(0.75)]["log_mb"].median()
    summary_lines.append(f"    M/B trend: early={early_mb:.3f}, late={late_mb:.3f} "
                         f"({'UP' if late_mb > early_mb else 'DOWN'} {late_mb - early_mb:+.3f})")

# Gap analysis
for var, label in [("log_pe", "P/E"), ("log_mb", "M/B")]:
    summary_lines.append(f"\n{'=' * 70}")
    summary_lines.append(f"{label} Gap: US minus Region (recent 5 years)")
    summary_lines.append("=" * 70)

    recent = df[df["fyear"] >= df["fyear"].max() - 4]
    us_recent_ew = recent[recent["region"] == "US"][var].median()
    us_recent_vw = vw_mean_within_country(recent[recent["region"] == "US"], var)

    for region in ["Europe", "Japan", "China", "Other Asia"]:
        r_recent = recent[recent["region"] == region]
        if len(r_recent) < 50:
            continue
        r_median = r_recent[var].median()
        if pd.isna(r_median) or pd.isna(us_recent_ew):
            continue
        ew_gap = us_recent_ew - r_median
        summary_lines.append(f"  US - {region:<12s}: {ew_gap:+.3f} log points "
                             f"(~{(np.exp(ew_gap) - 1)*100:+.0f}% in levels, EW median)")

        r_vw = vw_mean_within_country(r_recent, var)
        if not np.isnan(r_vw) and not np.isnan(us_recent_vw):
            vw_gap = us_recent_vw - r_vw
            summary_lines.append(f"  US - {region:<12s}: {vw_gap:+.3f} log points "
                                 f"(~{(np.exp(vw_gap) - 1)*100:+.0f}% in levels, VW mean)")

summary_text = "\n".join(summary_lines)
print(summary_text)

(OUT_DIR / "regional_summary.txt").write_text(summary_text, encoding="utf-8")
print(f"\n  Summary saved to {OUT_DIR / 'regional_summary.txt'}")
print("  Done.")
