"""
industry_scatter.py -- Industry-level ROA vs valuation scatter plots.

Generates:
  - scatter_industry_roa_vs_pe.pdf/png  (demeaned by country-year)
  - scatter_industry_roa_vs_mb.pdf/png  (demeaned by country-year)

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("output/regressions")

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
}

# GICS Industry Group code -> short name
GICS_NAMES = {
    1010: "Energy",
    1510: "Materials",
    2010: "Cap Goods",
    2020: "Comm Svc",
    2030: "Transport",
    2510: "Auto",
    2520: "Durables",
    2530: "Consumer Svc",
    2540: "Apparel",
    2550: "Retailing",
    3010: "Food Retail",
    3020: "Food & Bev",
    3030: "Household",
    3510: "Health Equip",
    3520: "Pharma",
    4010: "Banks",
    4020: "Div Financials",
    4030: "Insurance",
    4040: "Real Estate",
    4510: "Software",
    4520: "Tech HW",
    4530: "Semis",
    5010: "Telecom",
    5020: "Media",
    5510: "Utilities",
    6010: "Equity REITs",
    6020: "Mortgage REITs",
}

# ---------------------------------------------------------------------------
# Load panel
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs, {df.shape[1]} cols")

ind_col = "ggroup"

# ---------------------------------------------------------------------------
# Demean by country-year
# ---------------------------------------------------------------------------
for var in ["log_pe", "log_mb", "roa"]:
    if var not in df.columns:
        continue
    group_mean = df.groupby(["fic", "fyear"], observed=True)[var].transform("mean")
    df[f"{var}_dm"] = df[var] - group_mean

# ---------------------------------------------------------------------------
# Industry averages (demeaned) -- all firms for M/B, profitable only for P/E
# ---------------------------------------------------------------------------
# M/B and ROA: all firms
ind_mb = df.groupby(ind_col, observed=True).agg(
    roa_dm=("roa_dm", "mean"),
    log_mb_dm=("log_mb_dm", "mean"),
    n=("roa_dm", "count"),
).reset_index()

# P/E: profitable firms only
pe_sub = df.dropna(subset=["log_pe_dm"])
ind_pe = pe_sub.groupby(ind_col, observed=True).agg(
    roa_dm=("roa_dm", "mean"),
    log_pe_dm=("log_pe_dm", "mean"),
    n=("log_pe_dm", "count"),
).reset_index()

# Filter to industries with >= 1000 obs
min_obs = 1000
ind_mb = ind_mb[ind_mb["n"] >= min_obs].copy()
ind_pe = ind_pe[ind_pe["n"] >= min_obs].copy()

# Add names
ind_mb["name"] = ind_mb[ind_col].map(GICS_NAMES)
ind_pe["name"] = ind_pe[ind_col].map(GICS_NAMES)

# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
r_pe = ind_pe[["roa_dm", "log_pe_dm"]].corr().iloc[0, 1]
r_mb = ind_mb[["roa_dm", "log_mb_dm"]].corr().iloc[0, 1]
print(f"\nDemeaned industry correlations (within-country):")
print(f"  ROA vs P/E: {r_pe:+.2f}  (n={len(ind_pe)} industries)")
print(f"  ROA vs M/B: {r_mb:+.2f}  (n={len(ind_mb)} industries)")

# Raw (not demeaned)
ind_raw = df.groupby(ind_col, observed=True).agg(
    roa=("roa", "mean"), log_pe=("log_pe", "mean"), log_mb=("log_mb", "mean"),
    n=("roa", "count"),
).reset_index()
ind_raw = ind_raw[ind_raw["n"] >= min_obs]
r_pe_raw = ind_raw[["roa", "log_pe"]].dropna().corr().iloc[0, 1]
r_mb_raw = ind_raw[["roa", "log_mb"]].dropna().corr().iloc[0, 1]
print(f"\nRaw industry correlations:")
print(f"  ROA vs P/E: {r_pe_raw:+.2f}")
print(f"  ROA vs M/B: {r_mb_raw:+.2f}")


def _label_scatter(ax, data, x_col, y_col, n_label=5):
    """Label the most extreme points by y-value."""
    to_label = pd.concat([data.nlargest(n_label, y_col), data.nsmallest(n_label, y_col)])
    to_label = to_label.drop_duplicates(subset=[ind_col])
    for _, row in to_label.iterrows():
        name = row.get("name", str(int(row[ind_col])))
        if name is None or (isinstance(name, float) and np.isnan(name)):
            name = str(int(row[ind_col]))
        ax.annotate(
            name, (row[x_col], row[y_col]),
            fontsize=6.5, ha="left", va="bottom",
            xytext=(4, 2), textcoords="offset points",
            color=PALETTE["primary"],
        )


def _fit_line(ax, x, y):
    """Add OLS fit line."""
    mask = np.isfinite(x) & np.isfinite(y)
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), color=PALETTE["alert"],
            linewidth=1.5, linestyle="--", alpha=0.8)


# ---------------------------------------------------------------------------
# Scatter: Industry ROA vs P/E (demeaned)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(ind_pe["roa_dm"], ind_pe["log_pe_dm"],
           color=PALETTE["accent"], s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_scatter(ax, ind_pe, "roa_dm", "log_pe_dm")
_fit_line(ax, ind_pe["roa_dm"].values, ind_pe["log_pe_dm"].values)

ax.set_xlabel("Demeaned ROA (industry avg)", fontsize=10)
ax.set_ylabel("Demeaned log(P/E) (industry avg)", fontsize=10)
ax.set_title(f"Industry-Level: ROA vs P/E\n(demeaned by country$\\times$year, r = {r_pe:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_industry_roa_vs_pe.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_industry_roa_vs_pe.png", dpi=300)
plt.close(fig)
print("\nSaved scatter_industry_roa_vs_pe")

# ---------------------------------------------------------------------------
# Scatter: Industry ROA vs M/B (demeaned)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(ind_mb["roa_dm"], ind_mb["log_mb_dm"],
           color=PALETTE["accent"], s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_scatter(ax, ind_mb, "roa_dm", "log_mb_dm")
_fit_line(ax, ind_mb["roa_dm"].values, ind_mb["log_mb_dm"].values)

ax.set_xlabel("Demeaned ROA (industry avg)", fontsize=10)
ax.set_ylabel("Demeaned log(M/B) (industry avg)", fontsize=10)
ax.set_title(f"Industry-Level: ROA vs M/B\n(demeaned by country$\\times$year, r = {r_mb:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_industry_roa_vs_mb.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_industry_roa_vs_mb.png", dpi=300)
plt.close(fig)
print("Saved scatter_industry_roa_vs_mb")

# ---------------------------------------------------------------------------
# Print top/bottom industries for reference
# ---------------------------------------------------------------------------
print("\n--- Top 5 industries by demeaned P/E ---")
for _, r in ind_pe.nlargest(5, "log_pe_dm").iterrows():
    print(f"  {r['name']:20s}  ROA_dm={r['roa_dm']:+.3f}  P/E_dm={r['log_pe_dm']:+.3f}")

print("\n--- Bottom 5 industries by demeaned P/E ---")
for _, r in ind_pe.nsmallest(5, "log_pe_dm").iterrows():
    print(f"  {r['name']:20s}  ROA_dm={r['roa_dm']:+.3f}  P/E_dm={r['log_pe_dm']:+.3f}")

print("\n--- Top 5 industries by demeaned M/B ---")
for _, r in ind_mb.nlargest(5, "log_mb_dm").iterrows():
    print(f"  {r['name']:20s}  ROA_dm={r['roa_dm']:+.3f}  M/B_dm={r['log_mb_dm']:+.3f}")

print("\n--- Bottom 5 industries by demeaned M/B ---")
for _, r in ind_mb.nsmallest(5, "log_mb_dm").iterrows():
    print(f"  {r['name']:20s}  ROA_dm={r['roa_dm']:+.3f}  M/B_dm={r['log_mb_dm']:+.3f}")

print("\nDone.")
