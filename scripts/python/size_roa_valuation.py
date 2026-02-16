"""
size_roa_valuation.py -- Test whether country-level firm size mediates
the negative ROA-valuation correlation.

Hypothesis: In countries with poor capital markets, firms are suboptimally
small (capital-starved), earning high ROA on limited capital, while discount
rates are high (low P/E). If correct:
  - Low-P/E countries should have smaller firms
  - High-ROA countries should have smaller firms
  - Controlling for avg firm size should absorb the negative ROA-P/E link

Outputs:
  - scatter_size_vs_pe.pdf/png
  - scatter_size_vs_roa.pdf/png
  - scatter_roa_vs_pe_size_resid.pdf/png  (after partialing out size)
  - size_roa_valuation_results.txt

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT_DIR = Path("output/regressions")

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
    "highlight": "#8e44ad",
}

# ---------------------------------------------------------------------------
# Load panel and country effects
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs")

# Load country effects from year-by-year regressions
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

# Average country effects across years (same as in the scatter plots)
avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_mb = ce_mb.groupby("fic")["country_effect"].mean().rename("mb_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")

# ---------------------------------------------------------------------------
# Country-level size measures
# ---------------------------------------------------------------------------
# 1. Average log(assets) per country (time-series average of cross-sectional mean)
size_by_cy = df.groupby(["fic", "fyear"])["log_at"].agg(["mean", "median", "count"])
size_by_cy.columns = ["mean_log_at", "median_log_at", "n_firms"]
size_by_cy = size_by_cy.reset_index()

# Filter: at least 5 years and 20 firms/year
country_years = size_by_cy.groupby("fic").agg(
    n_years=("fyear", "count"),
    avg_n_firms=("n_firms", "mean"),
).reset_index()
valid_countries = country_years[
    (country_years["n_years"] >= 5) & (country_years["avg_n_firms"] >= 20)
]["fic"]
print(f"Countries with >= 5 years and >= 20 firms/year: {len(valid_countries)}")

size_avg = size_by_cy[size_by_cy["fic"].isin(valid_countries)].groupby("fic").agg(
    mean_log_at=("mean_log_at", "mean"),
    median_log_at=("median_log_at", "mean"),
    avg_n_firms=("n_firms", "mean"),
).reset_index()

# 2. Also compute: fraction of small firms (below global median)
global_median_at = df["log_at"].median()
df["is_small"] = (df["log_at"] < global_median_at).astype(int)
small_share = df[df["fic"].isin(valid_countries)].groupby("fic")["is_small"].mean()
small_share.name = "small_firm_share"

# ---------------------------------------------------------------------------
# Merge everything
# ---------------------------------------------------------------------------
country = size_avg.merge(avg_pe, on="fic", how="inner")
country = country.merge(avg_mb, on="fic", how="inner")
country = country.merge(avg_roa, on="fic", how="inner")
country = country.merge(small_share, on="fic", how="inner")

print(f"\nCountries in final sample: {len(country)}")
print(country[["fic", "mean_log_at", "pe_effect", "roa_effect"]].sort_values("mean_log_at").to_string())

# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
print("\n=== CORRELATIONS ===")
vars_of_interest = ["mean_log_at", "median_log_at", "pe_effect", "mb_effect",
                    "roa_effect", "small_firm_share"]
corr = country[vars_of_interest].corr()
print("\nCorrelation matrix:")
print(corr.round(2).to_string())

print(f"\n  Size vs P/E:  r = {country['mean_log_at'].corr(country['pe_effect']):+.2f}")
print(f"  Size vs M/B:  r = {country['mean_log_at'].corr(country['mb_effect']):+.2f}")
print(f"  Size vs ROA:  r = {country['mean_log_at'].corr(country['roa_effect']):+.2f}")
print(f"  ROA vs P/E:   r = {country['roa_effect'].corr(country['pe_effect']):+.2f}")
print(f"  Small share vs P/E: r = {country['small_firm_share'].corr(country['pe_effect']):+.2f}")

# ---------------------------------------------------------------------------
# Drop countries with missing size data
# ---------------------------------------------------------------------------
country = country.dropna(subset=["mean_log_at"]).copy()
print(f"Countries with size data: {len(country)}")

# ---------------------------------------------------------------------------
# Partial correlation: ROA vs P/E after controlling for size
# ---------------------------------------------------------------------------
# Regress ROA on size, get residuals
X_size = sm.add_constant(country["mean_log_at"])
roa_resid = sm.OLS(country["roa_effect"], X_size).fit().resid
pe_resid = sm.OLS(country["pe_effect"], X_size).fit().resid
mb_resid = sm.OLS(country["mb_effect"], X_size).fit().resid

r_partial_pe = np.corrcoef(roa_resid, pe_resid)[0, 1]
r_partial_mb = np.corrcoef(roa_resid, mb_resid)[0, 1]
print(f"\n  Partial corr (ROA vs P/E | size):  r = {r_partial_pe:+.2f}")
print(f"  Partial corr (ROA vs M/B | size):  r = {r_partial_mb:+.2f}")

# ---------------------------------------------------------------------------
# Regression: P/E = a + b*ROA + c*Size
# ---------------------------------------------------------------------------
print("\n=== REGRESSIONS ===")

# Spec 1: P/E ~ ROA
X1 = sm.add_constant(country["roa_effect"])
m1 = sm.OLS(country["pe_effect"], X1).fit()
print(f"\nSpec 1: P/E ~ ROA")
print(f"  ROA coef: {m1.params.iloc[1]:.3f} (t={m1.tvalues.iloc[1]:.2f})")
print(f"  R²: {m1.rsquared:.3f}")

# Spec 2: P/E ~ ROA + Size
X2 = sm.add_constant(country[["roa_effect", "mean_log_at"]])
m2 = sm.OLS(country["pe_effect"], X2).fit()
print(f"\nSpec 2: P/E ~ ROA + Size")
print(f"  ROA coef:  {m2.params.iloc[1]:.3f} (t={m2.tvalues.iloc[1]:.2f})")
print(f"  Size coef: {m2.params.iloc[2]:.3f} (t={m2.tvalues.iloc[2]:.2f})")
print(f"  R²: {m2.rsquared:.3f}")

# Spec 3: P/E ~ ROA + Small firm share
X3 = sm.add_constant(country[["roa_effect", "small_firm_share"]])
m3 = sm.OLS(country["pe_effect"], X3).fit()
print(f"\nSpec 3: P/E ~ ROA + Small firm share")
print(f"  ROA coef:        {m3.params.iloc[1]:.3f} (t={m3.tvalues.iloc[1]:.2f})")
print(f"  Small share coef:{m3.params.iloc[2]:.3f} (t={m3.tvalues.iloc[2]:.2f})")
print(f"  R²: {m3.rsquared:.3f}")

# Same for M/B
X2b = sm.add_constant(country[["roa_effect", "mean_log_at"]])
m2b = sm.OLS(country["mb_effect"], X2b).fit()
print(f"\nSpec 2b: M/B ~ ROA + Size")
print(f"  ROA coef:  {m2b.params.iloc[1]:.3f} (t={m2b.tvalues.iloc[1]:.2f})")
print(f"  Size coef: {m2b.params.iloc[2]:.3f} (t={m2b.tvalues.iloc[2]:.2f})")
print(f"  R²: {m2b.rsquared:.3f}")


# ---------------------------------------------------------------------------
# Focus countries for labeling
# ---------------------------------------------------------------------------
FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]


def _label_points(ax, data, x_col, y_col, focus=FOCUS):
    """Label focus countries."""
    for _, row in data.iterrows():
        if row["fic"] in focus:
            ax.annotate(
                row["fic"], (row[x_col], row[y_col]),
                fontsize=7, fontweight="bold", ha="left", va="bottom",
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
# Plot 1: Country avg firm size vs P/E effect
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(country["mean_log_at"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "mean_log_at", "pe_effect")
_fit_line(ax, country["mean_log_at"].values, country["pe_effect"].values)
r_val = country["mean_log_at"].corr(country["pe_effect"])
ax.set_xlabel("Average log(Assets) in country", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Country Firm Size vs P/E Effect (r = {r_val:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_size_vs_pe.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_size_vs_pe.png", dpi=300)
plt.close(fig)
print("\nSaved scatter_size_vs_pe")

# ---------------------------------------------------------------------------
# Plot 2: Country avg firm size vs ROA effect
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(country["mean_log_at"], country["roa_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "mean_log_at", "roa_effect")
_fit_line(ax, country["mean_log_at"].values, country["roa_effect"].values)
r_val = country["mean_log_at"].corr(country["roa_effect"])
ax.set_xlabel("Average log(Assets) in country", fontsize=10)
ax.set_ylabel("Country ROA effect (vs USA)", fontsize=10)
ax.set_title(f"Country Firm Size vs ROA Effect (r = {r_val:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_size_vs_roa.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_size_vs_roa.png", dpi=300)
plt.close(fig)
print("Saved scatter_size_vs_roa")

# ---------------------------------------------------------------------------
# Plot 3: ROA vs P/E after partialing out size
# ---------------------------------------------------------------------------
country["roa_resid"] = roa_resid.values
country["pe_resid"] = pe_resid.values

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: raw
ax = axes[0]
ax.scatter(country["roa_effect"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "roa_effect", "pe_effect")
_fit_line(ax, country["roa_effect"].values, country["pe_effect"].values)
r_raw = country["roa_effect"].corr(country["pe_effect"])
ax.set_xlabel("Country ROA effect", fontsize=10)
ax.set_ylabel("Country P/E effect", fontsize=10)
ax.set_title(f"Raw (r = {r_raw:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

# Right: after partialing out size
ax = axes[1]
ax.scatter(country["roa_resid"], country["pe_resid"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "roa_resid", "pe_resid")
_fit_line(ax, country["roa_resid"].values, country["pe_resid"].values)
ax.set_xlabel("ROA effect (residual after size)", fontsize=10)
ax.set_ylabel("P/E effect (residual after size)", fontsize=10)
ax.set_title(f"After partialing out avg firm size (r = {r_partial_pe:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Does Firm Size Explain the Negative ROA–P/E Correlation?", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_roa_vs_pe_size_resid.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_roa_vs_pe_size_resid.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_roa_vs_pe_size_resid")

# ---------------------------------------------------------------------------
# Plot 4: Small firm share vs P/E
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(country["small_firm_share"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "small_firm_share", "pe_effect")
_fit_line(ax, country["small_firm_share"].values, country["pe_effect"].values)
r_val = country["small_firm_share"].corr(country["pe_effect"])
ax.set_xlabel("Share of firms below global median size", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Small Firm Share vs P/E Effect (r = {r_val:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_smallshare_vs_pe.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_smallshare_vs_pe.png", dpi=300)
plt.close(fig)
print("Saved scatter_smallshare_vs_pe")

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
with open(OUT_DIR / "size_roa_valuation_results.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("SIZE, ROA, AND VALUATION: COUNTRY-LEVEL ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write("CORRELATIONS:\n")
    f.write(f"  Size vs P/E effect:   r = {country['mean_log_at'].corr(country['pe_effect']):+.3f}\n")
    f.write(f"  Size vs M/B effect:   r = {country['mean_log_at'].corr(country['mb_effect']):+.3f}\n")
    f.write(f"  Size vs ROA effect:   r = {country['mean_log_at'].corr(country['roa_effect']):+.3f}\n")
    f.write(f"  ROA vs P/E (raw):     r = {country['roa_effect'].corr(country['pe_effect']):+.3f}\n")
    f.write(f"  ROA vs P/E (|size):   r = {r_partial_pe:+.3f}\n")
    f.write(f"  ROA vs M/B (raw):     r = {country['roa_effect'].corr(country['mb_effect']):+.3f}\n")
    f.write(f"  ROA vs M/B (|size):   r = {r_partial_mb:+.3f}\n")
    f.write(f"  Small share vs P/E:   r = {country['small_firm_share'].corr(country['pe_effect']):+.3f}\n\n")

    f.write("REGRESSIONS:\n\n")
    f.write(f"Spec 1: P/E ~ ROA\n")
    f.write(f"  ROA: {m1.params.iloc[1]:.3f} (t={m1.tvalues.iloc[1]:.2f}), R²={m1.rsquared:.3f}\n\n")
    f.write(f"Spec 2: P/E ~ ROA + avg log(Assets)\n")
    f.write(f"  ROA:  {m2.params.iloc[1]:.3f} (t={m2.tvalues.iloc[1]:.2f})\n")
    f.write(f"  Size: {m2.params.iloc[2]:.3f} (t={m2.tvalues.iloc[2]:.2f})\n")
    f.write(f"  R²={m2.rsquared:.3f}\n\n")
    f.write(f"Spec 3: P/E ~ ROA + small firm share\n")
    f.write(f"  ROA:         {m3.params.iloc[1]:.3f} (t={m3.tvalues.iloc[1]:.2f})\n")
    f.write(f"  Small share: {m3.params.iloc[2]:.3f} (t={m3.tvalues.iloc[2]:.2f})\n")
    f.write(f"  R²={m3.rsquared:.3f}\n\n")

    f.write(f"N countries: {len(country)}\n")

print("\nSaved size_roa_valuation_results.txt")
print("\nDone.")
