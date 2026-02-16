"""
lag_roa_test.py -- Test if the negative ROA-P/E correlation is mechanical.

The concern: P/E = MktCap / NI, ROA = NI / Assets.
NI appears in both → temporary NI shock raises ROA and lowers P/E.

Solution: use ROA_{t-1} instead of ROA_t. Last year's NI is not in
this year's P/E denominator, so any remaining correlation is real.

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT_DIR = Path("output/regressions")

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
    "highlight": "#8e44ad",
}

FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]

# ---------------------------------------------------------------------------
# 1. Load panel and create lagged ROA
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs")

# Sort and create lag
df = df.sort_values(["gvkey", "fyear"])
df["lag_roa"] = df.groupby("gvkey")["roa"].shift(1)

n_lag = df["lag_roa"].notna().sum()
print(f"Observations with lagged ROA: {n_lag:,} ({100*n_lag/len(df):.1f}%)")

# ---------------------------------------------------------------------------
# 2. Year-by-year regressions: extract country effects on lag_roa
# ---------------------------------------------------------------------------
# Same spec as for ROA: lag_roa ~ log_at + C(fic, Treatment(reference='USA')) + C(ggroup)
print("\nRunning year-by-year regressions for lagged ROA...")

results_lag_roa = []
years = sorted(df["fyear"].dropna().unique())

for yr in years:
    sub = df[(df["fyear"] == yr)].dropna(subset=["lag_roa", "log_at", "ggroup"]).copy()
    sub["fic"] = sub["fic"].astype(str)
    sub["ggroup"] = sub["ggroup"].astype(int).astype(str)

    n_countries = sub["fic"].nunique()
    if n_countries < 5 or len(sub) < 500:
        continue
    if "USA" not in sub["fic"].values:
        continue

    try:
        model = smf.ols(
            'lag_roa ~ log_at + C(fic, Treatment(reference="USA")) + C(ggroup)',
            data=sub,
        ).fit()

        # Extract country effects: param names like C(fic, ...)[T.ARE]
        import re
        for param, coef in model.params.items():
            m = re.search(r"\[T\.(\w+)\]", param)
            if m and "fic" in param:
                fic = m.group(1)
                se = model.bse[param]
                results_lag_roa.append({
                    "fyear": yr, "fic": fic,
                    "country_effect": coef, "se": se,
                    "n_obs": len(sub),
                })

        # USA = 0 (reference)
        results_lag_roa.append({
            "fyear": yr, "fic": "USA",
            "country_effect": 0.0, "se": 0.0,
            "n_obs": len(sub),
        })
    except Exception as e:
        print(f"  Year {yr}: {e}")
        continue

ce_lag_roa = pd.DataFrame(results_lag_roa)
print(f"Country-year effects: {len(ce_lag_roa)} (lag ROA)")

# ---------------------------------------------------------------------------
# 3. Average country effects
# ---------------------------------------------------------------------------
# Load existing P/E and contemporaneous ROA effects
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")
avg_lag_roa = ce_lag_roa.groupby("fic")["country_effect"].mean().rename("lag_roa_effect")

country = pd.DataFrame(avg_pe).join([avg_roa, avg_lag_roa], how="inner")
country = country.reset_index()
print(f"\nCountries with all three effects: {len(country)}")

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
r_contemp = country["roa_effect"].corr(country["pe_effect"])
r_lagged = country["lag_roa_effect"].corr(country["pe_effect"])
r_roa_lag = country["roa_effect"].corr(country["lag_roa_effect"])

print(f"\n=== CORRELATIONS ===")
print(f"  Contemporaneous ROA vs P/E: r = {r_contemp:+.2f}")
print(f"  Lagged ROA vs P/E:          r = {r_lagged:+.2f}")
print(f"  ROA vs Lag ROA:              r = {r_roa_lag:+.2f}  (persistence)")

# ---------------------------------------------------------------------------
# 5. Also check at country-year level (more power)
# ---------------------------------------------------------------------------
# Merge P/E and lag ROA effects at the country-year level
cy_pe = ce_pe[["fyear", "fic", "country_effect"]].rename(columns={"country_effect": "pe_cy"})
cy_lag = ce_lag_roa[["fyear", "fic", "country_effect"]].rename(columns={"country_effect": "lag_roa_cy"})
cy_roa = ce_roa[["fyear", "fic", "country_effect"]].rename(columns={"country_effect": "roa_cy"})

cy = cy_pe.merge(cy_lag, on=["fyear", "fic"], how="inner")
cy = cy.merge(cy_roa, on=["fyear", "fic"], how="inner")

r_cy_contemp = cy["roa_cy"].corr(cy["pe_cy"])
r_cy_lagged = cy["lag_roa_cy"].corr(cy["pe_cy"])
print(f"\n  Country-year level:")
print(f"    Contemporaneous ROA vs P/E: r = {r_cy_contemp:+.2f}  (n={len(cy)})")
print(f"    Lagged ROA vs P/E:          r = {r_cy_lagged:+.2f}  (n={len(cy)})")

# ---------------------------------------------------------------------------
# 6. Scatter plots
# ---------------------------------------------------------------------------
def _label_points(ax, data, x_col, y_col, focus=FOCUS):
    for _, row in data.iterrows():
        if row["fic"] in focus:
            ax.annotate(
                row["fic"], (row[x_col], row[y_col]),
                fontsize=7, fontweight="bold", ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points",
                color=PALETTE["primary"],
            )


def _fit_line(ax, x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    z = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xline, np.polyval(z, xline), color=PALETTE["alert"],
            linewidth=1.5, linestyle="--", alpha=0.8)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: contemporaneous ROA vs P/E
ax = axes[0]
ax.scatter(country["roa_effect"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "roa_effect", "pe_effect")
_fit_line(ax, country["roa_effect"].values, country["pe_effect"].values)
ax.set_xlabel("Country ROA$_t$ effect", fontsize=10)
ax.set_ylabel("Country P/E$_t$ effect", fontsize=10)
ax.set_title(f"Contemporaneous ROA (r = {r_contemp:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

# Right: lagged ROA vs P/E
ax = axes[1]
ax.scatter(country["lag_roa_effect"], country["pe_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "lag_roa_effect", "pe_effect")
_fit_line(ax, country["lag_roa_effect"].values, country["pe_effect"].values)
ax.set_xlabel("Country ROA$_{t-1}$ effect", fontsize=10)
ax.set_ylabel("Country P/E$_t$ effect", fontsize=10)
ax.set_title(f"Lagged ROA (r = {r_lagged:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Is the Negative ROA–P/E Correlation Mechanical?", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_lag_roa_vs_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_lag_roa_vs_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_lag_roa_vs_pe")

# ---------------------------------------------------------------------------
# 7. Print key countries
# ---------------------------------------------------------------------------
print("\n=== KEY COUNTRIES ===")
print(f"{'Country':>5s}  {'ROA_t':>8s}  {'ROA_t-1':>8s}  {'P/E_t':>8s}")
for _, row in country.sort_values("pe_effect", ascending=False).head(15).iterrows():
    print(f"{row['fic']:>5s}  {row['roa_effect']:+8.3f}  {row['lag_roa_effect']:+8.3f}  {row['pe_effect']:+8.3f}")

print("\nDone.")
