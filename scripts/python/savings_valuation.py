"""
savings_valuation.py -- Gross savings / GDP as a predictor of P/E.

Savings/GDP is a significant cross-country predictor of P/E (t ≈ 2).
This script runs a proper analysis: robustness, M/B, outlier checks,
scatter plots, and time variation.

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import wbgapi as wb

np.random.seed(20260216)

OUT_DIR = Path("output/regressions")

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
    "highlight": "#8e44ad",
}

FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]

# ---------------------------------------------------------------------------
# 1. Fetch World Bank data
# ---------------------------------------------------------------------------
WB_CODES = [
    "USA", "GBR", "JPN", "CHN", "DEU", "FRA", "IND", "KOR", "BRA", "AUS",
    "CAN", "ITA", "ESP", "NLD", "CHE", "SWE", "NOR", "DNK", "FIN", "BEL",
    "AUT", "IRL", "PRT", "GRC", "POL", "CZE", "HUN", "ROU", "BGR", "HRV",
    "SVN", "EST", "LTU", "RUS", "UKR", "TUR", "ISR", "SAU", "ARE", "QAT",
    "KWT", "BHR", "OMN", "EGY", "ZAF", "NGA", "KEN", "MAR", "TUN", "MYS",
    "SGP", "THA", "IDN", "PHL", "VNM", "HKG", "PAK", "BGD", "LKA", "MEX",
    "ARG", "CHL", "COL", "PER", "NZL", "JOR", "MUS", "ZWE", "JAM", "LUX",
    "MLT", "CYP", "SRB",
]

print("Fetching World Bank data...")

# Gross domestic savings (% of GDP) - NY.GDS.TOTL.ZS
raw_sav = wb.data.DataFrame("NY.GDS.TOTL.ZS", economy=WB_CODES, time=range(2007, 2024))
avg_sav = raw_sav.mean(axis=1)
avg_sav.index = avg_sav.index.get_level_values("economy")
savings_avg_map = dict(zip(avg_sav.index, avg_sav.values))

# Also get year-by-year savings for time-varying analysis
sav_yearly = raw_sav.copy()
sav_yearly.index = sav_yearly.index.get_level_values("economy")
# Reshape: rows=country, cols=years
sav_yearly.columns = [int(c.replace("YR", "")) for c in sav_yearly.columns]

# Gross savings (% of GNI) - NY.GNS.ICTR.ZS (alternative measure)
raw_gns = wb.data.DataFrame("NY.GNS.ICTR.ZS", economy=WB_CODES, time=range(2007, 2024))
avg_gns = raw_gns.mean(axis=1)
avg_gns.index = avg_gns.index.get_level_values("economy")
gns_map = dict(zip(avg_gns.index, avg_gns.values))

# GDP per capita for a control
raw_gdppc = wb.data.DataFrame("NY.GDP.PCAP.PP.CD", economy=WB_CODES, time=range(2007, 2024))
avg_gdppc = raw_gdppc.mean(axis=1)
avg_gdppc.index = avg_gdppc.index.get_level_values("economy")
gdppc_map = dict(zip(avg_gdppc.index, avg_gdppc.values))

print(f"  Savings/GDP: {sum(1 for v in savings_avg_map.values() if np.isfinite(v))} countries")
print(f"  Savings/GNI: {sum(1 for v in gns_map.values() if np.isfinite(v))} countries")

# ---------------------------------------------------------------------------
# 2. Load country effects
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")
ce_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")
avg_mb = ce_mb.groupby("fic")["country_effect"].mean().rename("mb_effect")

# Build country-level dataset
country = pd.DataFrame(avg_pe).join([avg_roa, avg_mb], how="outer")
country.index.name = "fic"

# Add savings
country["savings_gdp"] = country.index.map(savings_avg_map)
country["savings_gni"] = country.index.map(gns_map)
country["log_gdppc"] = np.log(country.index.map(gdppc_map))

country = country.reset_index()
sub = country.dropna(subset=["savings_gdp", "pe_effect"])
print(f"\nCountries with savings + P/E: {len(sub)}")

# ---------------------------------------------------------------------------
# 3. Main correlations and regressions
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVINGS/GDP AND VALUATIONS")
print("=" * 80)

r_pe = sub["savings_gdp"].corr(sub["pe_effect"])
r_mb = sub["savings_gdp"].corr(sub["mb_effect"])
r_roa = sub["savings_gdp"].corr(sub["roa_effect"])
print(f"\n  Savings/GDP vs P/E:  r = {r_pe:+.2f}  (n={len(sub)})")
print(f"  Savings/GDP vs M/B:  r = {r_mb:+.2f}")
print(f"  Savings/GDP vs ROA:  r = {r_roa:+.2f}")

# --- Regressions ---
print("\n--- P/E regressions ---")

# Spec 1: P/E ~ savings
X1 = sm.add_constant(sub["savings_gdp"])
m1 = sm.OLS(sub["pe_effect"], X1).fit()
print(f"\nP/E ~ Savings/GDP")
print(f"  Savings: {m1.params.iloc[1]:+.4f} (t={m1.tvalues.iloc[1]:+.2f}), R²={m1.rsquared:.3f}")

# Spec 2: P/E ~ savings + GDP per capita
sub2 = sub.dropna(subset=["log_gdppc"])
X2 = sm.add_constant(sub2[["savings_gdp", "log_gdppc"]])
m2 = sm.OLS(sub2["pe_effect"], X2).fit()
print(f"\nP/E ~ Savings/GDP + log(GDP per capita)")
print(f"  Savings: {m2.params.iloc[1]:+.4f} (t={m2.tvalues.iloc[1]:+.2f})")
print(f"  GDP/cap: {m2.params.iloc[2]:+.4f} (t={m2.tvalues.iloc[2]:+.2f})")
print(f"  R²={m2.rsquared:.3f}")

# Spec 3: P/E ~ savings + ROA
X3 = sm.add_constant(sub[["savings_gdp", "roa_effect"]])
m3 = sm.OLS(sub["pe_effect"], X3).fit()
print(f"\nP/E ~ Savings/GDP + ROA")
print(f"  Savings: {m3.params.iloc[1]:+.4f} (t={m3.tvalues.iloc[1]:+.2f})")
print(f"  ROA:     {m3.params.iloc[2]:+.2f} (t={m3.tvalues.iloc[2]:+.2f})")
print(f"  R²={m3.rsquared:.3f}")

# Spec 4: P/E ~ savings + ROA + GDP/cap
sub4 = sub.dropna(subset=["log_gdppc"])
X4 = sm.add_constant(sub4[["savings_gdp", "roa_effect", "log_gdppc"]])
m4 = sm.OLS(sub4["pe_effect"], X4).fit()
print(f"\nP/E ~ Savings/GDP + ROA + log(GDP per capita)")
print(f"  Savings: {m4.params.iloc[1]:+.4f} (t={m4.tvalues.iloc[1]:+.2f})")
print(f"  ROA:     {m4.params.iloc[2]:+.2f} (t={m4.tvalues.iloc[2]:+.2f})")
print(f"  GDP/cap: {m4.params.iloc[3]:+.4f} (t={m4.tvalues.iloc[3]:+.2f})")
print(f"  R²={m4.rsquared:.3f}")

# --- M/B regressions ---
print("\n--- M/B regressions ---")
sub_mb = sub.dropna(subset=["mb_effect"])

X_mb1 = sm.add_constant(sub_mb["savings_gdp"])
m_mb1 = sm.OLS(sub_mb["mb_effect"], X_mb1).fit()
print(f"\nM/B ~ Savings/GDP")
print(f"  Savings: {m_mb1.params.iloc[1]:+.4f} (t={m_mb1.tvalues.iloc[1]:+.2f}), R²={m_mb1.rsquared:.3f}")

X_mb2 = sm.add_constant(sub_mb[["savings_gdp", "roa_effect"]])
m_mb2 = sm.OLS(sub_mb["mb_effect"], X_mb2).fit()
print(f"\nM/B ~ Savings/GDP + ROA")
print(f"  Savings: {m_mb2.params.iloc[1]:+.4f} (t={m_mb2.tvalues.iloc[1]:+.2f})")
print(f"  ROA:     {m_mb2.params.iloc[2]:+.2f} (t={m_mb2.tvalues.iloc[2]:+.2f})")
print(f"  R²={m_mb2.rsquared:.3f}")

# ---------------------------------------------------------------------------
# 4. Alternative savings measure (Savings/GNI)
# ---------------------------------------------------------------------------
print("\n--- Alternative: Savings/GNI ---")
sub_gni = sub.dropna(subset=["savings_gni"])
r_gni_pe = sub_gni["savings_gni"].corr(sub_gni["pe_effect"])
X_gni = sm.add_constant(sub_gni["savings_gni"])
m_gni = sm.OLS(sub_gni["pe_effect"], X_gni).fit()
print(f"  r(Savings/GNI, P/E) = {r_gni_pe:+.2f}  (n={len(sub_gni)})")
print(f"  t = {m_gni.tvalues.iloc[1]:+.2f}")

# ---------------------------------------------------------------------------
# 5. Robustness: drop Gulf states (extreme savers, oil economies)
# ---------------------------------------------------------------------------
print("\n--- Robustness: excluding Gulf states ---")
gulf = {"QAT", "SAU", "ARE", "KWT", "BHR", "OMN"}
sub_nogulf = sub[~sub["fic"].isin(gulf)]
r_nogulf = sub_nogulf["savings_gdp"].corr(sub_nogulf["pe_effect"])
X_ng = sm.add_constant(sub_nogulf["savings_gdp"])
m_ng = sm.OLS(sub_nogulf["pe_effect"], X_ng).fit()
print(f"  r = {r_nogulf:+.2f}  (n={len(sub_nogulf)})")
print(f"  t = {m_ng.tvalues.iloc[1]:+.2f}")

# Excluding Gulf + small island economies
small = {"MLT", "CYP", "MUS", "JAM", "BMU", "CYM", "MHL", "LUX", "BHR", "QAT"}
sub_clean = sub[~sub["fic"].isin(gulf | small)]
r_clean = sub_clean["savings_gdp"].corr(sub_clean["pe_effect"])
X_cl = sm.add_constant(sub_clean["savings_gdp"])
m_cl = sm.OLS(sub_clean["pe_effect"], X_cl).fit()
print(f"\n  Excl Gulf + small islands: r = {r_clean:+.2f}  (n={len(sub_clean)})")
print(f"  t = {m_cl.tvalues.iloc[1]:+.2f}")

# Robustness: OECD only
oecd = {"USA", "GBR", "JPN", "DEU", "FRA", "ITA", "CAN", "AUS", "KOR",
        "ESP", "NLD", "CHE", "SWE", "NOR", "DNK", "FIN", "BEL", "AUT",
        "IRL", "PRT", "GRC", "POL", "CZE", "HUN", "SVN", "EST", "LTU",
        "ISR", "NZL", "CHL", "COL", "MEX", "TUR"}
sub_oecd = sub[sub["fic"].isin(oecd)]
r_oecd = sub_oecd["savings_gdp"].corr(sub_oecd["pe_effect"])
X_oecd = sm.add_constant(sub_oecd["savings_gdp"])
m_oecd = sm.OLS(sub_oecd["pe_effect"], X_oecd).fit()
print(f"\n  OECD only: r = {r_oecd:+.2f}  (n={len(sub_oecd)})")
print(f"  t = {m_oecd.tvalues.iloc[1]:+.2f}")

# ---------------------------------------------------------------------------
# 6. Time-varying analysis: country-year level
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TIME-VARYING ANALYSIS (COUNTRY-YEAR LEVEL)")
print("=" * 80)

# Reshape savings to long format
sav_long = sav_yearly.stack().reset_index()
sav_long.columns = ["fic", "fyear", "savings_gdp"]

# Merge with country-year effects
cy_pe = ce_pe[["fyear", "fic", "country_effect"]].rename(columns={"country_effect": "pe_cy"})
cy_roa = ce_roa[["fyear", "fic", "country_effect"]].rename(columns={"country_effect": "roa_cy"})

cy = cy_pe.merge(cy_roa, on=["fyear", "fic"], how="inner")
cy = cy.merge(sav_long, on=["fyear", "fic"], how="inner")
cy = cy.dropna()
print(f"Country-year observations: {len(cy)}")

r_cy_pe = cy["savings_gdp"].corr(cy["pe_cy"])
r_cy_roa = cy["savings_gdp"].corr(cy["roa_cy"])
print(f"  Savings vs P/E (country-year):  r = {r_cy_pe:+.2f}  (n={len(cy)})")
print(f"  Savings vs ROA (country-year):  r = {r_cy_roa:+.2f}")

# With year FE (demean by year)
cy["pe_dm"] = cy.groupby("fyear")["pe_cy"].transform(lambda x: x - x.mean())
cy["roa_dm"] = cy.groupby("fyear")["roa_cy"].transform(lambda x: x - x.mean())
cy["sav_dm"] = cy.groupby("fyear")["savings_gdp"].transform(lambda x: x - x.mean())

r_dm = cy["sav_dm"].corr(cy["pe_dm"])
print(f"  Year-demeaned: savings vs P/E:  r = {r_dm:+.2f}")

# ---------------------------------------------------------------------------
# 7. Scatter plots
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


# Main scatter: Savings vs P/E and M/B
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(sub["savings_gdp"], sub["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, sub, "savings_gdp", "pe_effect")
_fit_line(ax, sub["savings_gdp"].values, sub["pe_effect"].values)
ax.set_xlabel("Gross Domestic Savings (% of GDP)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Savings vs P/E (r = {r_pe:+.2f}, t = {m1.tvalues.iloc[1]:.1f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
ax.scatter(sub_mb["savings_gdp"], sub_mb["mb_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, sub_mb, "savings_gdp", "mb_effect")
_fit_line(ax, sub_mb["savings_gdp"].values, sub_mb["mb_effect"].values)
r_mb_val = sub_mb["savings_gdp"].corr(sub_mb["mb_effect"])
ax.set_xlabel("Gross Domestic Savings (% of GDP)", fontsize=10)
ax.set_ylabel("Country M/B effect (vs USA)", fontsize=10)
ax.set_title(f"Savings vs M/B (r = {r_mb_val:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("National Savings Rate and Cross-Country Valuations",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_savings_vs_valuation.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_savings_vs_valuation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_savings_vs_valuation")

print("\nDone.")
