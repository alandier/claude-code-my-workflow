"""
equity_market_depth.py -- Equity market depth measures and the ROA-P/E puzzle.

Measures of how much of the economy is captured by listed equity markets:
  1. Market Cap / GDP (already tested, re-run for consistency)
  2. Number of listed firms / GDP (market breadth)
  3. Listed firms' total assets / GDP (coverage of real economy)
  4. Listed firms' total assets / GDP, in USD (same but currency-comparable)
  5. Number of listed firms per million population

Then test: do these measures correlate with P/E effects, ROA effects,
and do they absorb the ROA-P/E correlation?

Author: Augustin Landier, HEC Paris
"""

import re
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
# 1. Load panel data
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs, {df['fic'].nunique()} countries")

# ---------------------------------------------------------------------------
# 2. Fetch World Bank data
# ---------------------------------------------------------------------------
WB_CODE_MAP = {
    "USA": "USA", "GBR": "GBR", "JPN": "JPN", "CHN": "CHN",
    "DEU": "DEU", "FRA": "FRA", "IND": "IND", "KOR": "KOR",
    "BRA": "BRA", "AUS": "AUS", "CAN": "CAN", "ITA": "ITA",
    "ESP": "ESP", "NLD": "NLD", "CHE": "CHE", "SWE": "SWE",
    "NOR": "NOR", "DNK": "DNK", "FIN": "FIN", "BEL": "BEL",
    "AUT": "AUT", "IRL": "IRL", "PRT": "PRT", "GRC": "GRC",
    "POL": "POL", "CZE": "CZE", "HUN": "HUN", "ROU": "ROU",
    "BGR": "BGR", "HRV": "HRV", "SVN": "SVN", "EST": "EST",
    "LTU": "LTU", "RUS": "RUS", "UKR": "UKR",
    "TUR": "TUR", "ISR": "ISR", "SAU": "SAU", "ARE": "ARE",
    "QAT": "QAT", "KWT": "KWT", "BHR": "BHR", "OMN": "OMN",
    "EGY": "EGY", "ZAF": "ZAF", "NGA": "NGA", "KEN": "KEN",
    "MAR": "MAR", "TUN": "TUN", "MYS": "MYS", "SGP": "SGP",
    "THA": "THA", "IDN": "IDN", "PHL": "PHL", "VNM": "VNM",
    "TWN": "TWN", "HKG": "HKG", "PAK": "PAK", "BGD": "BGD",
    "LKA": "LKA", "MEX": "MEX", "ARG": "ARG", "CHL": "CHL",
    "COL": "COL", "PER": "PER", "NZL": "NZL", "JOR": "JOR",
    "MUS": "MUS", "ZWE": "ZWE", "JAM": "JAM",
    "CYM": None, "BMU": None, "VGB": None, "GGY": None,
    "JEY": None, "IMN": None, "LUX": "LUX", "MHL": None,
    "MLT": "MLT", "CYP": "CYP", "SRB": "SRB", "PSE": None,
}
EUROZONE = {"DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL", "PRT",
            "GRC", "FIN", "LUX", "SVN", "CYP", "MLT", "EST", "LTU", "HRV"}

fic_to_wb = {k: v for k, v in WB_CODE_MAP.items() if v is not None}
valid_wb = list(set(fic_to_wb.values()))

print("Fetching World Bank data...")

# Exchange rates (for USD conversion)
raw_fx = wb.data.DataFrame("PA.NUS.FCRF", economy=valid_wb, time=range(2007, 2024))
avg_fx = raw_fx.mean(axis=1)
avg_fx.index = avg_fx.index.get_level_values("economy")
fx_map = dict(zip(avg_fx.index, avg_fx.values))
eur_rate = fx_map.get("DEU", 1.0)
for c in EUROZONE:
    wb_c = fic_to_wb.get(c)
    if wb_c:
        fx_map[wb_c] = eur_rate
fx_map["USA"] = 1.0

# GDP (current USD)
raw_gdp = wb.data.DataFrame("NY.GDP.MKTP.CD", economy=valid_wb, time=range(2007, 2024))
avg_gdp = raw_gdp.mean(axis=1)
avg_gdp.index = avg_gdp.index.get_level_values("economy")
gdp_map = dict(zip(avg_gdp.index, avg_gdp.values))  # in USD

# Population
raw_pop = wb.data.DataFrame("SP.POP.TOTL", economy=valid_wb, time=range(2007, 2024))
avg_pop = raw_pop.mean(axis=1)
avg_pop.index = avg_pop.index.get_level_values("economy")
pop_map = dict(zip(avg_pop.index, avg_pop.values))

# Market cap / GDP (already have but re-fetch for consistency)
raw_mktcap = wb.data.DataFrame("CM.MKT.LCAP.GD.ZS", economy=valid_wb, time=range(2007, 2024))
avg_mktcap = raw_mktcap.mean(axis=1)
avg_mktcap.index = avg_mktcap.index.get_level_values("economy")
mktcap_gdp_map = dict(zip(avg_mktcap.index, avg_mktcap.values))

# Number of listed companies
raw_listed = wb.data.DataFrame("CM.MKT.LDOM.NO", economy=valid_wb, time=range(2007, 2024))
avg_listed = raw_listed.mean(axis=1)
avg_listed.index = avg_listed.index.get_level_values("economy")
listed_map = dict(zip(avg_listed.index, avg_listed.values))

print(f"  Exchange rates: {len(fx_map)} countries")
print(f"  GDP: {len(gdp_map)} countries")
print(f"  Population: {len(pop_map)} countries")
print(f"  Market cap/GDP: {sum(1 for v in mktcap_gdp_map.values() if np.isfinite(v))} countries")
print(f"  Listed companies: {sum(1 for v in listed_map.values() if np.isfinite(v))} countries")

# ---------------------------------------------------------------------------
# 3. Compute country-level measures from Compustat
# ---------------------------------------------------------------------------
# Convert assets to USD
df["wb_code"] = df["fic"].map(fic_to_wb)
df["fx_rate"] = df["wb_code"].map(fx_map)
df["at_usd"] = df["at"] / df["fx_rate"]  # millions USD

# Per country-year: total assets of listed firms, number of firms
cy = df.groupby(["fic", "fyear"], observed=True).agg(
    total_at_usd=("at_usd", "sum"),
    n_firms=("gvkey", "nunique"),
    total_at_local=("at", "sum"),
).reset_index()

# Average across years per country
country_stats = cy.groupby("fic").agg(
    avg_total_at_usd=("total_at_usd", "mean"),  # avg total listed assets (USD M)
    avg_n_firms=("n_firms", "mean"),  # avg number of listed firms
    avg_total_at_local=("total_at_local", "mean"),
).reset_index()

# Add WB data
country_stats["wb_code"] = country_stats["fic"].map(fic_to_wb)
country_stats["gdp_usd"] = country_stats["wb_code"].map(gdp_map)  # USD
country_stats["population"] = country_stats["wb_code"].map(pop_map)
country_stats["mktcap_gdp_wb"] = country_stats["wb_code"].map(mktcap_gdp_map)
country_stats["n_listed_wb"] = country_stats["wb_code"].map(listed_map)

# Compute measures
# GDP in millions for consistency with assets
country_stats["gdp_usd_m"] = country_stats["gdp_usd"] / 1e6

# 1. Market cap / GDP (from WB) — already have
# 2. Number of listed firms / GDP (in trillions)
country_stats["firms_per_gdp"] = country_stats["avg_n_firms"] / (country_stats["gdp_usd"] / 1e12)

# 3. Listed firms' total assets / GDP (both in USD)
country_stats["assets_gdp"] = country_stats["avg_total_at_usd"] / country_stats["gdp_usd_m"]

# 4. Number of firms per million population
country_stats["firms_per_pop"] = country_stats["avg_n_firms"] / (country_stats["population"] / 1e6)

# 5. Log versions for regressions
country_stats["log_assets_gdp"] = np.log(country_stats["assets_gdp"])
country_stats["log_firms_per_gdp"] = np.log(country_stats["firms_per_gdp"])
country_stats["log_firms_per_pop"] = np.log(country_stats["firms_per_pop"])
country_stats["log_mktcap_gdp"] = np.log(country_stats["mktcap_gdp_wb"] / 100)  # convert from % to ratio

# ---------------------------------------------------------------------------
# 4. Load country effects
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")

country = country_stats.set_index("fic").join([avg_pe, avg_roa], how="inner")
country = country.reset_index()

print(f"\nCountries with all data: {len(country)}")

# ---------------------------------------------------------------------------
# 5. Print key countries
# ---------------------------------------------------------------------------
print(f"\n{'Country':>5s}  {'MktCap/GDP':>10s}  {'Assets/GDP':>10s}  {'Firms/GDP':>10s}  "
      f"{'Firms/Pop':>10s}  {'P/E':>6s}  {'ROA':>6s}")
for _, row in country.sort_values("pe_effect", ascending=False).head(20).iterrows():
    mktcap = f"{row['mktcap_gdp_wb']:.0f}%" if np.isfinite(row.get('mktcap_gdp_wb', np.nan)) else "n/a"
    print(f"{row['fic']:>5s}  {mktcap:>10s}  {row['assets_gdp']:>10.2f}  "
          f"{row['firms_per_gdp']:>10.1f}  {row['firms_per_pop']:>10.1f}  "
          f"{row['pe_effect']:>+6.2f}  {row['roa_effect']:>+6.3f}")

# ---------------------------------------------------------------------------
# 6. Correlations
# ---------------------------------------------------------------------------
measures = {
    "Market Cap / GDP (WB)": "log_mktcap_gdp",
    "Listed Assets / GDP": "log_assets_gdp",
    "N Firms / GDP": "log_firms_per_gdp",
    "N Firms / Pop (M)": "log_firms_per_pop",
}

print("\n" + "=" * 80)
print("CORRELATIONS WITH P/E AND ROA EFFECTS")
print("=" * 80)
print(f"\n{'Measure':<25s}  {'vs P/E':>7s}  {'vs ROA':>7s}  "
      f"{'Partial r':>10s}  {'N':>4s}")
print("-" * 60)

results = []
for label, col in measures.items():
    sub = country[["pe_effect", "roa_effect", col]].dropna()
    n = len(sub)
    r_pe = sub[col].corr(sub["pe_effect"])
    r_roa = sub[col].corr(sub["roa_effect"])

    # Partial r(ROA, P/E | measure)
    X = sm.add_constant(sub[col])
    roa_resid = sm.OLS(sub["roa_effect"], X).fit().resid
    pe_resid = sm.OLS(sub["pe_effect"], X).fit().resid
    r_partial = np.corrcoef(roa_resid, pe_resid)[0, 1]

    print(f"{label:<25s}  {r_pe:>+7.2f}  {r_roa:>+7.2f}  {r_partial:>+10.2f}  {n:>4d}")
    results.append({
        "label": label, "col": col,
        "r_pe": r_pe, "r_roa": r_roa, "r_partial": r_partial, "n": n,
    })

# Raw correlation for reference
r_raw = country["roa_effect"].corr(country["pe_effect"])
print(f"\n{'Raw ROA vs P/E':<25s}  {'':>7s}  {'':>7s}  {r_raw:>+10.2f}  {len(country):>4d}")

# ---------------------------------------------------------------------------
# 7. Multiple regression: which measures absorb the most?
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("REGRESSIONS: P/E EFFECT ~ ROA EFFECT + MARKET DEPTH MEASURES")
print("=" * 80)

# Spec 1: P/E ~ ROA
sub_all = country[["pe_effect", "roa_effect"] + [v for v in measures.values()]].dropna()
print(f"\nCommon sample: {len(sub_all)} countries")

X1 = sm.add_constant(sub_all["roa_effect"])
m1 = sm.OLS(sub_all["pe_effect"], X1).fit()
print(f"\nSpec 1: P/E ~ ROA")
print(f"  ROA: {m1.params.iloc[1]:.2f} (t={m1.tvalues.iloc[1]:.2f}), R²={m1.rsquared:.3f}")

# Spec 2: P/E ~ ROA + each measure individually
for label, col in measures.items():
    X = sm.add_constant(sub_all[["roa_effect", col]])
    m = sm.OLS(sub_all["pe_effect"], X).fit()
    print(f"\nSpec: P/E ~ ROA + {label}")
    print(f"  ROA:     {m.params.iloc[1]:+.2f} (t={m.tvalues.iloc[1]:+.2f})")
    print(f"  {label[:20]:20s}: {m.params.iloc[2]:+.4f} (t={m.tvalues.iloc[2]:+.2f})")
    print(f"  R²={m.rsquared:.3f}")

# Spec 3: P/E ~ ROA + all measures
cols = list(measures.values())
X_all = sm.add_constant(sub_all[["roa_effect"] + cols])
m_all = sm.OLS(sub_all["pe_effect"], X_all).fit()
print(f"\nSpec: P/E ~ ROA + all measures")
for i, name in enumerate(["const", "ROA"] + list(measures.keys())):
    print(f"  {name:25s}: {m_all.params.iloc[i]:+.4f} (t={m_all.tvalues.iloc[i]:+.2f})")
print(f"  R²={m_all.rsquared:.3f}")

# ---------------------------------------------------------------------------
# 8. Scatter plots
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

# 2x2 scatter: each measure vs P/E effect
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for ax, (label, col) in zip(axes.flat, measures.items()):
    sub = country[[col, "pe_effect", "fic"]].dropna()
    r = sub[col].corr(sub["pe_effect"])
    ax.scatter(sub[col], sub["pe_effect"],
               color=PALETTE["accent"], s=30, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    _label_points(ax, sub, col, "pe_effect")
    _fit_line(ax, sub[col].values, sub["pe_effect"].values)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("P/E effect", fontsize=9)
    ax.set_title(f"r = {r:+.2f}", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Equity Market Depth Measures vs Country P/E Effect", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_equity_depth_vs_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_equity_depth_vs_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_equity_depth_vs_pe")

# Same for ROA
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for ax, (label, col) in zip(axes.flat, measures.items()):
    sub = country[[col, "roa_effect", "fic"]].dropna()
    r = sub[col].corr(sub["roa_effect"])
    ax.scatter(sub[col], sub["roa_effect"],
               color=PALETTE["accent"], s=30, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    _label_points(ax, sub, col, "roa_effect")
    _fit_line(ax, sub[col].values, sub["roa_effect"].values)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("ROA effect", fontsize=9)
    ax.set_title(f"r = {r:+.2f}", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Equity Market Depth Measures vs Country ROA Effect", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_equity_depth_vs_roa.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_equity_depth_vs_roa.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_equity_depth_vs_roa")

print("\nDone.")
