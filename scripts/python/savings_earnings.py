"""
savings_earnings.py -- Savings relative to corporate earnings as P/E predictor.

Three measures of savings pressure on equity valuations:
  1. Savings / GDP (baseline, from World Bank)
  2. Savings / GNI (alternative, from World Bank)
  3. Savings / Total Listed Earnings (new: how much savings chases each $ of profit)

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

EUROZONE = {"DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL", "PRT",
            "GRC", "FIN", "LUX", "SVN", "CYP", "MLT", "EST", "LTU", "HRV"}

# ---------------------------------------------------------------------------
# 1. Load panel data + exchange rates for USD conversion
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs")

print("Fetching World Bank data...")

# Exchange rates
raw_fx = wb.data.DataFrame("PA.NUS.FCRF", economy=WB_CODES, time=range(2007, 2024))
avg_fx = raw_fx.mean(axis=1)
avg_fx.index = avg_fx.index.get_level_values("economy")
fx_map = dict(zip(avg_fx.index, avg_fx.values))
eur_rate = fx_map.get("DEU", 1.0)
for c in EUROZONE:
    if c in WB_CODES:
        fx_map[c] = eur_rate
fx_map["USA"] = 1.0

# Year-by-year exchange rates for time-varying conversion
fx_yearly = raw_fx.copy()
fx_yearly.index = fx_yearly.index.get_level_values("economy")
fx_yearly.columns = [int(c.replace("YR", "")) for c in fx_yearly.columns]

# Savings (current USD) - NY.GDS.TOTL.CD
raw_sav_usd = wb.data.DataFrame("NY.GDS.TOTL.CD", economy=WB_CODES, time=range(2007, 2024))
sav_usd_yearly = raw_sav_usd.copy()
sav_usd_yearly.index = sav_usd_yearly.index.get_level_values("economy")
sav_usd_yearly.columns = [int(c.replace("YR", "")) for c in sav_usd_yearly.columns]
# Average savings in USD
avg_sav_usd = sav_usd_yearly.mean(axis=1)
sav_usd_map = dict(zip(avg_sav_usd.index, avg_sav_usd.values))

# Savings/GDP (%)
raw_sav_pct = wb.data.DataFrame("NY.GDS.TOTL.ZS", economy=WB_CODES, time=range(2007, 2024))
avg_sav_pct = raw_sav_pct.mean(axis=1)
avg_sav_pct.index = avg_sav_pct.index.get_level_values("economy")
sav_gdp_map = dict(zip(avg_sav_pct.index, avg_sav_pct.values))

# Savings/GNI (%)
raw_gns = wb.data.DataFrame("NY.GNS.ICTR.ZS", economy=WB_CODES, time=range(2007, 2024))
avg_gns = raw_gns.mean(axis=1)
avg_gns.index = avg_gns.index.get_level_values("economy")
sav_gni_map = dict(zip(avg_gns.index, avg_gns.values))

# GDP per capita (for controls)
raw_gdppc = wb.data.DataFrame("NY.GDP.PCAP.PP.CD", economy=WB_CODES, time=range(2007, 2024))
avg_gdppc = raw_gdppc.mean(axis=1)
avg_gdppc.index = avg_gdppc.index.get_level_values("economy")
gdppc_map = dict(zip(avg_gdppc.index, avg_gdppc.values))

# ---------------------------------------------------------------------------
# 2. Compute total listed earnings per country (in USD)
# ---------------------------------------------------------------------------
# Convert net income (ib) to USD using exchange rates
df["fx_rate"] = df["fic"].map(fx_map)
df["ib_usd"] = df["ib"] / df["fx_rate"]  # millions USD

# Total earnings per country-year (only positive ib firms, to match P/E sample)
# But also compute total for ALL firms for completeness
cy_earn = df.groupby(["fic", "fyear"], observed=True).agg(
    total_ib_usd=("ib_usd", "sum"),           # all firms
    total_ib_pos_usd=("ib_usd", lambda x: x[x > 0].sum()),  # profitable only
    n_firms=("gvkey", "nunique"),
    n_profitable=("ib_usd", lambda x: (x > 0).sum()),
).reset_index()

# Average across years
country_earn = cy_earn.groupby("fic").agg(
    avg_total_ib_usd=("total_ib_usd", "mean"),
    avg_total_ib_pos_usd=("total_ib_pos_usd", "mean"),
    avg_n_firms=("n_firms", "mean"),
    avg_n_profitable=("n_profitable", "mean"),
).reset_index()

print(f"Countries with earnings data: {len(country_earn)}")

# ---------------------------------------------------------------------------
# 3. Compute savings / earnings ratio
# ---------------------------------------------------------------------------
country_earn["sav_usd"] = country_earn["fic"].map(sav_usd_map)  # in USD

# Savings / total listed earnings (all firms)
# sav_usd is in USD, avg_total_ib_usd is in millions USD -> convert savings to millions
country_earn["sav_usd_m"] = country_earn["sav_usd"] / 1e6
country_earn["sav_over_earnings"] = country_earn["sav_usd_m"] / country_earn["avg_total_ib_pos_usd"]

# Also compute savings/GDP and savings/GNI
country_earn["savings_gdp"] = country_earn["fic"].map(sav_gdp_map)
country_earn["savings_gni"] = country_earn["fic"].map(sav_gni_map)
country_earn["log_gdppc"] = np.log(country_earn["fic"].map(gdppc_map))

# Log transform (savings/earnings can be very large for countries with small listed sectors)
country_earn["log_sav_earn"] = np.log(country_earn["sav_over_earnings"])

# ---------------------------------------------------------------------------
# 4. Merge with country effects
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")
ce_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")
avg_mb = ce_mb.groupby("fic")["country_effect"].mean().rename("mb_effect")

country = country_earn.set_index("fic").join([avg_pe, avg_roa, avg_mb], how="inner")
country = country.reset_index()

# Drop countries with extreme/negative savings or tiny earnings
country = country[country["sav_over_earnings"] > 0]
country = country[np.isfinite(country["log_sav_earn"])]
print(f"Countries with all data: {len(country)}")

# ---------------------------------------------------------------------------
# 5. Display key countries
# ---------------------------------------------------------------------------
print(f"\n{'Country':>5s}  {'Sav/GDP':>7s}  {'Sav/GNI':>7s}  {'Sav/Earn':>9s}  "
      f"{'P/E':>6s}  {'ROA':>7s}  {'N firms':>7s}")
for _, row in country.sort_values("sav_over_earnings", ascending=False).head(20).iterrows():
    print(f"{row['fic']:>5s}  {row['savings_gdp']:>6.1f}%  {row['savings_gni']:>6.1f}%  "
          f"{row['sav_over_earnings']:>9.1f}  "
          f"{row['pe_effect']:>+6.2f}  {row['roa_effect']:>+7.3f}  {row['avg_n_firms']:>7.0f}")

# ---------------------------------------------------------------------------
# 6. Correlations
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CORRELATIONS: THREE SAVINGS MEASURES")
print("=" * 80)

measures = {
    "Savings/GDP (%)": "savings_gdp",
    "Savings/GNI (%)": "savings_gni",
    "log(Savings/Earnings)": "log_sav_earn",
}

for label, col in measures.items():
    sub = country[[col, "pe_effect", "mb_effect", "roa_effect"]].dropna()
    r_pe = sub[col].corr(sub["pe_effect"])
    r_mb = sub[col].corr(sub["mb_effect"])
    r_roa = sub[col].corr(sub["roa_effect"])
    print(f"\n{label}:")
    print(f"  vs P/E:  r = {r_pe:+.2f}  (n={len(sub)})")
    print(f"  vs M/B:  r = {r_mb:+.2f}")
    print(f"  vs ROA:  r = {r_roa:+.2f}")

# ---------------------------------------------------------------------------
# 7. Regressions
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("REGRESSIONS")
print("=" * 80)

sub = country[["pe_effect", "mb_effect", "roa_effect",
               "savings_gdp", "savings_gni", "log_sav_earn", "log_gdppc"]].dropna()
print(f"Common sample: {len(sub)} countries")

specs = [
    ("P/E ~ Sav/GDP", "pe_effect", ["savings_gdp"]),
    ("P/E ~ Sav/GNI", "pe_effect", ["savings_gni"]),
    ("P/E ~ log(Sav/Earn)", "pe_effect", ["log_sav_earn"]),
    ("P/E ~ Sav/GDP + ROA", "pe_effect", ["savings_gdp", "roa_effect"]),
    ("P/E ~ Sav/GNI + ROA", "pe_effect", ["savings_gni", "roa_effect"]),
    ("P/E ~ log(Sav/Earn) + ROA", "pe_effect", ["log_sav_earn", "roa_effect"]),
    ("P/E ~ Sav/GDP + Sav/Earn + ROA", "pe_effect", ["savings_gdp", "log_sav_earn", "roa_effect"]),
    ("M/B ~ Sav/GDP", "mb_effect", ["savings_gdp"]),
    ("M/B ~ log(Sav/Earn)", "mb_effect", ["log_sav_earn"]),
    ("M/B ~ log(Sav/Earn) + ROA", "mb_effect", ["log_sav_earn", "roa_effect"]),
]

for label, dep, controls in specs:
    sub_reg = sub[[dep] + controls].dropna()
    X = sm.add_constant(sub_reg[controls])
    m = sm.OLS(sub_reg[dep], X).fit()
    print(f"\n{label}  (n={len(sub_reg)}, R²={m.rsquared:.3f})")
    for i, name in enumerate(["const"] + controls):
        print(f"  {name:20s}: {m.params.iloc[i]:+.4f} (t={m.tvalues.iloc[i]:+.2f})")

# ---------------------------------------------------------------------------
# 8. Robustness: drop Gulf + small islands
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ROBUSTNESS")
print("=" * 80)

exclude = {"QAT", "SAU", "ARE", "KWT", "BHR", "OMN", "MLT", "CYP", "MUS", "JAM", "LUX"}
sub_clean = sub[~sub.index.isin(
    country[country["fic"].isin(exclude)].index
)].copy()
# Safer: merge back fic
sub_with_fic = sub.merge(country[["fic"]], left_index=True, right_index=True)
sub_clean = sub_with_fic[~sub_with_fic["fic"].isin(exclude)]
print(f"Excluding Gulf + small islands: {len(sub_clean)} countries")

for label, col in [("Sav/GDP", "savings_gdp"), ("Sav/GNI", "savings_gni"),
                    ("log(Sav/Earn)", "log_sav_earn")]:
    X = sm.add_constant(sub_clean[col])
    m = sm.OLS(sub_clean["pe_effect"], X).fit()
    r = sub_clean[col].corr(sub_clean["pe_effect"])
    print(f"  P/E ~ {label}: r={r:+.2f}, t={m.tvalues.iloc[1]:+.2f}")

# ---------------------------------------------------------------------------
# 9. Partial correlations: ROA vs P/E | savings measures
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PARTIAL CORRELATIONS: ROA vs P/E AFTER CONTROLLING FOR SAVINGS")
print("=" * 80)

for label, col in measures.items():
    sub_p = country[["pe_effect", "roa_effect", col]].dropna()
    X = sm.add_constant(sub_p[col])
    roa_r = sm.OLS(sub_p["roa_effect"], X).fit().resid
    pe_r = sm.OLS(sub_p["pe_effect"], X).fit().resid
    r_partial = np.corrcoef(roa_r, pe_r)[0, 1]
    r_raw = sub_p["roa_effect"].corr(sub_p["pe_effect"])
    print(f"  {label:25s}: raw r={r_raw:+.2f} → partial r={r_partial:+.2f}")

# ---------------------------------------------------------------------------
# 10. Scatter plots
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


# 3-panel scatter: all three measures vs P/E
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (label, col) in zip(axes, measures.items()):
    sub_plot = country[[col, "pe_effect", "fic"]].dropna()
    r = sub_plot[col].corr(sub_plot["pe_effect"])
    X = sm.add_constant(sub_plot[col])
    t = sm.OLS(sub_plot["pe_effect"], X).fit().tvalues.iloc[1]
    ax.scatter(sub_plot[col], sub_plot["pe_effect"],
               color=PALETTE["accent"], s=35, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    _label_points(ax, sub_plot, col, "pe_effect")
    _fit_line(ax, sub_plot[col].values, sub_plot["pe_effect"].values)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("P/E effect" if ax == axes[0] else "", fontsize=9)
    ax.set_title(f"r = {r:+.2f}, t = {t:.1f}", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("National Savings and P/E: Three Measures", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_savings_three_measures.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_savings_three_measures.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_savings_three_measures")

# Savings/Earnings vs P/E (single panel, cleaner)
sub_se = country[["log_sav_earn", "pe_effect", "fic"]].dropna()
r_se = sub_se["log_sav_earn"].corr(sub_se["pe_effect"])
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub_se["log_sav_earn"], sub_se["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, sub_se, "log_sav_earn", "pe_effect")
_fit_line(ax, sub_se["log_sav_earn"].values, sub_se["pe_effect"].values)
ax.set_xlabel("log(National Savings / Listed Firms' Earnings)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Savings per Unit of Earnings vs P/E (r = {r_se:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_savings_earnings_vs_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_savings_earnings_vs_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_savings_earnings_vs_pe")

print("\nDone.")
