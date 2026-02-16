"""
growth_pe_test.py -- Do low-P/E countries have lower long-run growth?

"Quiet life" hypothesis: firms in low-P/E countries are less ambitious,
invest less (R&D), and the market anticipates lower growth → lower P/E.

Tests:
  1. GDP growth (World Bank, realized as proxy for expected)
  2. Realized corporate earnings growth from our panel
  3. IMF WEO 5-year forward GDP growth forecasts (if available)

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
# 1. Load country P/E and ROA effects
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")

country = pd.DataFrame(avg_pe).join(avg_roa, how="inner")
print(f"Countries with P/E + ROA: {len(country)}")

# ---------------------------------------------------------------------------
# 2. World Bank GDP growth
# ---------------------------------------------------------------------------
print("\nFetching World Bank GDP growth data...")

indicators = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",       # GDP growth (annual %)
    "NY.GDP.PCAP.KD.ZG": "gdppc_growth",      # GDP per capita growth
}

for code, name in indicators.items():
    try:
        df = wb.data.DataFrame(code, time=range(2000, 2024), labels=False)
        df = df.T  # years as rows, countries as columns
        # Average across years
        avg = df.mean().rename(name)
        # Index is country ISO3
        country = country.join(avg, how="left")
        valid = country[name].notna().sum()
        print(f"  {name}: {valid} countries matched")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# ---------------------------------------------------------------------------
# 3. Realized corporate earnings growth from panel
# ---------------------------------------------------------------------------
print("\nComputing realized corporate earnings growth from panel...")
panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "ib", "at"],
)
panel["fic"] = panel["fic"].astype(str)

# Aggregate earnings by country-year (sum of ib for firms with at > 0)
agg = (
    panel[panel["at"] > 0]
    .groupby(["fic", "fyear"])
    .agg(total_ib=("ib", "sum"), n_firms=("ib", "count"))
    .reset_index()
)

# Compute annual earnings growth by country
agg = agg.sort_values(["fic", "fyear"])
agg["ib_growth"] = agg.groupby("fic")["total_ib"].pct_change()
# Winsorize growth at 1/99 (some extreme values from small countries)
lo, hi = agg["ib_growth"].quantile([0.01, 0.99])
agg["ib_growth"] = agg["ib_growth"].clip(lo, hi)

# Average growth per country (2000-2024 to match GDP data window)
earn_growth = (
    agg[(agg["fyear"] >= 2000) & (agg["fyear"] <= 2024)]
    .groupby("fic")["ib_growth"]
    .mean()
    .rename("earn_growth")
)
country = country.join(earn_growth, how="left")
print(f"  earn_growth: {country['earn_growth'].notna().sum()} countries")

# Also: 5-year forward earnings growth (for each year, how much did
# aggregate earnings grow over the next 5 years?)
# This tests if P/E *predicts* future growth
agg_wide = agg.pivot(index="fyear", columns="fic", values="total_ib")
fwd5 = (agg_wide.shift(-5) / agg_wide).pow(1/5) - 1  # annualized 5yr growth
fwd5_avg = fwd5.mean().rename("fwd5_earn_growth")  # average across years
country = country.join(fwd5_avg, how="left")
print(f"  fwd5_earn_growth: {country['fwd5_earn_growth'].notna().sum()} countries")

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CORRELATIONS: Growth measures vs P/E and ROA")
print("=" * 70)

growth_vars = [
    ("gdp_growth", "GDP growth (WB, 2000-23)"),
    ("gdppc_growth", "GDP/cap growth (WB, 2000-23)"),
    ("earn_growth", "Realized earnings growth"),
    ("fwd5_earn_growth", "5-yr forward earnings growth"),
]

print(f"\n  {'Variable':<35s}  {'r(P/E)':>7s}  {'t':>6s}  {'r(ROA)':>7s}  {'n':>4s}")
print(f"  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*4}")

for var, label in growth_vars:
    sub = country[["pe_effect", "roa_effect", var]].dropna()
    n = len(sub)
    if n < 10:
        print(f"  {label:<35s}  {'--':>7s}  {'--':>6s}  {'--':>7s}  {n:>4d}")
        continue
    r_pe = sub["pe_effect"].corr(sub[var])
    r_roa = sub["roa_effect"].corr(sub[var])
    t_pe = r_pe * np.sqrt((n - 2) / (1 - r_pe**2))
    print(f"  {label:<35s}  {r_pe:>+.2f}   {t_pe:>+5.2f}  {r_roa:>+.2f}   {n:>4d}")

# Regressions: P/E ~ growth
print("\n  Regressions: P/E ~ growth measure")
for var, label in growth_vars:
    sub = country[["pe_effect", var]].dropna()
    if len(sub) < 10:
        continue
    X = sm.add_constant(sub[var])
    m = sm.OLS(sub["pe_effect"], X).fit()
    print(f"  {label:<35s}  coef={m.params.iloc[1]:+.3f} (t={m.tvalues.iloc[1]:+.2f}), R²={m.rsquared:.3f}")

# P/E ~ growth + ROA (does growth absorb the ROA puzzle?)
print("\n  P/E ~ growth + ROA:")
for var, label in growth_vars:
    sub = country[["pe_effect", "roa_effect", var]].dropna()
    if len(sub) < 10:
        continue
    X = sm.add_constant(sub[["roa_effect", var]])
    m = sm.OLS(sub["pe_effect"], X).fit()
    print(f"  {label:<35s}")
    print(f"    ROA:    {m.params.iloc[1]:+.2f} (t={m.tvalues.iloc[1]:+.2f})")
    print(f"    Growth: {m.params.iloc[2]:+.3f} (t={m.tvalues.iloc[2]:+.2f}), R²={m.rsquared:.3f}")

# ---------------------------------------------------------------------------
# 5. Key countries
# ---------------------------------------------------------------------------
print(f"\n{'Ctry':>5s}  {'P/E':>6s}  {'ROA':>7s}  {'GDP%':>6s}  {'Earn%':>6s}  {'Fwd5%':>6s}")
for fic in FOCUS:
    if fic in country.index:
        r = country.loc[fic]
        gdp = f"{r['gdp_growth']:.1f}" if pd.notna(r.get('gdp_growth')) else "  --"
        eg = f"{r['earn_growth']*100:.1f}" if pd.notna(r.get('earn_growth')) else "  --"
        f5 = f"{r['fwd5_earn_growth']*100:.1f}" if pd.notna(r.get('fwd5_earn_growth')) else "  --"
        print(f"{fic:>5s}  {r['pe_effect']:>+6.2f}  {r['roa_effect']:>+7.3f}"
              f"  {gdp:>6s}  {eg:>6s}  {f5:>6s}")

# ---------------------------------------------------------------------------
# 6. Scatter plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

def _label(ax, data, x, y, focus=FOCUS):
    for fic, row in data.iterrows():
        if fic in focus:
            ax.annotate(fic, (row[x], row[y]),
                        fontsize=7, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points",
                        color=PALETTE["primary"])

def _fit(ax, x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    z = np.polyfit(x[mask], y[mask], 1)
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
            linewidth=1.5, linestyle="--", alpha=0.8)

# GDP growth vs P/E
sub_gdp = country[["pe_effect", "gdp_growth"]].dropna()
r_gdp = sub_gdp["pe_effect"].corr(sub_gdp["gdp_growth"])
ax = axes[0]
ax.scatter(sub_gdp["gdp_growth"], sub_gdp["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, sub_gdp, "gdp_growth", "pe_effect")
_fit(ax, sub_gdp["gdp_growth"].values, sub_gdp["pe_effect"].values)
ax.set_xlabel("Avg GDP growth 2000-23 (%)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"GDP Growth vs P/E (r = {r_gdp:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

# Forward earnings growth vs P/E
sub_fwd = country[["pe_effect", "fwd5_earn_growth"]].dropna()
r_fwd = sub_fwd["pe_effect"].corr(sub_fwd["fwd5_earn_growth"])
ax = axes[1]
ax.scatter(sub_fwd["fwd5_earn_growth"] * 100, sub_fwd["pe_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
# Need to adjust for label function with scaled x
sub_fwd_plot = sub_fwd.copy()
sub_fwd_plot["fwd5_pct"] = sub_fwd_plot["fwd5_earn_growth"] * 100
_label(ax, sub_fwd_plot, "fwd5_pct", "pe_effect")
_fit(ax, (sub_fwd["fwd5_earn_growth"] * 100).values, sub_fwd["pe_effect"].values)
ax.set_xlabel("5-yr forward earnings growth (ann. %)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Fwd Earnings Growth vs P/E (r = {r_fwd:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Long-Run Growth and Cross-Country Valuations", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_growth_vs_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_growth_vs_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_growth_vs_pe")

print("\nDone.")
