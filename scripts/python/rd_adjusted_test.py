"""
rd_adjusted_test.py -- Does R&D accounting explain the profitability-valuation puzzle?

Net income (ib) is after R&D expense. High-R&D countries have:
  - Lower ROA = ib/at  (mechanically)
  - Higher P/E = mktcap/ib  (mechanically)

Test: add back R&D to earnings:
  - Adjusted ROA = (ib + xrd) / at
  - Adjusted P/E = log(mktcap / (ib + xrd))
If the negative ROA-P/E correlation disappears, R&D accounting is the explanation.

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(20260216)

DATA_DIR = Path(
    "~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/"
).expanduser()

OUT_DIR = Path("output/regressions")

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
    "highlight": "#8e44ad",
}

FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]

# ---------------------------------------------------------------------------
# 1. Load panel + add R&D
# ---------------------------------------------------------------------------
print("Loading panel data...")
panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "at", "ib", "market_cap", "ggroup",
             "log_at", "log_pe", "roa", "leverage", "lag_earn_growth"],
)
panel["fic"] = panel["fic"].astype(str)

# Load xrd
print("Loading R&D from Compustat...")
us = pd.read_csv(
    DATA_DIR / "compustat_america.csv",
    usecols=["GVKEY", "fyear", "xrd"],
    dtype={"GVKEY": str},
    low_memory=False,
)
us = us.rename(columns={"GVKEY": "gvkey"})

gl = pd.read_csv(
    DATA_DIR / "compustat_global.csv",
    usecols=["gvkey", "fyear", "xrd"],
    dtype={"gvkey": str},
    low_memory=False,
)

xrd = pd.concat([us, gl], ignore_index=True).drop_duplicates(subset=["gvkey", "fyear"])
panel = panel.merge(xrd, on=["gvkey", "fyear"], how="left")

# Fill missing R&D with 0 (firms that don't report R&D likely have ~0)
panel["xrd_filled"] = panel["xrd"].fillna(0)

# Adjusted measures
panel["ib_adj"] = panel["ib"] + panel["xrd_filled"]
panel["roa_adj"] = panel["ib_adj"] / panel["at"]
panel["log_pe_adj"] = np.where(
    panel["ib_adj"] > 0,
    np.log(panel["market_cap"] / panel["ib_adj"]),
    np.nan,
)

# Winsorize
for col in ["roa_adj", "log_pe_adj"]:
    lo, hi = panel[col].quantile([0.01, 0.99])
    panel[col] = panel[col].clip(lo, hi)

has_xrd = panel["xrd"].notna()
print(f"Panel: {len(panel):,} total, {has_xrd.sum():,} with reported R&D")
print(f"Adjusted P/E available: {panel['log_pe_adj'].notna().sum():,}")

# ---------------------------------------------------------------------------
# 2. Year-by-year regressions: standard vs adjusted
# ---------------------------------------------------------------------------
specs = {
    "pe_std": ("log_pe", "Standard P/E"),
    "pe_adj": ("log_pe_adj", "R&D-adjusted P/E"),
    "roa_std": ("roa", "Standard ROA"),
    "roa_adj": ("roa_adj", "R&D-adjusted ROA"),
}

all_effects = {}

for key, (dep_var, label) in specs.items():
    print(f"\nExtracting country effects for {label}...")
    results = []
    is_pe = "pe" in key

    for yr, grp in panel.groupby("fyear"):
        sub = grp[[dep_var, "log_at", "leverage", "lag_earn_growth",
                    "fic", "ggroup"]].dropna()
        if sub["fic"].nunique() < 10:
            continue
        try:
            if is_pe:
                formula = (f"{dep_var} ~ leverage + log_at + lag_earn_growth"
                           " + C(fic, Treatment(reference='USA')) + C(ggroup)")
            else:
                formula = (f"{dep_var} ~ leverage + log_at"
                           " + C(fic, Treatment(reference='USA')) + C(ggroup)")
            m = sm.OLS.from_formula(formula, data=sub).fit()
            for name, coef in m.params.items():
                if "C(fic" in name and "[T." in name:
                    iso = name.split("[T.")[1].rstrip("]")
                    results.append({"fyear": yr, "fic": iso, "country_effect": coef})
            results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
        except Exception as e:
            print(f"  Year {yr:.0f}: {e}")

    ce = pd.DataFrame(results)
    all_effects[key] = ce
    print(f"  {len(ce):,} country-year effects, {ce['fic'].nunique()} countries")

# ---------------------------------------------------------------------------
# 3. Average country effects and compare correlations
# ---------------------------------------------------------------------------
avg = {}
for key, ce in all_effects.items():
    avg[key] = ce.groupby("fic")["country_effect"].mean().rename(f"{key}_effect")

country = pd.DataFrame(avg["pe_std"]).join(
    [avg["pe_adj"], avg["roa_std"], avg["roa_adj"]], how="inner"
)
print(f"\nCountries with all effects: {len(country)}")

# ---------------------------------------------------------------------------
# 4. The key test
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("KEY TEST: Does R&D adjustment kill the ROA-P/E correlation?")
print("=" * 70)

r1 = country["roa_std_effect"].corr(country["pe_std_effect"])
r2 = country["roa_adj_effect"].corr(country["pe_adj_effect"])
r3 = country["roa_std_effect"].corr(country["pe_adj_effect"])
r4 = country["roa_adj_effect"].corr(country["pe_std_effect"])

n = len(country)
print(f"\n  {'Specification':<40s}  {'r':>6s}  {'t':>6s}")
print(f"  {'-'*40}  {'-'*6}  {'-'*6}")

for label, x_col, y_col in [
    ("Standard ROA vs Standard P/E", "roa_std_effect", "pe_std_effect"),
    ("Adjusted ROA vs Adjusted P/E", "roa_adj_effect", "pe_adj_effect"),
    ("Standard ROA vs Adjusted P/E", "roa_std_effect", "pe_adj_effect"),
    ("Adjusted ROA vs Standard P/E", "roa_adj_effect", "pe_std_effect"),
]:
    r = country[x_col].corr(country[y_col])
    t = r * np.sqrt((n - 2) / (1 - r**2))
    print(f"  {label:<40s}  {r:>+.2f}  {t:>+5.2f}")

# Also test: does R&D explain the P/E cross-section?
X = sm.add_constant(country[["roa_std_effect"]])
m_std = sm.OLS(country["pe_std_effect"], X).fit()
X = sm.add_constant(country[["roa_adj_effect"]])
m_adj = sm.OLS(country["pe_adj_effect"], X).fit()

print(f"\n  Regression: P/E ~ ROA")
print(f"  Standard:  coef={m_std.params.iloc[1]:+.2f} (t={m_std.tvalues.iloc[1]:+.2f}), R²={m_std.rsquared:.3f}")
print(f"  Adjusted:  coef={m_adj.params.iloc[1]:+.2f} (t={m_adj.tvalues.iloc[1]:+.2f}), R²={m_adj.rsquared:.3f}")

# ---------------------------------------------------------------------------
# 5. Key countries comparison
# ---------------------------------------------------------------------------
print(f"\n{'':>5s}  {'--- Standard ---':>17s}  {'--- R&D-Adjusted ---':>21s}")
print(f"{'Ctry':>5s}  {'P/E':>6s}  {'ROA':>7s}  {'P/E*':>7s}  {'ROA*':>7s}  {'xrd/at':>7s}")

# Add avg R&D intensity
xrd_rate = panel[panel["xrd"].notna()].groupby("fic")["xrd_filled"].sum() / \
           panel[panel["xrd"].notna()].groupby("fic")["at"].sum()
xrd_rate = xrd_rate.rename("xrd_intensity")
country = country.join(xrd_rate)

for fic in FOCUS:
    if fic in country.index:
        row = country.loc[fic]
        xrd_str = f"{row['xrd_intensity']:.3f}" if pd.notna(row.get('xrd_intensity')) else "  --"
        print(f"{fic:>5s}  {row['pe_std_effect']:>+6.2f}  {row['roa_std_effect']:>+7.3f}"
              f"  {row['pe_adj_effect']:>+7.2f}  {row['roa_adj_effect']:>+7.3f}"
              f"  {xrd_str:>7s}")

# ---------------------------------------------------------------------------
# 6. Scatter plots: standard vs adjusted
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

# Standard
ax = axes[0]
ax.scatter(country["roa_std_effect"], country["pe_std_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, country, "roa_std_effect", "pe_std_effect")
_fit(ax, country["roa_std_effect"].values, country["pe_std_effect"].values)
ax.set_xlabel("Country ROA effect", fontsize=10)
ax.set_ylabel("Country P/E effect", fontsize=10)
ax.set_title(f"Standard (r = {r1:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

# R&D-adjusted
ax = axes[1]
ax.scatter(country["roa_adj_effect"], country["pe_adj_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, country, "roa_adj_effect", "pe_adj_effect")
_fit(ax, country["roa_adj_effect"].values, country["pe_adj_effect"].values)
ax.set_xlabel("Country ROA* effect (ib+xrd)/at", fontsize=10)
ax.set_ylabel("Country P/E* effect mktcap/(ib+xrd)", fontsize=10)
ax.set_title(f"R&D-adjusted (r = {r2:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Does R&D Accounting Explain the Profitability Puzzle?", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_rd_adjusted.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_rd_adjusted.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_rd_adjusted")

print("\nDone.")
