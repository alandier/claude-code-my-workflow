"""
pension_funds_test.py -- Do pension fund assets explain cross-country P/E?

Hypothesis: countries with large funded pension systems (401k, superannuation,
etc.) channel more savings into equities, pushing up valuations.

Data: OECD Pension Markets in Focus (2023) — pension fund assets as % of GDP.
Autonomous pension funds only (excludes government reserve funds like GPIF).

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
# 1. Pension fund assets as % of GDP
# ---------------------------------------------------------------------------
# Source: OECD Global Pension Statistics 2023 / Pension Markets in Focus
# Autonomous pension funds (excludes government reserve funds)
# Approximate values, averaged over recent years (2020-2023)
PENSION_GDP = {
    # Anglo-Saxon — large funded systems
    "USA": 140,   # 401(k), IRA, DB plans
    "GBR": 105,   # DB + DC occupational pensions
    "AUS": 128,   # Superannuation (mandatory 11.5%)
    "CAN": 87,    # RPP, RRSP
    "NZL": 31,    # KiwiSaver + other
    "IRL": 28,    # Occupational pensions
    # Nordic
    "DNK": 72,    # Private pensions (excl ATP)
    "FIN": 57,    # Earnings-related (partially funded)
    "SWE": 91,    # Premium pension + occupational
    "NOR": 12,    # Occupational (most via government fund)
    # Continental Europe
    "NLD": 175,   # Largest in Europe
    "CHE": 139,   # Pillar 2 (mandatory occupational)
    "DEU": 7,     # Mostly pay-as-you-go
    "FRA": 11,    # Mostly pay-as-you-go
    "ITA": 10,    # TFR + complementary funds
    "ESP": 12,    # Small funded pillar
    "BEL": 9,     # Small funded pillar
    "AUT": 6,     # Small funded pillar
    "PRT": 9,
    "GRC": 1,     # Almost entirely PAYG
    "LUX": 3,
    # Eastern Europe
    "POL": 4,
    "CZE": 9,
    "HUN": 5,
    "EST": 16,
    "LTU": 4,
    "SVN": 7,
    "HRV": 8,
    "BGR": 8,
    "ROU": 3,
    # Asia
    "JPN": 28,    # Corporate DB/DC (excl GPIF)
    "KOR": 36,    # National Pension (partially funded) + corporate
    "HKG": 50,    # MPF (mandatory)
    "SGP": 70,    # CPF
    "MYS": 56,    # EPF
    "THA": 8,
    "IDN": 3,
    "PHL": 5,
    "IND": 5,     # EPFO + NPS (small relative to GDP)
    "CHN": 2,     # Pillar 2 enterprise annuity (small)
    "TWN": 15,    # Labor pension + old system
    "PAK": 1,
    "BGD": 1,
    "VNM": 2,
    "LKA": 10,
    # Latin America
    "CHL": 59,    # AFP system (mandatory DC)
    "COL": 28,    # AFP + Colpensiones
    "MEX": 18,    # AFORE system
    "BRA": 25,    # Closed + open funds
    "PER": 20,    # AFP system
    "ARG": 5,     # Nationalized in 2008
    # Middle East & Africa
    "ISR": 60,    # Provident + pension funds
    "ZAF": 85,    # Large funded system
    "SAU": 15,    # GOSI
    "ARE": 10,
    "TUR": 3,
    "EGY": 2,
    "NGA": 3,
    "KEN": 10,
    "MAR": 10,
    "TUN": 3,
    # Other
    "MUS": 30,
    "JOR": 25,
    "QAT": 5,
    "KWT": 10,
    "BHR": 5,
    "OMN": 5,
    "SRB": 2,
}

pension_df = pd.DataFrame([
    {"fic": k, "pension_gdp": v} for k, v in PENSION_GDP.items()
])
print(f"Pension data: {len(pension_df)} countries")

# ---------------------------------------------------------------------------
# 2. Load country effects
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")
ce_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")
avg_mb = ce_mb.groupby("fic")["country_effect"].mean().rename("mb_effect")

country = pd.DataFrame(avg_pe).join([avg_roa, avg_mb], how="outer")
country = country.merge(pension_df, left_index=True, right_on="fic", how="inner")
country["log_pension_gdp"] = np.log(country["pension_gdp"])
print(f"Countries with pension + effects: {len(country)}")

# ---------------------------------------------------------------------------
# 3. Key countries
# ---------------------------------------------------------------------------
print(f"\n{'Country':>5s}  {'Pension/GDP':>11s}  {'P/E':>6s}  {'M/B':>6s}  {'ROA':>7s}")
for _, row in country.sort_values("pension_gdp", ascending=False).head(20).iterrows():
    print(f"{row['fic']:>5s}  {row['pension_gdp']:>10.0f}%  "
          f"{row['pe_effect']:>+6.2f}  {row['mb_effect']:>+6.2f}  {row['roa_effect']:>+7.3f}")

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)

r_pe = country["log_pension_gdp"].corr(country["pe_effect"])
r_mb = country["log_pension_gdp"].corr(country["mb_effect"])
r_roa = country["log_pension_gdp"].corr(country["roa_effect"])
print(f"\n  log(Pension/GDP) vs P/E effect:  r = {r_pe:+.2f}  (n={len(country)})")
print(f"  log(Pension/GDP) vs M/B effect:  r = {r_mb:+.2f}")
print(f"  log(Pension/GDP) vs ROA effect:  r = {r_roa:+.2f}")

# Also with levels (not log)
r_pe_lev = country["pension_gdp"].corr(country["pe_effect"])
r_mb_lev = country["pension_gdp"].corr(country["mb_effect"])
r_roa_lev = country["pension_gdp"].corr(country["roa_effect"])
print(f"\n  Pension/GDP (level) vs P/E:  r = {r_pe_lev:+.2f}")
print(f"  Pension/GDP (level) vs M/B:  r = {r_mb_lev:+.2f}")
print(f"  Pension/GDP (level) vs ROA:  r = {r_roa_lev:+.2f}")

# Partial: ROA vs P/E | pension
sub = country[["pe_effect", "roa_effect", "log_pension_gdp"]].dropna()
X = sm.add_constant(sub["log_pension_gdp"])
roa_resid = sm.OLS(sub["roa_effect"], X).fit().resid
pe_resid = sm.OLS(sub["pe_effect"], X).fit().resid
r_partial = np.corrcoef(roa_resid, pe_resid)[0, 1]
r_raw = sub["roa_effect"].corr(sub["pe_effect"])
print(f"\n  ROA vs P/E (raw):           r = {r_raw:+.2f}")
print(f"  ROA vs P/E | pension:       r = {r_partial:+.2f}")

# ---------------------------------------------------------------------------
# 5. Regressions
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("REGRESSIONS")
print("=" * 80)

# P/E ~ pension
X1 = sm.add_constant(country["log_pension_gdp"])
m1 = sm.OLS(country["pe_effect"], X1).fit()
print(f"\nSpec 1: P/E ~ log(Pension/GDP)")
print(f"  Pension: {m1.params.iloc[1]:+.3f} (t={m1.tvalues.iloc[1]:+.2f}), R²={m1.rsquared:.3f}")

# P/E ~ ROA
X2 = sm.add_constant(country["roa_effect"])
m2 = sm.OLS(country["pe_effect"], X2).fit()
print(f"\nSpec 2: P/E ~ ROA")
print(f"  ROA: {m2.params.iloc[1]:+.2f} (t={m2.tvalues.iloc[1]:+.2f}), R²={m2.rsquared:.3f}")

# P/E ~ ROA + pension
X3 = sm.add_constant(country[["roa_effect", "log_pension_gdp"]])
m3 = sm.OLS(country["pe_effect"], X3).fit()
print(f"\nSpec 3: P/E ~ ROA + log(Pension/GDP)")
print(f"  ROA:     {m3.params.iloc[1]:+.2f} (t={m3.tvalues.iloc[1]:+.2f})")
print(f"  Pension: {m3.params.iloc[2]:+.3f} (t={m3.tvalues.iloc[2]:+.2f})")
print(f"  R²={m3.rsquared:.3f}")

# M/B ~ pension
X4 = sm.add_constant(country["log_pension_gdp"])
m4 = sm.OLS(country["mb_effect"].dropna(), X4.loc[country["mb_effect"].notna()]).fit()
print(f"\nSpec 4: M/B ~ log(Pension/GDP)")
print(f"  Pension: {m4.params.iloc[1]:+.3f} (t={m4.tvalues.iloc[1]:+.2f}), R²={m4.rsquared:.3f}")

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


# Pension vs P/E and M/B
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(country["log_pension_gdp"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "log_pension_gdp", "pe_effect")
_fit_line(ax, country["log_pension_gdp"].values, country["pe_effect"].values)
ax.set_xlabel("log(Pension Fund Assets / GDP)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"P/E (r = {r_pe:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
sub_mb = country[["log_pension_gdp", "mb_effect", "fic"]].dropna()
r_mb_log = sub_mb["log_pension_gdp"].corr(sub_mb["mb_effect"])
ax.scatter(sub_mb["log_pension_gdp"], sub_mb["mb_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, sub_mb, "log_pension_gdp", "mb_effect")
_fit_line(ax, sub_mb["log_pension_gdp"].values, sub_mb["mb_effect"].values)
ax.set_xlabel("log(Pension Fund Assets / GDP)", fontsize=10)
ax.set_ylabel("Country M/B effect (vs USA)", fontsize=10)
ax.set_title(f"M/B (r = {r_mb_log:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Pension Fund Assets and Cross-Country Valuations", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_pension_vs_valuation.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_pension_vs_valuation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_pension_vs_valuation")

# Pension vs ROA
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(country["log_pension_gdp"], country["roa_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "log_pension_gdp", "roa_effect")
_fit_line(ax, country["log_pension_gdp"].values, country["roa_effect"].values)
ax.set_xlabel("log(Pension Fund Assets / GDP)", fontsize=10)
ax.set_ylabel("Country ROA effect (vs USA)", fontsize=10)
ax.set_title(f"Pension vs ROA (r = {r_roa:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_pension_vs_roa.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_pension_vs_roa.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_pension_vs_roa")

print("\nDone.")
