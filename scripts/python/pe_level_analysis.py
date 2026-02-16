"""
pe_level_analysis.py -- Three improvements:
1. P/E in levels (not log) for scatter plots
2. How much of cross-country P/E do savings + R&D explain?
3. Add IMF GDP growth forecast as a control

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import wbgapi as wb

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
# 1. Extract country P/E effects in LEVELS (not log)
# ---------------------------------------------------------------------------
print("Loading panel data...")
panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "at", "ib", "market_cap",
             "ggroup", "log_at", "leverage", "lag_earn_growth"],
)
panel["fic"] = panel["fic"].astype(str)

# P/E in levels (winsorized)
panel["pe"] = np.where(panel["ib"] > 0, panel["market_cap"] / panel["ib"], np.nan)
lo, hi = panel["pe"].quantile([0.01, 0.99])
panel["pe"] = panel["pe"].clip(lo, hi)

print(f"P/E available: {panel['pe'].notna().sum():,}")
print(f"P/E median: {panel['pe'].median():.1f}, mean: {panel['pe'].mean():.1f}")

# Year-by-year regressions with P/E in levels
print("\nExtracting country P/E effects (levels)...")
results = []
for yr, grp in panel.groupby("fyear"):
    sub = grp[["pe", "leverage", "log_at", "lag_earn_growth", "fic", "ggroup"]].dropna()
    if sub["fic"].nunique() < 10:
        continue
    try:
        formula = ("pe ~ leverage + log_at + lag_earn_growth"
                   " + C(fic, Treatment(reference='USA')) + C(ggroup)")
        m = sm.OLS.from_formula(formula, data=sub).fit()
        for name, coef in m.params.items():
            if "C(fic" in name and "[T." in name:
                iso = name.split("[T.")[1].rstrip("]")
                results.append({"fyear": yr, "fic": iso, "country_effect": coef})
        results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
    except Exception as e:
        print(f"  Year {yr:.0f}: {e}")

ce_pe_lev = pd.DataFrame(results)
ce_pe_lev.to_parquet(OUT_DIR / "country_effects_pe_level.parquet", index=False)
avg_pe_lev = ce_pe_lev.groupby("fic")["country_effect"].mean().rename("pe_lev_effect")
print(f"Countries: {ce_pe_lev['fic'].nunique()}")

# For interpretation: USA median P/E
usa_med_pe = panel.loc[panel["fic"] == "USA", "pe"].median()
print(f"USA median P/E: {usa_med_pe:.1f}")
print("So effect of -5 means country P/E is ~5 points below USA")

# ---------------------------------------------------------------------------
# 2. Load R&D effects, savings, IMF growth
# ---------------------------------------------------------------------------
# R&D effects
ce_xrd = pd.read_parquet(OUT_DIR / "country_effects_xrd_at.parquet")
avg_xrd = (ce_xrd.groupby("fic").filter(lambda x: len(x) >= 5)
           .groupby("fic")["country_effect"].mean().rename("xrd_effect"))

# Savings/GNI
sav = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2000, 2024), labels=False)
sav = sav.T.mean().rename("savings_gni")

# IMF WEO GDP growth forecasts (2025-2029)
print("\nFetching IMF growth forecasts...")
r = requests.get("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH", timeout=15)
imf_data = r.json()["values"]["NGDP_RPCH"]
imf_rows = []
for iso, years in imf_data.items():
    fwd = [years.get(str(y)) for y in range(2025, 2030)]
    fwd = [v for v in fwd if v is not None]
    if len(fwd) >= 3:
        imf_rows.append({"fic": iso, "imf_growth": np.mean(fwd)})
imf = pd.DataFrame(imf_rows).set_index("fic")["imf_growth"]

# Combine
country = pd.DataFrame(avg_pe_lev)
country = country.join([avg_xrd, sav, imf], how="left")
print(f"Countries with all 3 predictors: "
      f"{country[['xrd_effect','savings_gni','imf_growth']].dropna().shape[0]}")

# ---------------------------------------------------------------------------
# 3. Key scatter plots in P/E levels
# ---------------------------------------------------------------------------
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

# --- Savings vs P/E ---
sub = country[["savings_gni", "pe_lev_effect"]].dropna()
r_sav = sub["savings_gni"].corr(sub["pe_lev_effect"])
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub["savings_gni"], sub["pe_lev_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, sub, "savings_gni", "pe_lev_effect")
_fit(ax, sub["savings_gni"].values, sub["pe_lev_effect"].values)
ax.set_xlabel("Gross Savings / GNI (%)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Savings and P/E (r = {r_sav:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_savings_vs_pe_level.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_savings_vs_pe_level.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSavings vs P/E (levels): r = {r_sav:+.2f}")

# --- R&D vs P/E ---
sub = country[["xrd_effect", "pe_lev_effect"]].dropna()
r_xrd = sub["xrd_effect"].corr(sub["pe_lev_effect"])
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub["xrd_effect"] * 100, sub["pe_lev_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
sub_plot = sub.copy()
sub_plot["xrd_pct"] = sub_plot["xrd_effect"] * 100
_label(ax, sub_plot, "xrd_pct", "pe_lev_effect")
_fit(ax, (sub["xrd_effect"] * 100).values, sub["pe_lev_effect"].values)
ax.set_xlabel("Country R&D/Assets effect (pp, vs USA)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"R&D and P/E (r = {r_xrd:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_rd_vs_pe_level.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_rd_vs_pe_level.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"R&D vs P/E (levels): r = {r_xrd:+.2f}")

# --- IMF growth vs P/E ---
sub = country[["imf_growth", "pe_lev_effect"]].dropna()
r_imf = sub["imf_growth"].corr(sub["pe_lev_effect"])
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub["imf_growth"], sub["pe_lev_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, sub, "imf_growth", "pe_lev_effect")
_fit(ax, sub["imf_growth"].values, sub["pe_lev_effect"].values)
ax.set_xlabel("IMF GDP growth forecast 2025-29 (%)", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"GDP Growth Forecast and P/E (r = {r_imf:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_growth_vs_pe_level.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_growth_vs_pe_level.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"IMF growth vs P/E (levels): r = {r_imf:+.2f}")

# ---------------------------------------------------------------------------
# 4. Horse race with all predictors (P/E levels)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("REGRESSIONS (P/E in levels)")
print("=" * 70)

specs = [
    ("R&D only",                   ["xrd_effect"]),
    ("Savings only",               ["savings_gni"]),
    ("Growth only",                ["imf_growth"]),
    ("R&D + Savings",              ["xrd_effect", "savings_gni"]),
    ("R&D + Growth",               ["xrd_effect", "imf_growth"]),
    ("Savings + Growth",           ["savings_gni", "imf_growth"]),
    ("R&D + Savings + Growth",     ["xrd_effect", "savings_gni", "imf_growth"]),
]

for label, rhs in specs:
    sub = country[["pe_lev_effect"] + rhs].dropna()
    X = sm.add_constant(sub[rhs])
    m = sm.OLS(sub["pe_lev_effect"], X).fit()
    print(f"\n  {label} (n={len(sub)}):  R²={m.rsquared:.3f}")
    for i, var in enumerate(rhs):
        print(f"    {var:>15s}: {m.params.iloc[i+1]:+.3f} (t={m.tvalues.iloc[i+1]:+.2f})")

# ---------------------------------------------------------------------------
# 5. Residual country variation after controls
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESIDUAL COUNTRY VARIATION")
print("=" * 70)

# Std dev of country P/E effects
raw_std = country["pe_lev_effect"].std()
print(f"\nStd dev of country P/E effects (raw): {raw_std:.2f}")

# After R&D + Savings
sub = country[["pe_lev_effect", "xrd_effect", "savings_gni"]].dropna()
X = sm.add_constant(sub[["xrd_effect", "savings_gni"]])
m = sm.OLS(sub["pe_lev_effect"], X).fit()
resid_std = m.resid.std()
print(f"After R&D + Savings:   residual std = {resid_std:.2f}  "
      f"(reduction = {(1 - resid_std/raw_std)*100:.0f}%, R² = {m.rsquared:.3f})")

# After all three
sub3 = country[["pe_lev_effect", "xrd_effect", "savings_gni", "imf_growth"]].dropna()
X3 = sm.add_constant(sub3[["xrd_effect", "savings_gni", "imf_growth"]])
m3 = sm.OLS(sub3["pe_lev_effect"], X3).fit()
resid_std3 = m3.resid.std()
print(f"After R&D + Sav + Grw: residual std = {resid_std3:.2f}  "
      f"(reduction = {(1 - resid_std3/raw_std)*100:.0f}%, R² = {m3.rsquared:.3f})")

# Largest residuals (countries most unexplained)
sub["resid"] = m.resid
print(f"\nLargest unexplained P/E effects (after R&D + Savings):")
print(f"  {'Country':>5s}  {'Raw P/E':>8s}  {'Fitted':>8s}  {'Residual':>8s}")
for fic in sub["resid"].abs().nlargest(10).index:
    r = sub.loc[fic]
    print(f"  {fic:>5s}  {r['pe_lev_effect']:>+8.1f}  "
          f"{r['pe_lev_effect'] - r['resid']:>+8.1f}  {r['resid']:>+8.1f}")

# ---------------------------------------------------------------------------
# 6. Key countries table
# ---------------------------------------------------------------------------
print(f"\n{'Ctry':>5s}  {'P/E eff':>8s}  {'R&D':>7s}  {'Sav%':>6s}  {'GDP%':>6s}")
for fic in FOCUS:
    if fic in country.index:
        r = country.loc[fic]
        xrd = f"{r['xrd_effect']*100:+.2f}" if pd.notna(r.get('xrd_effect')) else "   --"
        sav = f"{r['savings_gni']:.0f}" if pd.notna(r.get('savings_gni')) else "  --"
        grw = f"{r['imf_growth']:.1f}" if pd.notna(r.get('imf_growth')) else "  --"
        print(f"{fic:>5s}  {r['pe_lev_effect']:>+8.1f}  {xrd:>7s}  {sav:>6s}  {grw:>6s}")

print("\nDone.")
