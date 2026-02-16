"""
pe_plots_updated.py -- Redo scatter plots with P/E ratio (not log)
on y-axis, and add growth as a control.

Regressions stay in log(P/E) (statistically sound), but plots show
exp(country_effect) = multiplicative P/E factor vs USA.
E.g., France = 0.70 means French firms trade at 70% of US P/E.

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
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
# 1. Load country effects (log scale) and convert
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_xrd = pd.read_parquet(OUT_DIR / "country_effects_xrd_at.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("log_pe_effect")
avg_xrd = (ce_xrd.groupby("fic").filter(lambda x: len(x) >= 5)
           .groupby("fic")["country_effect"].mean().rename("xrd_effect"))

# Convert log P/E effect to P/E ratio vs USA
# exp(0) = 1.0 for USA, exp(-0.35) = 0.70 for France, exp(+0.71) = 2.03 for China
country = pd.DataFrame(avg_pe)
country["pe_ratio"] = np.exp(country["log_pe_effect"])
country = country.join(avg_xrd, how="left")

# Savings/GNI
sav = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2000, 2024), labels=False)
country = country.join(sav.T.mean().rename("savings_gni"), how="left")

# IMF growth
r = requests.get("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH", timeout=15)
imf_data = r.json()["values"]["NGDP_RPCH"]
imf_rows = {}
for iso, years in imf_data.items():
    fwd = [years.get(str(y)) for y in range(2025, 2030)]
    fwd = [v for v in fwd if v is not None]
    if len(fwd) >= 3:
        imf_rows[iso] = np.mean(fwd)
country = country.join(pd.Series(imf_rows, name="imf_growth"), how="left")

print(f"Countries: {len(country)}")
print(f"P/E ratio range: {country['pe_ratio'].min():.2f} to {country['pe_ratio'].max():.2f}")
print(f"  USA = {country.loc['USA','pe_ratio']:.2f}")
print(f"  CHN = {country.loc['CHN','pe_ratio']:.2f}")
print(f"  FRA = {country.loc['FRA','pe_ratio']:.2f}")
print(f"  BRA = {country.loc['BRA','pe_ratio']:.2f}")

# ---------------------------------------------------------------------------
# 2. Scatter plots with P/E ratio on y-axis
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

# --- Savings vs P/E ratio ---
sub = country[["savings_gni", "pe_ratio"]].dropna()
r_sav = sub["savings_gni"].corr(sub["pe_ratio"])
t_sav = r_sav * np.sqrt((len(sub) - 2) / (1 - r_sav**2))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub["savings_gni"], sub["pe_ratio"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, sub, "savings_gni", "pe_ratio")
_fit(ax, sub["savings_gni"].values, sub["pe_ratio"].values)
ax.set_xlabel("Gross Savings / GNI (%)", fontsize=10)
ax.set_ylabel("P/E relative to USA", fontsize=10)
ax.set_title(f"National Savings and P/E (r = {r_sav:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_savings_vs_pe_ratio.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_savings_vs_pe_ratio.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSavings vs P/E ratio: r = {r_sav:+.2f}, t = {t_sav:+.2f}")

# --- R&D vs P/E ratio ---
sub = country[["xrd_effect", "pe_ratio"]].dropna()
r_xrd = sub["xrd_effect"].corr(sub["pe_ratio"])
t_xrd = r_xrd * np.sqrt((len(sub) - 2) / (1 - r_xrd**2))

fig, ax = plt.subplots(figsize=(6, 4))
sub_plot = sub.copy()
sub_plot["xrd_pct"] = sub_plot["xrd_effect"] * 100
ax.scatter(sub_plot["xrd_pct"], sub_plot["pe_ratio"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, sub_plot, "xrd_pct", "pe_ratio")
_fit(ax, sub_plot["xrd_pct"].values, sub_plot["pe_ratio"].values)
ax.set_xlabel("Country R&D/Assets effect (pp, vs USA)", fontsize=10)
ax.set_ylabel("P/E relative to USA", fontsize=10)
ax.set_title(f"R&D Intensity and P/E (r = {r_xrd:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_rd_vs_pe_ratio.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_rd_vs_pe_ratio.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"R&D vs P/E ratio: r = {r_xrd:+.2f}, t = {t_xrd:+.2f}")

# --- IMF growth vs P/E ratio ---
sub = country[["imf_growth", "pe_ratio"]].dropna()
r_imf = sub["imf_growth"].corr(sub["pe_ratio"])
t_imf = r_imf * np.sqrt((len(sub) - 2) / (1 - r_imf**2))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub["imf_growth"], sub["pe_ratio"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label(ax, sub, "imf_growth", "pe_ratio")
_fit(ax, sub["imf_growth"].values, sub["pe_ratio"].values)
ax.set_xlabel("IMF GDP growth forecast 2025-29 (%)", fontsize=10)
ax.set_ylabel("P/E relative to USA", fontsize=10)
ax.set_title(f"GDP Growth Forecast and P/E (r = {r_imf:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_imf_growth_vs_pe_ratio.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_imf_growth_vs_pe_ratio.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"IMF growth vs P/E ratio: r = {r_imf:+.2f}, t = {t_imf:+.2f}")

# ---------------------------------------------------------------------------
# 3. Horse race (log P/E effects, which are the valid regressions)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HORSE RACE (log P/E effects -- valid regressions)")
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
    sub = country[["log_pe_effect"] + rhs].dropna()
    X = sm.add_constant(sub[rhs])
    m = sm.OLS(sub["log_pe_effect"], X).fit()
    print(f"\n  {label} (n={len(sub)}):  R²={m.rsquared:.3f}")
    for i, var in enumerate(rhs):
        print(f"    {var:>15s}: {m.params.iloc[i+1]:+.4f} (t={m.tvalues.iloc[i+1]:+.2f})")

# ---------------------------------------------------------------------------
# 4. Residual country variation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESIDUAL VARIATION (log scale)")
print("=" * 70)

raw_std = country["log_pe_effect"].std()
print(f"\nStd dev of log P/E effects (raw): {raw_std:.3f}")
print(f"  → in P/E terms: ×{np.exp(raw_std):.2f} / ÷{np.exp(raw_std):.2f}")

for label, rhs in [
    ("R&D + Savings", ["xrd_effect", "savings_gni"]),
    ("R&D + Savings + Growth", ["xrd_effect", "savings_gni", "imf_growth"]),
]:
    sub = country[["log_pe_effect"] + rhs].dropna()
    X = sm.add_constant(sub[rhs])
    m = sm.OLS(sub["log_pe_effect"], X).fit()
    resid_std = m.resid.std()
    pct = (1 - resid_std / sub["log_pe_effect"].std()) * 100
    print(f"\nAfter {label}:")
    print(f"  R² = {m.rsquared:.3f}, residual std = {resid_std:.3f} "
          f"(reduction = {pct:.0f}%)")

    # Largest residuals
    residuals = m.resid.rename("resid")
    top = residuals.abs().nlargest(8)
    print(f"  Largest unexplained: ", end="")
    parts = []
    for fic in top.index:
        parts.append(f"{fic} ({residuals[fic]:+.2f})")
    print(", ".join(parts))

# ---------------------------------------------------------------------------
# 5. Key countries
# ---------------------------------------------------------------------------
print(f"\n{'Ctry':>5s}  {'logPE':>6s}  {'PE ratio':>8s}  {'R&D%':>6s}  {'Sav%':>5s}  {'GDP%':>5s}")
for fic in FOCUS:
    if fic in country.index:
        r = country.loc[fic]
        pe_r = f"{r['pe_ratio']:.2f}"
        xrd = f"{r['xrd_effect']*100:+.1f}" if pd.notna(r.get('xrd_effect')) else "  --"
        sav = f"{r['savings_gni']:.0f}" if pd.notna(r.get('savings_gni')) else " --"
        grw = f"{r['imf_growth']:.1f}" if pd.notna(r.get('imf_growth')) else " --"
        print(f"{fic:>5s}  {r['log_pe_effect']:>+6.2f}  {pe_r:>8s}  {xrd:>6s}  {sav:>5s}  {grw:>5s}")

print("\nDone.")
