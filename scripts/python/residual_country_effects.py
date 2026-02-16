"""
residual_country_effects.py -- Country P/E effects after firm-level R&D control

Current spec: log(P/E) ~ leverage + log(at) + lag_earn_growth + C(fic) + C(ggroup)
New spec:     log(P/E) ~ leverage + log(at) + lag_earn_growth + xrd/at + C(fic) + C(ggroup)

Question: how much do country effects shrink when we control for firm-level R&D?
What predicts the residual country effects?

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
# 1. Load panel + add firm-level R&D
# ---------------------------------------------------------------------------
print("Loading panel data...")
panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "at", "ib", "market_cap",
             "ggroup", "log_at", "log_pe", "leverage", "lag_earn_growth"],
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

# Firm-level R&D intensity (fill missing with 0 — non-reporters likely have ~0)
panel["xrd_at"] = panel["xrd"].fillna(0) / panel["at"]
panel.loc[panel["at"] <= 0, "xrd_at"] = np.nan

# Winsorize
lo, hi = panel["xrd_at"].quantile([0.01, 0.99])
panel["xrd_at"] = panel["xrd_at"].clip(lo, hi)

print(f"Panel: {len(panel):,} obs")
print(f"  xrd reported: {panel['xrd'].notna().sum():,}")
print(f"  xrd_at valid: {panel['xrd_at'].notna().sum():,}")

# ---------------------------------------------------------------------------
# 2. Year-by-year regressions: baseline vs R&D-controlled
# ---------------------------------------------------------------------------
specs = {
    "baseline": "log_pe ~ leverage + log_at + lag_earn_growth"
                " + C(fic, Treatment(reference='USA')) + C(ggroup)",
    "rd_ctrl":  "log_pe ~ leverage + log_at + lag_earn_growth + xrd_at"
                " + C(fic, Treatment(reference='USA')) + C(ggroup)",
}

all_effects = {}
for spec_name, formula in specs.items():
    print(f"\nExtracting country effects: {spec_name}...")
    results = []
    for yr, grp in panel.groupby("fyear"):
        sub = grp.dropna(subset=["log_pe", "leverage", "log_at", "lag_earn_growth",
                                  "xrd_at", "fic", "ggroup"])
        if sub["fic"].nunique() < 10:
            continue
        try:
            m = sm.OLS.from_formula(formula, data=sub).fit()
            for name, coef in m.params.items():
                if "C(fic" in name and "[T." in name:
                    iso = name.split("[T.")[1].rstrip("]")
                    results.append({"fyear": yr, "fic": iso, "country_effect": coef})
            results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})

            # Print xrd_at coefficient for first and last year
            if spec_name == "rd_ctrl" and yr in [2000, 2010, 2020, 2024]:
                xrd_coef = m.params.get("xrd_at", np.nan)
                xrd_t = m.tvalues.get("xrd_at", np.nan)
                print(f"  {yr:.0f}: xrd_at coef = {xrd_coef:+.2f} (t = {xrd_t:+.2f}), n = {len(sub):,}")
        except Exception as e:
            print(f"  Year {yr:.0f}: {e}")

    ce = pd.DataFrame(results)
    ce.to_parquet(OUT_DIR / f"country_effects_pe_{spec_name}.parquet", index=False)
    all_effects[spec_name] = ce
    print(f"  {ce['fic'].nunique()} countries, {len(ce):,} country-year effects")

# ---------------------------------------------------------------------------
# 3. Compare: baseline vs R&D-controlled country effects
# ---------------------------------------------------------------------------
avg_base = all_effects["baseline"].groupby("fic")["country_effect"].mean().rename("baseline")
avg_rd = all_effects["rd_ctrl"].groupby("fic")["country_effect"].mean().rename("rd_ctrl")

compare = pd.DataFrame(avg_base).join(avg_rd, how="inner")
compare["change"] = compare["rd_ctrl"] - compare["baseline"]
compare["pct_change"] = (compare["change"] / compare["baseline"].abs()) * 100

print("\n" + "=" * 70)
print("COMPARISON: Baseline vs R&D-controlled country effects")
print("=" * 70)

# Overall dispersion
std_base = compare["baseline"].std()
std_rd = compare["rd_ctrl"].std()
print(f"\nCross-country std dev:")
print(f"  Baseline:     {std_base:.3f}")
print(f"  R&D-ctrl:     {std_rd:.3f}")
print(f"  Reduction:    {(1 - std_rd/std_base)*100:.1f}%")

# Correlation between the two
r = compare["baseline"].corr(compare["rd_ctrl"])
print(f"\nCorrelation(baseline, R&D-ctrl): {r:.3f}")

# Key countries
print(f"\n{'Ctry':>5s}  {'Baseline':>9s}  {'R&D-ctrl':>9s}  {'Change':>8s}  {'PE ratio':>9s}  {'PE* ratio':>9s}")
for fic in FOCUS:
    if fic in compare.index:
        row = compare.loc[fic]
        pe1 = np.exp(row["baseline"])
        pe2 = np.exp(row["rd_ctrl"])
        print(f"{fic:>5s}  {row['baseline']:>+9.3f}  {row['rd_ctrl']:>+9.3f}  "
              f"{row['change']:>+8.3f}  {pe1:>9.2f}  {pe2:>9.2f}")

# Countries that move the most
print("\nLargest changes (R&D control shrinks/grows effect):")
for fic in compare["change"].abs().nlargest(10).index:
    row = compare.loc[fic]
    print(f"  {fic:>5s}: {row['baseline']:+.3f} → {row['rd_ctrl']:+.3f} (Δ = {row['change']:+.3f})")

# ---------------------------------------------------------------------------
# 4. Dispersion over time: baseline vs R&D-controlled
# ---------------------------------------------------------------------------
disp_base = (all_effects["baseline"].groupby("fyear")["country_effect"]
             .std().rename("baseline"))
disp_rd = (all_effects["rd_ctrl"].groupby("fyear")["country_effect"]
           .std().rename("rd_ctrl"))
disp = pd.DataFrame(disp_base).join(disp_rd)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(disp.index, disp["baseline"], color=PALETTE["accent"],
        linewidth=2, label="Baseline (no R&D control)")
ax.plot(disp.index, disp["rd_ctrl"], color=PALETTE["alert"],
        linewidth=2, label="After firm-level R&D control")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Cross-country std dev of P/E effects", fontsize=10)
ax.set_title("P/E Dispersion: Before and After R&D Control", fontsize=11)
ax.legend(fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "dispersion_baseline_vs_rd.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "dispersion_baseline_vs_rd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved dispersion_baseline_vs_rd")

# ---------------------------------------------------------------------------
# 5. Country effects time series: baseline vs R&D-controlled
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = plt.cm.tab10(np.linspace(0, 1, len(FOCUS)))

for idx, (spec_name, label) in enumerate([
    ("baseline", "Baseline"), ("rd_ctrl", "After R&D control")
]):
    ax = axes[idx]
    ce = all_effects[spec_name]
    for i, fic in enumerate(FOCUS):
        sub = ce[ce["fic"] == fic].sort_values("fyear")
        if len(sub) > 0:
            ax.plot(sub["fyear"], sub["country_effect"], label=fic,
                    color=colors[i], linewidth=1.5, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(ncol=5, fontsize=7, loc="upper left")
    ax.set_ylim(-1.5, 1.5)

fig.tight_layout()
fig.savefig(OUT_DIR / "country_effects_pe_baseline_vs_rd.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "country_effects_pe_baseline_vs_rd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved country_effects_pe_baseline_vs_rd")

# ---------------------------------------------------------------------------
# 6. What predicts the RESIDUAL country effects?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WHAT PREDICTS RESIDUAL (R&D-CONTROLLED) COUNTRY EFFECTS?")
print("=" * 70)

# Savings/GNI
sav = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2000, 2024), labels=False)
compare = compare.join(sav.T.mean().rename("savings_gni"), how="left")

# IMF growth
r = requests.get("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH", timeout=15)
imf_data = r.json()["values"]["NGDP_RPCH"]
imf_rows = {}
for iso, years in imf_data.items():
    fwd = [years.get(str(y)) for y in range(2025, 2030)]
    fwd = [v for v in fwd if v is not None]
    if len(fwd) >= 3:
        imf_rows[iso] = np.mean(fwd)
compare = compare.join(pd.Series(imf_rows, name="imf_growth"), how="left")

# Country-level R&D (for reference)
ce_xrd = pd.read_parquet(OUT_DIR / "country_effects_xrd_at.parquet")
avg_xrd = (ce_xrd.groupby("fic").filter(lambda x: len(x) >= 5)
           .groupby("fic")["country_effect"].mean().rename("xrd_effect"))
compare = compare.join(avg_xrd, how="left")

for dep, dep_label in [("baseline", "Baseline P/E"), ("rd_ctrl", "R&D-controlled P/E")]:
    print(f"\n  --- {dep_label} ---")
    for var, var_label in [("savings_gni", "Savings/GNI"),
                           ("imf_growth", "IMF growth"),
                           ("xrd_effect", "Country R&D")]:
        sub = compare[[dep, var]].dropna()
        if len(sub) < 10:
            continue
        r_val = sub[dep].corr(sub[var])
        X = sm.add_constant(sub[var])
        m = sm.OLS(sub[dep], X).fit()
        print(f"    {var_label:<15s}: r = {r_val:+.2f}, t = {m.tvalues.iloc[1]:+.2f} (n={len(sub)})")

# Horse race on residual effects
print("\n  Horse race on R&D-controlled effects:")
for label, rhs in [
    ("Savings only", ["savings_gni"]),
    ("Growth only", ["imf_growth"]),
    ("Savings + Growth", ["savings_gni", "imf_growth"]),
]:
    sub = compare[["rd_ctrl"] + rhs].dropna()
    if len(sub) < 10:
        continue
    X = sm.add_constant(sub[rhs])
    m = sm.OLS(sub["rd_ctrl"], X).fit()
    print(f"\n    {label} (n={len(sub)}): R²={m.rsquared:.3f}")
    for i, var in enumerate(rhs):
        print(f"      {var:>15s}: {m.params.iloc[i+1]:+.4f} (t={m.tvalues.iloc[i+1]:+.2f})")

# ---------------------------------------------------------------------------
# 7. Scatter: Savings vs residual P/E effect
# ---------------------------------------------------------------------------
import matplotlib.ticker as mticker

sub = compare[["rd_ctrl", "savings_gni"]].dropna()
sub["pe_ratio"] = np.exp(sub["rd_ctrl"])
r_sav = sub["savings_gni"].corr(sub["pe_ratio"])

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(sub["savings_gni"], sub["pe_ratio"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
for fic, row in sub.iterrows():
    if fic in FOCUS:
        ax.annotate(fic, (row["savings_gni"], row["pe_ratio"]),
                    fontsize=7, fontweight="bold", ha="left", va="bottom",
                    xytext=(4, 2), textcoords="offset points",
                    color=PALETTE["primary"])
# Fit line
mask = np.isfinite(sub["savings_gni"].values) & np.isfinite(sub["pe_ratio"].values)
z = np.polyfit(sub["savings_gni"].values[mask], sub["pe_ratio"].values[mask], 1)
xl = np.linspace(sub["savings_gni"].min(), sub["savings_gni"].max(), 100)
ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
        linewidth=1.5, linestyle="--", alpha=0.8)
ax.set_xlabel("Gross Savings / GNI (%)", fontsize=10)
ax.set_ylabel("P/E relative to USA (after R&D control)", fontsize=10)
ax.set_title(f"Savings and Residual P/E (r = {r_sav:+.2f}, n = {len(sub)})", fontsize=11)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_savings_vs_residual_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_savings_vs_residual_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved scatter_savings_vs_residual_pe (r = {r_sav:+.2f})")

print("\nDone.")
