"""
investment_test.py -- Do cross-country P/E effects predict investment rates?

Tobin's Q theory: higher market valuations → more investment.
If P/E reflects the cost of equity capital, countries with higher P/E
should have higher corporate investment rates.

We test three measures separately:
  - capx/at  (capital expenditure)
  - xrd/at   (R&D spending)
  - (capx + xrd)/at  (total investment)

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
# 1. Load panel data + add capx from raw files
# ---------------------------------------------------------------------------
print("Loading panel data...")
panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "at", "ggroup", "log_at", "log_pe",
             "roa", "leverage", "lag_earn_growth"],
)

# Load capx and xrd from US
print("Loading capx, xrd from Compustat America...")
us = pd.read_csv(
    DATA_DIR / "compustat_america.csv",
    usecols=["GVKEY", "fyear", "capx", "xrd"],
    dtype={"GVKEY": str},
    low_memory=False,
)
us = us.rename(columns={"GVKEY": "gvkey"})

# Load capx and xrd from Global
print("Loading capx, xrd from Compustat Global...")
gl = pd.read_csv(
    DATA_DIR / "compustat_global.csv",
    usecols=["gvkey", "fyear", "capx", "xrd"],
    dtype={"gvkey": str},
    low_memory=False,
)

inv = pd.concat([us, gl], ignore_index=True)
inv = inv.drop_duplicates(subset=["gvkey", "fyear"])
print(f"  capx non-null: {inv['capx'].notna().sum():,}")
print(f"  xrd  non-null: {inv['xrd'].notna().sum():,}")

# Merge
panel = panel.merge(inv, on=["gvkey", "fyear"], how="left")

# Investment measures
panel["capx_at"] = panel["capx"] / panel["at"]
panel["xrd_at"] = panel["xrd"] / panel["at"]
panel["total_inv"] = (panel["capx"].fillna(0) + panel["xrd"].fillna(0)) / panel["at"]
# total_inv only valid when at least one is non-null
panel.loc[panel["capx"].isna() & panel["xrd"].isna(), "total_inv"] = np.nan

# Winsorize each at 1/99
for col in ["capx_at", "xrd_at", "total_inv"]:
    lo, hi = panel[col].quantile([0.01, 0.99])
    panel[col] = panel[col].clip(lo, hi)

valid_capx = panel["capx_at"].notna() & panel["at"].gt(0) & panel["ggroup"].notna()
valid_xrd = panel["xrd_at"].notna() & panel["at"].gt(0) & panel["ggroup"].notna()
print(f"Panel with capx data: {valid_capx.sum():,} firm-years")
print(f"Panel with xrd data:  {valid_xrd.sum():,} firm-years")

# ---------------------------------------------------------------------------
# 2. Year-by-year regressions to extract country effects on investment
# ---------------------------------------------------------------------------
inv_measures = {
    "capx_at": ("Capital Expenditure / Assets", valid_capx),
    "xrd_at": ("R&D / Assets", valid_xrd),
}

all_ce = {}
for measure, (label, mask) in inv_measures.items():
    print(f"\nExtracting country effects for {label}...")
    results = []
    for yr, grp in panel.loc[mask].groupby("fyear"):
        sub = grp[[measure, "log_at", "fic", "ggroup"]].dropna()
        if sub["fic"].nunique() < 10:
            continue
        try:
            sub["fic"] = sub["fic"].astype(str)
            formula = f"{measure} ~ log_at + C(fic, Treatment(reference='USA')) + C(ggroup)"
            m = sm.OLS.from_formula(formula, data=sub).fit()
            for name, coef in m.params.items():
                if "C(fic" in name and "[T." in name:
                    iso = name.split("[T.")[1].rstrip("]")
                    results.append({"fyear": yr, "fic": iso, "country_effect": coef})
            results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
        except Exception as e:
            print(f"  Year {yr:.0f}: {e}")

    ce = pd.DataFrame(results)
    ce.to_parquet(OUT_DIR / f"country_effects_{measure}.parquet", index=False)
    all_ce[measure] = ce
    print(f"  {len(ce):,} country-year effects")

# ---------------------------------------------------------------------------
# 3. Average country effects
# ---------------------------------------------------------------------------
# Require at least 5 years of data per country for stable averages
min_years = 5
avg_capx = (
    all_ce["capx_at"].groupby("fic")
    .filter(lambda x: len(x) >= min_years)
    .groupby("fic")["country_effect"].mean().rename("capx_effect")
)
avg_xrd = (
    all_ce["xrd_at"].groupby("fic")
    .filter(lambda x: len(x) >= min_years)
    .groupby("fic")["country_effect"].mean().rename("xrd_effect")
)

# Load P/E and ROA country effects
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")

country = pd.DataFrame(avg_pe).join([avg_roa, avg_capx, avg_xrd], how="outer")
has_capx = country["capx_effect"].notna() & country["pe_effect"].notna()
has_xrd = country["xrd_effect"].notna() & country["pe_effect"].notna()
print(f"\nCountries with P/E + CapEx: {has_capx.sum()}")
print(f"Countries with P/E + R&D:  {has_xrd.sum()}")

# ---------------------------------------------------------------------------
# 4. Correlations and regressions
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CORRELATIONS")
print("=" * 70)

for inv_col, inv_label in [("capx_effect", "CapEx"), ("xrd_effect", "R&D")]:
    sub = country[["pe_effect", "roa_effect", inv_col]].dropna()
    n = len(sub)
    if n < 5:
        print(f"\n  --- {inv_label} --- SKIPPED (n={n})")
        continue
    r_pe = sub["pe_effect"].corr(sub[inv_col])
    r_roa = sub["roa_effect"].corr(sub[inv_col])
    print(f"\n  --- {inv_label} (n={n}) ---")
    print(f"  P/E vs {inv_label}:   r = {r_pe:+.2f}")
    print(f"  ROA vs {inv_label}:   r = {r_roa:+.2f}")

    # Regression: Inv ~ P/E
    X1 = sm.add_constant(sub["pe_effect"])
    m1 = sm.OLS(sub[inv_col], X1).fit()
    print(f"  {inv_label} ~ P/E:  coef={m1.params.iloc[1]:+.4f} (t={m1.tvalues.iloc[1]:+.2f}), R²={m1.rsquared:.3f}")

    # Inv ~ ROA
    X2 = sm.add_constant(sub["roa_effect"])
    m2 = sm.OLS(sub[inv_col], X2).fit()
    print(f"  {inv_label} ~ ROA:  coef={m2.params.iloc[1]:+.4f} (t={m2.tvalues.iloc[1]:+.2f}), R²={m2.rsquared:.3f}")

    # Inv ~ P/E + ROA
    X3 = sm.add_constant(sub[["pe_effect", "roa_effect"]])
    m3 = sm.OLS(sub[inv_col], X3).fit()
    print(f"  {inv_label} ~ P/E + ROA:")
    print(f"    P/E: {m3.params.iloc[1]:+.4f} (t={m3.tvalues.iloc[1]:+.2f})")
    print(f"    ROA: {m3.params.iloc[2]:+.4f} (t={m3.tvalues.iloc[2]:+.2f})")
    print(f"    R²={m3.rsquared:.3f}")

# ---------------------------------------------------------------------------
# 5. Key countries
# ---------------------------------------------------------------------------
print(f"\n{'Country':>5s}  {'P/E':>6s}  {'ROA':>7s}  {'CapEx':>7s}  {'R&D':>7s}")
for fic in FOCUS:
    if fic in country.index:
        row = country.loc[fic]
        capx_str = f"{row['capx_effect']:>+7.4f}" if pd.notna(row['capx_effect']) else "     --"
        xrd_str = f"{row['xrd_effect']:>+7.4f}" if pd.notna(row['xrd_effect']) else "     --"
        pe_str = f"{row['pe_effect']:>+6.2f}" if pd.notna(row['pe_effect']) else "    --"
        roa_str = f"{row['roa_effect']:>+7.3f}" if pd.notna(row['roa_effect']) else "     --"
        print(f"{fic:>5s}  {pe_str}  {roa_str}  {capx_str}  {xrd_str}")

# ---------------------------------------------------------------------------
# 6. Scatter plots
# ---------------------------------------------------------------------------
def _label_points(ax, data, x_col, y_col, focus=FOCUS):
    for fic, row in data.iterrows():
        if fic in focus:
            ax.annotate(
                fic, (row[x_col], row[y_col]),
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


# P/E vs CapEx and P/E vs R&D
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sub_capx = country[["pe_effect", "capx_effect"]].dropna()
sub_xrd = country[["pe_effect", "xrd_effect"]].dropna()
r_pe_capx = sub_capx["pe_effect"].corr(sub_capx["capx_effect"])
r_pe_xrd = sub_xrd["pe_effect"].corr(sub_xrd["xrd_effect"])

ax = axes[0]
ax.scatter(sub_capx["pe_effect"], sub_capx["capx_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7,
           edgecolors="white", linewidth=0.5)
_label_points(ax, sub_capx, "pe_effect", "capx_effect")
_fit_line(ax, sub_capx["pe_effect"].values, sub_capx["capx_effect"].values)
ax.set_xlabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_ylabel("Country CapEx/Assets effect", fontsize=10)
ax.set_title(f"P/E vs CapEx (r = {r_pe_capx:+.2f}, n={len(sub_capx)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
ax.scatter(sub_xrd["pe_effect"], sub_xrd["xrd_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7,
           edgecolors="white", linewidth=0.5)
_label_points(ax, sub_xrd, "pe_effect", "xrd_effect")
_fit_line(ax, sub_xrd["pe_effect"].values, sub_xrd["xrd_effect"].values)
ax.set_xlabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_ylabel("Country R&D/Assets effect", fontsize=10)
ax.set_title(f"P/E vs R&D (r = {r_pe_xrd:+.2f}, n={len(sub_xrd)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("P/E and Corporate Investment", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_pe_vs_capx_rd.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_pe_vs_capx_rd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_pe_vs_capx_rd")

# ROA vs CapEx and ROA vs R&D
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sub_capx_r = country[["roa_effect", "capx_effect"]].dropna()
sub_xrd_r = country[["roa_effect", "xrd_effect"]].dropna()
r_roa_capx = sub_capx_r["roa_effect"].corr(sub_capx_r["capx_effect"])
r_roa_xrd = sub_xrd_r["roa_effect"].corr(sub_xrd_r["xrd_effect"])

ax = axes[0]
ax.scatter(sub_capx_r["roa_effect"], sub_capx_r["capx_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7,
           edgecolors="white", linewidth=0.5)
_label_points(ax, sub_capx_r, "roa_effect", "capx_effect")
_fit_line(ax, sub_capx_r["roa_effect"].values, sub_capx_r["capx_effect"].values)
ax.set_xlabel("Country ROA effect (vs USA)", fontsize=10)
ax.set_ylabel("Country CapEx/Assets effect", fontsize=10)
ax.set_title(f"ROA vs CapEx (r = {r_roa_capx:+.2f}, n={len(sub_capx_r)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
ax.scatter(sub_xrd_r["roa_effect"], sub_xrd_r["xrd_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7,
           edgecolors="white", linewidth=0.5)
_label_points(ax, sub_xrd_r, "roa_effect", "xrd_effect")
_fit_line(ax, sub_xrd_r["roa_effect"].values, sub_xrd_r["xrd_effect"].values)
ax.set_xlabel("Country ROA effect (vs USA)", fontsize=10)
ax.set_ylabel("Country R&D/Assets effect", fontsize=10)
ax.set_title(f"ROA vs R&D (r = {r_roa_xrd:+.2f}, n={len(sub_xrd_r)})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("ROA and Corporate Investment", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_roa_vs_capx_rd.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_roa_vs_capx_rd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_roa_vs_capx_rd")

# ---------------------------------------------------------------------------
# 7. Time series of investment country effects (focus countries)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(FOCUS)))

for idx, (measure, label) in enumerate([("capx_at", "CapEx/Assets"), ("xrd_at", "R&D/Assets")]):
    ax = axes[idx]
    ce = all_ce[measure]
    for i, fic in enumerate(FOCUS):
        sub = ce[ce["fic"] == fic].sort_values("fyear")
        if len(sub) > 0:
            ax.plot(sub["fyear"], sub["country_effect"], label=fic,
                    color=colors[i], linewidth=1.5, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(f"{label} effect (vs USA)", fontsize=10)
    ax.set_title(f"Country {label} Effects", fontsize=11)
    ax.legend(ncol=5, fontsize=7, loc="upper right")

fig.tight_layout()
fig.savefig(OUT_DIR / "country_effects_capx_rd.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "country_effects_capx_rd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved country_effects_capx_rd")

print("\nDone.")
