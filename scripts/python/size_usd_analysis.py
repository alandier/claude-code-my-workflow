"""
size_usd_analysis.py -- Re-run size analysis with USD-converted assets.

Converts local-currency assets to USD using World Bank exchange rates,
then re-tests whether country-level firm size explains the ROA-P/E puzzle.

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import wbgapi as wb

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
print(f"Panel: {df.shape[0]:,} obs")

# ---------------------------------------------------------------------------
# 2. Fetch exchange rates from World Bank
# ---------------------------------------------------------------------------
# PA.NUS.FCRF = Official exchange rate (LCU per US$, period average)
# Map fic (ISO-3) to WB codes
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

# Eurozone countries: report in EUR, need EUR/USD rate
# Before EUR (pre-1999): legacy currencies. For simplicity, use EUR rate for
# all Eurozone countries across all years (most data is post-2007 anyway).
EUROZONE = {"DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL", "PRT",
            "GRC", "FIN", "LUX", "SVN", "CYP", "MLT", "EST", "LTU", "HRV"}

valid_wb = [v for v in set(WB_CODE_MAP.values()) if v is not None]
print(f"Fetching exchange rates for {len(valid_wb)} countries...")

# Get exchange rates: LCU per USD, averaged over 2010-2023
raw_fx = wb.data.DataFrame("PA.NUS.FCRF", economy=valid_wb, time=range(2007, 2024))
avg_fx = raw_fx.mean(axis=1)
avg_fx.index = avg_fx.index.get_level_values("economy")
avg_fx.name = "lcu_per_usd"
fx_df = avg_fx.reset_index()
fx_df.columns = ["wb_code", "lcu_per_usd"]

print(f"Got exchange rates for {len(fx_df)} countries")
print("\nSample rates (LCU per USD):")
for c in ["USA", "JPN", "GBR", "KOR", "IDN", "VNM", "IND", "BRA"]:
    row = fx_df[fx_df["wb_code"] == c]
    if len(row):
        print(f"  {c}: {row['lcu_per_usd'].values[0]:.2f}")

# Map to fic
fic_to_wb = {k: v for k, v in WB_CODE_MAP.items() if v is not None}
df["wb_code"] = df["fic"].map(fic_to_wb)

# Merge exchange rates at country level (use average rate, not year-by-year,
# since we want a stable size comparison)
fx_map = dict(zip(fx_df["wb_code"], fx_df["lcu_per_usd"]))

# For Eurozone: use EMU/EUR rate
# EUR/USD is approximately the "Euro area" rate, but WB codes it differently
# Let's get EUR rate from Germany
eur_rate = fx_map.get("DEU", 1.0)  # DEU reports in EUR
print(f"\nEUR/USD rate (from DEU): {eur_rate:.4f}")

# For Eurozone, all should use EUR rate
for c in EUROZONE:
    if c in fic_to_wb:
        wb_c = fic_to_wb[c]
        fx_map[wb_c] = eur_rate

# USA = 1.0
fx_map["USA"] = 1.0

# ---------------------------------------------------------------------------
# 3. Convert assets to USD
# ---------------------------------------------------------------------------
df["fx_rate"] = df["wb_code"].map(fx_map)

# at is in millions of LCU. Convert: at_usd = at / fx_rate (millions USD)
df["at_usd"] = df["at"] / df["fx_rate"]
df["log_at_usd"] = np.log(df["at_usd"])

# Check: how many have valid conversion
n_valid = df["log_at_usd"].notna().sum()
n_total = len(df)
print(f"\nUSD conversion: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")

# Sanity check: compare local vs USD for key countries
print("\nAvg log(at) LOCAL vs USD:")
for c in ["USA", "JPN", "KOR", "IDN", "VNM", "DEU", "GBR", "IND", "BRA", "CHN"]:
    sub = df[df["fic"] == c]
    local = sub["log_at"].mean()
    usd = sub["log_at_usd"].mean()
    print(f"  {c}: local={local:.1f}  USD={usd:.1f}  diff={local-usd:+.1f}")

# ---------------------------------------------------------------------------
# 4. Country-level averages in USD
# ---------------------------------------------------------------------------
# Load country effects
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_mb = ce_mb.groupby("fic")["country_effect"].mean().rename("mb_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")

# Size in USD: average across firm-years per country
size_usd = df.groupby("fic", observed=True)["log_at_usd"].mean().rename("mean_log_at_usd")
size_local = df.groupby("fic", observed=True)["log_at"].mean().rename("mean_log_at_local")

country = pd.DataFrame(avg_pe).join([avg_mb, avg_roa, size_usd, size_local], how="outer")
country = country.dropna(subset=["mean_log_at_usd"]).reset_index()
print(f"\nCountries with USD size data: {len(country)}")

# ---------------------------------------------------------------------------
# 5. Correlations: local vs USD
# ---------------------------------------------------------------------------
print("\n=== CORRELATIONS: LOCAL CURRENCY SIZE ===")
print(f"  Size (local) vs P/E:  r = {country['mean_log_at_local'].corr(country['pe_effect']):+.2f}")
print(f"  Size (local) vs ROA:  r = {country['mean_log_at_local'].corr(country['roa_effect']):+.2f}")

print("\n=== CORRELATIONS: USD SIZE ===")
r_size_pe = country["mean_log_at_usd"].corr(country["pe_effect"])
r_size_roa = country["mean_log_at_usd"].corr(country["roa_effect"])
r_size_mb = country["mean_log_at_usd"].corr(country["mb_effect"])
print(f"  Size (USD) vs P/E:  r = {r_size_pe:+.2f}")
print(f"  Size (USD) vs M/B:  r = {r_size_mb:+.2f}")
print(f"  Size (USD) vs ROA:  r = {r_size_roa:+.2f}")

# Partial correlation: ROA vs P/E | size (USD)
sub = country[["pe_effect", "roa_effect", "mean_log_at_usd"]].dropna()
X = sm.add_constant(sub["mean_log_at_usd"])
roa_r = sm.OLS(sub["roa_effect"], X).fit().resid
pe_r = sm.OLS(sub["pe_effect"], X).fit().resid
r_partial = np.corrcoef(roa_r, pe_r)[0, 1]
r_raw = sub["roa_effect"].corr(sub["pe_effect"])
print(f"\n  ROA vs P/E (raw):         r = {r_raw:+.2f}")
print(f"  ROA vs P/E (| size USD):  r = {r_partial:+.2f}")

# Also with local for comparison
sub_l = country[["pe_effect", "roa_effect", "mean_log_at_local"]].dropna()
X_l = sm.add_constant(sub_l["mean_log_at_local"])
roa_rl = sm.OLS(sub_l["roa_effect"], X_l).fit().resid
pe_rl = sm.OLS(sub_l["pe_effect"], X_l).fit().resid
r_partial_local = np.corrcoef(roa_rl, pe_rl)[0, 1]
print(f"  ROA vs P/E (| size local): r = {r_partial_local:+.2f}")

# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------
print("\n=== REGRESSIONS ===")

# Spec 1: P/E ~ ROA
X1 = sm.add_constant(sub["roa_effect"])
m1 = sm.OLS(sub["pe_effect"], X1).fit()
print(f"\nSpec 1: P/E ~ ROA  (n={len(sub)})")
print(f"  ROA: {m1.params.iloc[1]:.2f} (t={m1.tvalues.iloc[1]:.2f}), R²={m1.rsquared:.3f}")

# Spec 2: P/E ~ ROA + Size (USD)
X2 = sm.add_constant(sub[["roa_effect", "mean_log_at_usd"]])
m2 = sm.OLS(sub["pe_effect"], X2).fit()
print(f"\nSpec 2: P/E ~ ROA + Size (USD)  (n={len(sub)})")
print(f"  ROA:       {m2.params.iloc[1]:.2f} (t={m2.tvalues.iloc[1]:.2f})")
print(f"  Size USD:  {m2.params.iloc[2]:.4f} (t={m2.tvalues.iloc[2]:.2f})")
print(f"  R²={m2.rsquared:.3f}")

# Spec 3: P/E ~ ROA + Size (local)
X3 = sm.add_constant(sub_l[["roa_effect", "mean_log_at_local"]])
m3 = sm.OLS(sub_l["pe_effect"], X3).fit()
print(f"\nSpec 3: P/E ~ ROA + Size (local)  (n={len(sub_l)})")
print(f"  ROA:        {m3.params.iloc[1]:.2f} (t={m3.tvalues.iloc[1]:.2f})")
print(f"  Size local: {m3.params.iloc[2]:.4f} (t={m3.tvalues.iloc[2]:.2f})")
print(f"  R²={m3.rsquared:.3f}")


# ---------------------------------------------------------------------------
# 7. Plots
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


# --- Scatter: Size USD vs P/E ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(country["mean_log_at_usd"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "mean_log_at_usd", "pe_effect")
_fit_line(ax, country["mean_log_at_usd"].values, country["pe_effect"].values)
ax.set_xlabel("Average log(Assets in USD) in country", fontsize=10)
ax.set_ylabel("Country P/E effect (vs USA)", fontsize=10)
ax.set_title(f"Firm Size (USD) vs P/E Effect (r = {r_size_pe:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_size_usd_vs_pe.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_size_usd_vs_pe.png", dpi=300)
plt.close(fig)
print("\nSaved scatter_size_usd_vs_pe")

# --- Scatter: Size USD vs ROA ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(country["mean_log_at_usd"], country["roa_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "mean_log_at_usd", "roa_effect")
_fit_line(ax, country["mean_log_at_usd"].values, country["roa_effect"].values)
ax.set_xlabel("Average log(Assets in USD) in country", fontsize=10)
ax.set_ylabel("Country ROA effect (vs USA)", fontsize=10)
ax.set_title(f"Firm Size (USD) vs ROA Effect (r = {r_size_roa:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_size_usd_vs_roa.pdf", dpi=300)
fig.savefig(OUT_DIR / "scatter_size_usd_vs_roa.png", dpi=300)
plt.close(fig)
print("Saved scatter_size_usd_vs_roa")

# --- Side-by-side: ROA vs P/E raw vs after partialing out USD size ---
country_sub = sub.copy()
country_sub = country_sub.merge(country[["fic"]], left_index=True, right_index=True, how="left")
# Re-merge fic
country_sub = country[["fic", "pe_effect", "roa_effect", "mean_log_at_usd"]].dropna()
X = sm.add_constant(country_sub["mean_log_at_usd"])
country_sub["roa_resid"] = sm.OLS(country_sub["roa_effect"], X).fit().resid.values
country_sub["pe_resid"] = sm.OLS(country_sub["pe_effect"], X).fit().resid.values

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(country_sub["roa_effect"], country_sub["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country_sub, "roa_effect", "pe_effect")
_fit_line(ax, country_sub["roa_effect"].values, country_sub["pe_effect"].values)
ax.set_xlabel("Country ROA effect", fontsize=10)
ax.set_ylabel("Country P/E effect", fontsize=10)
ax.set_title(f"Raw (r = {r_raw:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
ax.scatter(country_sub["roa_resid"], country_sub["pe_resid"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country_sub, "roa_resid", "pe_resid")
_fit_line(ax, country_sub["roa_resid"].values, country_sub["pe_resid"].values)
ax.set_xlabel("ROA effect (residual after USD size)", fontsize=10)
ax.set_ylabel("P/E effect (residual after USD size)", fontsize=10)
ax.set_title(f"After partialing out firm size in USD (r = {r_partial:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("ROA–P/E Correlation: Before and After Controlling for USD Firm Size",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_roa_pe_usd_size_resid.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_roa_pe_usd_size_resid.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_roa_pe_usd_size_resid")

# --- Comparison: local vs USD size ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(country["mean_log_at_local"], country["pe_effect"],
           color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "mean_log_at_local", "pe_effect")
_fit_line(ax, country["mean_log_at_local"].values, country["pe_effect"].values)
r_local = country["mean_log_at_local"].corr(country["pe_effect"])
ax.set_xlabel("Avg log(Assets) — local currency", fontsize=10)
ax.set_ylabel("P/E effect", fontsize=10)
ax.set_title(f"Local currency (r = {r_local:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
ax.scatter(country["mean_log_at_usd"], country["pe_effect"],
           color=PALETTE["highlight"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, country, "mean_log_at_usd", "pe_effect")
_fit_line(ax, country["mean_log_at_usd"].values, country["pe_effect"].values)
ax.set_xlabel("Avg log(Assets) — USD", fontsize=10)
ax.set_ylabel("P/E effect", fontsize=10)
ax.set_title(f"USD-converted (r = {r_size_pe:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Does Currency Conversion Change the Size–P/E Relationship?",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_size_local_vs_usd.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_size_local_vs_usd.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_size_local_vs_usd")

print("\nDone.")
