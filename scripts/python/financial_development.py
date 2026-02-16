"""
financial_development.py -- Link ROA-valuation puzzle to financial development.

Tests whether financial development indicators (market cap/GDP, credit/GDP,
rule of law) explain the negative ROA-P/E correlation at the country level.

Hypothesis: underdeveloped capital markets → capital-starved firms →
high ROA + high discount rates (low P/E).

Data sources:
  - World Bank (wbgapi): market cap/GDP, credit/GDP, rule of law
  - La Porta et al. (2008): anti-self-dealing index (hand-coded)
  - Panel regressions output: country effects on P/E, M/B, ROA

Outputs:
  - scatter_findev_vs_pe.pdf/png
  - scatter_findev_vs_roa.pdf/png
  - scatter_findev_roa_pe_resid.pdf/png
  - financial_development_results.txt

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
# 1. Load country effects and size data
# ---------------------------------------------------------------------------
ce_pe = pd.read_parquet(OUT_DIR / "country_effects_pe.parquet")
ce_mb = pd.read_parquet(OUT_DIR / "country_effects_mb.parquet")
ce_roa = pd.read_parquet(OUT_DIR / "country_effects_roa.parquet")

avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_mb = ce_mb.groupby("fic")["country_effect"].mean().rename("mb_effect")
avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa_effect")

# Also load size from previous analysis
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
size_by_cy = df.groupby(["fic", "fyear"], observed=True)["log_at"].mean().reset_index()
size_avg = size_by_cy.groupby("fic")["log_at"].mean().rename("mean_log_at")

country = pd.DataFrame(avg_pe).join([avg_mb, avg_roa, size_avg], how="outer")
country = country.reset_index()
print(f"Countries with effects: {len(country)}")

# ---------------------------------------------------------------------------
# 2. Map Compustat country codes (ISO-3) to World Bank codes (ISO-3/2)
# ---------------------------------------------------------------------------
# Compustat uses ISO-3166 alpha-3; World Bank also uses ISO-3 but with some
# differences. Most map directly. Handle known exceptions.
WB_CODE_MAP = {
    "USA": "USA", "GBR": "GBR", "JPN": "JPN", "CHN": "CHN",
    "DEU": "DEU", "FRA": "FRA", "IND": "IND", "KOR": "KOR",
    "BRA": "BRA", "AUS": "AUS", "CAN": "CAN", "ITA": "ITA",
    "ESP": "ESP", "NLD": "NLD", "CHE": "CHE", "SWE": "SWE",
    "NOR": "NOR", "DNK": "DNK", "FIN": "FIN", "BEL": "BEL",
    "AUT": "AUT", "IRL": "IRL", "PRT": "PRT", "GRC": "GRC",
    "POL": "POL", "CZE": "CZE", "HUN": "HUN", "ROU": "ROU",
    "BGR": "BGR", "HRV": "HRV", "SVN": "SVN", "EST": "EST",
    "LTU": "LTU", "LVA": "LVA", "RUS": "RUS", "UKR": "UKR",
    "TUR": "TUR", "ISR": "ISR", "SAU": "SAU", "ARE": "ARE",
    "QAT": "QAT", "KWT": "KWT", "BHR": "BHR", "OMN": "OMN",
    "EGY": "EGY", "ZAF": "ZAF", "NGA": "NGA", "KEN": "KEN",
    "MAR": "MAR", "TUN": "TUN", "MYS": "MYS", "SGP": "SGP",
    "THA": "THA", "IDN": "IDN", "PHL": "PHL", "VNM": "VNM",
    "TWN": "TWN", "HKG": "HKG", "PAK": "PAK", "BGD": "BGD",
    "LKA": "LKA", "MEX": "MEX", "ARG": "ARG", "CHL": "CHL",
    "COL": "COL", "PER": "PER", "NZL": "NZL", "JOR": "JOR",
    "MUS": "MUS", "ZWE": "ZWE", "JAM": "JAM",
    # Offshore/special jurisdictions - no WB data
    "CYM": None, "BMU": None, "VGB": None, "GGY": None,
    "JEY": None, "IMN": None, "LUX": "LUX", "MHL": None,
    "MLT": "MLT", "CYP": "CYP", "SRB": "SRB", "PSE": "PSE",
}

country["wb_code"] = country["fic"].map(WB_CODE_MAP)

# ---------------------------------------------------------------------------
# 3. Fetch World Bank indicators
# ---------------------------------------------------------------------------
# Key indicators:
#   CM.MKT.LCAP.GD.ZS  - Stock market capitalization (% of GDP)
#   FS.AST.PRVT.GD.ZS  - Domestic credit to private sector (% of GDP)
#   NY.GDP.PCAP.PP.KD   - GDP per capita, PPP (constant 2021 intl $)
#   RL.EST              - Rule of Law estimate (WGI)

WB_INDICATORS = {
    "CM.MKT.LCAP.GD.ZS": "mktcap_gdp",
    "FS.AST.PRVT.GD.ZS": "credit_gdp",
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
}

# Governance indicators use a different database
WGI_INDICATORS = {
    "RL.EST": "rule_of_law",
}

valid_wb = country["wb_code"].dropna().unique().tolist()
print(f"Countries with WB codes: {len(valid_wb)}")

# Fetch main indicators (average over 2010-2023 for stability)
print("Fetching World Bank indicators...")
for wb_code, col_name in WB_INDICATORS.items():
    try:
        raw = wb.data.DataFrame(wb_code, economy=valid_wb, time=range(2010, 2024))
        # Average across years
        avg_val = raw.mean(axis=1)
        avg_val.index = avg_val.index.get_level_values("economy")
        avg_val.name = col_name
        # Map back to fic
        wb_to_fic = {v: k for k, v in WB_CODE_MAP.items() if v is not None}
        avg_df = avg_val.reset_index()
        avg_df.columns = ["wb_code", col_name]
        country = country.merge(avg_df, on="wb_code", how="left")
        n_valid = country[col_name].notna().sum()
        print(f"  {col_name}: {n_valid} countries with data")
    except Exception as e:
        print(f"  Warning: failed to fetch {wb_code}: {e}")
        country[col_name] = np.nan

# Fetch WGI (rule of law) - different API structure
print("Fetching governance indicators...")
try:
    raw = wb.data.DataFrame("RL.EST", economy=valid_wb, time=range(2010, 2023),
                            db=2)  # db=2 is WDI, try without db first
    avg_val = raw.mean(axis=1)
    avg_val.index = avg_val.index.get_level_values("economy")
    avg_val.name = "rule_of_law"
    avg_df = avg_val.reset_index()
    avg_df.columns = ["wb_code", "rule_of_law"]
    country = country.merge(avg_df, on="wb_code", how="left")
    print(f"  rule_of_law: {country['rule_of_law'].notna().sum()} countries")
except Exception as e:
    print(f"  Rule of law fetch failed: {e}")
    # Try alternative indicator
    try:
        raw = wb.data.DataFrame("CC.EST", economy=valid_wb, time=range(2010, 2023))
        avg_val = raw.mean(axis=1)
        avg_val.index = avg_val.index.get_level_values("economy")
        avg_val.name = "rule_of_law"
        avg_df = avg_val.reset_index()
        avg_df.columns = ["wb_code", "rule_of_law"]
        country = country.merge(avg_df, on="wb_code", how="left")
        print(f"  control_of_corruption (proxy): {country['rule_of_law'].notna().sum()}")
    except Exception as e2:
        print(f"  Also failed: {e2}")
        country["rule_of_law"] = np.nan

# ---------------------------------------------------------------------------
# 4. La Porta et al. (2008) Anti-Self-Dealing Index
# ---------------------------------------------------------------------------
# From Djankov, La Porta, Lopez-de-Silanes, Shleifer (2008, JFE)
# "The law and economics of self-dealing"
# Higher = better investor protection (0-1 scale)
ANTI_SELF_DEALING = {
    "GBR": 0.95, "SGP": 0.92, "HKG": 0.96, "NZL": 0.95, "CAN": 0.64,
    "USA": 0.65, "IND": 0.58, "IRL": 0.79, "ZAF": 0.81, "MYS": 0.95,
    "KEN": 0.34, "AUS": 0.76, "ISR": 0.71, "PAK": 0.41, "THA": 0.81,
    "FRA": 0.38, "JPN": 0.50, "DEU": 0.28, "KOR": 0.47, "NOR": 0.42,
    "DNK": 0.46, "SWE": 0.33, "FIN": 0.46, "CHE": 0.27, "NLD": 0.20,
    "BEL": 0.54, "AUT": 0.21, "ITA": 0.42, "ESP": 0.37, "PRT": 0.44,
    "GRC": 0.22, "TUR": 0.43, "BRA": 0.27, "MEX": 0.17, "ARG": 0.34,
    "CHL": 0.63, "COL": 0.57, "PER": 0.45, "IDN": 0.65, "PHL": 0.22,
    "CHN": 0.76, "EGY": 0.20, "JOR": 0.16, "NGA": 0.43, "LKA": 0.39,
    "POL": 0.29, "CZE": 0.33, "HUN": 0.18, "ROU": 0.44, "BGR": 0.65,
    "HRV": 0.25, "RUS": 0.44, "UKR": 0.23,
}

country["anti_self_dealing"] = country["fic"].map(ANTI_SELF_DEALING)
print(f"\nAnti-self-dealing: {country['anti_self_dealing'].notna().sum()} countries")

# ---------------------------------------------------------------------------
# 5. Composite financial development index
# ---------------------------------------------------------------------------
# Standardize and average available indicators
for col in ["mktcap_gdp", "credit_gdp", "gdp_pc_ppp", "rule_of_law", "anti_self_dealing"]:
    if col in country.columns:
        valid = country[col].dropna()
        if len(valid) > 5:
            country[f"{col}_z"] = (country[col] - valid.mean()) / valid.std()

# Composite = average of available z-scores
z_cols = [c for c in country.columns if c.endswith("_z")]
country["fin_dev_composite"] = country[z_cols].mean(axis=1)
print(f"Composite financial development: {country['fin_dev_composite'].notna().sum()} countries")

# ---------------------------------------------------------------------------
# 6. Correlations
# ---------------------------------------------------------------------------
print("\n=== CORRELATIONS WITH P/E EFFECT ===")
for col in ["mktcap_gdp", "credit_gdp", "gdp_pc_ppp", "rule_of_law",
            "anti_self_dealing", "fin_dev_composite", "mean_log_at"]:
    if col in country.columns:
        valid = country[["pe_effect", col]].dropna()
        if len(valid) > 10:
            r = valid["pe_effect"].corr(valid[col])
            print(f"  {col:25s}  r = {r:+.2f}  (n={len(valid)})")

print("\n=== CORRELATIONS WITH ROA EFFECT ===")
for col in ["mktcap_gdp", "credit_gdp", "gdp_pc_ppp", "rule_of_law",
            "anti_self_dealing", "fin_dev_composite", "mean_log_at"]:
    if col in country.columns:
        valid = country[["roa_effect", col]].dropna()
        if len(valid) > 10:
            r = valid["roa_effect"].corr(valid[col])
            print(f"  {col:25s}  r = {r:+.2f}  (n={len(valid)})")

# ---------------------------------------------------------------------------
# 7. Regressions: P/E ~ ROA + financial development
# ---------------------------------------------------------------------------
print("\n=== REGRESSIONS: P/E effect ===")

# Spec 1: P/E ~ ROA (baseline)
sub = country[["pe_effect", "roa_effect"]].dropna()
X1 = sm.add_constant(sub["roa_effect"])
m1 = sm.OLS(sub["pe_effect"], X1).fit()
print(f"\nSpec 1: P/E ~ ROA  (n={len(sub)})")
print(f"  ROA: {m1.params.iloc[1]:.2f} (t={m1.tvalues.iloc[1]:.2f}), R²={m1.rsquared:.3f}")

# Spec 2: P/E ~ ROA + mktcap/GDP
for fin_col, fin_label in [("mktcap_gdp", "Mkt Cap/GDP"),
                            ("credit_gdp", "Credit/GDP"),
                            ("gdp_pc_ppp", "GDP per capita"),
                            ("rule_of_law", "Rule of Law"),
                            ("anti_self_dealing", "Anti-Self-Dealing"),
                            ("fin_dev_composite", "Fin Dev Composite"),
                            ("mean_log_at", "Avg Firm Size")]:
    sub = country[["pe_effect", "roa_effect", fin_col]].dropna()
    if len(sub) < 15:
        continue
    X = sm.add_constant(sub[["roa_effect", fin_col]])
    m = sm.OLS(sub["pe_effect"], X).fit()
    print(f"\nSpec: P/E ~ ROA + {fin_label}  (n={len(sub)})")
    print(f"  ROA:         {m.params.iloc[1]:.2f} (t={m.tvalues.iloc[1]:.2f})")
    print(f"  {fin_label:15s} {m.params.iloc[2]:.4f} (t={m.tvalues.iloc[2]:.2f})")
    print(f"  R²={m.rsquared:.3f}")

# Horse race: P/E ~ ROA + size + mktcap/GDP
sub = country[["pe_effect", "roa_effect", "mean_log_at", "mktcap_gdp"]].dropna()
if len(sub) >= 15:
    X = sm.add_constant(sub[["roa_effect", "mean_log_at", "mktcap_gdp"]])
    m = sm.OLS(sub["pe_effect"], X).fit()
    print(f"\nHorse race: P/E ~ ROA + Size + MktCap/GDP  (n={len(sub)})")
    print(f"  ROA:        {m.params.iloc[1]:.2f} (t={m.tvalues.iloc[1]:.2f})")
    print(f"  Size:       {m.params.iloc[2]:.4f} (t={m.tvalues.iloc[2]:.2f})")
    print(f"  MktCap/GDP: {m.params.iloc[3]:.4f} (t={m.tvalues.iloc[3]:.2f})")
    print(f"  R²={m.rsquared:.3f}")

# ---------------------------------------------------------------------------
# 8. Partial correlations: ROA vs P/E after controlling for fin dev
# ---------------------------------------------------------------------------
print("\n=== PARTIAL CORRELATIONS: ROA vs P/E | fin dev ===")
for fin_col, fin_label in [("mktcap_gdp", "Mkt Cap/GDP"),
                            ("credit_gdp", "Credit/GDP"),
                            ("anti_self_dealing", "Anti-Self-Dealing"),
                            ("fin_dev_composite", "Fin Dev Composite"),
                            ("mean_log_at", "Avg Firm Size")]:
    sub = country[["pe_effect", "roa_effect", fin_col]].dropna()
    if len(sub) < 15:
        continue
    X = sm.add_constant(sub[fin_col])
    roa_r = sm.OLS(sub["roa_effect"], X).fit().resid
    pe_r = sm.OLS(sub["pe_effect"], X).fit().resid
    r_partial = np.corrcoef(roa_r, pe_r)[0, 1]
    r_raw = sub["roa_effect"].corr(sub["pe_effect"])
    print(f"  {fin_label:20s}  raw r={r_raw:+.2f}  partial r={r_partial:+.2f}  (n={len(sub)})")


# ---------------------------------------------------------------------------
# 9. Plots
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


# --- Plot A: Financial development vs P/E (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

plot_specs = [
    ("mktcap_gdp", "Stock Market Cap / GDP (%)", "pe_effect"),
    ("credit_gdp", "Private Credit / GDP (%)", "pe_effect"),
    ("anti_self_dealing", "Anti-Self-Dealing Index", "pe_effect"),
    ("gdp_pc_ppp", "GDP per capita (PPP, $K)", "pe_effect"),
]

for ax, (x_col, x_label, y_col) in zip(axes.flat, plot_specs):
    sub = country[[x_col, y_col, "fic"]].dropna()
    x_vals = sub[x_col].values
    if x_col == "gdp_pc_ppp":
        x_vals = x_vals / 1000  # Convert to $K
        sub = sub.copy()
        sub[x_col] = sub[x_col] / 1000
    r = np.corrcoef(sub[x_col].values, sub[y_col].values)[0, 1]
    ax.scatter(sub[x_col], sub[y_col], color=PALETTE["accent"],
               s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
    _label_points(ax, sub, x_col, y_col)
    _fit_line(ax, sub[x_col].values, sub[y_col].values)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("P/E effect (vs USA)", fontsize=9)
    ax.set_title(f"r = {r:+.2f} (n={len(sub)})", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Financial Development and Country P/E Effects", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_findev_vs_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_findev_vs_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved scatter_findev_vs_pe")

# --- Plot B: Financial development vs ROA (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

plot_specs_roa = [
    ("mktcap_gdp", "Stock Market Cap / GDP (%)", "roa_effect"),
    ("credit_gdp", "Private Credit / GDP (%)", "roa_effect"),
    ("anti_self_dealing", "Anti-Self-Dealing Index", "roa_effect"),
    ("gdp_pc_ppp", "GDP per capita (PPP, $K)", "roa_effect"),
]

for ax, (x_col, x_label, y_col) in zip(axes.flat, plot_specs_roa):
    sub = country[[x_col, y_col, "fic"]].dropna()
    if x_col == "gdp_pc_ppp":
        sub = sub.copy()
        sub[x_col] = sub[x_col] / 1000
    r = np.corrcoef(sub[x_col].values, sub[y_col].values)[0, 1]
    ax.scatter(sub[x_col], sub[y_col], color=PALETTE["accent"],
               s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
    _label_points(ax, sub, x_col, y_col)
    _fit_line(ax, sub[x_col].values, sub[y_col].values)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("ROA effect (vs USA)", fontsize=9)
    ax.set_title(f"r = {r:+.2f} (n={len(sub)})", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Financial Development and Country ROA Effects", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_findev_vs_roa.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_findev_vs_roa.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_findev_vs_roa")

# --- Plot C: ROA vs P/E, raw vs after partialing out composite fin dev ---
sub = country[["pe_effect", "roa_effect", "fin_dev_composite", "fic"]].dropna()
X = sm.add_constant(sub["fin_dev_composite"])
roa_r = sm.OLS(sub["roa_effect"], X).fit().resid
pe_r = sm.OLS(sub["pe_effect"], X).fit().resid
sub = sub.copy()
sub["roa_resid"] = roa_r.values
sub["pe_resid"] = pe_r.values
r_raw = sub["roa_effect"].corr(sub["pe_effect"])
r_partial = np.corrcoef(roa_r, pe_r)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter(sub["roa_effect"], sub["pe_effect"], color=PALETTE["accent"],
           s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, sub, "roa_effect", "pe_effect")
_fit_line(ax, sub["roa_effect"].values, sub["pe_effect"].values)
ax.set_xlabel("Country ROA effect", fontsize=10)
ax.set_ylabel("Country P/E effect", fontsize=10)
ax.set_title(f"Raw (r = {r_raw:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

ax = axes[1]
ax.scatter(sub["roa_resid"], sub["pe_resid"], color=PALETTE["highlight"],
           s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
_label_points(ax, sub, "roa_resid", "pe_resid")
_fit_line(ax, sub["roa_resid"].values, sub["pe_resid"].values)
ax.set_xlabel("ROA effect (residual)", fontsize=10)
ax.set_ylabel("P/E effect (residual)", fontsize=10)
ax.set_title(f"After partialing out fin dev composite (r = {r_partial:+.2f})", fontsize=11)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Does Financial Development Explain the Negative ROA–P/E Link?", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_findev_roa_pe_resid.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_findev_roa_pe_resid.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_findev_roa_pe_resid")

# ---------------------------------------------------------------------------
# 10. Save results
# ---------------------------------------------------------------------------
with open(OUT_DIR / "financial_development_results.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("FINANCIAL DEVELOPMENT AND THE ROA-VALUATION PUZZLE\n")
    f.write("=" * 70 + "\n\n")

    f.write("CORRELATIONS WITH P/E EFFECT:\n")
    for col in ["mktcap_gdp", "credit_gdp", "gdp_pc_ppp", "rule_of_law",
                "anti_self_dealing", "fin_dev_composite", "mean_log_at"]:
        if col in country.columns:
            valid = country[["pe_effect", col]].dropna()
            if len(valid) > 10:
                r = valid["pe_effect"].corr(valid[col])
                f.write(f"  {col:25s}  r = {r:+.3f}  (n={len(valid)})\n")

    f.write("\nCORRELATIONS WITH ROA EFFECT:\n")
    for col in ["mktcap_gdp", "credit_gdp", "gdp_pc_ppp", "rule_of_law",
                "anti_self_dealing", "fin_dev_composite", "mean_log_at"]:
        if col in country.columns:
            valid = country[["roa_effect", col]].dropna()
            if len(valid) > 10:
                r = valid["roa_effect"].corr(valid[col])
                f.write(f"  {col:25s}  r = {r:+.3f}  (n={len(valid)})\n")

    f.write(f"\nPARTIAL CORRELATIONS: ROA vs P/E after controlling for...\n")
    for fin_col, fin_label in [("mktcap_gdp", "Mkt Cap/GDP"),
                                ("credit_gdp", "Credit/GDP"),
                                ("anti_self_dealing", "Anti-Self-Dealing"),
                                ("fin_dev_composite", "Fin Dev Composite"),
                                ("mean_log_at", "Avg Firm Size")]:
        sub2 = country[["pe_effect", "roa_effect", fin_col]].dropna()
        if len(sub2) < 15:
            continue
        Xp = sm.add_constant(sub2[fin_col])
        roa_rp = sm.OLS(sub2["roa_effect"], Xp).fit().resid
        pe_rp = sm.OLS(sub2["pe_effect"], Xp).fit().resid
        rp = np.corrcoef(roa_rp, pe_rp)[0, 1]
        r_raw2 = sub2["roa_effect"].corr(sub2["pe_effect"])
        f.write(f"  {fin_label:20s}  raw={r_raw2:+.2f}  partial={rp:+.2f}  (n={len(sub2)})\n")

print("\nSaved financial_development_results.txt")
print("\nDone.")
