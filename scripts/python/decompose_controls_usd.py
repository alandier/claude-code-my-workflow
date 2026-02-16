"""
decompose_controls_usd.py -- Redo control decomposition with USD-converted size.

The previous decomposition used log(assets) in local currency as the size
control. Since currency denomination inflates log(at) for countries like
Japan (¥) and Korea (₩), we re-run using USD-converted assets to check
whether the Simpson's paradox finding is robust.

Author: Augustin Landier, HEC Paris
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wbgapi as wb

np.random.seed(20260216)

OUT_DIR = Path("output/regressions")

# ---------------------------------------------------------------------------
# 1. Load panel and add USD-converted size
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs")

# --- Exchange rates from World Bank ---
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

EUROZONE = {"DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL", "PRT",
            "GRC", "FIN", "LUX", "SVN", "CYP", "MLT", "EST", "LTU", "HRV"}

valid_wb = [v for v in set(WB_CODE_MAP.values()) if v is not None]
print(f"Fetching exchange rates for {len(valid_wb)} countries...")

raw_fx = wb.data.DataFrame("PA.NUS.FCRF", economy=valid_wb, time=range(2007, 2024))
avg_fx = raw_fx.mean(axis=1)
avg_fx.index = avg_fx.index.get_level_values("economy")
fx_map = dict(zip(avg_fx.index, avg_fx.values))

# Eurozone: all use EUR rate from DEU
eur_rate = fx_map.get("DEU", 1.0)
for c in EUROZONE:
    if c in WB_CODE_MAP and WB_CODE_MAP[c] is not None:
        fx_map[WB_CODE_MAP[c]] = eur_rate

# USA = 1.0
fx_map["USA"] = 1.0

# Map fic -> fx rate
fic_to_wb = {k: v for k, v in WB_CODE_MAP.items() if v is not None}
df["wb_code"] = df["fic"].map(fic_to_wb)
df["fx_rate"] = df["wb_code"].map(fx_map)
df["at_usd"] = df["at"] / df["fx_rate"]
df["log_at_usd"] = np.log(df["at_usd"])

n_valid = df["log_at_usd"].notna().sum()
print(f"USD conversion: {n_valid:,} / {len(df):,} ({100*n_valid/len(df):.1f}%)")

# ---------------------------------------------------------------------------
# 2. Year-by-year regressions with different control sets
# ---------------------------------------------------------------------------
# We'll run regressions using log_at_usd instead of log_at (local currency)

def extract_country_effects(formula, data, years):
    """Run year-by-year regressions, extract country effects vs USA."""
    results = []
    for yr in years:
        sub = data[data["fyear"] == yr].copy()
        sub = sub.dropna(subset=["fic", "ggroup"])
        sub["fic"] = sub["fic"].astype(str)
        sub["ggroup"] = sub["ggroup"].astype(int).astype(str)

        # Drop rows with NaN in any variable used in formula
        # Parse variable names from formula
        dep_var = formula.split("~")[0].strip()
        sub = sub.dropna(subset=[dep_var])
        if "log_at_usd" in formula:
            sub = sub.dropna(subset=["log_at_usd"])
        if "log_at" in formula and "log_at_usd" not in formula:
            sub = sub.dropna(subset=["log_at"])
        if "leverage" in formula:
            sub = sub.dropna(subset=["leverage"])
        if "lag_earn_growth" in formula:
            sub = sub.dropna(subset=["lag_earn_growth"])

        n_countries = sub["fic"].nunique()
        if n_countries < 5 or len(sub) < 500:
            continue
        if "USA" not in sub["fic"].values:
            continue

        try:
            model = smf.ols(formula, data=sub).fit()
            for param, coef in model.params.items():
                m = re.search(r"\[T\.(\w+)\]", param)
                if m and "fic" in param:
                    fic = m.group(1)
                    results.append({"fyear": yr, "fic": fic, "country_effect": coef})
            # USA = reference = 0
            results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
        except Exception as e:
            print(f"  Year {yr} ({dep_var}): {e}")
            continue

    return pd.DataFrame(results)


years = sorted(df["fyear"].dropna().unique())

# Define control sets -- now using log_at_usd instead of log_at
SPECS = {
    # (P/E controls, ROA controls)
    "No controls": (
        'log_pe ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "+ size (USD)": (
        'log_pe ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "P/E: +size, ROA: none": (
        'log_pe ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "P/E: none, ROA: +size": (
        'log_pe ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "+ size + lev (USD)": (
        'log_pe ~ log_at_usd + leverage + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_usd + leverage + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "Full spec (USD)": (
        'log_pe ~ log_at_usd + leverage + lag_earn_growth + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_usd + leverage + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
}

# Also re-run with local currency for comparison
SPECS_LOCAL = {
    "No controls": (
        'log_pe ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "+ size (local)": (
        'log_pe ~ log_at + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "Full spec (local)": (
        'log_pe ~ log_at + leverage + lag_earn_growth + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at + leverage + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
}

# Restrict to observations with USD size for consistent sample
df_usd = df[df["log_at_usd"].notna()].copy()
print(f"\nUsing {len(df_usd):,} obs with USD-converted size")

print("\n" + "=" * 80)
print("DECOMPOSITION WITH USD-CONVERTED SIZE")
print("=" * 80)

results_table = []

for label, (pe_formula, roa_formula) in SPECS.items():
    print(f"\n--- {label} ---")

    ce_pe = extract_country_effects(pe_formula, df_usd, years)
    ce_roa = extract_country_effects(roa_formula, df_usd, years)

    if len(ce_pe) == 0 or len(ce_roa) == 0:
        print("  No results")
        continue

    avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe")
    avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa")
    merged = pd.DataFrame(avg_pe).join(avg_roa, how="inner").dropna()
    r = merged["roa"].corr(merged["pe"])
    n = len(merged)
    print(f"  r(ROA, P/E) = {r:+.2f}  (n={n})")
    results_table.append({"spec": label, "r": r, "n": n, "currency": "USD"})

# Also run local currency specs for comparison
print("\n" + "=" * 80)
print("COMPARISON WITH LOCAL CURRENCY SIZE")
print("=" * 80)

for label, (pe_formula, roa_formula) in SPECS_LOCAL.items():
    print(f"\n--- {label} ---")

    ce_pe = extract_country_effects(pe_formula, df_usd, years)
    ce_roa = extract_country_effects(roa_formula, df_usd, years)

    if len(ce_pe) == 0 or len(ce_roa) == 0:
        print("  No results")
        continue

    avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe")
    avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa")
    merged = pd.DataFrame(avg_pe).join(avg_roa, how="inner").dropna()
    r = merged["roa"].corr(merged["pe"])
    n = len(merged)
    print(f"  r(ROA, P/E) = {r:+.2f}  (n={n})")
    results_table.append({"spec": label, "r": r, "n": n, "currency": "local"})

# ---------------------------------------------------------------------------
# 3. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY: LOCAL vs USD SIZE IN DECOMPOSITION")
print("=" * 80)
print(f"\n{'Specification':<35s}  {'Currency':<8s}  {'r(ROA,P/E)':>10s}  {'N':>4s}")
print("-" * 65)
for row in results_table:
    print(f"{row['spec']:<35s}  {row['currency']:<8s}  {row['r']:>+10.2f}  {row['n']:>4d}")

print("\nDone.")
