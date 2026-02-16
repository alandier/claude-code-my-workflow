"""
gdp_size_test.py -- Is the size-driven Simpson's paradox really a country-size effect?

Hypothesis: a $10B firm means different things in a large vs small economy.
Tests:
  1. After extracting country effects (with USD firm size), partial out GDP
  2. Use firm size relative to GDP: log(assets_USD / GDP) as the size control
  3. Compare: absolute USD size vs relative size vs GDP alone

Author: Augustin Landier, HEC Paris
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wbgapi as wb

np.random.seed(20260216)

OUT_DIR = Path("output/regressions")

# ---------------------------------------------------------------------------
# 1. Load panel + USD conversion (same as decompose_controls_usd.py)
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "panel_data.parquet")
print(f"Panel: {df.shape[0]:,} obs")

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

# --- Exchange rates ---
print("Fetching exchange rates...")
raw_fx = wb.data.DataFrame("PA.NUS.FCRF", economy=valid_wb, time=range(2007, 2024))
avg_fx = raw_fx.mean(axis=1)
avg_fx.index = avg_fx.index.get_level_values("economy")
fx_map = dict(zip(avg_fx.index, avg_fx.values))

eur_rate = fx_map.get("DEU", 1.0)
for c in EUROZONE:
    if c in WB_CODE_MAP and WB_CODE_MAP[c] is not None:
        fx_map[WB_CODE_MAP[c]] = eur_rate
fx_map["USA"] = 1.0

fic_to_wb = {k: v for k, v in WB_CODE_MAP.items() if v is not None}
df["wb_code"] = df["fic"].map(fic_to_wb)
df["fx_rate"] = df["wb_code"].map(fx_map)
df["at_usd"] = df["at"] / df["fx_rate"]
df["log_at_usd"] = np.log(df["at_usd"])

# --- GDP (current USD, annual) ---
print("Fetching GDP data...")
# NY.GDP.MKTP.CD = GDP (current US$)
raw_gdp = wb.data.DataFrame("NY.GDP.MKTP.CD", economy=valid_wb, time=range(2007, 2024))
# Average GDP over the period for each country
avg_gdp = raw_gdp.mean(axis=1)
avg_gdp.index = avg_gdp.index.get_level_values("economy")
gdp_map = dict(zip(avg_gdp.index, avg_gdp.values))

# Map fic -> GDP
df["gdp_usd"] = df["wb_code"].map(gdp_map)
# GDP is in USD. at_usd is in millions USD. Convert GDP to millions for consistency.
df["gdp_usd_m"] = df["gdp_usd"] / 1e6
df["log_gdp"] = np.log(df["gdp_usd_m"])

# Firm size relative to GDP: log(at_usd / gdp_usd) = log_at_usd - log_gdp_usd
# This measures how big the firm is relative to the economy
df["log_at_rel_gdp"] = df["log_at_usd"] - df["log_gdp"]

n_valid = df[["log_at_usd", "log_gdp", "log_at_rel_gdp"]].notna().all(axis=1).sum()
print(f"Obs with USD size + GDP: {n_valid:,} / {len(df):,}")

# Check: country-level GDP and firm size
print("\nCountry-level averages:")
print(f"{'Country':>5s}  {'log(at_USD)':>11s}  {'log(GDP)':>9s}  {'log(at/GDP)':>11s}")
for c in ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS",
          "SGP", "ISR", "NZL", "MUS"]:
    sub = df[df["fic"] == c]
    if len(sub) == 0:
        continue
    at = sub["log_at_usd"].mean()
    gdp = sub["log_gdp"].mean()
    rel = sub["log_at_rel_gdp"].mean()
    print(f"{c:>5s}  {at:>11.2f}  {gdp:>9.2f}  {rel:>11.2f}")

# ---------------------------------------------------------------------------
# 2. Year-by-year regressions with different size measures
# ---------------------------------------------------------------------------
def extract_country_effects(formula, data, years):
    """Run year-by-year regressions, extract country effects vs USA."""
    results = []
    for yr in years:
        sub = data[data["fyear"] == yr].copy()
        sub = sub.dropna(subset=["fic", "ggroup"])
        sub["fic"] = sub["fic"].astype(str)
        sub["ggroup"] = sub["ggroup"].astype(int).astype(str)

        dep_var = formula.split("~")[0].strip()
        drop_cols = [dep_var]
        for v in ["log_at_usd", "log_at_rel_gdp", "log_gdp", "leverage", "lag_earn_growth"]:
            if v in formula:
                drop_cols.append(v)
        sub = sub.dropna(subset=drop_cols)

        n_countries = sub["fic"].nunique()
        if n_countries < 5 or len(sub) < 500 or "USA" not in sub["fic"].values:
            continue

        try:
            model = smf.ols(formula, data=sub).fit()
            for param, coef in model.params.items():
                m = re.search(r"\[T\.(\w+)\]", param)
                if m and "fic" in param:
                    results.append({"fyear": yr, "fic": m.group(1), "country_effect": coef})
            results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
        except Exception as e:
            print(f"  Year {yr} ({dep_var}): {e}")
    return pd.DataFrame(results)


df_valid = df[df[["log_at_usd", "log_gdp", "log_at_rel_gdp"]].notna().all(axis=1)].copy()
years = sorted(df_valid["fyear"].dropna().unique())
print(f"\nUsing {len(df_valid):,} obs, {len(years)} years")

# --- Specifications ---
SPECS = {
    "No controls": (
        'log_pe ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "+ size (USD)": (
        'log_pe ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "+ size/GDP (relative)": (
        'log_pe ~ log_at_rel_gdp + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_rel_gdp + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
    "+ size (USD) + GDP": (
        'log_pe ~ log_at_usd + log_gdp + C(fic, Treatment(reference="USA")) + C(ggroup)',
        'roa ~ log_at_usd + log_gdp + C(fic, Treatment(reference="USA")) + C(ggroup)',
    ),
}

print("\n" + "=" * 80)
print("DECOMPOSITION: ABSOLUTE SIZE vs RELATIVE SIZE vs GDP")
print("=" * 80)

results_table = []

for label, (pe_formula, roa_formula) in SPECS.items():
    print(f"\n--- {label} ---")

    ce_pe = extract_country_effects(pe_formula, df_valid, years)
    ce_roa = extract_country_effects(roa_formula, df_valid, years)

    if len(ce_pe) == 0 or len(ce_roa) == 0:
        print("  No results")
        continue

    avg_pe = ce_pe.groupby("fic")["country_effect"].mean().rename("pe")
    avg_roa = ce_roa.groupby("fic")["country_effect"].mean().rename("roa")
    merged = pd.DataFrame(avg_pe).join(avg_roa, how="inner").dropna()
    r = merged["roa"].corr(merged["pe"])
    n = len(merged)
    print(f"  r(ROA, P/E) = {r:+.2f}  (n={n})")
    results_table.append({"spec": label, "r": r, "n": n})

# ---------------------------------------------------------------------------
# 3. Country-level test: partial out GDP from country effects
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("COUNTRY-LEVEL: PARTIAL OUT GDP FROM EXTRACTED EFFECTS")
print("=" * 80)

# Extract country effects with USD size control (our main spec)
ce_pe_size = extract_country_effects(
    'log_pe ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
    df_valid, years,
)
ce_roa_size = extract_country_effects(
    'roa ~ log_at_usd + C(fic, Treatment(reference="USA")) + C(ggroup)',
    df_valid, years,
)

avg_pe = ce_pe_size.groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_roa = ce_roa_size.groupby("fic")["country_effect"].mean().rename("roa_effect")

# Get country-level GDP
gdp_country = df_valid.groupby("fic", observed=True)["log_gdp"].mean().rename("log_gdp")

country = pd.DataFrame(avg_pe).join([avg_roa, gdp_country], how="inner").dropna()
country = country.reset_index()
print(f"\nCountries with effects + GDP: {len(country)}")

# Correlations
r_pe_gdp = country["pe_effect"].corr(country["log_gdp"])
r_roa_gdp = country["roa_effect"].corr(country["log_gdp"])
r_roa_pe = country["roa_effect"].corr(country["pe_effect"])
print(f"\n  P/E effect vs GDP:   r = {r_pe_gdp:+.2f}")
print(f"  ROA effect vs GDP:   r = {r_roa_gdp:+.2f}")
print(f"  ROA effect vs P/E:   r = {r_roa_pe:+.2f}")

# Partial correlation: ROA vs P/E | GDP
X = sm.add_constant(country["log_gdp"])
roa_resid = sm.OLS(country["roa_effect"], X).fit().resid
pe_resid = sm.OLS(country["pe_effect"], X).fit().resid
r_partial_gdp = np.corrcoef(roa_resid, pe_resid)[0, 1]
print(f"  ROA vs P/E | GDP:    r = {r_partial_gdp:+.2f}")

# Also get country-level avg firm size in USD
size_country = df_valid.groupby("fic", observed=True)["log_at_usd"].mean().rename("avg_size_usd")
country2 = country.merge(size_country.reset_index(), on="fic")

# Partial: ROA vs P/E | GDP + avg firm size
X2 = sm.add_constant(country2[["log_gdp", "avg_size_usd"]])
roa_resid2 = sm.OLS(country2["roa_effect"], X2).fit().resid
pe_resid2 = sm.OLS(country2["pe_effect"], X2).fit().resid
r_partial_both = np.corrcoef(roa_resid2, pe_resid2)[0, 1]
print(f"  ROA vs P/E | GDP + avg size:  r = {r_partial_both:+.2f}")

# GDP vs avg firm size correlation
r_gdp_size = country2["log_gdp"].corr(country2["avg_size_usd"])
print(f"\n  GDP vs avg firm size (USD):  r = {r_gdp_size:+.2f}")

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'Specification':<35s}  {'r(ROA,P/E)':>10s}  {'N':>4s}")
print("-" * 55)
for row in results_table:
    print(f"{row['spec']:<35s}  {row['r']:>+10.2f}  {row['n']:>4d}")

print(f"\nCountry-level partial correlations (after firm-level USD size control):")
print(f"  ROA vs P/E (raw):              r = {r_roa_pe:+.2f}")
print(f"  ROA vs P/E | GDP:              r = {r_partial_gdp:+.2f}")
print(f"  ROA vs P/E | GDP + avg size:   r = {r_partial_both:+.2f}")

print("\nDone.")
