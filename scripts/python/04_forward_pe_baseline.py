"""
04_forward_pe_baseline.py -- Forward P/E as baseline dependent variable.

Structure:
  1. Build forward P/E panel (IBES FY1/FY2 + Compustat)
  2. Fetch country-year macro controls (IMF WEO growth, WB real rate)
  3. Pooled panel regressions with progressive controls
  4. Year-by-year regressions for time-varying country effects
  5. Second-stage: explain residual country FE with savings + fin dev
  6. Generate figures

Data availability constraints:
  - IBES consensus: 2000+ (but most international coverage from ~2003)
  - Compustat Global returns (market cap): 2007+
  - IMF WEO editions: Oct 2007 -- Apr 2025
  - World Bank real rate: ~2000-2024 (varies by country)
  - Effective international panel: ~2007-2024
  - US panel: ~2000-2024

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import wbgapi as wb
import weo

np.random.seed(20260217)

DATA_DIR = Path(
    "~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/"
).expanduser()

OUT_DIR = Path("output/regressions")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "wrds_cache"

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
    "highlight": "#8e44ad",
}

FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]

# La Porta et al. (2008) Anti-Self-Dealing Index
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

IBES_SMALL_UNIT = {"BPN": 100}  # British Pence → divide by 100

# ---------------------------------------------------------------------------
# 1. Load IBES cached data + build link
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading IBES data and building link tables")
print("=" * 70)


def load_cached(name):
    """Load parquet from WRDS cache."""
    path = CACHE / f"{name}.parquet"
    df = pd.read_parquet(path)
    print(f"  {name}: {len(df):,} rows")
    return df


# Link tables
us_link = load_cached("iclink_ccm_link")[["ticker", "gvkey"]].drop_duplicates()
intl_link = load_cached("ibtic_gvkey_link")[["ticker", "gvkey"]].drop_duplicates()
link = pd.concat([us_link, intl_link], ignore_index=True).drop_duplicates(
    "ticker", keep="first"
)
print(f"  Combined link: {len(link):,} unique tickers")

# IBES consensus data
us_fy1 = load_cached("ibes_us_fy1")
int_fy1 = load_cached("ibes_int_fy1")
us_fy2 = load_cached("ibes_us_fy2")
int_fy2 = load_cached("ibes_int_fy2")
us_ltg = load_cached("ibes_us_ltg")
int_ltg = load_cached("ibes_int_ltg")


# ---------------------------------------------------------------------------
# 2. Prepare firm-year IBES data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Preparing firm-year consensus data")
print("=" * 70)


def prepare_fy(raw, link, eps_col="fy1_eps", source=""):
    """For each gvkey-fyear, keep latest consensus before fiscal year end."""
    if len(raw) == 0:
        return pd.DataFrame(columns=["gvkey", "fyear", eps_col])
    df = raw.merge(link, on="ticker", how="inner")
    # BPN currency fix
    if "curr_act" in df.columns:
        for cur, factor in IBES_SMALL_UNIT.items():
            mask = df["curr_act"] == cur
            if mask.any():
                df.loc[mask, eps_col] = df.loc[mask, eps_col] / factor
                print(f"    Converted {mask.sum():,} {source} rows from {cur}")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    df["fyear"] = df["fpedats"].dt.year
    df = df[df["statpers"] <= df["fpedats"]]
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    out = df[["gvkey", "fyear", eps_col]].copy()
    if "numest" in df.columns:
        out["numest"] = df["numest"].values
    print(f"  {source}: {len(out):,} firm-years")
    return out


def prepare_fy2(raw, link, source=""):
    """FY2: fpedats is two years ahead, align with FY1 panel year."""
    if len(raw) == 0:
        return pd.DataFrame(columns=["gvkey", "fyear", "fy2_eps"])
    df = raw.merge(link, on="ticker", how="inner")
    if "curr_act" in df.columns:
        for cur, factor in IBES_SMALL_UNIT.items():
            mask = df["curr_act"] == cur
            if mask.any():
                df.loc[mask, "fy2_eps"] = df.loc[mask, "fy2_eps"] / factor
                print(f"    Converted {mask.sum():,} {source} FY2 rows from {cur}")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    df["fyear"] = df["fpedats"].dt.year - 1  # align with FY1 panel year
    df = df[df["statpers"] <= df["fpedats"]]
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    out = df[["gvkey", "fyear", "fy2_eps"]].copy()
    print(f"  {source} FY2: {len(out):,} firm-years")
    return out


def prepare_ltg(raw, link, source=""):
    """For each gvkey-year, keep latest LTG consensus."""
    if len(raw) == 0:
        return pd.DataFrame(columns=["gvkey", "fyear", "ltg"])
    df = raw.merge(link, on="ticker", how="inner")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fyear"] = df["statpers"].dt.year
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    out = df[["gvkey", "fyear", "ltg"]].copy()
    print(f"  {source} LTG: {len(out):,} firm-years")
    return out


fy1 = pd.concat(
    [prepare_fy(us_fy1, link, "fy1_eps", "US"),
     prepare_fy(int_fy1, link, "fy1_eps", "Intl")],
    ignore_index=True,
).drop_duplicates(["gvkey", "fyear"], keep="first")

fy2 = pd.concat(
    [prepare_fy2(us_fy2, link, "US"),
     prepare_fy2(int_fy2, link, "Intl")],
    ignore_index=True,
).drop_duplicates(["gvkey", "fyear"], keep="first")

ltg = pd.concat(
    [prepare_ltg(us_ltg, link, "US"),
     prepare_ltg(int_ltg, link, "Intl")],
    ignore_index=True,
).drop_duplicates(["gvkey", "fyear"], keep="first")

print(f"\nTotal FY1: {len(fy1):,}, FY2: {len(fy2):,}, LTG: {len(ltg):,}")

# ---------------------------------------------------------------------------
# 3. Load panel data and merge
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Building forward P/E panel")
print("=" * 70)

panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "at", "ib", "market_cap",
             "ggroup", "log_at", "log_pe", "leverage", "lag_earn_growth"],
)
panel["fic"] = panel["fic"].astype(str)
print(f"Panel: {len(panel):,} obs, {panel['fic'].nunique()} countries")

# R&D intensity
print("Loading R&D...")
xrd_us = pd.read_csv(
    DATA_DIR / "compustat_america.csv",
    usecols=["GVKEY", "fyear", "xrd", "at"],
    dtype={"GVKEY": str},
    low_memory=False,
).rename(columns={"GVKEY": "gvkey"})
xrd_gl = pd.read_csv(
    DATA_DIR / "compustat_global.csv",
    usecols=["gvkey", "fyear", "xrd", "at"],
    dtype={"gvkey": str, "fyear": "Int64"},
    low_memory=False,
)
xrd = pd.concat([xrd_us, xrd_gl], ignore_index=True)
xrd["xrd"] = xrd["xrd"].fillna(0)
xrd.loc[xrd["at"] <= 0, "at"] = np.nan
xrd["xrd_at"] = xrd["xrd"] / xrd["at"]
xrd = xrd[["gvkey", "fyear", "xrd_at"]].dropna()
xrd = xrd[np.isfinite(xrd["xrd_at"])]
xrd["fyear"] = xrd["fyear"].astype(float)

# Shares outstanding
print("Loading shares outstanding...")
us_sh = pd.read_csv(
    DATA_DIR / "compustat_america.csv",
    usecols=["GVKEY", "fyear", "csho"],
    dtype={"GVKEY": str},
    low_memory=False,
).rename(columns={"GVKEY": "gvkey"}).dropna(subset=["csho"])
gl_sh = pd.read_csv(
    DATA_DIR / "compustat_global.csv",
    usecols=["gvkey", "fyear", "cshoi"],
    dtype={"gvkey": str, "fyear": "Int64"},
    low_memory=False,
).dropna(subset=["cshoi", "fyear"]).rename(columns={"cshoi": "csho"})
shares = pd.concat([us_sh, gl_sh], ignore_index=True).drop_duplicates(
    ["gvkey", "fyear"], keep="first"
)

# Merge all
panel = panel.merge(xrd, on=["gvkey", "fyear"], how="left")
panel["xrd_at"] = panel["xrd_at"].fillna(0)
panel = panel.merge(shares, on=["gvkey", "fyear"], how="left")
panel = panel.merge(fy1, on=["gvkey", "fyear"], how="left")
panel = panel.merge(fy2[["gvkey", "fyear", "fy2_eps"]], on=["gvkey", "fyear"], how="left")
panel = panel.merge(ltg, on=["gvkey", "fyear"], how="left")

# Compute forward P/E (FY1)
panel["fwd_earnings"] = panel["fy1_eps"] * panel["csho"]
panel.loc[panel["fwd_earnings"] <= 0, "fwd_earnings"] = np.nan
panel["fwd_pe"] = panel["market_cap"] / panel["fwd_earnings"]
lo, hi = panel["fwd_pe"].dropna().quantile([0.01, 0.99])
panel["fwd_pe"] = panel["fwd_pe"].clip(lo, hi)
panel["log_fwd_pe"] = np.log(panel["fwd_pe"])

# Compute forward P/E (FY2)
panel["fwd_earnings_fy2"] = panel["fy2_eps"] * panel["csho"]
panel.loc[panel["fwd_earnings_fy2"] <= 0, "fwd_earnings_fy2"] = np.nan
panel["fwd_pe_fy2"] = panel["market_cap"] / panel["fwd_earnings_fy2"]
lo2, hi2 = panel["fwd_pe_fy2"].dropna().quantile([0.01, 0.99])
panel["fwd_pe_fy2"] = panel["fwd_pe_fy2"].clip(lo2, hi2)
panel["log_fwd_pe_fy2"] = np.log(panel["fwd_pe_fy2"])

n_fwd = panel["log_fwd_pe"].notna().sum()
n_fwd2 = panel["log_fwd_pe_fy2"].notna().sum()
n_ltg = panel["ltg"].notna().sum()
c_fwd = panel.loc[panel["log_fwd_pe"].notna(), "fic"].nunique()
yr_fwd = panel.loc[panel["log_fwd_pe"].notna(), "fyear"]

print(f"\n  Forward P/E (FY1): {n_fwd:,} obs, {c_fwd} countries, "
      f"years {yr_fwd.min():.0f}-{yr_fwd.max():.0f}")
print(f"  Forward P/E (FY2): {n_fwd2:,} obs")
print(f"  LTG available:     {n_ltg:,} obs")

# Data availability summary
print("\n  --- Data Availability by Variable ---")
for col, label in [("log_pe", "Trailing P/E"), ("log_fwd_pe", "Forward P/E FY1"),
                    ("log_fwd_pe_fy2", "Forward P/E FY2"), ("ltg", "IBES LTG")]:
    sub = panel.loc[panel[col].notna()]
    if len(sub) > 0:
        print(f"  {label:20s}: {len(sub):>8,} obs, "
              f"years {sub['fyear'].min():.0f}-{sub['fyear'].max():.0f}, "
              f"{sub['fic'].nunique()} countries")

# ---------------------------------------------------------------------------
# 4. Fetch country-year macro controls
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Fetching country-year macro controls")
print("=" * 70)

# --- 4a. IMF WEO growth forecasts ---
# For each country-fyear, compute 5-year-ahead average GDP growth forecast
# from the most recent WEO edition before fiscal year end.
print("\nFetching IMF WEO growth forecasts...")

WEO_CACHE = OUT_DIR / "weo_growth_forecasts.parquet"
if WEO_CACHE.exists():
    weo_panel = pd.read_parquet(WEO_CACHE)
    print(f"  Loaded from cache: {len(weo_panel):,} country-years")
else:
    # Download editions from Oct 2007 to Oct 2024
    # Note: weo package only has machine-readable CSVs from Oct 2007 onward.
    # Each edition contains both historical (realized) and forecast values.
    # For ex-ante forecasts, we use only forward-looking years from each edition.
    editions = []
    for year in range(2007, 2025):
        for release in [1, 2]:  # 1=April, 2=October
            editions.append((year, release))

    weo_rows = []
    for yr, rel in editions:
        try:
            path, _ = weo.download(yr, rel)
            w = weo.WEO(path)
            # API changed between editions: try both unit names
            try:
                gdp = w.get("Gross domestic product, constant prices",
                            "Percent change")
            except Exception:
                gdp = w.get("Gross domestic product, constant prices",
                            "Annual percent change")
            edition_label = f"{yr}-{'Apr' if rel == 1 else 'Oct'}"

            for iso3 in gdp.columns:
                # 5-year-ahead average: years t+1 to t+5
                for base_year in range(2000, 2030):
                    forecasts = []
                    for ahead in range(1, 6):
                        target_yr = base_year + ahead
                        try:
                            val = gdp.loc[str(target_yr), iso3]
                            if pd.notna(val):
                                forecasts.append(float(val))
                        except (KeyError, TypeError):
                            pass
                    if len(forecasts) >= 3:  # need at least 3 of 5 years
                        weo_rows.append({
                            "fic": iso3,
                            "fyear": float(base_year),
                            "weo_year": yr,
                            "weo_release": rel,
                            "imf_growth_5y": np.mean(forecasts),
                        })
            print(f"  {edition_label}: OK ({len(gdp.columns)} countries, "
                  f"years {gdp.index[0]}-{gdp.index[-1]})")
        except Exception as e:
            print(f"  {yr}-{'Apr' if rel == 1 else 'Oct'}: FAILED ({e})")

    weo_all = pd.DataFrame(weo_rows)

    # For each country-fyear, keep the most recent edition published
    # before or during the fiscal year (ensures ex-ante forecast)
    weo_all["edition_order"] = weo_all["weo_year"] * 2 + weo_all["weo_release"]
    weo_all = weo_all[weo_all["weo_year"] <= weo_all["fyear"]]
    weo_panel = (weo_all.sort_values("edition_order")
                 .drop_duplicates(["fic", "fyear"], keep="last")
                 [["fic", "fyear", "imf_growth_5y"]])
    weo_panel.to_parquet(WEO_CACHE, index=False)
    print(f"  Saved: {len(weo_panel):,} country-years")

print(f"  WEO panel: {weo_panel['fic'].nunique()} countries, "
      f"years {weo_panel['fyear'].min():.0f}-{weo_panel['fyear'].max():.0f}")

# --- 4b. World Bank real interest rate (country-year panel) ---
print("\nFetching World Bank real interest rate...")
WB_RATE_CACHE = OUT_DIR / "wb_real_rate_panel.parquet"
if WB_RATE_CACHE.exists():
    wb_rate = pd.read_parquet(WB_RATE_CACHE)
    print(f"  Loaded from cache: {len(wb_rate):,} country-years")
else:
    try:
        raw = wb.data.DataFrame("FR.INR.RINR", time=range(2000, 2025), labels=False)
        # raw: index=economy, columns=time periods (YR2000 etc.)
        wb_rate_rows = []
        for iso3 in raw.index.get_level_values("economy").unique():
            row = raw.loc[iso3]
            for col in row.index:
                yr_str = str(col).replace("YR", "")
                try:
                    yr = int(yr_str)
                    val = row[col]
                    if pd.notna(val):
                        wb_rate_rows.append({"fic": iso3, "fyear": float(yr),
                                             "real_rate": float(val)})
                except (ValueError, TypeError):
                    pass
        wb_rate = pd.DataFrame(wb_rate_rows)
        wb_rate.to_parquet(WB_RATE_CACHE, index=False)
        print(f"  Downloaded: {len(wb_rate):,} country-years, "
              f"{wb_rate['fic'].nunique()} countries")
    except Exception as e:
        print(f"  WARNING: Failed to fetch real rate: {e}")
        wb_rate = pd.DataFrame(columns=["fic", "fyear", "real_rate"])

# --- 4c. Cross-sectional country-level variables (for second stage) ---
print("\nFetching cross-sectional country variables...")

# Savings / GNI (average 2005-2023)
sav = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2005, 2024), labels=False)
sav_avg = sav.T.mean().rename("savings_gni")
sav_avg.index.name = "fic"
print(f"  Savings/GNI: {sav_avg.notna().sum()} countries")

# Market cap / GDP
mktcap = wb.data.DataFrame("CM.MKT.LCAP.GD.ZS", time=range(2010, 2024), labels=False)
mktcap_avg = mktcap.T.mean().rename("mktcap_gdp")
mktcap_avg.index.name = "fic"
print(f"  Mktcap/GDP: {mktcap_avg.notna().sum()} countries")

# Credit / GDP
credit = wb.data.DataFrame("FS.AST.PRVT.GD.ZS", time=range(2010, 2024), labels=False)
credit_avg = credit.T.mean().rename("credit_gdp")
credit_avg.index.name = "fic"
print(f"  Credit/GDP: {credit_avg.notna().sum()} countries")

# GDP per capita PPP
gdppc = wb.data.DataFrame("NY.GDP.PCAP.PP.KD", time=range(2010, 2024), labels=False)
gdppc_avg = gdppc.T.mean().rename("gdp_pc_ppp")
gdppc_avg.index.name = "fic"
print(f"  GDP/cap PPP: {gdppc_avg.notna().sum()} countries")

# ---------------------------------------------------------------------------
# 5. Merge macro controls into panel
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Merging macro controls into panel")
print("=" * 70)

panel = panel.merge(weo_panel, on=["fic", "fyear"], how="left")
panel = panel.merge(wb_rate, on=["fic", "fyear"], how="left")

# Coverage of macro controls
for col, label in [("imf_growth_5y", "IMF growth"), ("real_rate", "Real rate")]:
    sub = panel.loc[panel[col].notna() & panel["log_fwd_pe"].notna()]
    print(f"  {label:15s} + fwd PE: {len(sub):>8,} obs, "
          f"{sub['fic'].nunique()} countries, "
          f"years {sub['fyear'].min():.0f}-{sub['fyear'].max():.0f}")

# ---------------------------------------------------------------------------
# 6. Pooled panel regressions (first stage)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Pooled panel regressions")
print("=" * 70)

# Define specifications with progressive controls
SPECS = [
    ("(1) Base",
     "log_fwd_pe ~ leverage + log_at + lag_earn_growth"
     " + C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear)"),
    ("(2) + R&D",
     "log_fwd_pe ~ leverage + log_at + lag_earn_growth + xrd_at"
     " + C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear)"),
    ("(3) + LTG",
     "log_fwd_pe ~ leverage + log_at + lag_earn_growth + xrd_at + ltg"
     " + C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear)"),
    ("(4) + Macro",
     "log_fwd_pe ~ leverage + log_at + lag_earn_growth + xrd_at + ltg"
     " + real_rate + imf_growth_5y"
     " + C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear)"),
    ("(5) FY2 + all",
     "log_fwd_pe_fy2 ~ leverage + log_at + lag_earn_growth + xrd_at + ltg"
     " + C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear)"),
]

reg_results = {}
for spec_name, formula in SPECS:
    dep = formula.split("~")[0].strip()
    # Identify required columns from formula
    req_cols = [dep, "leverage", "log_at", "lag_earn_growth", "fic", "ggroup", "fyear"]
    for extra in ["xrd_at", "ltg", "real_rate", "imf_growth_5y"]:
        if extra in formula:
            req_cols.append(extra)

    sub = panel.dropna(subset=req_cols)
    if sub["fic"].nunique() < 5:
        print(f"\n{spec_name}: too few countries, skipping")
        continue

    print(f"\n{spec_name}: n={len(sub):,}, countries={sub['fic'].nunique()}, "
          f"years={sub['fyear'].min():.0f}-{sub['fyear'].max():.0f}")

    m = sm.OLS.from_formula(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["fic"]}
    )

    # Extract country effects
    country_fe = {}
    for name, coef in m.params.items():
        if "C(fic" in name and "[T." in name:
            iso = name.split("[T.")[1].rstrip("]")
            country_fe[iso] = coef
    country_fe["USA"] = 0.0
    sigma_fe = pd.Series(country_fe).std()

    # Report key coefficients
    print(f"  R² = {m.rsquared:.3f}, σ(country FE) = {sigma_fe:.3f}")
    for var in ["leverage", "log_at", "lag_earn_growth", "xrd_at", "ltg",
                "real_rate", "imf_growth_5y"]:
        if var in m.params.index:
            print(f"    {var:20s}: {m.params[var]:+.4f} "
                  f"(t={m.tvalues[var]:+.2f})")

    reg_results[spec_name] = {
        "model": m, "n": len(sub), "r2": m.rsquared,
        "sigma_fe": sigma_fe, "country_fe": country_fe,
        "dep": dep, "n_countries": sub["fic"].nunique(),
        "year_range": f"{sub['fyear'].min():.0f}-{sub['fyear'].max():.0f}",
    }

# Print comparison table
print("\n--- Progressive Controls Summary ---")
print(f"  {'Spec':20s}  {'N':>8s}  {'R²':>6s}  {'σ(FE)':>6s}  {'Countries':>9s}  {'Years':>12s}")
for spec_name, r in reg_results.items():
    print(f"  {spec_name:20s}  {r['n']:>8,}  {r['r2']:>6.3f}  "
          f"{r['sigma_fe']:>6.3f}  {r['n_countries']:>9d}  {r['year_range']:>12s}")

# ---------------------------------------------------------------------------
# 7. Year-by-year regressions (time-varying country effects)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Year-by-year country effects (forward P/E)")
print("=" * 70)

# Year-by-year spec: forward P/E with firm-level controls + country FE
# Note: real_rate and imf_growth are country-level → absorbed by country FE
# in cross-section, so only firm-level controls here
yby_formula = ("log_fwd_pe ~ leverage + log_at + lag_earn_growth + xrd_at + ltg"
               " + C(fic, Treatment(reference='USA')) + C(ggroup)")
yby_cols = ["log_fwd_pe", "leverage", "log_at", "lag_earn_growth",
            "xrd_at", "ltg", "fic", "ggroup"]

results_yby = []
for yr, grp in panel.groupby("fyear"):
    sub = grp.dropna(subset=yby_cols)
    if sub["fic"].nunique() < 10 or len(sub) < 100:
        continue
    try:
        m = sm.OLS.from_formula(yby_formula, data=sub).fit()
        for name, coef in m.params.items():
            if "C(fic" in name and "[T." in name:
                iso = name.split("[T.")[1].rstrip("]")
                results_yby.append({"fyear": yr, "fic": iso, "country_effect": coef})
        results_yby.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
    except Exception as e:
        print(f"  Year {yr:.0f}: {e}")

ce_fwd = pd.DataFrame(results_yby)
ce_fwd.to_parquet(OUT_DIR / "country_effects_fwd_baseline.parquet", index=False)
print(f"  Forward P/E effects: {ce_fwd['fic'].nunique()} countries, "
      f"{ce_fwd['fyear'].nunique()} years, {len(ce_fwd):,} obs")

# Also run trailing P/E year-by-year (same sample) for comparison
yby_trail = ("log_pe ~ leverage + log_at + lag_earn_growth + xrd_at + ltg"
             " + C(fic, Treatment(reference='USA')) + C(ggroup)")
yby_trail_cols = ["log_pe", "leverage", "log_at", "lag_earn_growth",
                  "xrd_at", "ltg", "fic", "ggroup"]

results_trail = []
for yr, grp in panel.groupby("fyear"):
    sub = grp.dropna(subset=yby_trail_cols)
    if sub["fic"].nunique() < 10 or len(sub) < 100:
        continue
    try:
        m = sm.OLS.from_formula(yby_trail, data=sub).fit()
        for name, coef in m.params.items():
            if "C(fic" in name and "[T." in name:
                iso = name.split("[T.")[1].rstrip("]")
                results_trail.append({"fyear": yr, "fic": iso, "country_effect": coef})
        results_trail.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
    except Exception as e:
        print(f"  Trailing year {yr:.0f}: {e}")

ce_trail = pd.DataFrame(results_trail)
ce_trail.to_parquet(OUT_DIR / "country_effects_trailing_baseline.parquet", index=False)
print(f"  Trailing P/E effects: {ce_trail['fic'].nunique()} countries, "
      f"{ce_trail['fyear'].nunique()} years")

# ---------------------------------------------------------------------------
# 8. Characterize residual country FE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Residual country effects analysis")
print("=" * 70)

# Use pooled country FE from spec (3) = forward P/E with R&D + LTG
# (the main baseline before macro controls)
baseline_spec = "(3) + LTG"
if baseline_spec in reg_results:
    pooled_fe = pd.Series(reg_results[baseline_spec]["country_fe"]).rename("fe_pooled")
else:
    # Fallback to whatever is available
    fallback = list(reg_results.keys())[-1]
    pooled_fe = pd.Series(reg_results[fallback]["country_fe"]).rename("fe_pooled")
    print(f"  Using {fallback} as baseline")

# Also compute time-averaged year-by-year effects
avg_fwd = ce_fwd.groupby("fic")["country_effect"].mean().rename("fe_yby")
avg_trail = ce_trail.groupby("fic")["country_effect"].mean().rename("fe_trail_yby")

country = pd.DataFrame(pooled_fe)
country = country.join(avg_fwd, how="left")
country = country.join(avg_trail, how="left")

# Merge cross-sectional variables
country = country.join(sav_avg, how="left")
country = country.join(mktcap_avg, how="left")
country = country.join(credit_avg, how="left")
country = country.join(gdppc_avg, how="left")
country["anti_self_dealing"] = country.index.map(ANTI_SELF_DEALING)

# Drop countries with missing FE
country = country.dropna(subset=["fe_pooled"])
print(f"\nCountries with pooled FE: {len(country)}")
print(f"  σ(pooled FE) = {country['fe_pooled'].std():.3f}")
print(f"  Range: [{country['fe_pooled'].min():.2f}, {country['fe_pooled'].max():.2f}]")

# Focus country summary
print(f"\n  {'Ctry':>5s}  {'Pooled FE':>10s}  {'PE ratio':>9s}  {'YbY avg':>8s}")
for fic in FOCUS:
    if fic in country.index:
        r = country.loc[fic]
        yby = f"{r['fe_yby']:+.3f}" if pd.notna(r.get('fe_yby')) else "n/a"
        print(f"  {fic:>5s}  {r['fe_pooled']:>+10.3f}  "
              f"{np.exp(r['fe_pooled']):>9.2f}  {yby:>8s}")

# ---------------------------------------------------------------------------
# 9. Second-stage: explain residual country FE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: Second-stage cross-country regressions")
print("=" * 70)

# Univariate regressions
print("\n--- Univariate Correlations ---")
second_stage_vars = [
    ("savings_gni", "Savings/GNI"),
    ("mktcap_gdp", "Mkt Cap/GDP"),
    ("credit_gdp", "Credit/GDP"),
    ("anti_self_dealing", "Anti-Self-Dealing"),
    ("gdp_pc_ppp", "GDP/cap PPP"),
]

uni_results = []
for col, label in second_stage_vars:
    sub = country[["fe_pooled", col]].dropna().astype(float)
    if len(sub) < 10:
        continue
    r = sub["fe_pooled"].corr(sub[col])
    X = sm.add_constant(sub[col])
    m = sm.OLS(sub["fe_pooled"], X).fit()
    t = m.tvalues.iloc[1]
    print(f"  {label:20s}: r={r:+.2f}, t={t:+.2f}, R²={m.rsquared:.3f} (n={len(sub)})")
    uni_results.append({
        "variable": label, "r": r, "t": t, "r2": m.rsquared,
        "n": len(sub), "coef": m.params.iloc[1],
    })

# Multivariate regressions
print("\n--- Multivariate Regressions ---")
multi_specs = [
    ("Savings only", ["savings_gni"]),
    ("Fin dev (mktcap+credit)", ["mktcap_gdp", "credit_gdp"]),
    ("Savings + Mktcap", ["savings_gni", "mktcap_gdp"]),
    ("Savings + Credit", ["savings_gni", "credit_gdp"]),
    ("Savings + ASD", ["savings_gni", "anti_self_dealing"]),
    ("Kitchen sink", ["savings_gni", "mktcap_gdp", "credit_gdp", "anti_self_dealing"]),
]

multi_results = []
for label, rhs in multi_specs:
    sub = country[["fe_pooled"] + rhs].dropna().astype(float)
    if len(sub) < 10:
        continue
    X = sm.add_constant(sub[rhs])
    m = sm.OLS(sub["fe_pooled"], X).fit()
    print(f"\n  {label} (n={len(sub)}): R²={m.rsquared:.3f}")
    for i, var in enumerate(rhs):
        print(f"    {var:25s}: {m.params.iloc[i+1]:+.4f} (t={m.tvalues.iloc[i+1]:+.2f})")
    multi_results.append({
        "spec": label, "n": len(sub), "r2": m.rsquared,
        "vars": rhs, "model": m,
    })

# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: Generating figures")
print("=" * 70)


def _label_points(ax, data, x_col, y_col, focus=FOCUS):
    """Add country labels for focus countries."""
    for fic in data.index:
        if fic in focus:
            ax.annotate(
                fic, (data.loc[fic, x_col], data.loc[fic, y_col]),
                fontsize=7, fontweight="bold", ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points",
                color=PALETTE["primary"],
            )


def _fit_line(ax, x, y):
    """Add OLS fit line."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    z = np.polyfit(x[mask], y[mask], 1)
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
            linewidth=1.5, linestyle="--", alpha=0.8)


# --- Fig 1: Country forward P/E effects over time ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(FOCUS)))
for i, fic in enumerate(FOCUS):
    sub = ce_fwd[ce_fwd["fic"] == fic].sort_values("fyear")
    if len(sub) >= 3:
        ax.plot(sub["fyear"], sub["country_effect"], label=fic,
                linewidth=1.5, color=colors[i])
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Country forward P/E effect (log, vs USA = 0)", fontsize=10)
ax.set_title("Forward P/E Country Effects Over Time\n"
             "(controls: size, leverage, earnings growth, R&D, LTG, industry FE)",
             fontsize=11)
ax.legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "country_effects_forward_baseline.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "country_effects_forward_baseline.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved country_effects_forward_baseline")

# --- Fig 2: Dispersion over time (forward vs trailing) ---
disp_fwd = ce_fwd.groupby("fyear")["country_effect"].std().rename("forward")
disp_trail = ce_trail.groupby("fyear")["country_effect"].std().rename("trailing")
disp = pd.DataFrame(disp_fwd).join(disp_trail, how="outer")

fig, ax = plt.subplots(figsize=(7, 4))
if "trailing" in disp.columns:
    ax.plot(disp.index, disp["trailing"], color="gray",
            linewidth=1.5, linestyle="--", alpha=0.6, label="Trailing P/E")
ax.plot(disp.index, disp["forward"], color=PALETTE["accent"],
        linewidth=2, label="Forward P/E (FY1)")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Cross-country std dev of P/E effects", fontsize=10)
ax.set_title("P/E Dispersion: Forward vs Trailing\n"
             "(both with R&D + LTG controls)", fontsize=11)
ax.legend(fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "dispersion_forward_baseline.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "dispersion_forward_baseline.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved dispersion_forward_baseline")

# --- Fig 3: Scatter: Savings/GNI vs residual country FE ---
sav_plot = country[["fe_pooled", "savings_gni"]].dropna().astype(float)
if len(sav_plot) >= 10:
    fig, ax = plt.subplots(figsize=(6, 4))
    pe_ratio = np.exp(sav_plot["fe_pooled"])
    ax.scatter(sav_plot["savings_gni"], pe_ratio,
               color=PALETTE["accent"], s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    _label_points(ax, sav_plot.assign(pe_ratio=pe_ratio), "savings_gni", "pe_ratio")
    _fit_line(ax, sav_plot["savings_gni"].values, pe_ratio.values)
    r_sv = sav_plot["fe_pooled"].corr(sav_plot["savings_gni"])
    n_sv = len(sav_plot)
    t_sv = r_sv * np.sqrt((n_sv - 2) / (1 - r_sv**2))
    ax.set_xlabel("Gross Savings / GNI (%)", fontsize=10)
    ax.set_ylabel("Forward P/E relative to USA", fontsize=10)
    ax.set_title(f"National Savings and Forward P/E "
                 f"(r = {r_sv:+.2f}, t = {t_sv:.2f}, n = {n_sv})", fontsize=11)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_savings_vs_residual_fe.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "scatter_savings_vs_residual_fe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_savings_vs_residual_fe")

# --- Fig 4: Financial development 2x2 panel ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plot_specs = [
    ("mktcap_gdp", "Stock Market Cap / GDP (%)"),
    ("credit_gdp", "Private Credit / GDP (%)"),
    ("anti_self_dealing", "Anti-Self-Dealing Index"),
    ("gdp_pc_ppp", "GDP per capita (PPP, $K)"),
]

for ax, (x_col, x_label) in zip(axes.flat, plot_specs):
    sub = country[["fe_pooled", x_col]].dropna().astype(float)
    if len(sub) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        continue
    x_vals = sub[x_col].values.copy()
    if x_col == "gdp_pc_ppp":
        x_vals = x_vals / 1000
        sub = sub.copy()
        sub[x_col] = sub[x_col] / 1000
    r = np.corrcoef(sub[x_col].values, sub["fe_pooled"].values)[0, 1]
    ax.scatter(sub[x_col], sub["fe_pooled"], color=PALETTE["accent"],
               s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
    _label_points(ax, sub, x_col, "fe_pooled")
    _fit_line(ax, sub[x_col].values, sub["fe_pooled"].values)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Forward P/E effect (vs USA)", fontsize=9)
    ax.set_title(f"r = {r:+.2f} (n={len(sub)})", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

fig.suptitle("Financial Development and Forward P/E Country Effects", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_findev_vs_residual_fe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "scatter_findev_vs_residual_fe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved scatter_findev_vs_residual_fe")

# ---------------------------------------------------------------------------
# 11. Save regression results for slides
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11: Saving results for slides")
print("=" * 70)

# Save pooled regression summary
with open(OUT_DIR / "forward_pe_baseline_results.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("FORWARD P/E BASELINE REGRESSIONS\n")
    f.write("=" * 70 + "\n\n")

    f.write("--- Pooled Panel Regressions ---\n")
    f.write(f"{'Spec':20s}  {'N':>8s}  {'R²':>6s}  {'σ(FE)':>6s}  "
            f"{'Ctry':>5s}  {'Years':>12s}\n")
    for spec_name, r in reg_results.items():
        f.write(f"{spec_name:20s}  {r['n']:>8,}  {r['r2']:>6.3f}  "
                f"{r['sigma_fe']:>6.3f}  {r['n_countries']:>5d}  "
                f"{r['year_range']:>12s}\n")

    f.write("\n--- Key Coefficients from Spec (3) + LTG ---\n")
    if baseline_spec in reg_results:
        m = reg_results[baseline_spec]["model"]
        for var in ["leverage", "log_at", "lag_earn_growth", "xrd_at", "ltg"]:
            if var in m.params.index:
                se = m.bse[var]
                f.write(f"  {var:20s}: {m.params[var]:+.4f} "
                        f"(SE={se:.4f}, t={m.tvalues[var]:+.2f})\n")

    f.write("\n--- Country FE (pooled, baseline spec) ---\n")
    fe_sorted = pooled_fe.sort_values()
    for fic in fe_sorted.index:
        f.write(f"  {fic:>5s}: {fe_sorted[fic]:+.3f} "
                f"(PE ratio: {np.exp(fe_sorted[fic]):.2f})\n")

    f.write("\n--- Second-Stage Regressions ---\n")
    for r in multi_results:
        f.write(f"\n  {r['spec']} (n={r['n']}): R²={r['r2']:.3f}\n")
        m = r['model']
        for i, var in enumerate(r['vars']):
            f.write(f"    {var:25s}: {m.params.iloc[i+1]:+.4f} "
                    f"(t={m.tvalues.iloc[i+1]:+.2f})\n")

print("  Saved forward_pe_baseline_results.txt")

# Save country-level data for potential further use
country_export = country.copy()
country_export.index.name = "fic"
country_export.to_parquet(OUT_DIR / "forward_pe_country_data.parquet")
print("  Saved forward_pe_country_data.parquet")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
