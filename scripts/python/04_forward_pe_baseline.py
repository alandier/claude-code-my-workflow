"""
04_forward_pe_baseline.py -- Forward P/E and Expected Returns across countries.

Approach:
  - Forward P/E = IBES price / IBES EPS forecast (from pansum + statsum)
    Both in same IBES currency → ratio is currency-invariant, no market cap needed.
  - Expected return = IBES price target / IBES price - 1 (from ptgsum + pansum)
  - Timing: statpers = 3rd Thursday of each month. Price and EPS consensus are
    contemporaneous. We keep the latest statpers before fiscal year end.
  - Panel: Compustat Global + America fundamentals (1987+) × IBES consensus (2000+)
    No market cap required → full non-US coverage from 2000 (60+ countries).

Structure:
  1. Load IBES cached data (FY1, FY2, LTG, pansum prices, price targets)
  2. Build fundamentals panel from Compustat (no market cap required)
  3. Merge IBES with fundamentals → forward P/E + expected return panel
  4. Fetch country-year macro controls (WEO growth, WB real rate)
  5. Forward P/E regressions with progressive controls
  6. Expected return regressions
  7. Decomposition: discount rate (expected return) vs growth (LTG)
  8. Year-by-year country effects + second-stage cross-country regressions
  9. Figures

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

# Maximum year to include in year-by-year regressions (2025 has partial coverage)
MAX_YEAR = 2024

# Region mapping for market-cap-weighted aggregates
REGIONS = {
    "AMERICA": [
        "USA", "CAN", "BRA", "MEX", "CHL", "COL", "PER", "ARG",
    ],
    "EUROPE": [
        "GBR", "DEU", "FRA", "ITA", "ESP", "NLD", "CHE", "SWE", "NOR", "DNK",
        "FIN", "BEL", "AUT", "IRL", "PRT", "GRC", "POL", "CZE", "HUN", "ROU",
        "LUX", "ISR", "TUR", "RUS", "ZAF",
    ],
    "ASIA ex-CHINA": [
        "JPN", "KOR", "IND", "AUS", "TWN", "HKG", "SGP", "THA", "IDN",
        "MYS", "PHL", "NZL", "PAK",
    ],
    "CHINA": ["CHN"],
}

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


def load_cached(name):
    """Load parquet from WRDS cache."""
    path = CACHE / f"{name}.parquet"
    df = pd.read_parquet(path)
    print(f"  {name}: {len(df):,} rows")
    return df


# ===================================================================
# STEP 1: Load IBES data and link tables
# ===================================================================
print("=" * 70)
print("STEP 1: Loading IBES data and link tables")
print("=" * 70)

us_link = load_cached("iclink_ccm_link")[["ticker", "gvkey"]].drop_duplicates()
intl_link = load_cached("ibtic_gvkey_link")[["ticker", "gvkey"]].drop_duplicates()
link = pd.concat([us_link, intl_link], ignore_index=True).drop_duplicates(
    "ticker", keep="first"
)
print(f"  Combined link: {len(link):,} unique tickers")

us_fy1 = load_cached("ibes_us_fy1")
int_fy1 = load_cached("ibes_int_fy1")
us_fy2 = load_cached("ibes_us_fy2")
int_fy2 = load_cached("ibes_int_fy2")
us_ltg = load_cached("ibes_us_ltg")
int_ltg = load_cached("ibes_int_ltg")
pansum = load_cached("ibes_pansum")
ptgsum = load_cached("ibes_ptgsum")

# ===================================================================
# STEP 2: Build fundamentals panel from Compustat (no market cap)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2: Building fundamentals panel from Compustat")
print("=" * 70)

# --- US fundamentals ---
us_fund = pd.read_csv(
    DATA_DIR / "compustat_america.csv",
    usecols=["GVKEY", "fyear", "fic", "at", "dltt", "dlc", "ib", "xrd", "sich"],
    dtype={"GVKEY": str},
    low_memory=False,
).rename(columns={"GVKEY": "gvkey"})
us_fund["fic"] = us_fund["fic"].fillna("USA")

# --- Global fundamentals ---
gl_fund = pd.read_csv(
    DATA_DIR / "compustat_global.csv",
    usecols=["gvkey", "fyear", "fic", "at", "dltt", "dlc", "ib", "xrd", "sich"],
    dtype={"gvkey": str, "fyear": "Int64"},
    low_memory=False,
)

fund = pd.concat([us_fund, gl_fund], ignore_index=True)
fund["fyear"] = pd.to_numeric(fund["fyear"], errors="coerce")
fund = fund.dropna(subset=["fyear", "at"])
fund = fund[fund["at"] > 0]
fund["fic"] = fund["fic"].astype(str)
fund = fund.drop_duplicates(["gvkey", "fyear"], keep="first")

# SIC2 industry classification (first 2 digits of Historical SIC)
fund["sich"] = pd.to_numeric(fund["sich"], errors="coerce")
fund["sic2"] = (fund["sich"] // 100).astype("Int64")
fund.loc[fund["sic2"].isna(), "sic2"] = pd.NA

# Compute firm characteristics
fund["leverage"] = (fund["dltt"].fillna(0) + fund["dlc"].fillna(0)) / fund["at"]
fund["log_at"] = np.log(fund["at"])
fund["xrd_at"] = fund["xrd"].fillna(0) / fund["at"]

# Lag earnings growth: (ib_t - ib_{t-1}) / |ib_{t-1}|
fund = fund.sort_values(["gvkey", "fyear"])
fund["ib_lag"] = fund.groupby("gvkey")["ib"].shift(1)
fund["lag_earn_growth"] = (fund["ib"] - fund["ib_lag"]) / fund["ib_lag"].abs()
fund.loc[fund["ib_lag"].abs() < 1e-6, "lag_earn_growth"] = np.nan

# Winsorize lag_earn_growth at 1/99 percentiles
lg = fund["lag_earn_growth"].dropna()
lo, hi = lg.quantile([0.01, 0.99])
fund["lag_earn_growth"] = fund["lag_earn_growth"].clip(lo, hi)

keep = ["gvkey", "fyear", "fic", "sic2", "log_at", "leverage",
        "lag_earn_growth", "xrd_at"]
fund = fund[keep].copy()

sic2_cov = fund["sic2"].notna().sum()
print(f"  Fundamentals: {len(fund):,} firm-years, {fund['fic'].nunique()} countries")
print(f"  SIC2 coverage: {sic2_cov:,} ({100*sic2_cov/len(fund):.1f}%), {fund['sic2'].nunique()} sectors")
print(f"  Pre-2007: {(fund['fyear'] < 2007).sum():,}")
print(f"  Years: {fund['fyear'].min():.0f}-{fund['fyear'].max():.0f}")

# ===================================================================
# STEP 3: Build IBES forward P/E + expected return panel
# ===================================================================
print("\n" + "=" * 70)
print("STEP 3: Building IBES-based forward P/E and expected return")
print("=" * 70)

# Prepare pansum prices and price targets
pansum_p = pansum[["ticker", "statpers", "price"]].copy()
pansum_p["statpers"] = pd.to_datetime(pansum_p["statpers"])

ptgsum["statpers"] = pd.to_datetime(ptgsum["statpers"])
ptg_price = ptgsum.merge(pansum_p, on=["ticker", "statpers"], how="inner")
ptg_price["exp_ret"] = ptg_price["medptg"] / ptg_price["price"] - 1


def prepare_fy(raw, eps_col="fy1_eps", source=""):
    """For each ticker-fyear: merge price + ptg, compute P/E and exp_ret."""
    if len(raw) == 0:
        return pd.DataFrame()
    df = raw.copy()
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    # Merge pansum price
    df = df.merge(pansum_p, on=["ticker", "statpers"], how="left")
    # Merge price target (expected return)
    df = df.merge(
        ptg_price[["ticker", "statpers", "exp_ret"]],
        on=["ticker", "statpers"], how="left",
    )
    # Forward P/E = price / EPS (same IBES currency, no conversion needed)
    mask_pe = (df[eps_col] > 0) & (df["price"] > 0)
    df["fwd_pe"] = np.nan
    df.loc[mask_pe, "fwd_pe"] = df.loc[mask_pe, "price"] / df.loc[mask_pe, eps_col]
    # BPN fix for EPS level (still needed for other uses)
    if "curr_act" in df.columns:
        for cur, factor in IBES_SMALL_UNIT.items():
            mask = df["curr_act"] == cur
            if mask.any():
                df.loc[mask, eps_col] = df.loc[mask, eps_col] / factor
    # Keep latest consensus before fiscal year end → one per gvkey-fyear
    df = df.merge(link, on="ticker", how="inner")
    df["fyear"] = df["fpedats"].dt.year
    df = df[df["statpers"] <= df["fpedats"]]
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    out = df[["gvkey", "fyear", "fwd_pe", "exp_ret"]].copy()
    n_pe = out["fwd_pe"].notna().sum()
    n_er = out["exp_ret"].notna().sum()
    print(f"  {source}: {len(out):,} firm-years, "
          f"{n_pe:,} with P/E, {n_er:,} with exp_ret")
    return out


def prepare_fy2(raw, source=""):
    """FY2 consensus → forward P/E only (FY2)."""
    if len(raw) == 0:
        return pd.DataFrame()
    df = raw.copy()
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    df = df.merge(pansum_p, on=["ticker", "statpers"], how="left")
    mask_pe = (df["fy2_eps"] > 0) & (df["price"] > 0)
    df["fwd_pe_fy2"] = np.nan
    df.loc[mask_pe, "fwd_pe_fy2"] = df.loc[mask_pe, "price"] / df.loc[mask_pe, "fy2_eps"]
    df = df.merge(link, on="ticker", how="inner")
    df["fyear"] = df["fpedats"].dt.year - 1  # align with FY1 year
    df = df[df["statpers"] <= df["fpedats"]]
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    out = df[["gvkey", "fyear", "fwd_pe_fy2"]].copy()
    print(f"  {source} FY2: {len(out):,} firm-years")
    return out


def prepare_ltg(raw, source=""):
    """LTG consensus."""
    if len(raw) == 0:
        return pd.DataFrame()
    df = raw.merge(link, on="ticker", how="inner")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fyear"] = df["statpers"].dt.year
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    out = df[["gvkey", "fyear", "ltg"]].copy()
    out["ltg"] = pd.to_numeric(out["ltg"], errors="coerce")
    print(f"  {source} LTG: {len(out):,} firm-years")
    return out


fy1 = pd.concat(
    [prepare_fy(us_fy1, "fy1_eps", "US FY1"),
     prepare_fy(int_fy1, "fy1_eps", "Intl FY1")],
    ignore_index=True,
).drop_duplicates(["gvkey", "fyear"], keep="first")

fy2 = pd.concat(
    [prepare_fy2(us_fy2, "US"),
     prepare_fy2(int_fy2, "Intl")],
    ignore_index=True,
).drop_duplicates(["gvkey", "fyear"], keep="first")

ltg = pd.concat(
    [prepare_ltg(us_ltg, "US"),
     prepare_ltg(int_ltg, "Intl")],
    ignore_index=True,
).drop_duplicates(["gvkey", "fyear"], keep="first")

print(f"\nTotal FY1: {len(fy1):,}, FY2: {len(fy2):,}, LTG: {len(ltg):,}")

# ===================================================================
# STEP 4: Merge fundamentals + IBES → analysis panel
# ===================================================================
print("\n" + "=" * 70)
print("STEP 4: Merging fundamentals with IBES data")
print("=" * 70)

panel = fund.merge(fy1, on=["gvkey", "fyear"], how="inner")
panel = panel.merge(fy2[["gvkey", "fyear", "fwd_pe_fy2"]], on=["gvkey", "fyear"], how="left")
panel = panel.merge(ltg, on=["gvkey", "fyear"], how="left")

# Winsorize forward P/E and expected return
# P/E at 1/99 (log transform further compresses tails)
for col in ["fwd_pe", "fwd_pe_fy2"]:
    vals = panel[col].dropna()
    if len(vals) > 100:
        lo, hi = vals.quantile([0.01, 0.99])
        panel[col] = panel[col].clip(lo, hi)
# Expected return: tighter 2.5/97.5 winsorization (right tail is extreme)
vals_er = panel["exp_ret"].dropna()
if len(vals_er) > 100:
    lo_er, hi_er = vals_er.quantile([0.025, 0.975])
    panel["exp_ret"] = panel["exp_ret"].clip(lo_er, hi_er)

panel["log_fwd_pe"] = np.log(panel["fwd_pe"])
panel["log_fwd_pe_fy2"] = np.log(panel["fwd_pe_fy2"])

# Medium-term growth (MTG): implied EPS growth from FY1 to FY2
# MTG = (EPS_fy2 - EPS_fy1) / |mean(EPS_fy1, EPS_fy2)|
# Since EPS = Price / PE, and Price cancels: MTG = (1/PE2 - 1/PE1) / |(1/PE2 + 1/PE1)/2|
ey1 = 1.0 / panel["fwd_pe"]      # earnings yield FY1
ey2 = 1.0 / panel["fwd_pe_fy2"]  # earnings yield FY2
avg_ey = (ey1 + ey2) / 2.0
panel["mtg"] = (ey2 - ey1) / avg_ey.abs()
# Exclude where average EPS is near zero (sign-change firms)
panel.loc[avg_ey.abs() < 1e-6, "mtg"] = np.nan
# Winsorize MTG at 1/99 percentiles
mtg_vals = panel["mtg"].dropna()
if len(mtg_vals) > 100:
    lo_mtg, hi_mtg = mtg_vals.quantile([0.01, 0.99])
    panel["mtg"] = panel["mtg"].clip(lo_mtg, hi_mtg)

# Ensure numeric
for col in ["leverage", "log_at", "lag_earn_growth", "xrd_at", "ltg", "exp_ret", "mtg"]:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

# Coverage summary
n_pe = panel["log_fwd_pe"].notna().sum()
n_pe2 = panel["log_fwd_pe_fy2"].notna().sum()
n_er = panel["exp_ret"].notna().sum()
n_ltg = panel["ltg"].notna().sum()
n_mtg = panel["mtg"].notna().sum()
sub_pe = panel[panel["log_fwd_pe"].notna()]
sub_er = panel[panel["exp_ret"].notna()]

print(f"\n  Panel: {len(panel):,} firm-years, {panel['fic'].nunique()} countries")
print(f"  Forward P/E (FY1): {n_pe:,} obs, {sub_pe['fic'].nunique()} countries, "
      f"years {sub_pe['fyear'].min():.0f}-{sub_pe['fyear'].max():.0f}")
print(f"  Forward P/E (FY2): {n_pe2:,} obs")
print(f"  Expected return:   {n_er:,} obs, {sub_er['fic'].nunique()} countries, "
      f"years {sub_er['fyear'].min():.0f}-{sub_er['fyear'].max():.0f}")
print(f"  LTG:               {n_ltg:,} obs")
print(f"  MTG (FY2-FY1):     {n_mtg:,} obs, "
      f"{panel[panel['mtg'].notna()]['fic'].nunique()} countries")

# Coverage by year
print(f"\n  --- Coverage by Year ---")
for yr in range(2000, 2025):
    s = panel[(panel["fyear"] == yr) & panel["log_fwd_pe"].notna()]
    if len(s) > 0:
        print(f"  {yr}: {s['gvkey'].nunique():>6,} firms, "
              f"{s['fic'].nunique():>3} countries")

# --- Outlier diagnostics ---
print(f"\n  --- Outlier Diagnostics ---")
for col, label in [("fwd_pe", "Forward P/E (FY1)"), ("exp_ret", "Expected Return"),
                    ("fwd_pe_fy2", "Forward P/E (FY2)")]:
    v = panel[col].dropna()
    print(f"  {label:25s}: n={len(v):,}, "
          f"mean={v.mean():.2f}, median={v.median():.2f}, "
          f"P5={v.quantile(0.05):.2f}, P95={v.quantile(0.95):.2f}, "
          f"min={v.min():.2f}, max={v.max():.2f}")

# Country sample sizes (flag small ones)
ctry_n = sub_pe.groupby("fic").size().sort_values(ascending=False)
small_countries = ctry_n[ctry_n < 50]
print(f"\n  Countries with < 50 obs: {len(small_countries)}")
for fic, n in small_countries.items():
    print(f"    {fic}: {n}")

# Outlier diagnostic figures
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
# Forward P/E distribution
v_pe = panel["fwd_pe"].dropna()
axes[0].hist(v_pe, bins=100, color=PALETTE["accent"], alpha=0.8, edgecolor="white")
axes[0].axvline(v_pe.median(), color=PALETTE["alert"], linestyle="--", linewidth=1.5,
                label=f"median = {v_pe.median():.1f}")
axes[0].set_xlabel("Forward P/E (FY1)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Forward P/E")
axes[0].legend(fontsize=8)
# Expected return distribution
v_er = panel["exp_ret"].dropna()
axes[1].hist(v_er * 100, bins=100, color=PALETTE["highlight"], alpha=0.8, edgecolor="white")
axes[1].axvline(v_er.median() * 100, color=PALETTE["alert"], linestyle="--", linewidth=1.5,
                label=f"median = {v_er.median()*100:.1f}%")
axes[1].set_xlabel("Expected Return (PTG/P − 1, %)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Expected Return")
axes[1].legend(fontsize=8)
# P/E by focus country (box plot)
focus_pe = panel[panel["fic"].isin(FOCUS) & panel["fwd_pe"].notna()].copy()
focus_pe["fic"] = pd.Categorical(focus_pe["fic"], categories=FOCUS, ordered=True)
focus_pe.boxplot(column="fwd_pe", by="fic", ax=axes[2],
                 showfliers=False, grid=False, patch_artist=True,
                 boxprops=dict(facecolor=PALETTE["accent"], alpha=0.6))
axes[2].set_xlabel("")
axes[2].set_ylabel("Forward P/E")
axes[2].set_title("Forward P/E by Country")
plt.suptitle("")  # remove auto title from boxplot
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"outlier_diagnostics.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved outlier_diagnostics figure")

# ===================================================================
# STEP 5: Fetch macro controls
# ===================================================================
print("\n" + "=" * 70)
print("STEP 5: Fetching macro controls")
print("=" * 70)

# --- IMF WEO growth forecasts ---
print("\nFetching IMF WEO growth forecasts...")
WEO_CACHE = OUT_DIR / "weo_growth_forecasts.parquet"
if WEO_CACHE.exists():
    weo_panel = pd.read_parquet(WEO_CACHE)
    print(f"  Loaded from cache: {len(weo_panel):,} country-years")
else:
    editions = [(yr, rel) for yr in range(2007, 2025) for rel in [1, 2]]
    weo_rows = []
    for yr, rel in editions:
        try:
            path, _ = weo.download(yr, rel)
            w = weo.WEO(path)
            try:
                gdp = w.get("Gross domestic product, constant prices", "Percent change")
            except Exception:
                gdp = w.get("Gross domestic product, constant prices", "Annual percent change")
            for iso3 in gdp.columns:
                for base_year in range(2000, 2030):
                    forecasts = []
                    for ahead in range(1, 6):
                        try:
                            val = gdp.loc[str(base_year + ahead), iso3]
                            if pd.notna(val):
                                forecasts.append(float(val))
                        except (KeyError, TypeError):
                            pass
                    if len(forecasts) >= 3:
                        weo_rows.append({
                            "fic": iso3, "fyear": float(base_year),
                            "weo_year": yr, "weo_release": rel,
                            "imf_growth_5y": np.mean(forecasts),
                        })
            print(f"  {yr}-{'Apr' if rel == 1 else 'Oct'}: OK")
        except Exception as e:
            print(f"  {yr}-{'Apr' if rel == 1 else 'Oct'}: FAILED ({e})")
    weo_all = pd.DataFrame(weo_rows)
    weo_all["edition_order"] = weo_all["weo_year"] * 2 + weo_all["weo_release"]
    weo_all = weo_all[weo_all["weo_year"] <= weo_all["fyear"]]
    weo_panel = (weo_all.sort_values("edition_order")
                 .drop_duplicates(["fic", "fyear"], keep="last")
                 [["fic", "fyear", "imf_growth_5y"]])
    weo_panel.to_parquet(WEO_CACHE, index=False)
print(f"  WEO: {weo_panel['fic'].nunique()} countries, "
      f"years {weo_panel['fyear'].min():.0f}-{weo_panel['fyear'].max():.0f}")

# --- Bond yields: OECD 10Y govt bond yields (primary) + BIS policy rates (gap-fill) ---
# OECD STES IRLT: 47 countries, monthly, back to 1960 for core OECD
# BIS CBPOL: 49 countries, monthly, central bank policy rates (short-term proxy)
# Combined: ~55 countries with some form of risk-free rate
# Previously used Compustat ltgdr (33 countries) — OECD+BIS adds India, Brazil,
# Israel, Chile, Colombia, Malaysia, Turkey, Saudi Arabia, etc.
import requests
import zipfile
import io
from io import StringIO

OECD_BOND_CACHE = OUT_DIR / "oecd_bond_yield_10y.parquet"
BIS_POLICY_CACHE = OUT_DIR / "bis_policy_rate.parquet"
WB_INFL_CACHE = OUT_DIR / "wb_inflation_panel.parquet"

# ISO2 → ISO3 mapping for BIS data (BIS uses ISO2)
ISO2_TO_ISO3 = {
    "US": "USA", "JP": "JPN", "GB": "GBR", "DE": "DEU", "FR": "FRA",
    "CN": "CHN", "IN": "IND", "BR": "BRA", "KR": "KOR", "AU": "AUS",
    "CA": "CAN", "IT": "ITA", "ES": "ESP", "NL": "NLD", "CH": "CHE",
    "SE": "SWE", "NO": "NOR", "DK": "DNK", "FI": "FIN", "BE": "BEL",
    "AT": "AUT", "IE": "IRL", "PT": "PRT", "GR": "GRC", "NZ": "NZL",
    "SG": "SGP", "HK": "HKG", "TW": "TWN", "IL": "ISR", "ZA": "ZAF",
    "MX": "MEX", "TR": "TUR", "PL": "POL", "CZ": "CZE", "HU": "HUN",
    "CL": "CHL", "CO": "COL", "PE": "PER", "AR": "ARG", "RU": "RUS",
    "TH": "THA", "MY": "MYS", "ID": "IDN", "PH": "PHL", "SA": "SAU",
    "RO": "ROU", "BG": "BGR", "HR": "HRV", "RS": "SRB", "EG": "EGY",
    "NG": "NGA", "KE": "KEN", "PK": "PAK", "BD": "BGD", "VN": "VNM",
    "UA": "UKR", "KZ": "KAZ", "QA": "QAT", "AE": "ARE", "KW": "KWT",
    "BH": "BHR", "OM": "OMN", "JO": "JOR", "LK": "LKA", "SK": "SVK",
    "SI": "SVN", "LT": "LTU", "LV": "LVA", "EE": "EST", "IS": "ISL",
    "LU": "LUX", "MT": "MLT", "CY": "CYP",
}

print("\nFetching OECD 10Y government bond yields...")
if OECD_BOND_CACHE.exists():
    oecd_bond = pd.read_parquet(OECD_BOND_CACHE)
    print(f"  Loaded from cache: {len(oecd_bond):,} obs, "
          f"{oecd_bond['fic'].nunique()} countries")
else:
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
        ".M.IRLT.PA._Z._Z._Z._Z.N"
        "?startPeriod=1990-01&endPeriod=2025-12"
        "&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    oecd_raw = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    oecd_raw.columns = ["fic", "date", "bond_yield_10y"]
    oecd_raw["date"] = pd.to_datetime(oecd_raw["date"])
    oecd_raw["bond_yield_10y"] = pd.to_numeric(oecd_raw["bond_yield_10y"], errors="coerce")
    oecd_raw = oecd_raw.dropna(subset=["bond_yield_10y"])
    oecd_raw["fyear"] = oecd_raw["date"].dt.year
    oecd_bond = (oecd_raw.groupby(["fic", "fyear"])["bond_yield_10y"]
                 .mean().reset_index())
    oecd_bond.to_parquet(OECD_BOND_CACHE, index=False)
    print(f"  Downloaded: {len(oecd_bond):,} obs, {oecd_bond['fic'].nunique()} countries")
    print(f"  Countries: {sorted(oecd_bond['fic'].unique())}")

print("\nFetching BIS central bank policy rates (gap-fill)...")
if BIS_POLICY_CACHE.exists():
    bis_rate = pd.read_parquet(BIS_POLICY_CACHE)
    print(f"  Loaded from cache: {len(bis_rate):,} obs, "
          f"{bis_rate['fic'].nunique()} countries")
else:
    url = "https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    ref_col = [c for c in df.columns if "REF_AREA" in c][0]
    time_col = [c for c in df.columns if "TIME_PERIOD" in c][0]
    val_col = [c for c in df.columns if "OBS_VALUE" in c][0]
    freq_col = [c for c in df.columns if "FREQ" in c][0]
    monthly = df[df[freq_col].str.startswith("M")].copy()
    monthly["iso2"] = monthly[ref_col].str[:2]
    monthly["fic"] = monthly["iso2"].map(ISO2_TO_ISO3)
    monthly = monthly[monthly["fic"].notna()]
    monthly["date"] = pd.to_datetime(monthly[time_col])
    monthly["policy_rate"] = pd.to_numeric(monthly[val_col], errors="coerce")
    monthly = monthly.dropna(subset=["policy_rate"])
    monthly = monthly[monthly["date"] >= "2000-01-01"]
    monthly["fyear"] = monthly["date"].dt.year
    bis_rate = (monthly.groupby(["fic", "fyear"])["policy_rate"]
                .mean().reset_index())
    bis_rate.to_parquet(BIS_POLICY_CACHE, index=False)
    print(f"  Downloaded: {len(bis_rate):,} obs, {bis_rate['fic'].nunique()} countries")
    print(f"  Countries: {sorted(bis_rate['fic'].unique())}")

# Combine: OECD bond yield preferred, BIS policy rate as fallback
# For countries with OECD data, use 10Y bond yield
# For countries without OECD data (Malaysia, Turkey, Saudi Arabia, etc.), use policy rate
oecd_countries = set(oecd_bond["fic"].unique())
bis_only = bis_rate[~bis_rate["fic"].isin(oecd_countries)].copy()
bis_only = bis_only.rename(columns={"policy_rate": "bond_yield_10y"})
wb_bond = pd.concat([oecd_bond, bis_only[["fic", "fyear", "bond_yield_10y"]]],
                     ignore_index=True)
print(f"\n  Combined bond yield: {len(wb_bond):,} obs, "
      f"{wb_bond['fic'].nunique()} countries")
print(f"  OECD (10Y yield): {len(oecd_countries)} countries")
print(f"  BIS gap-fill (policy rate): {bis_only['fic'].nunique()} countries "
      f"({sorted(bis_only['fic'].unique())})")

# --- Inflation: World Bank CPI (broadest coverage, 200+ countries) ---
print("\nFetching World Bank inflation...")
if WB_INFL_CACHE.exists():
    wb_inflation = pd.read_parquet(WB_INFL_CACHE)
    print(f"  Loaded from cache: {len(wb_inflation):,} obs, "
          f"{wb_inflation['fic'].nunique()} countries")
else:
    infl_raw = wb.data.DataFrame("FP.CPI.TOTL.ZG", time=range(2000, 2025), labels=False)
    wb_inflation = infl_raw.T.stack().reset_index()
    wb_inflation.columns = ["fyear", "fic", "inflation"]
    wb_inflation["fyear"] = wb_inflation["fyear"].str.replace("YR", "").astype(int)
    wb_inflation["inflation"] = pd.to_numeric(wb_inflation["inflation"], errors="coerce")
    wb_inflation = wb_inflation.dropna(subset=["inflation"])
    wb_inflation.to_parquet(WB_INFL_CACHE, index=False)
    print(f"  Downloaded: {len(wb_inflation):,} obs, "
          f"{wb_inflation['fic'].nunique()} countries")

wb_rate = wb_bond.merge(wb_inflation, on=["fic", "fyear"], how="outer")

# --- Cross-sectional variables ---
print("\nFetching cross-sectional country variables...")
sav_avg = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2005, 2024), labels=False).T.mean()
sav_avg = sav_avg.rename("savings_gni")
sav_avg.index.name = "fic"
mktcap_avg = wb.data.DataFrame("CM.MKT.LCAP.GD.ZS", time=range(2010, 2024), labels=False).T.mean()
mktcap_avg = mktcap_avg.rename("mktcap_gdp")
mktcap_avg.index.name = "fic"
credit_avg = wb.data.DataFrame("FS.AST.PRVT.GD.ZS", time=range(2010, 2024), labels=False).T.mean()
credit_avg = credit_avg.rename("credit_gdp")
credit_avg.index.name = "fic"
gdppc_avg = wb.data.DataFrame("NY.GDP.PCAP.PP.KD", time=range(2010, 2024), labels=False).T.mean()
gdppc_avg = gdppc_avg.rename("gdp_pc_ppp")
gdppc_avg.index.name = "fic"
print(f"  Savings: {sav_avg.notna().sum()}, Mktcap: {mktcap_avg.notna().sum()}, "
      f"Credit: {credit_avg.notna().sum()}, GDP/cap: {gdppc_avg.notna().sum()}")

# --- Market cap in current USD (country-year panel, for regional weighting) ---
WB_MKTCAP_USD_CACHE = OUT_DIR / "wb_mktcap_usd_panel.parquet"
if WB_MKTCAP_USD_CACHE.exists():
    mktcap_usd = pd.read_parquet(WB_MKTCAP_USD_CACHE)
    print(f"  Mktcap USD panel from cache: {len(mktcap_usd):,} obs")
else:
    # wb.data.DataFrame returns countries as rows, "YR2000"... as columns
    mktcap_raw = wb.data.DataFrame(
        "CM.MKT.LCAP.CD", time=range(2000, 2025), labels=False
    )
    rows = []
    for fic in mktcap_raw.index:
        for col in mktcap_raw.columns:
            val = mktcap_raw.loc[fic, col]
            if pd.notna(val):
                yr = float(str(col).replace("YR", ""))
                rows.append({"fic": fic, "fyear": yr, "mktcap_usd": float(val)})
    mktcap_usd = pd.DataFrame(rows)
    mktcap_usd.to_parquet(WB_MKTCAP_USD_CACHE, index=False)
    print(f"  Mktcap USD panel: {len(mktcap_usd):,} obs, "
          f"{mktcap_usd['fic'].nunique()} countries")
print(f"  Mktcap USD: {mktcap_usd['fic'].nunique()} countries, "
      f"years {mktcap_usd['fyear'].min():.0f}-{mktcap_usd['fyear'].max():.0f}")

# Merge macro into panel
panel = panel.merge(weo_panel, on=["fic", "fyear"], how="left")
panel = panel.merge(wb_rate, on=["fic", "fyear"], how="left")

# ===================================================================
# Helper: iterative demeaning for two-way FE
# ===================================================================

def _iterative_demean(sub, cols, group1_col, group2_col, max_iter=200, tol=1e-8):
    """Iterative demeaning (alternating projections) to absorb two-way FE.

    Returns dict of demeaned columns (as numpy arrays) and number of iterations.
    """
    dm = {c: sub[c].astype(float).values.copy() for c in cols}
    n_iter = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1
        remaining = 0.0
        for c in cols:
            sub["_tmp"] = dm[c]
            gm1 = sub.groupby(group1_col)["_tmp"].transform("mean").values
            dm[c] -= gm1
            remaining = max(remaining, np.abs(gm1).max())

            sub["_tmp"] = dm[c]
            gm2 = sub.groupby(group2_col)["_tmp"].transform("mean").values
            dm[c] -= gm2
            remaining = max(remaining, np.abs(gm2).max())
        if remaining < tol:
            break
    sub.drop(columns=["_tmp"], errors="ignore", inplace=True)
    return dm, n_iter


# ===================================================================
# STEP 5b: Variance decomposition by FE type
# ===================================================================
print("\n" + "=" * 70)
print("STEP 5b: Variance decomposition by FE type")
print("=" * 70)

# Prepare sample: complete cases for the main spec (+ MTG)
_vd_controls = ["leverage", "log_at", "lag_earn_growth", "xrd_at", "mtg"]
_vd = panel.dropna(subset=["log_fwd_pe"] + _vd_controls + ["fic", "fyear"]).copy()
for _c in ["log_fwd_pe"] + _vd_controls:
    _vd = _vd[np.isfinite(_vd[_c].astype(float))]

# Create group variables
_vd["sic2"] = _vd["sic2"].fillna(-1).astype(int).astype(str) if "sic2" in _vd.columns else "0"
_vd["sic2_year"] = _vd["sic2"] + "_" + _vd["fyear"].astype(int).astype(str)
_vd["fic_year"] = _vd["fic"] + "_" + _vd["fyear"].astype(int).astype(str)

_y = _vd["log_fwd_pe"].astype(float)
_sst = ((_y - _y.mean()) ** 2).sum()
print(f"  Sample: {len(_vd):,} obs, {_vd['fic'].nunique()} countries, SST = {_sst:.1f}")

# One-way FE R² (simple group mean demeaning)
_vd_r2 = {}
for name, gcol in [("Year", "fyear"), ("SIC2", "sic2"), ("Country", "fic"),
                    ("SIC2×Year", "sic2_year"), ("Country×Year", "fic_year")]:
    _gm = _vd.groupby(gcol)["log_fwd_pe"].transform("mean")
    _ssr = ((_y - _gm) ** 2).sum()
    _r2 = 1 - _ssr / _sst
    _vd_r2[name] = _r2
    print(f"  {name:20s}: R² = {_r2:.3f} ({_vd[gcol].nunique():,} groups)")

# Two-way FE: Country + SIC2×Year (old approach) and Country×Year + SIC2×Year (new)
for name, g1, g2 in [("Country + SIC2×Year", "sic2_year", "fic"),
                      ("Country×Year + SIC2×Year", "sic2_year", "fic_year")]:
    _dm_dict, _n_it = _iterative_demean(_vd.copy(), ["log_fwd_pe"], g1, g2)
    _ssr = (_dm_dict["log_fwd_pe"] ** 2).sum()
    _r2 = 1 - _ssr / _sst
    _vd_r2[name] = _r2
    print(f"  {name:35s}: R² = {_r2:.3f} ({_n_it} iter)")

# Incremental contribution
_incr_cy = _vd_r2["Country×Year + SIC2×Year"] - _vd_r2["SIC2×Year"]
_incr_c = _vd_r2["Country + SIC2×Year"] - _vd_r2["SIC2×Year"]
_incr_cy_over_c = _vd_r2["Country×Year + SIC2×Year"] - _vd_r2["Country + SIC2×Year"]
print(f"\n  Incremental R² of Country (time-invariant) over SIC2×Year:  {_incr_c:.3f}")
print(f"  Incremental R² of Country×Year over SIC2×Year:              {_incr_cy:.3f}")
print(f"  Time-varying component (Country×Year vs Country):           {_incr_cy_over_c:.3f}")

# ===================================================================
# STEP 6: Forward P/E regressions
# ===================================================================
print("\n" + "=" * 70)
print("STEP 6: Forward P/E regressions")
print("=" * 70)


def run_ols(data, dep_var, controls, label, cluster_var="fic"):
    """OLS with Country×Year + SIC2×Year FE, clustered SE.

    Uses iterative demeaning (alternating projections) to absorb both
    high-dimensional FE sets, then runs OLS on the demeaned firm controls.
    Recovers time-averaged country effects for cross-country analysis.
    """
    sub = data.dropna(subset=[dep_var] + controls + ["fic", "fyear"]).copy()
    for c in [dep_var] + controls:
        sub = sub[np.isfinite(sub[c].astype(float))]
    if sub["fic"].nunique() < 5:
        print(f"\n{label}: too few countries, skipping")
        return None, None, None

    # Create group variables
    has_sic2 = "sic2" in sub.columns and sub["sic2"].notna().sum() > len(sub) * 0.5
    if has_sic2:
        sub["sic2"] = sub["sic2"].fillna(-1).astype(int).astype(str)
        sub["sic2_year"] = sub["sic2"] + "_" + sub["fyear"].astype(int).astype(str)
    else:
        sub["sic2_year"] = sub["fyear"].astype(int).astype(str)
    sub["fic_year"] = sub["fic"] + "_" + sub["fyear"].astype(int).astype(str)

    demean_cols = [dep_var] + controls
    sst = ((sub[dep_var].astype(float) - sub[dep_var].astype(float).mean()) ** 2).sum()

    # Iterative demeaning: absorb both Country×Year and SIC2×Year FE
    dm, n_iter = _iterative_demean(sub, demean_cols, "sic2_year", "fic_year")
    for c in demean_cols:
        sub[c + "_dm"] = dm[c]
    dm_dep = dep_var + "_dm"
    dm_controls = [c + "_dm" for c in controls]

    # OLS on demeaned data: firm controls only (no dummies needed)
    X = sub[dm_controls].astype(float)
    y = sub[dm_dep].astype(float)
    # No intercept: absorbed by FE demeaning
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub[cluster_var]})

    # R² computations (manual — no constant means statsmodels uses uncentered)
    ssr = (model.resid ** 2).sum()
    sst_within = (y ** 2).sum()  # y already demeaned, mean ≈ 0
    r2_within = 1 - ssr / sst_within if sst_within > 0 else 0.0
    r2_total = 1 - ssr / sst
    r2_fe_only = 1 - sst_within / sst  # FE contribution alone

    # Recover time-averaged country effects via partial residuals.
    # r_i = y_i - X_i'β  contains γ_{c,t} + δ_{s,t} + ε
    beta = model.params.values
    fitted_ctrl = (sub[controls].astype(float).values * beta).sum(axis=1)
    sub["_partial_resid"] = sub[dep_var].astype(float).values - fitted_ctrl

    # Iterative projection to separate γ_{c,t} from δ_{s,t}
    sub["_gamma_ct"] = 0.0
    sub["_delta_st"] = 0.0
    for _ in range(200):
        new_gamma = sub.groupby("fic_year")["_partial_resid"].transform("mean") \
                    - sub.groupby("fic_year")["_delta_st"].transform("mean")
        new_delta = sub.groupby("sic2_year")["_partial_resid"].transform("mean") \
                    - sub.groupby("sic2_year")["_gamma_ct"].transform("mean")
        chg = max((new_gamma - sub["_gamma_ct"]).abs().max(),
                  (new_delta - sub["_delta_st"]).abs().max())
        sub["_gamma_ct"] = new_gamma
        sub["_delta_st"] = new_delta
        if chg < 1e-8:
            break

    # Time-averaged country effects: γ̄_c = mean_t(γ_{c,t})
    ct_fe = sub.groupby(["fic", "fyear"])["_gamma_ct"].first().reset_index()
    country_fe_avg = ct_fe.groupby("fic")["_gamma_ct"].mean()
    # Normalize: USA = 0
    if "USA" in country_fe_avg.index:
        country_fe_avg -= country_fe_avg["USA"]
    fe_map = country_fe_avg.to_dict()
    sigma_fe = country_fe_avg.std()

    sub.drop(columns=["_tmp", "_partial_resid", "_gamma_ct", "_delta_st"],
             errors="ignore", inplace=True)

    n_ctry = sub["fic"].nunique()
    yr_range = f"{sub['fyear'].min():.0f}-{sub['fyear'].max():.0f}"
    print(f"\n{label}: n={len(sub):,}, countries={n_ctry}, years={yr_range}")
    print(f"  R²_total = {r2_total:.3f} (FE: {r2_fe_only:.3f}, within: {r2_within:.3f}), "
          f"σ(country FE) = {sigma_fe:.4f} ({n_iter} iter)")
    for c, c_dm in zip(controls, dm_controls):
        coef = model.params[c_dm]
        t = coef / model.bse[c_dm]
        print(f"    {c:20s}: {coef:+.4f} (t={t:+.2f})")

    return model, fe_map, {
        "n": len(sub), "r2": r2_total, "r2_within": r2_within,
        "r2_fe": r2_fe_only, "sigma_fe": sigma_fe,
        "n_countries": n_ctry, "year_range": yr_range,
    }


# --- Forward P/E regressions ---
pe_results = {}

_, fe_A, info_A = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth"],
    "(A) Base")
pe_results["(A) Base"] = info_A

_, fe_B, info_B = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at"],
    "(B) + R&D")
pe_results["(B) + R&D"] = info_B

_, fe_C, info_C = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "mtg"],
    "(C) + MTG")
pe_results["(C) + MTG"] = info_C

_, fe_C2, info_C2 = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "ltg"],
    "(C2) + LTG")
pe_results["(C2) + LTG"] = info_C2

# Note: Macro spec (D) dropped — country×year FE absorbs bond yield, inflation, IMF growth.

# FY2 variant
_, _, info_E = run_ols(panel, "log_fwd_pe_fy2",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "mtg"],
    "(E) FY2 + MTG")
pe_results["(E) FY2 + MTG"] = info_E

print("\n--- Forward P/E Summary ---")
print(f"  {'Spec':15s} {'N':>9s} {'R²tot':>7s} {'R²wth':>7s} {'σ(FE)':>7s} {'Ctry':>5s} {'Years':>12s}")
for name, r in pe_results.items():
    if r:
        print(f"  {name:15s} {r['n']:>9,} {r['r2']:>7.3f} {r.get('r2_within', 0):>7.3f} "
              f"{r['sigma_fe']:>7.4f} {r['n_countries']:>5d} {r['year_range']:>12s}")

# ===================================================================
# STEP 7: Expected return regressions
# ===================================================================
print("\n" + "=" * 70)
print("STEP 7: Expected return (PTG/P - 1) regressions")
print("=" * 70)

er_results = {}

_, er_fe_A, er_info_A = run_ols(panel, "exp_ret",
    ["leverage", "log_at", "lag_earn_growth", "mtg"],
    "(A) Base + MTG")
er_results["(A) Base + MTG"] = er_info_A

_, er_fe_B, er_info_B = run_ols(panel, "exp_ret",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "mtg"],
    "(B) + R&D")
er_results["(B) + R&D"] = er_info_B

_, er_fe_C2, er_info_C2 = run_ols(panel, "exp_ret",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "ltg"],
    "(C') + LTG")
er_results["(C') + LTG"] = er_info_C2

print("\n--- Expected Return Summary ---")
print(f"  {'Spec':15s} {'N':>9s} {'R²tot':>7s} {'R²wth':>7s} {'σ(FE)':>7s} {'Ctry':>5s} {'Years':>12s}")
for name, r in er_results.items():
    if r:
        print(f"  {name:15s} {r['n']:>9,} {r['r2']:>7.3f} {r.get('r2_within', 0):>7.3f} "
              f"{r['sigma_fe']:>7.4f} {r['n_countries']:>5d} {r['year_range']:>12s}")

# ===================================================================
# STEP 8: Decomposition — discount rate vs growth
# ===================================================================
print("\n" + "=" * 70)
print("STEP 8: Decomposition (discount rate vs growth)")
print("=" * 70)

_, _, dec_A = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at"],
    "P/E ~ firm controls")

_, _, dec_B = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "exp_ret"],
    "P/E ~ + expected return")

_, _, dec_C = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "mtg"],
    "P/E ~ + MTG")

_, _, dec_D = run_ols(panel, "log_fwd_pe",
    ["leverage", "log_at", "lag_earn_growth", "xrd_at", "exp_ret", "mtg"],
    "P/E ~ + exp_ret + MTG")

if dec_A and dec_B and dec_C and dec_D:
    sA, sB, sC, sD = dec_A["sigma_fe"], dec_B["sigma_fe"], dec_C["sigma_fe"], dec_D["sigma_fe"]
    print(f"\n--- Country FE Decomposition ---")
    print(f"  Firm controls only:    σ(FE) = {sA:.4f}")
    print(f"  + Expected return:     σ(FE) = {sB:.4f}  ({(sB-sA)/sA*100:+.1f}%)")
    print(f"  + MTG:                 σ(FE) = {sC:.4f}  ({(sC-sA)/sA*100:+.1f}%)")
    print(f"  + Both:                σ(FE) = {sD:.4f}  ({(sD-sA)/sA*100:+.1f}%)")
    print(f"\n  Exp return absorbs: {(sA-sB)/sA*100:.1f}% of country FE dispersion")
    print(f"  MTG absorbs:        {(sA-sC)/sA*100:.1f}%")
    print(f"  Both absorb:        {(sA-sD)/sA*100:.1f}%")

# ===================================================================
# STEP 9: Year-by-year country effects
# ===================================================================
print("\n" + "=" * 70)
print("STEP 9: Year-by-year country effects")
print("=" * 70)

# Year-by-year: use BASE controls only (no LTG) for stability — LTG severely
# limits per-year sample size, making country FE noisy. LTG effects tested in pooled.
yby_controls = ["leverage", "log_at", "lag_earn_growth", "xrd_at"]
MIN_FIRMS_PER_COUNTRY = 20  # require enough firms for reliable country FE

results_yby = []
for yr, grp in panel.groupby("fyear"):
    if yr > MAX_YEAR:
        continue
    sub = grp.dropna(subset=["log_fwd_pe"] + yby_controls + ["fic"]).copy()
    for c in ["log_fwd_pe"] + yby_controls:
        sub = sub[np.isfinite(sub[c].astype(float))]
    if sub["fic"].nunique() < 10 or len(sub) < 200:
        continue
    # Filter to countries with enough observations
    ctry_counts = sub.groupby("fic").size()
    keep_countries = ctry_counts[ctry_counts >= MIN_FIRMS_PER_COUNTRY].index
    sub = sub[sub["fic"].isin(keep_countries)]
    if sub["fic"].nunique() < 10:
        continue
    try:
        country_dm = pd.get_dummies(sub["fic"], prefix="c", drop_first=True, dtype=float)
        parts = [sub[yby_controls].astype(float), country_dm]
        # Add SIC2 industry FE
        if "sic2" in sub.columns and sub["sic2"].notna().sum() > len(sub) * 0.3:
            sub["sic2"] = sub["sic2"].fillna(-1).astype(int).astype(str)
            ind_dm = pd.get_dummies(sub["sic2"], prefix="s", drop_first=True, dtype=float)
            parts.append(ind_dm)
        X = pd.concat(parts, axis=1)
        X = sm.add_constant(X)
        m = sm.OLS(sub["log_fwd_pe"].astype(float), X).fit()
        fe_cols = [c for c in m.params.index if c.startswith("c_")]
        for c in fe_cols:
            iso = c.replace("c_", "")
            results_yby.append({"fyear": yr, "fic": iso, "country_effect": m.params[c]})
        ref = [c for c in sub["fic"].unique() if f"c_{c}" not in m.params.index]
        for c in ref:
            results_yby.append({"fyear": yr, "fic": c, "country_effect": 0.0})
    except Exception as e:
        print(f"  Year {yr:.0f}: {e}")

ce_fwd = pd.DataFrame(results_yby)

# Normalize to USA = 0 in each year
for yr in ce_fwd["fyear"].unique():
    mask_yr = ce_fwd["fyear"] == yr
    usa = ce_fwd.loc[mask_yr & (ce_fwd["fic"] == "USA"), "country_effect"]
    if len(usa) > 0:
        ce_fwd.loc[mask_yr, "country_effect"] -= usa.values[0]

print(f"  P/E year-by-year: {ce_fwd['fic'].nunique()} countries, "
      f"{ce_fwd['fyear'].nunique()} years (normalized: USA = 0)")

# --- Year-by-year with MTG (saturated spec for trajectory plots) ---
yby_sat_controls = ["leverage", "log_at", "lag_earn_growth", "xrd_at", "mtg"]
results_yby_sat = []
for yr, grp in panel.groupby("fyear"):
    if yr > MAX_YEAR:
        continue
    sub = grp.dropna(subset=["log_fwd_pe"] + yby_sat_controls + ["fic"]).copy()
    for c in ["log_fwd_pe"] + yby_sat_controls:
        sub = sub[np.isfinite(sub[c].astype(float))]
    if sub["fic"].nunique() < 10 or len(sub) < 200:
        continue
    ctry_counts = sub.groupby("fic").size()
    keep_countries = ctry_counts[ctry_counts >= MIN_FIRMS_PER_COUNTRY].index
    sub = sub[sub["fic"].isin(keep_countries)]
    if sub["fic"].nunique() < 10:
        continue
    try:
        country_dm = pd.get_dummies(sub["fic"], prefix="c", drop_first=True, dtype=float)
        parts = [sub[yby_sat_controls].astype(float), country_dm]
        if "sic2" in sub.columns and sub["sic2"].notna().sum() > len(sub) * 0.3:
            sub["sic2"] = sub["sic2"].fillna(-1).astype(int).astype(str)
            ind_dm = pd.get_dummies(sub["sic2"], prefix="s", drop_first=True, dtype=float)
            parts.append(ind_dm)
        X = pd.concat(parts, axis=1)
        X = sm.add_constant(X)
        m = sm.OLS(sub["log_fwd_pe"].astype(float), X).fit()
        fe_cols = [c for c in m.params.index if c.startswith("c_")]
        for c in fe_cols:
            iso = c.replace("c_", "")
            results_yby_sat.append({"fyear": yr, "fic": iso, "country_effect": m.params[c]})
        ref = [c for c in sub["fic"].unique() if f"c_{c}" not in m.params.index]
        for c in ref:
            results_yby_sat.append({"fyear": yr, "fic": c, "country_effect": 0.0})
    except Exception as e:
        print(f"  Sat Year {yr:.0f}: {e}")

ce_sat = pd.DataFrame(results_yby_sat)
for yr in ce_sat["fyear"].unique():
    mask_yr = ce_sat["fyear"] == yr
    usa = ce_sat.loc[mask_yr & (ce_sat["fic"] == "USA"), "country_effect"]
    if len(usa) > 0:
        ce_sat.loc[mask_yr, "country_effect"] -= usa.values[0]
print(f"  P/E saturated year-by-year (+MTG): {ce_sat['fic'].nunique()} countries, "
      f"{ce_sat['fyear'].nunique()} years")

# --- Year-by-year Expected Return country effects ---
er_yby_controls = ["leverage", "log_at", "lag_earn_growth", "xrd_at"]
results_er_yby = []
for yr, grp in panel.groupby("fyear"):
    if yr > MAX_YEAR:
        continue
    sub = grp.dropna(subset=["exp_ret"] + er_yby_controls + ["fic"]).copy()
    for c in ["exp_ret"] + er_yby_controls:
        sub = sub[np.isfinite(sub[c].astype(float))]
    if sub["fic"].nunique() < 10 or len(sub) < 200:
        continue
    ctry_counts = sub.groupby("fic").size()
    keep_countries = ctry_counts[ctry_counts >= MIN_FIRMS_PER_COUNTRY].index
    sub = sub[sub["fic"].isin(keep_countries)]
    if sub["fic"].nunique() < 10:
        continue
    try:
        country_dm = pd.get_dummies(sub["fic"], prefix="c", drop_first=True, dtype=float)
        parts = [sub[er_yby_controls].astype(float), country_dm]
        # Add SIC2 industry FE
        if "sic2" in sub.columns and sub["sic2"].notna().sum() > len(sub) * 0.3:
            sub["sic2"] = sub["sic2"].fillna(-1).astype(int).astype(str)
            ind_dm = pd.get_dummies(sub["sic2"], prefix="s", drop_first=True, dtype=float)
            parts.append(ind_dm)
        X = pd.concat(parts, axis=1)
        X = sm.add_constant(X)
        m = sm.OLS(sub["exp_ret"].astype(float), X).fit()
        fe_cols = [c for c in m.params.index if c.startswith("c_")]
        for c in fe_cols:
            iso = c.replace("c_", "")
            results_er_yby.append({"fyear": yr, "fic": iso, "er_effect": m.params[c]})
        ref = [c for c in sub["fic"].unique() if f"c_{c}" not in m.params.index]
        for c in ref:
            results_er_yby.append({"fyear": yr, "fic": c, "er_effect": 0.0})
    except Exception as e:
        print(f"  ER Year {yr:.0f}: {e}")

ce_er = pd.DataFrame(results_er_yby)
# Normalize to USA = 0
for yr in ce_er["fyear"].unique():
    mask_yr = ce_er["fyear"] == yr
    usa = ce_er.loc[mask_yr & (ce_er["fic"] == "USA"), "er_effect"]
    if len(usa) > 0:
        ce_er.loc[mask_yr, "er_effect"] -= usa.values[0]

print(f"  ER year-by-year: {ce_er['fic'].nunique()} countries, "
      f"{ce_er['fyear'].nunique()} years (normalized: USA = 0)")

# --- Regional aggregates (market-cap weighted) ---
print("\n  Computing market-cap-weighted regional aggregates...")
# Merge market cap weights
ce_fwd_w = ce_fwd.merge(mktcap_usd, on=["fic", "fyear"], how="left")
ce_er_w = ce_er.merge(mktcap_usd, on=["fic", "fyear"], how="left")

# Assign regions
fic_to_region = {}
for region, countries in REGIONS.items():
    for c in countries:
        fic_to_region[c] = region
ce_fwd_w["region"] = ce_fwd_w["fic"].map(fic_to_region)
ce_er_w["region"] = ce_er_w["fic"].map(fic_to_region)


def _weighted_mean(grp, val_col, wt_col="mktcap_usd"):
    """Market-cap-weighted mean, falling back to equal weight if no cap data."""
    valid = grp[[val_col, wt_col]].dropna()
    if len(valid) == 0:
        return np.nan
    if valid[wt_col].sum() <= 0:
        return grp[val_col].mean()
    return np.average(valid[val_col], weights=valid[wt_col])


# P/E regional aggregates
regional_pe = (ce_fwd_w[ce_fwd_w["region"].notna()]
               .groupby(["region", "fyear"])
               .apply(_weighted_mean, "country_effect", include_groups=False)
               .reset_index(name="country_effect"))

# ER regional aggregates
regional_er = (ce_er_w[ce_er_w["region"].notna()]
               .groupby(["region", "fyear"])
               .apply(_weighted_mean, "er_effect", include_groups=False)
               .reset_index(name="er_effect"))

for region in REGIONS:
    pe_sub = regional_pe[regional_pe["region"] == region]
    er_sub = regional_er[regional_er["region"] == region]
    n_pe = ce_fwd_w[(ce_fwd_w["region"] == region)]["fic"].nunique()
    print(f"  {region:18s}: {n_pe:>2} countries in data, "
          f"PE years={len(pe_sub)}, ER years={len(er_sub)}")

# ===================================================================
# STEP 10: Second-stage cross-country regressions
# ===================================================================
print("\n" + "=" * 70)
print("STEP 10: Second-stage cross-country regressions")
print("=" * 70)

# Use pooled FE from spec (C) = +LTG — normalize to USA = 0
pooled_fe = pd.Series(fe_C).rename("fe_pooled") if fe_C else pd.Series(fe_B).rename("fe_pooled")
if "USA" in pooled_fe.index:
    pooled_fe -= pooled_fe["USA"]
avg_yby = ce_fwd.groupby("fic")["country_effect"].mean().rename("fe_yby")

# Filter to countries with enough obs for reliable FE
MIN_TOTAL_OBS = 100
ctry_total = panel[panel["log_fwd_pe"].notna()].groupby("fic").size()
reliable_countries = ctry_total[ctry_total >= MIN_TOTAL_OBS].index
pooled_fe = pooled_fe[pooled_fe.index.isin(reliable_countries)]
print(f"  Reliable countries (>= {MIN_TOTAL_OBS} obs): {len(pooled_fe)}")

country = pd.DataFrame(pooled_fe)
country = country.join(avg_yby, how="left")
country = country.join(sav_avg, how="left")
country = country.join(mktcap_avg, how="left")
country = country.join(credit_avg, how="left")
country = country.join(gdppc_avg, how="left")
country["anti_self_dealing"] = country.index.map(ANTI_SELF_DEALING)
country["log_gdp_pc"] = np.log(country["gdp_pc_ppp"])
country = country.dropna(subset=["fe_pooled"])

# Also add expected return FE — normalize to USA = 0
if er_fe_B:
    er_fe_series = pd.Series(er_fe_B).rename("er_fe")
    if "USA" in er_fe_series.index:
        er_fe_series -= er_fe_series["USA"]
    country = country.join(er_fe_series, how="left")

print(f"\nCountries: {len(country)}, σ(FE) = {country['fe_pooled'].std():.4f}")
print(f"\n  Focus countries:")
print(f"  {'Ctry':>5s} {'P/E FE':>8s} {'P/E ratio':>10s} {'Exp Ret FE':>11s}")
for fic in FOCUS:
    if fic in country.index:
        r = country.loc[fic]
        er_str = f"{r['er_fe']:+.3f}" if pd.notna(r.get('er_fe')) else "n/a"
        print(f"  {fic:>5s} {r['fe_pooled']:>+8.3f} {np.exp(r['fe_pooled']):>10.2f} {er_str:>11s}")

# Trim mktcap/GDP outliers (financial hubs like HKG at 1200% distort cross-country regressions)
if "mktcap_gdp" in country.columns:
    p95 = country["mktcap_gdp"].quantile(0.95)
    n_trimmed = (country["mktcap_gdp"] > p95).sum()
    if n_trimmed > 0:
        trimmed_countries = country[country["mktcap_gdp"] > p95].index.tolist()
        country.loc[country["mktcap_gdp"] > p95, "mktcap_gdp"] = np.nan
        print(f"  Trimmed mktcap/GDP > {p95:.0f}%: {trimmed_countries}")

# Univariate
print(f"\n--- Univariate Correlations ---")
for col, label in [("savings_gni", "Savings/GNI"), ("mktcap_gdp", "Mkt Cap/GDP"),
                    ("credit_gdp", "Credit/GDP"), ("log_gdp_pc", "log(GDP/cap PPP)"),
                    ("anti_self_dealing", "Anti-Self-Dealing")]:
    sub = country[["fe_pooled", col]].dropna().astype(float)
    if len(sub) < 10:
        continue
    r = sub["fe_pooled"].corr(sub[col])
    X = sm.add_constant(sub[col])
    m = sm.OLS(sub["fe_pooled"], X).fit()
    t = m.tvalues.iloc[1]
    print(f"  {label:20s}: r={r:+.2f}, t={t:+.2f}, R²={m.rsquared:.3f} (n={len(sub)})")

# ===================================================================
# STEP 11: Figures
# ===================================================================
print("\n" + "=" * 70)
print("STEP 11: Generating figures")
print("=" * 70)


def _label_points(ax, data, x_col, y_col, focus=FOCUS):
    for fic in data.index:
        if fic in focus:
            ax.annotate(fic, (data.loc[fic, x_col], data.loc[fic, y_col]),
                        fontsize=7, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points",
                        color=PALETTE["primary"])


def _fit_line(ax, x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    z = np.polyfit(x[mask], y[mask], 1)
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
            linewidth=1.5, linestyle="--", alpha=0.8)


# Fig 1: Regional P/E effects over time (market-cap weighted, P/E ratio scale)
REGION_COLORS = {
    "AMERICA": "#2980b9",      # blue
    "EUROPE": "#27ae60",       # green
    "ASIA ex-CHINA": "#e67e22", # orange
    "CHINA": "#c0392b",        # red
}
fig, ax = plt.subplots(figsize=(8, 5))
for region in REGIONS:
    sub = regional_pe[regional_pe["region"] == region].sort_values("fyear")
    if len(sub) >= 3:
        ax.plot(sub["fyear"], np.exp(sub["country_effect"]), label=region,
                linewidth=2.5, color=REGION_COLORS.get(region, "gray"))
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":",
           label="USA = 1.0")
ax.set_xlabel("Year")
ax.set_ylabel("Forward P/E relative to USA")
ax.set_title("Forward P/E Country Effects by Region\n"
             "(market-cap weighted; controls: size, leverage, earnings growth, R&D)")
ax.legend(fontsize=8, loc="upper left")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"country_effects_forward_baseline.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved country_effects_forward_baseline")

# Fig 2: Regional Expected Return effects over time (market-cap weighted, pp)
fig, ax = plt.subplots(figsize=(8, 5))
for region in REGIONS:
    sub = regional_er[regional_er["region"] == region].sort_values("fyear")
    if len(sub) >= 3:
        ax.plot(sub["fyear"], sub["er_effect"] * 100, label=region,
                linewidth=2.5, color=REGION_COLORS.get(region, "gray"))
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":",
           label="USA = 0")
ax.set_xlabel("Year")
ax.set_ylabel("Expected Return effect vs USA (pp)")
ax.set_title("Expected Return Country Effects by Region\n"
             "(market-cap weighted; controls: size, leverage, earnings growth, R&D)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"country_effects_expret.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved country_effects_expret")

# Fig 3: Dispersion over time (both P/E and ER)
fig, ax1 = plt.subplots(figsize=(8, 4.5))
disp_pe = ce_fwd.groupby("fyear")["country_effect"].std()
disp_er = ce_er.groupby("fyear")["er_effect"].std()
ax1.plot(disp_pe.index, disp_pe.values, color=PALETTE["accent"], linewidth=2,
         label="Forward P/E (σ of log effects)")
ax1.set_xlabel("Year")
ax1.set_ylabel("σ(country effects) — P/E", color=PALETTE["accent"])
ax2 = ax1.twinx()
ax2.plot(disp_er.index, disp_er.values * 100, color=PALETTE["highlight"], linewidth=2,
         linestyle="--", label="Expected Return (σ of effects, pp)")
ax2.set_ylabel("σ(country effects) — Exp Return (pp)", color=PALETTE["highlight"])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax1.set_title("Cross-Country Dispersion Over Time")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"dispersion_forward_baseline.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved dispersion_forward_baseline")

# Fig 4: P/E FE vs Expected Return FE (the key scatter — PE scale)
if "er_fe" in country.columns:
    both = country[["fe_pooled", "er_fe"]].dropna()
    if len(both) >= 10:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(both["er_fe"] * 100, np.exp(both["fe_pooled"]),
                   color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white")
        _label_points(ax, both.assign(er100=both["er_fe"]*100,
                                       pe=np.exp(both["fe_pooled"])),
                      "er100", "pe")
        _fit_line(ax, (both["er_fe"] * 100).values, np.exp(both["fe_pooled"]).values)
        r = both["fe_pooled"].corr(both["er_fe"])
        n = len(both)
        t = r * np.sqrt((n-2)/(1-r**2))
        ax.set_xlabel("Expected Return Country Effect (pp, USA = 0)")
        ax.set_ylabel("Forward P/E relative to USA")
        ax.set_title(f"P/E vs Expected Return: Country Effects\n"
                     f"r = {r:+.2f}, t = {t:.2f}, n = {n}")
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(OUT_DIR / f"scatter_pe_vs_expret.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Saved scatter_pe_vs_expret")

# Fig 5: Savings vs P/E FE (PE scale)
sav_plot = country[["fe_pooled", "savings_gni"]].dropna().astype(float)
if len(sav_plot) >= 10:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sav_plot["savings_gni"], np.exp(sav_plot["fe_pooled"]),
               color=PALETTE["accent"], s=40, alpha=0.7, edgecolors="white")
    _label_points(ax, sav_plot.assign(pe=np.exp(sav_plot["fe_pooled"])),
                  "savings_gni", "pe")
    _fit_line(ax, sav_plot["savings_gni"].values, np.exp(sav_plot["fe_pooled"]).values)
    r = sav_plot["fe_pooled"].corr(sav_plot["savings_gni"])
    n = len(sav_plot)
    t = r * np.sqrt((n-2)/(1-r**2))
    ax.set_xlabel("Gross Savings / GNI (%)")
    ax.set_ylabel("Forward P/E relative to USA")
    ax.set_title(f"Savings and Forward P/E (r = {r:+.2f}, t = {t:.2f}, n = {n})")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"scatter_savings_vs_residual_fe.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_savings_vs_residual_fe")

# Fig 6: Expected return distribution by country (box plot)
er_focus = panel[panel["fic"].isin(FOCUS) & panel["exp_ret"].notna()].copy()
er_focus["fic"] = pd.Categorical(er_focus["fic"], categories=FOCUS, ordered=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
er_focus.boxplot(column="exp_ret", by="fic", ax=ax,
                 showfliers=False, grid=False, patch_artist=True,
                 boxprops=dict(facecolor=PALETTE["highlight"], alpha=0.6))
ax.set_xlabel("")
ax.set_ylabel("Expected Return (PTG/P − 1)")
ax.set_title("Expected Return Distribution by Country")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
plt.suptitle("")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"boxplot_expret_by_country.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved boxplot_expret_by_country")

# Fig 5: Decomposition bar chart
if dec_A and dec_B and dec_C and dec_D:
    sigmas = [dec_A["sigma_fe"], dec_B["sigma_fe"], dec_C["sigma_fe"], dec_D["sigma_fe"]]
    labels_dec = ["Firm\ncontrols", "+ Expected\nreturn", "+ MTG", "+ Both"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(4), sigmas, color=[PALETTE["primary"], PALETTE["accent"],
                                            PALETTE["highlight"], PALETTE["alert"]],
                  width=0.6, edgecolor="white")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_dec, fontsize=9)
    ax.set_ylabel("σ(country FE)")
    ax.set_title("Cross-Country P/E Dispersion:\nDiscount Rate vs Growth Decomposition")
    # Annotate bars
    for i, (bar, s) in enumerate(zip(bars, sigmas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{s:.3f}", ha="center", fontsize=9)
    if sigmas[0] > 0:
        ax.text(3, sigmas[3] - 0.02,
                f"−{(sigmas[0]-sigmas[3])/sigmas[0]*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold", color="white")
    ax.set_ylim(0, max(sigmas) * 1.15)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"decomposition_sigma_fe.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved decomposition_sigma_fe")

# Fig 6: 2x2 scatter of cross-country variables vs P/E FE
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
panels = [
    ("savings_gni", "Savings / GNI (%)", axes[0, 0]),
    ("mktcap_gdp", "Market Cap / GDP (%)", axes[0, 1]),
    ("credit_gdp", "Credit / GDP (%)", axes[1, 0]),
    ("log_gdp_pc", "log(GDP per capita, PPP)", axes[1, 1]),
]
for col, xlabel, ax in panels:
    sub = country[["fe_pooled", col]].dropna().astype(float)
    if len(sub) < 5:
        ax.set_visible(False)
        continue
    ax.scatter(sub[col], np.exp(sub["fe_pooled"]),
               color=PALETTE["accent"], s=30, alpha=0.7, edgecolors="white")
    _label_points(ax, sub.assign(pe=np.exp(sub["fe_pooled"])), col, "pe")
    _fit_line(ax, sub[col].values, np.exp(sub["fe_pooled"]).values)
    r = sub["fe_pooled"].corr(sub[col])
    n = len(sub)
    t = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 0
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("P/E ratio (USA = 1)", fontsize=9)
    ax.set_title(f"r = {r:+.2f}, t = {t:.1f}, n = {n}", fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
fig.suptitle("Cross-Country Variables vs Forward P/E Effect", fontsize=11, y=1.01)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"scatter_crosscountry_vs_pe.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved scatter_crosscountry_vs_pe")

# Fig 7: France, Germany & UK P/E trajectory (saturated spec with MTG)
fra_sat = ce_sat[ce_sat["fic"] == "FRA"].sort_values("fyear")
deu_sat = ce_sat[ce_sat["fic"] == "DEU"].sort_values("fyear")
gbr_sat = ce_sat[ce_sat["fic"] == "GBR"].sort_values("fyear")
if len(fra_sat) >= 3 and len(deu_sat) >= 3:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fra_sat["fyear"], np.exp(fra_sat["country_effect"]),
            label="France", linewidth=2.5, color="#2980b9")
    ax.plot(deu_sat["fyear"], np.exp(deu_sat["country_effect"]),
            label="Germany", linewidth=2.5, color="#c0392b")
    if len(gbr_sat) >= 3:
        ax.plot(gbr_sat["fyear"], np.exp(gbr_sat["country_effect"]),
                label="UK", linewidth=2.5, color="#27ae60")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="USA = 1.0")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Forward P/E relative to USA", fontsize=11)
    ax.set_title("France, Germany & UK: Forward P/E Discount vs USA\n"
                 "(controls: size, leverage, R&D, MTG, 2-digit SIC industry FE)",
                 fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(0.55, 1.15)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"trajectory_fra_deu.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved trajectory_fra_deu")

# Fig 8+9: Latest-year scatter — country FE and raw medians in (P/E, ER) space
# Find latest year with good coverage in both P/E and ER year-by-year effects
SNAPSHOT_FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR",
                  "BRA", "AUS", "CAN", "TWN", "CHE", "NLD", "SWE", "ITA",
                  "ESP", "ZAF", "MEX", "THA", "SGP", "HKG", "NOR", "IDN"]
MIN_COUNTRIES_SNAPSHOT = 20

# Find year: latest with ≥ MIN_COUNTRIES_SNAPSHOT in both PE and ER effects
pe_yr_counts = ce_fwd.groupby("fyear")["fic"].nunique()
er_yr_counts = ce_er.groupby("fyear")["fic"].nunique()
both_counts = pd.DataFrame({"pe": pe_yr_counts, "er": er_yr_counts}).dropna()
both_counts["min_n"] = both_counts.min(axis=1)
valid_years = both_counts[both_counts["min_n"] >= MIN_COUNTRIES_SNAPSHOT].index
# Use second-to-last valid year to avoid partial coverage in the latest
if len(valid_years) >= 2:
    snapshot_year = sorted(valid_years)[-2]
else:
    snapshot_year = sorted(valid_years)[-1]
print(f"\n  Snapshot year for scatter: {snapshot_year:.0f} "
      f"(PE: {pe_yr_counts.get(snapshot_year, 0)} ctry, "
      f"ER: {er_yr_counts.get(snapshot_year, 0)} ctry)")

# --- Fig 7: Country FE scatter (regression-based) ---
pe_snap = ce_fwd[ce_fwd["fyear"] == snapshot_year].set_index("fic")["country_effect"]
er_snap = ce_er[ce_er["fyear"] == snapshot_year].set_index("fic")["er_effect"]
fe_snap = pd.DataFrame({"pe_fe": pe_snap, "er_fe": er_snap}).dropna()

if len(fe_snap) >= 10:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(fe_snap["er_fe"] * 100, np.exp(fe_snap["pe_fe"]),
               color=PALETTE["accent"], s=50, alpha=0.7, edgecolors="white", zorder=3)
    # Label all countries
    for fic in fe_snap.index:
        x = fe_snap.loc[fic, "er_fe"] * 100
        y = np.exp(fe_snap.loc[fic, "pe_fe"])
        bold = fic in FOCUS
        ax.annotate(fic, (x, y), fontsize=6.5 if bold else 5.5,
                    fontweight="bold" if bold else "normal",
                    ha="left", va="bottom", xytext=(3, 2),
                    textcoords="offset points",
                    color=PALETTE["primary"] if bold else "gray")
    _fit_line(ax, (fe_snap["er_fe"] * 100).values, np.exp(fe_snap["pe_fe"]).values)
    r = fe_snap["pe_fe"].corr(fe_snap["er_fe"])
    n = len(fe_snap)
    t = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 0
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Expected Return Country Effect (pp, USA = 0)", fontsize=10)
    ax.set_ylabel("Forward P/E relative to USA", fontsize=10)
    ax.set_title(f"Country Effects: P/E vs Expected Return ({snapshot_year:.0f})\n"
                 f"r = {r:+.2f}, t = {t:.1f}, n = {n} countries "
                 f"(regression FE, controls: size, leverage, R&D, MTG)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"scatter_fe_pe_er_{snapshot_year:.0f}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter_fe_pe_er_{snapshot_year:.0f}")

# --- Fig 8: Raw median scatter (no regression) ---
# Compute country medians of forward P/E and expected return for snapshot year
snap_data = panel[(panel["fyear"] == snapshot_year) &
                  panel["fwd_pe"].notna() & panel["exp_ret"].notna()]
ctry_snap_n = snap_data.groupby("fic").size()
snap_countries = ctry_snap_n[ctry_snap_n >= MIN_FIRMS_PER_COUNTRY].index
snap_data = snap_data[snap_data["fic"].isin(snap_countries)]
med_pe = snap_data.groupby("fic")["fwd_pe"].median()
med_er = snap_data.groupby("fic")["exp_ret"].median()
raw_snap = pd.DataFrame({"med_pe": med_pe, "med_er": med_er}).dropna()
# Express relative to USA
if "USA" in raw_snap.index:
    raw_snap["pe_rel"] = raw_snap["med_pe"] / raw_snap.loc["USA", "med_pe"]
    raw_snap["er_diff"] = (raw_snap["med_er"] - raw_snap.loc["USA", "med_er"]) * 100

if len(raw_snap) >= 10 and "pe_rel" in raw_snap.columns:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(raw_snap["er_diff"], raw_snap["pe_rel"],
               color=PALETTE["highlight"], s=50, alpha=0.7, edgecolors="white", zorder=3)
    for fic in raw_snap.index:
        x = raw_snap.loc[fic, "er_diff"]
        y = raw_snap.loc[fic, "pe_rel"]
        bold = fic in FOCUS
        ax.annotate(fic, (x, y), fontsize=6.5 if bold else 5.5,
                    fontweight="bold" if bold else "normal",
                    ha="left", va="bottom", xytext=(3, 2),
                    textcoords="offset points",
                    color=PALETTE["primary"] if bold else "gray")
    _fit_line(ax, raw_snap["er_diff"].values, raw_snap["pe_rel"].values)
    r = raw_snap["pe_rel"].corr(raw_snap["er_diff"])
    n = len(raw_snap)
    t = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 0
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Median Expected Return − USA (pp)", fontsize=10)
    ax.set_ylabel("Median Forward P/E / USA", fontsize=10)
    ax.set_title(f"Raw Country Medians: P/E vs Expected Return ({snapshot_year:.0f})\n"
                 f"r = {r:+.2f}, t = {t:.1f}, n = {n} countries "
                 f"(no controls, just medians)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"scatter_raw_pe_er_{snapshot_year:.0f}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter_raw_pe_er_{snapshot_year:.0f}")

# ===================================================================
# STEP 12: Save results
# ===================================================================
print("\n" + "=" * 70)
print("STEP 12: Saving results")
print("=" * 70)

with open(OUT_DIR / "forward_pe_baseline_results.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("FORWARD P/E AND EXPECTED RETURN ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write("--- Forward P/E Regressions ---\n")
    f.write(f"{'Spec':15s} {'N':>9s} {'R²':>6s} {'σ(FE)':>7s} {'Ctry':>5s} {'Years':>12s}\n")
    for name, r in pe_results.items():
        if r:
            f.write(f"{name:15s} {r['n']:>9,} {r['r2']:>6.3f} {r['sigma_fe']:>7.4f} "
                    f"{r['n_countries']:>5d} {r['year_range']:>12s}\n")

    f.write("\n--- Expected Return Regressions ---\n")
    f.write(f"{'Spec':15s} {'N':>9s} {'R²':>6s} {'σ(FE)':>7s} {'Ctry':>5s} {'Years':>12s}\n")
    for name, r in er_results.items():
        if r:
            f.write(f"{name:15s} {r['n']:>9,} {r['r2']:>6.3f} {r['sigma_fe']:>7.4f} "
                    f"{r['n_countries']:>5d} {r['year_range']:>12s}\n")

    if dec_A and dec_B and dec_C and dec_D:
        f.write("\n--- Decomposition: Country FE Reduction ---\n")
        f.write(f"  Firm controls:      σ(FE) = {dec_A['sigma_fe']:.4f}\n")
        f.write(f"  + Expected return:  σ(FE) = {dec_B['sigma_fe']:.4f}  "
                f"({(dec_B['sigma_fe']-dec_A['sigma_fe'])/dec_A['sigma_fe']*100:+.1f}%)\n")
        f.write(f"  + MTG:              σ(FE) = {dec_C['sigma_fe']:.4f}  "
                f"({(dec_C['sigma_fe']-dec_A['sigma_fe'])/dec_A['sigma_fe']*100:+.1f}%)\n")
        f.write(f"  + Both:             σ(FE) = {dec_D['sigma_fe']:.4f}  "
                f"({(dec_D['sigma_fe']-dec_A['sigma_fe'])/dec_A['sigma_fe']*100:+.1f}%)\n")

    f.write(f"\n--- Country Effects (pooled, +MTG spec, USA=0, N>={MIN_TOTAL_OBS}) ---\n")
    fe_sorted = sorted(country["fe_pooled"].items(), key=lambda x: x[1])
    for fic, v in fe_sorted:
        if pd.notna(v):
            f.write(f"  {fic:>5s}: {v:+.3f} (P/E vs USA: {np.exp(v):.2f})\n")

print("  Saved forward_pe_baseline_results.txt")

country.index.name = "fic"
country.to_parquet(OUT_DIR / "forward_pe_country_data.parquet")
ce_fwd.to_parquet(OUT_DIR / "country_effects_fwd_baseline.parquet", index=False)
ce_er.to_parquet(OUT_DIR / "country_effects_er_baseline.parquet", index=False)
print("  Saved parquet files")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
