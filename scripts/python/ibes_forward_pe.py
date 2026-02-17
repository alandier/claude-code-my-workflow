"""
ibes_forward_pe.py -- Analyst forecasts and cross-country P/E

Key tests:
  1. Forward P/E (IBES FY1 consensus) → do country effects shrink?
  2. Forward P/E (IBES FY2 consensus) → two-year-ahead variant
  3. Analyst long-term growth (IBES LTG) → predicts country P/E?

Link: IBES → Compustat via ibtic (comp.g_security) + ICLINK/CCM (US)
WRDS connection required (credentials in ~/.pgpass).

Author: Augustin Landier, HEC Paris
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import wrds

np.random.seed(20260216)

DATA_DIR = Path(
    "~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/"
).expanduser()

OUT_DIR = Path("output/regressions")
CACHE = OUT_DIR / "wrds_cache"
CACHE.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#2c3e50",
    "accent": "#2980b9",
    "alert": "#c0392b",
    "highlight": "#8e44ad",
}

FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]


# ---------------------------------------------------------------------------
# 1. WRDS download (cached as parquet)
# ---------------------------------------------------------------------------
def cached_query(db, name, sql):
    """Run SQL on WRDS, cache result as parquet."""
    path = CACHE / f"{name}.parquet"
    if path.exists():
        print(f"  {name}: cached ({pd.read_parquet(path).shape[0]:,} rows)")
        return pd.read_parquet(path)
    print(f"  {name}: downloading from WRDS...")
    try:
        df = db.raw_sql(sql)
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
        df = pd.DataFrame()
    if len(df) > 0:
        df.to_parquet(path, index=False)
    print(f"  {name}: {len(df):,} rows")
    return df


def download_all():
    """Download all IBES data from WRDS."""
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username="jalandier")

    data = {}

    # --- Global IBES ticker → Compustat gvkey link via comp.g_security ---
    data["ibtic_link"] = cached_query(db, "ibtic_gvkey_link", """
        SELECT DISTINCT gvkey, ibtic AS ticker
        FROM comp.g_security
        WHERE ibtic IS NOT NULL
    """)

    # --- US link via ICLINK + CCM (covers US IBES tickers) ---
    data["us_link"] = cached_query(db, "iclink_ccm_link", """
        SELECT DISTINCT b.ticker, c.gvkey
        FROM wrdsapps.ibcrsphist b
        INNER JOIN crsp.ccmxpf_lnkhist c
          ON b.permno = c.lpermno
          AND c.linktype IN ('LC', 'LU', 'LS')
          AND c.linkprim IN ('P', 'C')
        WHERE b.score IN (0, 1, 2)
    """)

    # --- US FY1 consensus ---
    data["us_fy1"] = cached_query(db, "ibes_us_fy1", """
        SELECT ticker, statpers, fpedats,
               medest AS fy1_eps, numest
        FROM ibes.statsum_epsus
        WHERE fpi = '1'
          AND statpers >= '2000-01-01'
          AND numest >= 3
          AND medest IS NOT NULL
    """)

    # --- US LTG ---
    data["us_ltg"] = cached_query(db, "ibes_us_ltg", """
        SELECT ticker, statpers,
               medest AS ltg, numest
        FROM ibes.statsum_epsus
        WHERE fpi = '0'
          AND statpers >= '2000-01-01'
          AND numest >= 3
          AND medest IS NOT NULL
    """)

    # --- International FY1 consensus ---
    data["int_fy1"] = cached_query(db, "ibes_int_fy1", """
        SELECT ticker, statpers, fpedats,
               medest AS fy1_eps, numest, curr_act
        FROM ibes.statsum_epsint
        WHERE fpi = '1'
          AND statpers >= '2000-01-01'
          AND numest >= 3
          AND medest IS NOT NULL
    """)

    # --- International LTG ---
    data["int_ltg"] = cached_query(db, "ibes_int_ltg", """
        SELECT ticker, statpers,
               medest AS ltg, numest
        FROM ibes.statsum_epsint
        WHERE fpi = '0'
          AND statpers >= '2000-01-01'
          AND numest >= 3
          AND medest IS NOT NULL
    """)

    # --- US FY2 consensus ---
    data["us_fy2"] = cached_query(db, "ibes_us_fy2", """
        SELECT ticker, statpers, fpedats,
               medest AS fy2_eps, numest
        FROM ibes.statsum_epsus
        WHERE fpi = '2'
          AND statpers >= '2000-01-01'
          AND numest >= 3
          AND medest IS NOT NULL
    """)

    # --- International FY2 consensus ---
    data["int_fy2"] = cached_query(db, "ibes_int_fy2", """
        SELECT ticker, statpers, fpedats,
               medest AS fy2_eps, numest, curr_act
        FROM ibes.statsum_epsint
        WHERE fpi = '2'
          AND statpers >= '2000-01-01'
          AND numest >= 3
          AND medest IS NOT NULL
    """)

    db.close()
    print("WRDS download complete.\n")
    return data


# ---------------------------------------------------------------------------
# 2. Build unified ticker → gvkey link
# ---------------------------------------------------------------------------
def build_link(data):
    """Combine US (ICLINK/CCM) and international (ibtic) links."""
    # US link
    us = data["us_link"][["ticker", "gvkey"]].drop_duplicates()
    print(f"  US link (ICLINK/CCM): {len(us):,} pairs")

    # International link via ibtic
    intl = data["ibtic_link"][["ticker", "gvkey"]].drop_duplicates()
    print(f"  Global link (ibtic):  {len(intl):,} pairs")

    # Combine (US takes priority for overlaps)
    link = pd.concat([us, intl], ignore_index=True).drop_duplicates(
        "ticker", keep="first"
    )
    print(f"  Combined link:        {len(link):,} unique tickers")
    return link


# ---------------------------------------------------------------------------
# 3. Prepare firm-year FY1 and LTG
# ---------------------------------------------------------------------------
IBES_SMALL_UNIT = {"BPN": 100}  # British Pence → divide by 100 for GBP


def prepare_fy1(raw, link, source=""):
    """For each gvkey-fyear, keep latest FY1 consensus before fiscal year end."""
    if len(raw) == 0:
        return pd.DataFrame(columns=["gvkey", "fyear", "fy1_eps"])

    df = raw.merge(link, on="ticker", how="inner")

    # Convert small-unit currencies (e.g. BPN → GBP)
    if "curr_act" in df.columns:
        for cur, factor in IBES_SMALL_UNIT.items():
            mask = df["curr_act"] == cur
            if mask.any():
                df.loc[mask, "fy1_eps"] = df.loc[mask, "fy1_eps"] / factor
                print(f"    Converted {mask.sum():,} rows from {cur} (÷{factor})")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    df["fyear"] = df["fpedats"].dt.year

    # Keep consensus made before fiscal year end
    df = df[df["statpers"] <= df["fpedats"]]
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    df = df[["gvkey", "fyear", "fy1_eps", "numest"]].copy()
    print(f"  {source} FY1: {len(df):,} firm-years")
    return df


def prepare_ltg(raw, link, source=""):
    """For each gvkey-year, keep latest LTG consensus."""
    if len(raw) == 0:
        return pd.DataFrame(columns=["gvkey", "fyear", "ltg"])

    df = raw.merge(link, on="ticker", how="inner")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fyear"] = df["statpers"].dt.year
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    df = df[["gvkey", "fyear", "ltg"]].copy()
    print(f"  {source} LTG: {len(df):,} firm-years")
    return df


def prepare_fy2(raw, link, source=""):
    """For each gvkey-fyear, keep latest FY2 consensus before fiscal year end.

    FY2 fpedats points to the fiscal year end TWO years ahead, so
    panel_fyear = fpedats.dt.year - 1 aligns with the FY1 panel year.
    """
    if len(raw) == 0:
        return pd.DataFrame(columns=["gvkey", "fyear", "fy2_eps"])

    df = raw.merge(link, on="ticker", how="inner")

    # Convert small-unit currencies (e.g. BPN → GBP)
    if "curr_act" in df.columns:
        for cur, factor in IBES_SMALL_UNIT.items():
            mask = df["curr_act"] == cur
            if mask.any():
                df.loc[mask, "fy2_eps"] = df.loc[mask, "fy2_eps"] / factor
                print(f"    Converted {mask.sum():,} rows from {cur} (÷{factor})")

    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    df["fyear"] = df["fpedats"].dt.year - 1  # align with panel fyear

    # Keep consensus made before the FY1 fiscal year end (fpedats - 1 year)
    df = df[df["statpers"] <= df["fpedats"]]
    df = df.sort_values("statpers").drop_duplicates(["gvkey", "fyear"], keep="last")
    df = df[["gvkey", "fyear", "fy2_eps", "numest"]].copy()
    print(f"  {source} FY2: {len(df):,} firm-years")
    return df


# ===========================================================================
# Main
# ===========================================================================
# Step 1: Download
data = download_all()

# Step 2: Build link
print("Building IBES-Compustat link...")
link = build_link(data)

# Step 3: Prepare firm-year data
print("\nPreparing firm-year consensus...")
fy1_us = prepare_fy1(data["us_fy1"], link, "US")
fy1_int = prepare_fy1(data["int_fy1"], link, "International")
fy1 = pd.concat([fy1_us, fy1_int], ignore_index=True).drop_duplicates(
    ["gvkey", "fyear"], keep="first"
)

ltg_us = prepare_ltg(data["us_ltg"], link, "US")
ltg_int = prepare_ltg(data["int_ltg"], link, "International")
ltg = pd.concat([ltg_us, ltg_int], ignore_index=True).drop_duplicates(
    ["gvkey", "fyear"], keep="first"
)

fy2_us = prepare_fy2(data["us_fy2"], link, "US")
fy2_int = prepare_fy2(data["int_fy2"], link, "International")
fy2 = pd.concat([fy2_us, fy2_int], ignore_index=True).drop_duplicates(
    ["gvkey", "fyear"], keep="first"
)

print(f"\nTotal FY1: {len(fy1):,}")
print(f"Total FY2: {len(fy2):,}")
print(f"Total LTG: {len(ltg):,}")

# Step 4: Load panel + shares outstanding + merge
print("\nLoading panel data...")
panel = pd.read_parquet(
    OUT_DIR / "panel_data.parquet",
    columns=["gvkey", "fyear", "fic", "at", "ib", "market_cap",
             "ggroup", "log_at", "log_pe", "leverage", "lag_earn_growth"],
)
panel["fic"] = panel["fic"].astype(str)

# R&D intensity (firm-level control)
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
print(f"  R&D observations: {len(xrd):,}")

# Shares outstanding: US from compustat_america, Global from compustat_global
print("Loading shares outstanding...")
us_sh = pd.read_csv(
    DATA_DIR / "compustat_america.csv",
    usecols=["GVKEY", "fyear", "csho"],
    dtype={"GVKEY": str},
    low_memory=False,
).rename(columns={"GVKEY": "gvkey"}).dropna(subset=["csho"])

# Global: load cshoi from compustat_global (annual fundamentals)
print("  Loading cshoi from compustat_global...")
gl_sh = pd.read_csv(
    DATA_DIR / "compustat_global.csv",
    usecols=["gvkey", "fyear", "cshoi"],
    dtype={"gvkey": str, "fyear": "Int64"},
    low_memory=False,
).dropna(subset=["cshoi", "fyear"])
gl_sh = gl_sh.rename(columns={"cshoi": "csho"})
print(f"  Global shares: {len(gl_sh):,}")

shares = pd.concat([us_sh, gl_sh], ignore_index=True).drop_duplicates(
    ["gvkey", "fyear"], keep="first"
)
print(f"  Total shares: {len(shares):,}")

# Merge
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

print(f"\nPanel summary:")
print(f"  Total obs:          {len(panel):,}")
print(f"  FY1 EPS available:  {panel['fy1_eps'].notna().sum():,}")
print(f"  FY2 EPS available:  {panel['fy2_eps'].notna().sum():,}")
print(f"  Shares available:   {panel['csho'].notna().sum():,}")
print(f"  Forward P/E valid:  {panel['log_fwd_pe'].notna().sum():,}")
print(f"  Forward P/E FY2:    {panel['log_fwd_pe_fy2'].notna().sum():,}")
print(f"  LTG available:      {panel['ltg'].notna().sum():,}")
fwd_countries = panel.loc[panel['log_fwd_pe'].notna(), 'fic'].nunique()
fwd2_countries = panel.loc[panel['log_fwd_pe_fy2'].notna(), 'fic'].nunique()
ltg_countries = panel.loc[panel['ltg'].notna(), 'fic'].nunique()
print(f"  Countries with fwd PE:     {fwd_countries}")
print(f"  Countries with fwd PE FY2: {fwd2_countries}")
print(f"  Countries with LTG:        {ltg_countries}")

# Country-level coverage
print("\nForward P/E coverage (top 15 countries by obs):")
cov = panel.loc[panel["log_fwd_pe"].notna()].groupby("fic").size().sort_values(ascending=False)
for fic in cov.head(15).index:
    print(f"  {fic}: {cov[fic]:,}")

# ---------------------------------------------------------------------------
# 5. Year-by-year regressions: trailing vs forward P/E
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COUNTRY EFFECTS: TRAILING vs FORWARD P/E")
print("=" * 70)

formula_base = (" ~ leverage + log_at + lag_earn_growth"
                " + C(fic, Treatment(reference='USA')) + C(ggroup)")
formula_full = (" ~ leverage + log_at + lag_earn_growth + xrd_at + ltg"
                " + C(fic, Treatment(reference='USA')) + C(ggroup)")
specs = {
    "trailing":         "log_pe" + formula_base,
    "trailing_full":    "log_pe" + formula_full,
    "forward":          "log_fwd_pe" + formula_base,
    "forward_fy2":      "log_fwd_pe_fy2" + formula_base,
}

all_effects = {}
for spec_name, formula in specs.items():
    dep = formula.split("~")[0].strip()
    drop_cols = [dep, "leverage", "log_at", "lag_earn_growth", "fic", "ggroup"]
    if "xrd_at" in formula:
        drop_cols.append("xrd_at")
    if "ltg" in formula:
        drop_cols.append("ltg")
    print(f"\nExtracting country effects: {spec_name} ({dep})...")
    results = []
    for yr, grp in panel.groupby("fyear"):
        sub = grp.dropna(subset=drop_cols)
        if sub["fic"].nunique() < 10:
            continue
        try:
            m = sm.OLS.from_formula(formula, data=sub).fit()
            for name, coef in m.params.items():
                if "C(fic" in name and "[T." in name:
                    iso = name.split("[T.")[1].rstrip("]")
                    results.append({"fyear": yr, "fic": iso, "country_effect": coef})
            results.append({"fyear": yr, "fic": "USA", "country_effect": 0.0})
        except Exception as e:
            print(f"  Year {yr:.0f}: {e}")

    ce = pd.DataFrame(results)
    ce.to_parquet(OUT_DIR / f"country_effects_{spec_name}_pe.parquet", index=False)
    all_effects[spec_name] = ce
    print(f"  {ce['fic'].nunique()} countries, {len(ce):,} country-year effects")

# --- Compare ---
print("\n--- Country effect dispersion ---")
for spec_name, ce in all_effects.items():
    avg = ce.groupby("fic")["country_effect"].mean()
    print(f"  {spec_name:>18s}: σ = {avg.std():.3f}, "
          f"range = [{avg.min():.2f}, {avg.max():.2f}], "
          f"countries = {len(avg)}")

# Use trailing_full (with R&D + LTG controls) as primary baseline
avg_trail = all_effects["trailing_full"].groupby("fic")["country_effect"].mean().rename("trailing")
avg_trail_base = all_effects["trailing"].groupby("fic")["country_effect"].mean().rename("trailing_base")
avg_fwd = all_effects["forward"].groupby("fic")["country_effect"].mean().rename("forward")
avg_fwd2 = all_effects["forward_fy2"].groupby("fic")["country_effect"].mean().rename("forward_fy2")
compare = (pd.DataFrame(avg_trail)
           .join(avg_trail_base, how="left")
           .join(avg_fwd, how="inner")
           .join(avg_fwd2, how="left"))
compare["change"] = compare["forward"] - compare["trailing"]
compare["change_fy2"] = compare["forward_fy2"] - compare["trailing"]

r_tf = compare["trailing"].corr(compare["forward"])
r_tf2 = compare["trailing"].corr(compare["forward_fy2"].dropna())
std_trail = compare["trailing"].std()
std_fwd = compare["forward"].std()
std_fwd2 = compare["forward_fy2"].dropna().std() if compare["forward_fy2"].notna().any() else np.nan
reduction = (1 - std_fwd / std_trail) * 100
reduction2 = (1 - std_fwd2 / std_trail) * 100 if not np.isnan(std_fwd2) else np.nan

print(f"\n  Common countries: {len(compare)}")
print(f"  Corr(trailing, forward FY1): {r_tf:.3f}")
print(f"  Corr(trailing, forward FY2): {r_tf2:.3f}")
print(f"  σ trailing = {std_trail:.3f}, σ forward FY1 = {std_fwd:.3f}, "
      f"σ forward FY2 = {std_fwd2:.3f}")
print(f"  Dispersion reduction FY1: {reduction:+.1f}%")
print(f"  Dispersion reduction FY2: {reduction2:+.1f}%")

print(f"\n  {'Ctry':>5s}  {'Trailing':>9s}  {'FwdFY1':>9s}  {'FwdFY2':>9s}  "
      f"{'Chg FY1':>8s}  {'Chg FY2':>8s}  "
      f"{'PE ratio':>9s}  {'FwdPE1':>8s}  {'FwdPE2':>8s}")
for fic in FOCUS:
    if fic in compare.index:
        row = compare.loc[fic]
        fy2_str = f"{row['forward_fy2']:>+9.3f}" if pd.notna(row['forward_fy2']) else f"{'n/a':>9s}"
        chg2_str = f"{row['change_fy2']:>+8.3f}" if pd.notna(row['change_fy2']) else f"{'n/a':>8s}"
        fpe2_str = f"{np.exp(row['forward_fy2']):>8.2f}" if pd.notna(row['forward_fy2']) else f"{'n/a':>8s}"
        print(f"  {fic:>5s}  {row['trailing']:>+9.3f}  {row['forward']:>+9.3f}  "
              f"{fy2_str}  {row['change']:>+8.3f}  {chg2_str}  "
              f"{np.exp(row['trailing']):>9.2f}  {np.exp(row['forward']):>8.2f}  "
              f"{fpe2_str}")

# ---------------------------------------------------------------------------
# 6. LTG analysis: analyst long-term growth vs country P/E
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYST LONG-TERM GROWTH (LTG) vs COUNTRY P/E")
print("=" * 70)

# Average LTG by country (need ≥50 firm-years for reliable estimate)
ltg_by_country = (
    panel.dropna(subset=["ltg", "fic"])
    .groupby("fic")
    .filter(lambda x: len(x) >= 50)
    .groupby("fic")["ltg"]
    .mean()
    .rename("ltg")
)

# Use trailing_full effects (with R&D + LTG firm-level controls)
avg_pe = all_effects["trailing_full"].groupby("fic")["country_effect"].mean().rename("pe_effect")

country = pd.DataFrame(avg_pe).join(ltg_by_country, how="inner")
country = country.astype(float)
country["pe_ratio"] = np.exp(country["pe_effect"])

r_ltg = country["pe_effect"].corr(country["ltg"])
n_ltg = len(country)
t_ltg = r_ltg * np.sqrt((n_ltg - 2) / (1 - r_ltg**2)) if abs(r_ltg) < 1 else np.nan

print(f"\n  Countries with LTG + P/E: {n_ltg}")
print(f"  r(P/E, LTG) = {r_ltg:+.3f}, t = {t_ltg:+.2f}")

if n_ltg >= 10:
    X = sm.add_constant(country["ltg"])
    m = sm.OLS(country["pe_effect"], X).fit()
    print(f"  OLS: coef = {m.params.iloc[1]:+.4f} (t={m.tvalues.iloc[1]:+.2f}), "
          f"R² = {m.rsquared:.3f}")

print(f"\n  {'Ctry':>5s}  {'P/E':>7s}  {'PE ratio':>8s}  {'LTG%':>6s}")
for fic in FOCUS:
    if fic in country.index:
        r = country.loc[fic]
        print(f"  {fic:>5s}  {r['pe_effect']:>+7.3f}  {r['pe_ratio']:>8.2f}  "
              f"{r['ltg']:>6.1f}")

# --- Horse race: LTG + R&D + Savings ---
if n_ltg >= 10:
    print("\n  Horse race: LTG + R&D + Savings")
    # Country-level average R&D intensity (from panel)
    avg_xrd = (panel.dropna(subset=["xrd_at", "fic"])
               .groupby("fic").filter(lambda x: len(x) >= 50)
               .groupby("fic")["xrd_at"].mean().rename("xrd_effect"))

    import wbgapi as wb
    sav = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2000, 2024), labels=False)
    sav = sav.T.mean().rename("savings_gni")

    country = country.join(avg_xrd, how="left").join(sav, how="left")

    for label, rhs in [
        ("LTG only",          ["ltg"]),
        ("R&D only",          ["xrd_effect"]),
        ("LTG + R&D",         ["ltg", "xrd_effect"]),
        ("LTG + R&D + Sav",   ["ltg", "xrd_effect", "savings_gni"]),
    ]:
        sub = country[["pe_effect"] + rhs].dropna().astype(float)
        if len(sub) < 10:
            continue
        X = sm.add_constant(sub[rhs])
        m = sm.OLS(sub["pe_effect"], X).fit()
        print(f"\n    {label} (n={len(sub)}): R²={m.rsquared:.3f}")
        for i, var in enumerate(rhs):
            print(f"      {var:>15s}: {m.params.iloc[i+1]:+.4f} "
                  f"(t={m.tvalues.iloc[i+1]:+.2f})")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------

# Fig 1: Dispersion over time — trailing vs forward
disp = {}
for spec_name, ce in all_effects.items():
    disp[spec_name] = ce.groupby("fyear")["country_effect"].std()
disp = pd.DataFrame(disp)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(disp.index, disp["trailing"], color=PALETTE["accent"],
        linewidth=2, label="Trailing P/E")
if "forward" in disp.columns:
    ax.plot(disp.index, disp["forward"], color=PALETTE["alert"],
            linewidth=2, label="Forward P/E (IBES FY1)")
if "forward_fy2" in disp.columns:
    ax.plot(disp.index, disp["forward_fy2"], color=PALETTE["highlight"],
            linewidth=2, label="Forward P/E (IBES FY2)")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Cross-country std dev of P/E effects", fontsize=10)
ax.set_title("P/E Dispersion: Trailing vs Forward (FY1 & FY2)", fontsize=11)
ax.legend(fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "dispersion_trailing_vs_forward.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "dispersion_trailing_vs_forward.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved dispersion_trailing_vs_forward")

# Fig 2: LTG vs P/E scatter
if n_ltg >= 10:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(country["ltg"], country["pe_ratio"],
               color=PALETTE["highlight"], s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    for fic, row in country.iterrows():
        if fic in FOCUS:
            ax.annotate(fic, (row["ltg"], row["pe_ratio"]),
                        fontsize=7, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points",
                        color=PALETTE["primary"])
    mask = np.isfinite(country["ltg"].values) & np.isfinite(country["pe_ratio"].values)
    if mask.sum() >= 3:
        z = np.polyfit(country["ltg"].values[mask], country["pe_ratio"].values[mask], 1)
        xl = np.linspace(country["ltg"].min(), country["ltg"].max(), 100)
        ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
                linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("IBES Long-Term Growth Forecast (%)", fontsize=10)
    ax.set_ylabel("P/E relative to USA", fontsize=10)
    ax.set_title(f"Analyst Growth Forecasts and P/E (r = {r_ltg:+.2f}, n = {n_ltg})",
                 fontsize=11)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_ltg_vs_pe.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "scatter_ltg_vs_pe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved scatter_ltg_vs_pe")

# Fig 3: Trailing vs forward country effects
if len(compare) >= 10:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.exp(compare["trailing"]), np.exp(compare["forward"]),
               color=PALETTE["accent"], s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    for fic, row in compare.iterrows():
        if fic in FOCUS:
            ax.annotate(fic, (np.exp(row["trailing"]), np.exp(row["forward"])),
                        fontsize=7, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points",
                        color=PALETTE["primary"])
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("Trailing P/E ratio (vs USA)", fontsize=10)
    ax.set_ylabel("Forward P/E ratio (vs USA)", fontsize=10)
    ax.set_title(f"Trailing vs Forward P/E Effects (r = {r_tf:.2f})", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_trailing_vs_forward_pe.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "scatter_trailing_vs_forward_pe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved scatter_trailing_vs_forward_pe")

# Fig 4: Country effects time series (trailing_full, 10 focus countries)
ce_full = all_effects["trailing_full"]
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(FOCUS)))
for i, fic in enumerate(FOCUS):
    sub = ce_full[ce_full["fic"] == fic].sort_values("fyear")
    if len(sub) >= 3:
        ax.plot(sub["fyear"], sub["country_effect"], label=fic,
                linewidth=1.5, color=colors[i])
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Country P/E effect (log, vs USA = 0)", fontsize=10)
ax.set_title("Country P/E Effects Over Time (with R&D + LTG controls)", fontsize=11)
ax.legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "country_effects_pe.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "country_effects_pe.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSaved country_effects_pe (trailing_full, 10 countries)")

# Fig 5: Dispersion over time (trailing_full vs base trailing)
ce_base = all_effects["trailing"]
disp_full = ce_full.groupby("fyear")["country_effect"].std().rename("trailing_full")
disp_base = ce_base.groupby("fyear")["country_effect"].std().rename("trailing_base")
disp_time = pd.DataFrame(disp_full).join(disp_base, how="outer")

fig, ax = plt.subplots(figsize=(7, 4))
if "trailing_base" in disp_time.columns:
    ax.plot(disp_time.index, disp_time["trailing_base"], color="gray",
            linewidth=1.5, linestyle="--", alpha=0.6, label="Base (no firm-level controls)")
ax.plot(disp_time.index, disp_time["trailing_full"], color=PALETTE["accent"],
        linewidth=2, label="With R&D + LTG controls")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Cross-country std dev of P/E effects", fontsize=10)
ax.set_title("P/E Dispersion Over Time", fontsize=11)
ax.legend(fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "dispersion_over_time.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "dispersion_over_time.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved dispersion_over_time (trailing_full)")

# --- Residual analysis: country effects after country-level controls ---
print("\n" + "=" * 70)
print("RESIDUAL COUNTRY EFFECTS (trailing_full baseline)")
print("=" * 70)

import wbgapi as wb

# Country-level variables
avg_pe_res = all_effects["trailing_full"].groupby("fic")["country_effect"].mean().rename("pe_effect")
avg_xrd_res = (panel.dropna(subset=["xrd_at", "fic"])
               .groupby("fic").filter(lambda x: len(x) >= 50)
               .groupby("fic")["xrd_at"].mean().rename("xrd_mean"))
sav_res = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2000, 2024), labels=False)
sav_res = sav_res.T.mean().rename("savings_gni")

res_df = pd.DataFrame(avg_pe_res).join(avg_xrd_res, how="left").join(sav_res, how="left").dropna()
res_df = res_df.astype(float)

print(f"\n  Countries with all variables: {len(res_df)}")
print(f"  Raw σ(country effects): {res_df['pe_effect'].std():.3f}")

# Progressive residualization
for label, rhs in [
    ("+ R&D (country avg)", ["xrd_mean"]),
    ("+ R&D + Savings", ["xrd_mean", "savings_gni"]),
]:
    X = sm.add_constant(res_df[rhs])
    m = sm.OLS(res_df["pe_effect"], X).fit()
    res_df[f"resid_{label}"] = m.resid
    sigma = m.resid.std()
    pct = (1 - sigma / res_df["pe_effect"].std()) * 100
    print(f"  {label}: σ_resid = {sigma:.3f} ({pct:.0f}% reduction)")

# Show residuals for focus countries
print(f"\n  {'Ctry':>5s}  {'Raw':>7s}  {'After R&D+Sav':>14s}")
resid_col = [c for c in res_df.columns if c.startswith("resid_")][-1]
for fic in FOCUS:
    if fic in res_df.index:
        r = res_df.loc[fic]
        print(f"  {fic:>5s}  {r['pe_effect']:>+7.3f}  {r[resid_col]:>+14.3f}")

# --- Regenerate cross-country scatter: R&D vs P/E ---
if len(res_df) >= 10:
    fig, ax = plt.subplots(figsize=(6, 4))
    pe_ratio = np.exp(res_df["pe_effect"])
    ax.scatter(res_df["xrd_mean"] * 100, pe_ratio,
               color=PALETTE["accent"], s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    for fic in res_df.index:
        if fic in FOCUS:
            ax.annotate(fic, (res_df.loc[fic, "xrd_mean"] * 100, pe_ratio[fic]),
                        fontsize=7, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points",
                        color=PALETTE["primary"])
    r_rd = res_df["pe_effect"].corr(res_df["xrd_mean"])
    n_rd = len(res_df)
    t_rd = r_rd * np.sqrt((n_rd - 2) / (1 - r_rd**2))
    ax.set_xlabel("Country avg R&D / Assets (%)", fontsize=10)
    ax.set_ylabel("P/E relative to USA", fontsize=10)
    ax.set_title(f"R&D Intensity and P/E (r = {r_rd:+.2f}, t = {t_rd:.2f}, n = {n_rd})",
                 fontsize=11)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_rd_vs_pe_ratio.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "scatter_rd_vs_pe_ratio.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved scatter_rd_vs_pe_ratio (trailing_full)")

# --- Regenerate cross-country scatter: Savings vs P/E ---
sav_full = pd.DataFrame(avg_pe_res).join(sav_res, how="inner").dropna().astype(float)
if len(sav_full) >= 10:
    fig, ax = plt.subplots(figsize=(6, 4))
    pe_ratio_s = np.exp(sav_full["pe_effect"])
    ax.scatter(sav_full["savings_gni"], pe_ratio_s,
               color=PALETTE["accent"], s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    for fic in sav_full.index:
        if fic in FOCUS:
            ax.annotate(fic, (sav_full.loc[fic, "savings_gni"], pe_ratio_s[fic]),
                        fontsize=7, fontweight="bold", ha="left", va="bottom",
                        xytext=(4, 2), textcoords="offset points",
                        color=PALETTE["primary"])
    r_sv = sav_full["pe_effect"].corr(sav_full["savings_gni"])
    n_sv = len(sav_full)
    t_sv = r_sv * np.sqrt((n_sv - 2) / (1 - r_sv**2))
    z = np.polyfit(sav_full["savings_gni"], pe_ratio_s, 1)
    xl = np.linspace(sav_full["savings_gni"].min(), sav_full["savings_gni"].max(), 100)
    ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
            linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("Gross savings / GNI (%)", fontsize=10)
    ax.set_ylabel("P/E relative to USA", fontsize=10)
    ax.set_title(f"National Savings and P/E (r = {r_sv:+.2f}, t = {t_sv:.2f}, n = {n_sv})",
                 fontsize=11)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_savings_vs_pe_ratio.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "scatter_savings_vs_pe_ratio.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved scatter_savings_vs_pe_ratio (trailing_full)")

print("\nDone.")
