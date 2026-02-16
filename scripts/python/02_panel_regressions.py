"""
02_panel_regressions.py -- Cross-country valuation panel regressions.

Purpose: Construct firm-year panel with P/E, M/B, and ROA,
         run cross-country regressions with country/industry/year
         fixed effects, extract country coefficient evolution,
         and perform variance decomposition.

Inputs:
  - compustat_global.csv (via DATA_DIR)
  - compustat_america.csv (via DATA_DIR)
  - returns_global.csv (via DATA_DIR, for Global market cap)

Outputs:
  - output/regressions/panel_data.parquet
  - output/regressions/regression_results.txt
  - output/regressions/country_effects_pe.pdf/png
  - output/regressions/country_effects_mb.pdf/png
  - output/regressions/country_effects_roa.pdf/png
  - output/regressions/dispersion_over_time.pdf/png
  - output/regressions/variance_decomposition.pdf/png
  - output/regressions/industry_country_heatmap.pdf/png
  - output/regressions/industry_country_heatmap_mb.pdf/png

Author: Augustin Landier, HEC Paris
"""

# stdlib
import gc
import itertools
import math
from io import StringIO
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# 0. Setup & Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    "~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/"
).expanduser()

OUT_DIR = Path("output/regressions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUT_DIR / "panel_data.parquet"

PALETTE = {
    "primary": "#2c3e50",
    "secondary": "#7f8c8d",
    "accent": "#2980b9",
    "highlight": "#8e44ad",
    "alert": "#c0392b",
}

# 10 focus countries for time-series plots
FOCUS_COUNTRIES = [
    "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "AUS", "BRA", "CAN",
]

# Minimum observations per country-year for inclusion
MIN_OBS_COUNTRY_YEAR = 30

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})

_report_buf = StringIO()


def rprint(*args, **kwargs):
    """Print to both console and report buffer."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=_report_buf)


# ---------------------------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------------------------


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Winsorize a Series at given quantiles.

    Parameters
    ----------
    s : pd.Series
        Input series.
    lo, hi : float
        Lower and upper quantile bounds.

    Returns
    -------
    pd.Series
        Winsorized series.
    """
    return s.clip(s.quantile(lo), s.quantile(hi))


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure in dual format (PDF + PNG) at 300 DPI."""
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    rprint(f"  Saved: {name}.pdf/png")


def format_regression_table(results: dict, dep_var: str) -> str:
    """Format regression results into a text table.

    Parameters
    ----------
    results : dict
        Keys are spec names, values are statsmodels result objects or dicts.
    dep_var : str
        Dependent variable name for the header.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  Dependent variable: {dep_var}")
    lines.append(f"{'='*80}")

    header = f"  {'Variable':<25s}"
    for spec_name in results:
        header += f"  {spec_name:>15s}"
    lines.append(header)
    lines.append("  " + "-" * (25 + 17 * len(results)))

    # Extract common coefficients (non-FE), deduplicating
    seen = set()
    coef_names = []
    for res in results.values():
        if isinstance(res, dict):
            names = res.get("coef_names", [])
        else:
            names = [p for p in res.params.index
                     if not p.startswith("C(") and p != "Intercept"]
        for n in names:
            if n not in seen:
                coef_names.append(n)
                seen.add(n)

    for cname in coef_names:
        row = f"  {cname:<25s}"
        for res in results.values():
            if isinstance(res, dict):
                coef = res.get("coefs", {}).get(cname, None)
                se = res.get("ses", {}).get(cname, None)
            else:
                coef = res.params.get(cname, None)
                se = res.bse.get(cname, None)
            if coef is not None:
                # Use p-values from model when available
                if isinstance(res, dict):
                    pval = res.get("pvalues", {}).get(cname, 1.0)
                else:
                    pval = res.pvalues.get(cname, 1.0)
                stars = "***" if pval < 0.01 else "** " if pval < 0.05 else "*  " if pval < 0.10 else "   "
                row += f"  {coef:>12.4f}{stars}"
            else:
                row += f"  {'':>15s}"
        lines.append(row)
        # Standard errors row
        se_row = f"  {'':25s}"
        for res in results.values():
            if isinstance(res, dict):
                se = res.get("ses", {}).get(cname, None)
            else:
                se = res.bse.get(cname, None)
            if se is not None:
                se_row += f"  ({se:>10.4f})  "
            else:
                se_row += f"  {'':>15s}"
        lines.append(se_row)

    lines.append("  " + "-" * (25 + 17 * len(results)))

    # Summary stats
    summary_row = f"  {'N':<25s}"
    for res in results.values():
        n = res["n"] if isinstance(res, dict) else int(res.nobs)
        summary_row += f"  {n:>15,}"
    lines.append(summary_row)

    r2_row = f"  {'R-squared':<25s}"
    for res in results.values():
        r2 = res["r2"] if isinstance(res, dict) else res.rsquared
        r2_row += f"  {r2:>15.4f}"
    lines.append(r2_row)

    # FE indicators
    for fe_label in ["Country FE", "Industry FE", "Year FE", "Country x Year FE"]:
        fe_row = f"  {fe_label:<25s}"
        for res in results.values():
            if isinstance(res, dict):
                yes = res.get("fe", {}).get(fe_label, False)
            else:
                yes = fe_label in getattr(res, "_fe_labels", [])
            fe_row += f"  {'Yes':>15s}" if yes else f"  {'No':>15s}"
        lines.append(fe_row)

    lines.append(f"{'='*80}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Load and Prepare US Panel
# ---------------------------------------------------------------------------


def load_us_panel() -> pd.DataFrame:
    """Load Compustat NA and construct US firm-year panel.

    Returns
    -------
    pd.DataFrame
        US firm-year panel with: gvkey, fyear, fic, ggroup, market_cap,
        ceq, at, ib, log_mb, roa, log_at, lag_earn_growth.
    """
    rprint("\n  Loading US panel (compustat_america)...")
    cols = [
        "GVKEY", "fyear", "at", "ceq", "ib", "ggroup", "mkvalt",
        "prcc_f", "csho", "fic",
    ]
    df = pd.read_csv(
        DATA_DIR / "compustat_america.csv",
        usecols=cols,
        dtype={"GVKEY": str},
        low_memory=False,
    )
    df = df.rename(columns={"GVKEY": "gvkey"})

    # Country: fill missing with USA
    if "fic" not in df.columns or df["fic"].isna().all():
        df["fic"] = "USA"
    else:
        df["fic"] = df["fic"].fillna("USA")

    # Market cap: prefer mkvalt, fallback to prcc_f * csho
    df["market_cap"] = df["mkvalt"]
    mask_missing = df["market_cap"].isna()
    if "prcc_f" in df.columns and "csho" in df.columns:
        df.loc[mask_missing, "market_cap"] = (
            df.loc[mask_missing, "prcc_f"] * df.loc[mask_missing, "csho"]
        )

    # Industry: GICS Industry Group (~25 categories)
    df["ggroup"] = df["ggroup"].astype("Int64")

    # Filter
    df = df[(df["at"] > 0) & (df["ceq"] > 0) & (df["market_cap"] > 0)].copy()

    # Compute variables
    mb_raw = df["market_cap"] / df["ceq"]
    mb_win = winsorize(mb_raw)
    df["log_mb"] = np.log(mb_win)
    df["roa"] = winsorize(df["ib"] / df["at"])
    df["log_at"] = np.log(df["at"])

    # Leverage: fraction of assets financed by non-equity
    df["leverage"] = winsorize((df["at"] - df["ceq"]) / df["at"])

    # P/E: log(market_cap / earnings) for profitable firms only
    df["log_pe"] = np.nan
    pe_mask = df["ib"] > 0
    pe_raw = df.loc[pe_mask, "market_cap"] / df.loc[pe_mask, "ib"]
    df.loc[pe_mask, "log_pe"] = np.log(winsorize(pe_raw))

    # Lag earnings growth: (ib_{t-1} - ib_{t-2}) / |ib_{t-2}|
    df = df.sort_values(["gvkey", "fyear"])
    ib_lag1 = df.groupby("gvkey")["ib"].shift(1)
    ib_lag2 = df.groupby("gvkey")["ib"].shift(2)
    valid_denom = ib_lag2.abs() > 0.01  # avoid division by near-zero
    df["lag_earn_growth"] = np.where(
        valid_denom, (ib_lag1 - ib_lag2) / ib_lag2.abs(), np.nan
    )
    # Winsorize earnings growth at 1/99
    mask_valid = df["lag_earn_growth"].notna()
    df.loc[mask_valid, "lag_earn_growth"] = winsorize(
        df.loc[mask_valid, "lag_earn_growth"]
    )

    # US firms use US GAAP
    df["acctstd"] = "DA"

    # Keep needed columns
    keep = [
        "gvkey", "fyear", "fic", "ggroup", "market_cap", "ceq", "at", "ib",
        "log_mb", "log_pe", "roa", "log_at", "leverage", "lag_earn_growth",
        "acctstd",
    ]
    df = df[keep].copy()

    n_pe = df["log_pe"].notna().sum()
    rprint(f"  US panel: {len(df):,} obs, {df['gvkey'].nunique():,} firms, "
           f"fyear {df['fyear'].min():.0f}–{df['fyear'].max():.0f}")
    rprint(f"    P/E available (ib>0): {n_pe:,} ({n_pe/len(df)*100:.0f}%)")
    return df


# ---------------------------------------------------------------------------
# 3. Load and Prepare Global Panel
# ---------------------------------------------------------------------------


def load_global_panel() -> pd.DataFrame:
    """Load Compustat Global + Returns Global, merge for market cap.

    Market cap = cshoi (shares outstanding) * prccm (fiscal year-end price).
    Uses ``prirow`` (primary security for rest-of-world) to select the
    correct price from the returns file, avoiding cross-currency listings
    and different share-class denominations.

    Returns
    -------
    pd.DataFrame
        Global firm-year panel with same columns as US panel.
    """
    rprint("\n  Loading Global panel (compustat_global + returns_global)...")

    # --- Compustat Global fundamentals ---
    fund_cols = [
        "gvkey", "fyear", "datadate", "at", "ceq", "ib", "ggroup",
        "cshoi", "fic", "prirow", "curcd", "acctstd",
    ]
    df = pd.read_csv(
        DATA_DIR / "compustat_global.csv",
        usecols=fund_cols,
        dtype={"gvkey": str},
        low_memory=False,
    )
    rprint(f"  Compustat Global loaded: {len(df):,} rows")

    # Parse datadate → year-month key for merge
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df["ym"] = df["datadate"].dt.to_period("M")

    # --- Returns Global (for fiscal year-end price) ---
    ret_cols = ["gvkey", "datadate", "prccm", "iid", "curcdm"]
    df_ret = pd.read_csv(
        DATA_DIR / "returns_global.csv",
        usecols=ret_cols,
        dtype={"gvkey": str},
        low_memory=False,
    )
    rprint(f"  Returns Global loaded: {len(df_ret):,} rows")

    df_ret["datadate"] = pd.to_datetime(df_ret["datadate"], errors="coerce")
    df_ret["ym"] = df_ret["datadate"].dt.to_period("M")

    # Use prirow to select the primary security for each firm.
    # prirow maps to the iid in the returns file (e.g., "01W", "02W").
    # This avoids cross-currency listings (e.g., HUF listing for a EUR firm)
    # and different share-class denominations (e.g., pre/post redenomination
    # in Turkey).
    df_ret = df_ret.rename(columns={"iid": "prirow"})
    n_before = len(df)
    df = df.merge(
        df_ret[["gvkey", "ym", "prirow", "prccm", "curcdm"]],
        on=["gvkey", "ym", "prirow"],
        how="left",
    )
    n_matched = df["prccm"].notna().sum()
    rprint(f"  After prirow merge: {n_matched:,}/{n_before:,} "
           f"have fiscal year-end price ({n_matched/n_before*100:.1f}%)")

    # Validate currency alignment
    cur_match = (df["curcd"] == df["curcdm"]).sum()
    cur_total = df["curcdm"].notna().sum()
    rprint(f"  Currency alignment: {cur_match:,}/{cur_total:,} match "
           f"({cur_match/max(cur_total,1)*100:.1f}%)")

    del df_ret
    gc.collect()

    # Market cap = shares outstanding * price
    df["market_cap"] = df["cshoi"] * df["prccm"]

    # Industry: GICS Industry Group (~25 categories)
    df["ggroup"] = df["ggroup"].astype("Int64")

    # Filter: require positive assets, equity, and market cap.
    # Financial firms included; country FE absorb regulatory differences.
    # M/B in reporting currency; cross-country levels absorbed by country FE.
    n_before = len(df)
    df = df[
        (df["at"] > 0) & (df["ceq"] > 0) & (df["market_cap"] > 0)
    ].copy()
    rprint(f"  After filters: {len(df):,} obs (dropped {n_before - len(df):,})")

    # Compute variables
    mb_raw = df["market_cap"] / df["ceq"]
    mb_win = winsorize(mb_raw)
    df["log_mb"] = np.log(mb_win)
    df["roa"] = winsorize(df["ib"] / df["at"])
    df["log_at"] = np.log(df["at"])

    # Leverage
    df["leverage"] = winsorize((df["at"] - df["ceq"]) / df["at"])

    # P/E: log(market_cap / earnings) for profitable firms only
    df["log_pe"] = np.nan
    pe_mask = df["ib"] > 0
    pe_raw = df.loc[pe_mask, "market_cap"] / df.loc[pe_mask, "ib"]
    df.loc[pe_mask, "log_pe"] = np.log(winsorize(pe_raw))

    # Lag earnings growth
    df = df.sort_values(["gvkey", "fyear"])
    ib_lag1 = df.groupby("gvkey")["ib"].shift(1)
    ib_lag2 = df.groupby("gvkey")["ib"].shift(2)
    valid_denom = ib_lag2.abs() > 0.01
    df["lag_earn_growth"] = np.where(
        valid_denom, (ib_lag1 - ib_lag2) / ib_lag2.abs(), np.nan
    )
    mask_valid = df["lag_earn_growth"].notna()
    df.loc[mask_valid, "lag_earn_growth"] = winsorize(
        df.loc[mask_valid, "lag_earn_growth"]
    )

    keep = [
        "gvkey", "fyear", "fic", "ggroup", "market_cap", "ceq", "at", "ib",
        "log_mb", "log_pe", "roa", "log_at", "leverage", "lag_earn_growth",
        "acctstd",
    ]
    df = df[keep].copy()

    n_pe = df["log_pe"].notna().sum()
    rprint(f"  Global panel: {len(df):,} obs, {df['gvkey'].nunique():,} firms, "
           f"{df['fic'].nunique()} countries, "
           f"fyear {df['fyear'].min():.0f}–{df['fyear'].max():.0f}")
    rprint(f"    P/E available (ib>0): {n_pe:,} ({n_pe/len(df)*100:.0f}%)")

    # Report accounting standard breakdown
    acct = df["acctstd"].value_counts(dropna=False)
    rprint("    Accounting standards:")
    for std, cnt in acct.items():
        label = {"DI": "IFRS", "DS": "Domestic GAAP", "DA": "US GAAP"}.get(
            str(std), str(std))
        rprint(f"      {label}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# 4. Stack Panels and Construct Variables
# ---------------------------------------------------------------------------


def build_panel(df_us: pd.DataFrame, df_global: pd.DataFrame) -> pd.DataFrame:
    """Stack US and Global panels, apply final filters, save parquet.

    Parameters
    ----------
    df_us : pd.DataFrame
        US firm-year panel.
    df_global : pd.DataFrame
        Global firm-year panel.

    Returns
    -------
    pd.DataFrame
        Combined panel.
    """
    rprint("\n  Building combined panel...")

    # Drop any US firms from Global to avoid duplication
    if "fic" in df_global.columns:
        n_us_in_global = (df_global["fic"] == "USA").sum()
        if n_us_in_global > 0:
            rprint(f"  Dropping {n_us_in_global:,} USA rows from Global panel")
            df_global = df_global[df_global["fic"] != "USA"].copy()

    df = pd.concat([df_us, df_global], ignore_index=True)

    # Drop 2025 (partial year with incomplete coverage)
    n_2025 = (df["fyear"] == 2025).sum()
    if n_2025 > 0:
        df = df[df["fyear"] < 2025].copy()
        rprint(f"  Dropped {n_2025:,} obs from 2025 (partial year)")

    # Drop duplicate gvkey-fyear (keep first = US version)
    n_before = len(df)
    df = df.drop_duplicates(subset=["gvkey", "fyear"], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        rprint(f"  Dropped {n_dropped:,} duplicate gvkey-fyear rows")

    # Filter country-year cells with too few observations
    cy_counts = df.groupby(["fic", "fyear"]).size()
    valid_cy = cy_counts[cy_counts >= MIN_OBS_COUNTRY_YEAR].index
    valid_cy_set = set(valid_cy)
    df["_cy"] = list(zip(df["fic"], df["fyear"]))
    n_before = len(df)
    df = df[df["_cy"].isin(valid_cy_set)].drop(columns=["_cy"]).copy()
    rprint(f"  Filtered to country-years with >= {MIN_OBS_COUNTRY_YEAR} obs: "
           f"{n_before:,} → {len(df):,}")

    # Convert to categorical for memory
    df["fic"] = df["fic"].astype("category")
    df["ggroup"] = df["ggroup"].astype("category")

    # Save parquet
    df.to_parquet(PARQUET_PATH, index=False)
    rprint(f"  Panel saved to {PARQUET_PATH}")

    rprint(f"\n  Combined panel: {len(df):,} obs, {df['gvkey'].nunique():,} firms, "
           f"{df['fic'].nunique()} countries, "
           f"fyear {df['fyear'].min():.0f}–{df['fyear'].max():.0f}")
    return df


# ---------------------------------------------------------------------------
# 5. Summary Statistics
# ---------------------------------------------------------------------------


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print descriptive statistics for the regression panel."""
    rprint("\n" + "=" * 70)
    rprint("  SUMMARY STATISTICS")
    rprint("=" * 70)

    rprint(f"  Total firm-years: {len(df):,}")
    rprint(f"  Unique firms: {df['gvkey'].nunique():,}")
    rprint(f"  Countries: {df['fic'].nunique()}")
    rprint(f"  Year range: {df['fyear'].min():.0f}–{df['fyear'].max():.0f}")

    # US vs non-US breakdown
    us_mask = df["fic"] == "USA"
    rprint(f"  US obs: {us_mask.sum():,} ({us_mask.mean()*100:.1f}%)")
    rprint(f"  Non-US obs: {(~us_mask).sum():,} ({(~us_mask).mean()*100:.1f}%)")

    # Variable summaries
    stats_vars = ["log_pe", "log_mb", "leverage", "roa", "log_at", "lag_earn_growth"]
    rprint(f"\n  {'Variable':<20s} {'N':>8s} {'Mean':>8s} {'Std':>8s} "
           f"{'P10':>8s} {'Median':>8s} {'P90':>8s}")
    rprint("  " + "-" * 68)
    for v in stats_vars:
        s = df[v].dropna()
        rprint(f"  {v:<20s} {len(s):>8,} {s.mean():>8.3f} {s.std():>8.3f} "
               f"{s.quantile(0.10):>8.3f} {s.median():>8.3f} "
               f"{s.quantile(0.90):>8.3f}")

    # Top 10 countries
    rprint("\n  Top 10 countries by obs:")
    top = df["fic"].value_counts().head(10)
    for country, cnt in top.items():
        rprint(f"    {country:<6s} {cnt:>8,}")


# ---------------------------------------------------------------------------
# 6. M/B Pooled Regressions
# ---------------------------------------------------------------------------


def run_pe_regressions(df: pd.DataFrame) -> dict:
    """Run P/E regression specifications (profitable firms only).

    Spec 1: log_pe ~ leverage + log_at + lag_earn_growth + C(fic) + C(ggroup) + C(fyear)
    Spec 2: log_pe ~ leverage + log_at + lag_earn_growth + C(ggroup) + Country×Year FE

    Returns
    -------
    dict
        Spec names to result objects/dicts.
    """
    rprint("\n" + "=" * 70)
    rprint("  P/E REGRESSIONS (profitable firms only)")
    rprint("=" * 70)

    reg_df = df[["log_pe", "leverage", "log_at", "lag_earn_growth",
                 "fic", "ggroup", "fyear"]].dropna().copy()
    reg_df["fic"] = reg_df["fic"].astype(str)
    reg_df["ggroup"] = reg_df["ggroup"].astype(str)
    reg_df["fyear_cat"] = reg_df["fyear"].astype(str)

    rprint(f"  Regression sample: {len(reg_df):,} obs (ib > 0)")
    results = {}

    # --- Spec 1: Separate FE ---
    rprint("\n  Spec 1: Country + Industry + Year FE...")
    formula1 = ("log_pe ~ leverage + log_at + lag_earn_growth "
                "+ C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear_cat)")
    m1 = smf.ols(formula1, data=reg_df).fit(
        cov_type="cluster", cov_kwds={"groups": reg_df["fic"]}
    )
    results["Spec 1"] = m1
    m1._fe_labels = ["Country FE", "Industry FE", "Year FE"]
    rprint(f"    N={int(m1.nobs):,}, R²={m1.rsquared:.4f}")
    rprint(f"    leverage:         {m1.params['leverage']:.4f} (se={m1.bse['leverage']:.4f})")
    rprint(f"    log_at:           {m1.params['log_at']:.4f} (se={m1.bse['log_at']:.4f})")
    rprint(f"    lag_earn_growth:  {m1.params['lag_earn_growth']:.4f} "
           f"(se={m1.bse['lag_earn_growth']:.4f})")

    # --- Spec 2: Country×Year FE via Frisch-Waugh ---
    rprint("\n  Spec 2: Country×Year FE (Frisch-Waugh demeaning)...")
    demean_vars = ["log_pe", "leverage", "log_at", "lag_earn_growth"]
    cy_group = reg_df["fic"] + "_" + reg_df["fyear_cat"]

    reg_df_dm = reg_df.copy()
    for v in demean_vars:
        group_mean = reg_df_dm.groupby(cy_group)[v].transform("mean")
        reg_df_dm[v] = reg_df_dm[v] - group_mean

    formula2 = "log_pe ~ leverage + log_at + lag_earn_growth + C(ggroup) - 1"
    m2 = smf.ols(formula2, data=reg_df_dm).fit(
        cov_type="cluster", cov_kwds={"groups": reg_df["fic"]}
    )

    y_orig = reg_df["log_pe"].values
    y_hat_cy_mean = reg_df.groupby(cy_group)["log_pe"].transform("mean").values
    y_hat_full = y_hat_cy_mean + m2.fittedvalues.values
    ss_res = np.sum((y_orig - y_hat_full) ** 2)
    ss_tot = np.sum((y_orig - np.mean(y_orig)) ** 2)
    r2_full = 1.0 - ss_res / ss_tot
    assert 0 <= r2_full <= 1, f"Frisch-Waugh R² out of bounds: {r2_full:.4f}"

    spec2_result = {
        "coef_names": ["leverage", "log_at", "lag_earn_growth"],
        "coefs": {
            "leverage": m2.params.get("leverage", np.nan),
            "log_at": m2.params.get("log_at", np.nan),
            "lag_earn_growth": m2.params.get("lag_earn_growth", np.nan),
        },
        "ses": {
            "leverage": m2.bse.get("leverage", np.nan),
            "log_at": m2.bse.get("log_at", np.nan),
            "lag_earn_growth": m2.bse.get("lag_earn_growth", np.nan),
        },
        "pvalues": {
            "leverage": m2.pvalues.get("leverage", 1.0),
            "log_at": m2.pvalues.get("log_at", 1.0),
            "lag_earn_growth": m2.pvalues.get("lag_earn_growth", 1.0),
        },
        "n": int(m2.nobs),
        "r2": r2_full,
        "fe": {
            "Country FE": False, "Industry FE": True,
            "Year FE": False, "Country x Year FE": True,
        },
    }
    results["Spec 2"] = spec2_result
    rprint(f"    N={int(m2.nobs):,}, R²={r2_full:.4f}")
    rprint(f"    leverage:         {m2.params.get('leverage', np.nan):.4f}")
    rprint(f"    log_at:           {m2.params.get('log_at', np.nan):.4f}")
    rprint(f"    lag_earn_growth:  {m2.params.get('lag_earn_growth', np.nan):.4f}")

    table = format_regression_table(results, "log(P/E)")
    rprint(table)

    return results


# ---------------------------------------------------------------------------
# 6b. M/B Pooled Regressions
# ---------------------------------------------------------------------------


def run_mb_regressions(df: pd.DataFrame) -> dict:
    """Run M/B regression specifications (all firms with ceq > 0).

    Spec 1: log_mb ~ roa + leverage + log_at + lag_earn_growth + C(fic) + C(ggroup) + C(fyear)
    Spec 2: log_mb ~ roa + leverage + log_at + lag_earn_growth + C(ggroup) + Country×Year FE

    Returns
    -------
    dict
        Spec names to result objects/dicts.
    """
    rprint("\n" + "=" * 70)
    rprint("  M/B REGRESSIONS")
    rprint("=" * 70)

    reg_df = df[["log_mb", "roa", "leverage", "log_at", "lag_earn_growth",
                 "fic", "ggroup", "fyear"]].dropna().copy()
    reg_df["fic"] = reg_df["fic"].astype(str)
    reg_df["ggroup"] = reg_df["ggroup"].astype(str)
    reg_df["fyear_cat"] = reg_df["fyear"].astype(str)

    rprint(f"  Regression sample: {len(reg_df):,} obs")
    results = {}

    # --- Spec 1: Separate FE ---
    rprint("\n  Spec 1: Country + Industry + Year FE...")
    formula1 = ("log_mb ~ roa + leverage + log_at + lag_earn_growth "
                "+ C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear_cat)")
    m1 = smf.ols(formula1, data=reg_df).fit(
        cov_type="cluster", cov_kwds={"groups": reg_df["fic"]}
    )
    results["Spec 1"] = m1
    m1._fe_labels = ["Country FE", "Industry FE", "Year FE"]
    rprint(f"    N={int(m1.nobs):,}, R²={m1.rsquared:.4f}")
    rprint(f"    roa:              {m1.params['roa']:.4f} (se={m1.bse['roa']:.4f})")
    rprint(f"    leverage:         {m1.params['leverage']:.4f} (se={m1.bse['leverage']:.4f})")
    rprint(f"    log_at:           {m1.params['log_at']:.4f} (se={m1.bse['log_at']:.4f})")
    rprint(f"    lag_earn_growth:  {m1.params['lag_earn_growth']:.4f} "
           f"(se={m1.bse['lag_earn_growth']:.4f})")

    # --- Spec 2: Country×Year FE via Frisch-Waugh ---
    rprint("\n  Spec 2: Country×Year FE (Frisch-Waugh demeaning)...")
    demean_vars = ["log_mb", "roa", "leverage", "log_at", "lag_earn_growth"]
    cy_group = reg_df["fic"] + "_" + reg_df["fyear_cat"]

    reg_df_dm = reg_df.copy()
    for v in demean_vars:
        group_mean = reg_df_dm.groupby(cy_group)[v].transform("mean")
        reg_df_dm[v] = reg_df_dm[v] - group_mean

    formula2 = "log_mb ~ roa + leverage + log_at + lag_earn_growth + C(ggroup) - 1"
    m2 = smf.ols(formula2, data=reg_df_dm).fit(
        cov_type="cluster", cov_kwds={"groups": reg_df["fic"]}
    )

    y_orig = reg_df["log_mb"].values
    y_hat_cy_mean = reg_df.groupby(cy_group)["log_mb"].transform("mean").values
    y_hat_full = y_hat_cy_mean + m2.fittedvalues.values
    ss_res = np.sum((y_orig - y_hat_full) ** 2)
    ss_tot = np.sum((y_orig - np.mean(y_orig)) ** 2)
    r2_full = 1.0 - ss_res / ss_tot
    assert 0 <= r2_full <= 1, f"Frisch-Waugh R² out of bounds: {r2_full:.4f}"

    spec2_result = {
        "coef_names": ["roa", "leverage", "log_at", "lag_earn_growth"],
        "coefs": {
            "roa": m2.params.get("roa", np.nan),
            "leverage": m2.params.get("leverage", np.nan),
            "log_at": m2.params.get("log_at", np.nan),
            "lag_earn_growth": m2.params.get("lag_earn_growth", np.nan),
        },
        "ses": {
            "roa": m2.bse.get("roa", np.nan),
            "leverage": m2.bse.get("leverage", np.nan),
            "log_at": m2.bse.get("log_at", np.nan),
            "lag_earn_growth": m2.bse.get("lag_earn_growth", np.nan),
        },
        "pvalues": {
            "roa": m2.pvalues.get("roa", 1.0),
            "leverage": m2.pvalues.get("leverage", 1.0),
            "log_at": m2.pvalues.get("log_at", 1.0),
            "lag_earn_growth": m2.pvalues.get("lag_earn_growth", 1.0),
        },
        "n": int(m2.nobs),
        "r2": r2_full,
        "fe": {
            "Country FE": False, "Industry FE": True,
            "Year FE": False, "Country x Year FE": True,
        },
    }
    results["Spec 2"] = spec2_result
    rprint(f"    N={int(m2.nobs):,}, R²={r2_full:.4f}")
    rprint(f"    roa:              {m2.params.get('roa', np.nan):.4f}")
    rprint(f"    leverage:         {m2.params.get('leverage', np.nan):.4f}")
    rprint(f"    log_at:           {m2.params.get('log_at', np.nan):.4f}")
    rprint(f"    lag_earn_growth:  {m2.params.get('lag_earn_growth', np.nan):.4f}")

    table = format_regression_table(results, "log(M/B)")
    rprint(table)

    return results


# ---------------------------------------------------------------------------
# 7. ROA Pooled Regressions
# ---------------------------------------------------------------------------


def run_roa_regressions(df: pd.DataFrame) -> dict:
    """Run ROA regression specification.

    Spec 1: roa ~ log_at + C(fic) + C(ggroup) + C(fyear)

    Returns
    -------
    dict
        Spec name to result object.
    """
    rprint("\n" + "=" * 70)
    rprint("  ROA REGRESSIONS")
    rprint("=" * 70)

    reg_df = df[["roa", "leverage", "log_at", "fic", "ggroup", "fyear"]].dropna().copy()
    reg_df["fic"] = reg_df["fic"].astype(str)
    reg_df["ggroup"] = reg_df["ggroup"].astype(str)
    reg_df["fyear_cat"] = reg_df["fyear"].astype(str)

    rprint(f"  Regression sample: {len(reg_df):,} obs")
    results = {}

    rprint("\n  Spec 1: Country + Industry + Year FE...")
    formula = ("roa ~ leverage + log_at "
               "+ C(fic, Treatment(reference='USA')) + C(ggroup) + C(fyear_cat)")
    m1 = smf.ols(formula, data=reg_df).fit(
        cov_type="cluster", cov_kwds={"groups": reg_df["fic"]}
    )
    results["Spec 1"] = m1
    m1._fe_labels = ["Country FE", "Industry FE", "Year FE"]
    rprint(f"    N={int(m1.nobs):,}, R²={m1.rsquared:.4f}")
    rprint(f"    log_at: {m1.params['log_at']:.4f} (se={m1.bse['log_at']:.4f})")

    table = format_regression_table(results, "ROA")
    rprint(table)

    return results


# ---------------------------------------------------------------------------
# 8. Year-by-Year Regressions and Country Effect Extraction
# ---------------------------------------------------------------------------


def run_yearly_regressions(
    df: pd.DataFrame,
    dep_var: str,
    controls: list,
    label: str,
) -> pd.DataFrame:
    """Run cross-sectional regressions year by year, extract country FE.

    For each year: dep_var ~ controls + C(fic, ref=USA) + C(ggroup)

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    dep_var : str
        'log_mb' or 'roa'.
    controls : list
        Control variable names.
    label : str
        'mb' or 'roa' for output labeling.

    Returns
    -------
    pd.DataFrame
        Columns: fyear, fic, country_effect, se, n_obs.
    """
    rprint(f"\n  Year-by-year regressions for {dep_var}...")
    all_cols = [dep_var] + controls + ["fic", "ggroup", "fyear"]
    reg_df = df[all_cols].dropna().copy()
    reg_df["fic"] = reg_df["fic"].astype(str)
    reg_df["ggroup"] = reg_df["ggroup"].astype(str)

    years = sorted(reg_df["fyear"].unique())
    records = []

    control_str = " + ".join(controls) if controls else "1"
    formula = f"{dep_var} ~ {control_str} + C(fic, Treatment(reference='USA')) + C(ggroup)"

    for yr in years:
        yr_df = reg_df[reg_df["fyear"] == yr]
        if len(yr_df) < 100:
            continue

        # Ensure USA is in the sample for this year
        if "USA" not in yr_df["fic"].values:
            continue

        try:
            m = smf.ols(formula, data=yr_df).fit(cov_type="HC1")
        except (np.linalg.LinAlgError, ValueError) as e:
            rprint(f"    WARNING: Regression failed for year {yr}: {e}")
            continue

        # Extract country coefficients
        for param_name, coef in m.params.items():
            if param_name.startswith("C(fic"):
                # Parse country code from e.g. "C(fic, Treatment(reference='USA'))[T.JPN]"
                country = param_name.split("[T.")[-1].rstrip("]")
                records.append({
                    "fyear": yr,
                    "fic": country,
                    "country_effect": coef,
                    "se": m.bse[param_name],
                    "n_obs": int(m.nobs),
                })

        # USA is reference (effect = 0)
        records.append({
            "fyear": yr,
            "fic": "USA",
            "country_effect": 0.0,
            "se": 0.0,
            "n_obs": int(m.nobs),
        })

    effects_df = pd.DataFrame(records)
    rprint(f"  Extracted {len(effects_df):,} country-year effects "
           f"across {effects_df['fyear'].nunique()} years")
    return effects_df


def variance_decomposition(df: pd.DataFrame, dep_var: str) -> dict:
    """Compute Shapley-value R² decomposition for country, industry, year.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    dep_var : str
        'log_mb' or 'roa'.

    Returns
    -------
    dict
        Keys: 'Country', 'Industry', 'Year', values: average marginal R².
    """
    rprint(f"\n  Variance decomposition for {dep_var}...")
    reg_df = df[[dep_var, "fic", "ggroup", "fyear"]].dropna().copy()
    reg_df["fic"] = reg_df["fic"].astype(str)
    reg_df["ggroup"] = reg_df["ggroup"].astype(str)
    reg_df["fyear_cat"] = reg_df["fyear"].astype(str)

    factors = {
        "Country": "C(fic)",
        "Industry": "C(ggroup)",
        "Year": "C(fyear_cat)",
    }
    factor_names = list(factors.keys())

    # Compute R² for all subsets of factors
    r2_cache = {}
    for size in range(len(factor_names) + 1):
        for subset in itertools.combinations(factor_names, size):
            subset_key = frozenset(subset)
            if not subset:
                r2_cache[subset_key] = 0.0
                continue
            rhs = " + ".join(factors[f] for f in subset)
            formula = f"{dep_var} ~ {rhs}"
            try:
                m = smf.ols(formula, data=reg_df).fit()
                r2_cache[subset_key] = m.rsquared
            except (np.linalg.LinAlgError, ValueError) as e:
                rprint(f"    WARNING: R² regression failed for {subset}: {e}")
                r2_cache[subset_key] = 0.0

    # Shapley values: average marginal contribution
    n = len(factor_names)
    shapley = {f: 0.0 for f in factor_names}
    for f in factor_names:
        others = [x for x in factor_names if x != f]
        for size in range(n):
            for subset in itertools.combinations(others, size):
                subset_with = frozenset(list(subset) + [f])
                subset_without = frozenset(subset)
                marginal = r2_cache[subset_with] - r2_cache[subset_without]
                # Weight: |S|! * (n - |S| - 1)! / n!
                s = len(subset)
                weight = (
                    math.factorial(s)
                    * math.factorial(n - s - 1)
                    / math.factorial(n)
                )
                shapley[f] += weight * marginal

    rprint(f"  R² decomposition for {dep_var}:")
    for f, val in shapley.items():
        rprint(f"    {f:<12s} {val:.4f} ({val/sum(shapley.values())*100:.1f}%)")
    rprint(f"    Total R²:  {sum(shapley.values()):.4f}")

    return shapley


# ---------------------------------------------------------------------------
# 8b. Industry × Country Interaction Exploration
# ---------------------------------------------------------------------------

# GICS Sector labels for readable output
GICS_SECTOR_MAP = {
    10: "Energy", 15: "Materials", 20: "Industrials", 25: "Cons Disc",
    30: "Cons Staples", 35: "Health Care", 40: "Financials",
    45: "Info Tech", 50: "Comm Svc", 55: "Utilities", 60: "Real Estate",
}


def explore_industry_country_interaction(
    df: pd.DataFrame,
    dep_var: str = "log_pe",
    controls: list | None = None,
) -> pd.DataFrame:
    """Test whether country effects on a valuation metric vary across industries.

    Parameters
    ----------
    dep_var : str
        Dependent variable ('log_pe' or 'log_mb').
    controls : list
        Control variables for per-sector regressions.

    Returns
    -------
    pd.DataFrame
        Country effects by sector (rows=countries, columns=sectors).
    """
    if controls is None:
        controls = ["leverage", "log_at"]
    label = "P/E" if dep_var == "log_pe" else "M/B"

    rprint("\n" + "=" * 70)
    rprint(f"  INDUSTRY × COUNTRY INTERACTION ({label})")
    rprint("=" * 70)

    all_cols = [dep_var] + controls + ["fic", "ggroup", "fyear"]
    reg_df = df[all_cols].dropna().copy()
    reg_df["fic"] = reg_df["fic"].astype(str)
    reg_df["ggroup"] = reg_df["ggroup"].astype(str)
    reg_df["fyear_cat"] = reg_df["fyear"].astype(str)

    # Map ggroup to sector (first 2 digits)
    reg_df["gsector"] = reg_df["ggroup"].str[:2].astype(int)

    # --- R² comparison: additive vs interaction (via Frisch-Waugh) ---
    rprint("\n  R² comparison (additive vs country×sector interaction):")
    try:
        cs_group = reg_df["fic"] + "_" + reg_df["gsector"].astype(str)
        dm_vars = [dep_var] + controls
        reg_dm = reg_df.copy()
        for v in dm_vars:
            reg_dm[v] = reg_dm[v] - reg_dm.groupby(cs_group)[v].transform("mean")
        ctrl_str = " + ".join(controls)
        m_dm = smf.ols(f"{dep_var} ~ {ctrl_str} + C(ggroup) + C(fyear_cat) - 1",
                       data=reg_dm).fit()
        y_orig = reg_df[dep_var].values
        y_cs_mean = reg_df.groupby(cs_group)[dep_var].transform("mean").values
        y_hat = y_cs_mean + m_dm.fittedvalues.values
        ss_res = np.sum((y_orig - y_hat) ** 2)
        ss_tot = np.sum((y_orig - y_orig.mean()) ** 2)
        r2_int = 1.0 - ss_res / ss_tot
        rprint(f"    Country×Sector + Ind + Year R²:     {r2_int:.4f}")
    except (np.linalg.LinAlgError, ValueError) as e:
        rprint(f"    WARNING: R² comparison failed: {e}")

    # --- Per-sector regressions for key countries ---
    rprint(f"\n  Country effects on {label} by GICS sector (10 key countries):")
    key_countries = ["JPN", "CHN", "GBR", "DEU", "FRA", "IND", "KOR", "AUS", "BRA", "CAN"]
    sectors = sorted(reg_df["gsector"].unique())
    records = []

    ctrl_str = " + ".join(controls)
    for sec in sectors:
        sec_df = reg_df[reg_df["gsector"] == sec]
        if len(sec_df) < 500 or "USA" not in sec_df["fic"].values:
            continue
        formula = (f"{dep_var} ~ {ctrl_str} "
                   "+ C(fic, Treatment(reference='USA')) + C(fyear_cat)")
        try:
            m = smf.ols(formula, data=sec_df).fit(cov_type="HC1")
        except (np.linalg.LinAlgError, ValueError):
            continue
        for param_name, coef in m.params.items():
            if param_name.startswith("C(fic"):
                country = param_name.split("[T.")[-1].rstrip("]")
                if country in key_countries:
                    records.append({
                        "country": country,
                        "gsector": sec,
                        "sector_name": GICS_SECTOR_MAP.get(sec, str(sec)),
                        "effect": coef,
                        "n_obs": int(m.nobs),
                    })

    if not records:
        rprint("    No valid sector-level results.")
        return pd.DataFrame()

    sector_effects = pd.DataFrame(records)
    pivot = sector_effects.pivot_table(
        index="country", columns="sector_name", values="effect"
    )
    rprint(f"\n{pivot.round(3).to_string()}")

    # Cross-sector dispersion for each country
    rprint("\n  Cross-sector dispersion of country effects (std dev):")
    for country in key_countries:
        cdata = sector_effects[sector_effects["country"] == country]["effect"]
        if len(cdata) > 2:
            rprint(f"    {country:<6s} std={cdata.std():.3f}  "
                   f"range=[{cdata.min():.3f}, {cdata.max():.3f}]")

    return sector_effects


def plot_industry_country_heatmap(
    sector_effects: pd.DataFrame,
    dep_label: str = "pe",
) -> None:
    """Plot heatmap of country effects by GICS sector."""
    if sector_effects.empty:
        return

    pivot = sector_effects.pivot_table(
        index="country", columns="sector_name", values="effect"
    )
    if pivot.empty:
        return

    title_map = {"pe": "log(P/E)", "mb": "log(M/B)"}
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title(f"Country Effects on {title_map.get(dep_label, dep_label)} "
                 f"by GICS Sector (vs USA)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Effect")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.4 else "black")

    suffix = f"_{dep_label}" if dep_label != "pe" else ""
    save_figure(fig, f"industry_country_heatmap{suffix}")


# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------


def plot_country_effects(effects_df: pd.DataFrame, dep_label: str) -> None:
    """Plot country FE evolution over time for focus countries.

    Parameters
    ----------
    effects_df : pd.DataFrame
        Output of run_yearly_regressions.
    dep_label : str
        'mb' or 'roa'.
    """
    title_map = {"pe": "log(P/E)", "mb": "log(M/B)", "roa": "ROA"}
    fig, ax = plt.subplots(figsize=(10, 6))

    # Project-consistent muted palette for 10 countries
    colors = [
        PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"],
        PALETTE["highlight"], PALETTE["alert"],
        "#1abc9c", "#f39c12", "#3498db", "#e74c3c", "#9b59b6",
    ][:len(FOCUS_COUNTRIES)]

    for i, country in enumerate(FOCUS_COUNTRIES):
        cdata = effects_df[effects_df["fic"] == country].sort_values("fyear")
        if len(cdata) < 3:
            continue
        ax.plot(cdata["fyear"], cdata["country_effect"],
                color=colors[i], label=country, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Country Effect (vs USA)")
    ax.set_title(f"Country Effects on {title_map[dep_label]} Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    save_figure(fig, f"country_effects_{dep_label}")


def plot_dispersion(
    effects_pe: pd.DataFrame,
    effects_mb: pd.DataFrame,
    effects_roa: pd.DataFrame,
) -> None:
    """Plot cross-country dispersion (std dev of country effects) over time."""
    fig, ax = plt.subplots()

    for effects_df, label, color in [
        (effects_pe, "log(P/E)", PALETTE["accent"]),
        (effects_mb, "log(M/B)", PALETTE["highlight"]),
        (effects_roa, "ROA", PALETTE["primary"]),
    ]:
        disp = (
            effects_df[effects_df["fic"] != "USA"]
            .groupby("fyear")["country_effect"]
            .std()
        )
        ax.plot(disp.index, disp.values, color=color, label=label, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Std Dev of Country Effects")
    ax.set_title("Cross-Country Dispersion Over Time")
    ax.legend()

    save_figure(fig, "dispersion_over_time")


def plot_variance_decomposition(
    decomp_pe: dict, decomp_mb: dict, decomp_roa: dict
) -> None:
    """Stacked bar chart of R² decomposition."""
    fig, ax = plt.subplots()

    factors = ["Country", "Industry", "Year"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    x = [0, 1, 2]
    labels = ["log(P/E)", "log(M/B)", "ROA"]
    width = 0.5

    for decomp, xi in [(decomp_pe, 0), (decomp_mb, 1), (decomp_roa, 2)]:
        bottom = 0
        for factor, color in zip(factors, colors):
            val = decomp[factor]
            ax.bar(xi, val, width, bottom=bottom, color=color,
                   label=factor if xi == 0 else None)
            bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R-squared")
    ax.set_title("Variance Decomposition: Country vs Industry vs Year")
    ax.legend()

    save_figure(fig, "variance_decomposition")


# ---------------------------------------------------------------------------
# 10. Main Pipeline
# ---------------------------------------------------------------------------


def print_stylized_facts(
    effects_pe: pd.DataFrame,
    effects_mb: pd.DataFrame,
    effects_roa: pd.DataFrame,
    decomp_pe: dict,
    decomp_mb: dict,
    decomp_roa: dict,
) -> None:
    """Identify and print key stylized facts from the analysis."""
    rprint("\n" + "=" * 70)
    rprint("  STYLIZED FACTS")
    rprint("=" * 70)

    # 1. Country premium persistence (both P/E and M/B)
    for idx, (label, effects) in enumerate([("P/E", effects_pe), ("M/B", effects_mb)]):
        sub = "a" if idx == 0 else "b"
        rprint(f"\n  1{sub}. Country Premium Persistence ({label})")
        avg_effects = (
            effects[effects["fic"] != "USA"]
            .groupby("fic")["country_effect"]
            .mean()
            .sort_values()
        )
        rprint(f"     Lowest 5 (valuation discount vs USA):")
        for country, val in avg_effects.head(5).items():
            rprint(f"       {country:<6s} {val:+.3f}")
        rprint(f"     Highest 5 (valuation premium vs USA):")
        for country, val in avg_effects.tail(5).items():
            rprint(f"       {country:<6s} {val:+.3f}")

    # 2. Convergence/divergence
    rprint("\n  2. Cross-Country Dispersion Trend")
    for label, effects in [("P/E", effects_pe), ("M/B", effects_mb)]:
        disp = (
            effects[effects["fic"] != "USA"]
            .groupby("fyear")["country_effect"]
            .std()
        )
        if len(disp) >= 5:
            early = disp.head(3).mean()
            late = disp.tail(3).mean()
            trend = "CONVERGING" if late < early else "DIVERGING"
            rprint(f"     {label}: Early={early:.3f}, Late={late:.3f} → {trend}")

    # 3. Correlations between P/E, M/B, and ROA country effects
    rprint("\n  3. Country Effect Correlations")
    avg_pe = effects_pe.groupby("fic")["country_effect"].mean()
    avg_mb = effects_mb.groupby("fic")["country_effect"].mean()
    avg_roa = effects_roa.groupby("fic")["country_effect"].mean()
    common = avg_pe.index.intersection(avg_mb.index).intersection(avg_roa.index)
    if len(common) > 5:
        corr_pe_mb = avg_pe[common].corr(avg_mb[common])
        corr_pe_roa = avg_pe[common].corr(avg_roa[common])
        corr_mb_roa = avg_mb[common].corr(avg_roa[common])
        rprint(f"     P/E vs M/B:  {corr_pe_mb:+.3f}")
        rprint(f"     P/E vs ROA:  {corr_pe_roa:+.3f}")
        rprint(f"     M/B vs ROA:  {corr_mb_roa:+.3f}")

    # 4. Variance decomposition
    rprint("\n  4. Variance Decomposition")
    for label, decomp in [("P/E", decomp_pe), ("M/B", decomp_mb), ("ROA", decomp_roa)]:
        total = sum(decomp.values())
        dominant = max(decomp, key=decomp.get)
        rprint(f"     {label}: {dominant} explains the most "
               f"({decomp[dominant]/total*100:.1f}% of R²)")

    # 5. Japan discount (both metrics)
    rprint("\n  5. Japan Discount")
    for label, effects in [("P/E", effects_pe), ("M/B", effects_mb)]:
        jpn = effects[effects["fic"] == "JPN"].sort_values("fyear")
        if len(jpn) > 0:
            rprint(f"     Japan avg {label} effect: {jpn['country_effect'].mean():+.3f}")
            rprint(f"     Japan latest year ({label}): {jpn['country_effect'].iloc[-1]:+.3f}")
            persistent = (jpn["country_effect"] < 0).mean()
            rprint(f"     Negative in {persistent*100:.0f}% of years")


def main() -> None:
    """Run the full panel regression pipeline."""
    rprint("=" * 70)
    rprint("  GLOBAL VALUATION — Panel Regressions")
    rprint(f"  Date: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    rprint("=" * 70)

    # --- Step 0: Check parquet cache ---
    if PARQUET_PATH.exists():
        rprint("  Loading panel from parquet cache...")
        df = pd.read_parquet(PARQUET_PATH)
        df["fic"] = df["fic"].astype("category")
        df["ggroup"] = df["ggroup"].astype("category")
    else:
        df_us = load_us_panel()
        df_global = load_global_panel()
        df = build_panel(df_us, df_global)
        del df_us, df_global
        gc.collect()

    # --- Summary Statistics ---
    print_summary_statistics(df)

    # --- P/E Regressions ---
    pe_results = run_pe_regressions(df)

    # --- M/B Regressions ---
    mb_results = run_mb_regressions(df)

    # --- ROA Regressions ---
    roa_results = run_roa_regressions(df)

    # --- Leverage Cross-Country Analysis ---
    rprint("\n" + "=" * 70)
    rprint("  LEVERAGE CROSS-COUNTRY ANALYSIS")
    rprint("=" * 70)
    lev_by_country = (
        df.groupby("fic", observed=True)["leverage"]
        .agg(["median", "mean", "std", "count"])
        .sort_values("median", ascending=False)
    )
    lev_by_country = lev_by_country[lev_by_country["count"] >= 100]
    rprint(f"\n  Leverage by country (top 15 by median, ≥100 obs):")
    rprint(f"  {'Country':<8s} {'Median':>8s} {'Mean':>8s} {'Std':>8s} {'N':>8s}")
    rprint("  " + "-" * 40)
    for fic, row in lev_by_country.head(15).iterrows():
        rprint(f"  {fic:<8s} {row['median']:>8.3f} {row['mean']:>8.3f} "
               f"{row['std']:>8.3f} {int(row['count']):>8,}")
    rprint(f"\n  Bottom 5 by median leverage:")
    for fic, row in lev_by_country.tail(5).iterrows():
        rprint(f"  {fic:<8s} {row['median']:>8.3f} {row['mean']:>8.3f} "
               f"{row['std']:>8.3f} {int(row['count']):>8,}")
    overall_std = lev_by_country["median"].std()
    overall_range = lev_by_country["median"].max() - lev_by_country["median"].min()
    rprint(f"\n  Cross-country dispersion: std={overall_std:.3f}, "
           f"range={overall_range:.3f}")

    # --- Year-by-Year Regressions ---
    effects_pe = run_yearly_regressions(
        df, dep_var="log_pe",
        controls=["leverage", "log_at", "lag_earn_growth"],
        label="pe",
    )
    effects_mb = run_yearly_regressions(
        df, dep_var="log_mb",
        controls=["leverage", "log_at", "lag_earn_growth"],
        label="mb",
    )
    effects_roa = run_yearly_regressions(
        df, dep_var="roa",
        controls=["leverage", "log_at"],
        label="roa",
    )

    # --- Variance Decomposition ---
    decomp_pe = variance_decomposition(df, "log_pe")
    decomp_mb = variance_decomposition(df, "log_mb")
    decomp_roa = variance_decomposition(df, "roa")

    # --- Industry × Country Interaction ---
    sector_effects_pe = explore_industry_country_interaction(
        df, dep_var="log_pe", controls=["leverage", "log_at"],
    )
    sector_effects_mb = explore_industry_country_interaction(
        df, dep_var="log_mb", controls=["leverage", "log_at"],
    )

    # --- Figures ---
    rprint("\n" + "=" * 70)
    rprint("  FIGURES")
    rprint("=" * 70)
    plot_country_effects(effects_pe, "pe")
    plot_country_effects(effects_mb, "mb")
    plot_country_effects(effects_roa, "roa")
    plot_dispersion(effects_pe, effects_mb, effects_roa)
    plot_variance_decomposition(decomp_pe, decomp_mb, decomp_roa)
    plot_industry_country_heatmap(sector_effects_pe, "pe")
    plot_industry_country_heatmap(sector_effects_mb, "mb")

    # --- Stylized Facts ---
    print_stylized_facts(
        effects_pe, effects_mb, effects_roa,
        decomp_pe, decomp_mb, decomp_roa,
    )

    # --- Save report ---
    report_path = OUT_DIR / "regression_results.txt"
    report_path.write_text(_report_buf.getvalue(), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")

    # Save country effects for later use
    effects_pe.to_parquet(OUT_DIR / "country_effects_pe.parquet", index=False)
    effects_mb.to_parquet(OUT_DIR / "country_effects_mb.parquet", index=False)
    effects_roa.to_parquet(OUT_DIR / "country_effects_roa.parquet", index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
