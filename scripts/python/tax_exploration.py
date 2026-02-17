"""
tax_exploration.py -- Do taxes and interest rates explain country P/E differences?

Hypothesis: Higher combined dividend/capital gains taxes -> higher required
pre-tax returns -> lower P/E. Additionally, cross-country variation in real
interest rates may explain P/E differences via the discount-rate channel
(Gormsen and Lazarus, NBER w34814, 2026).

Tax data sources (3-tier):
  1. OECD SDMX API: CIT, combined dividend tax, dividend WHT (38-144 countries)
  2. Tax Foundation GitHub: statutory CIT rates (225 jurisdictions)
  3. Manual dataset: ~25 non-OECD countries (EY/PwC/KPMG 2023 guides)

Interest rate data:
  - World Bank: real interest rate (FR.INR.RINR), lending rate (FR.INR.LEND)
  - World Bank: long-term govt bond yield (proxy via FR.INR.RINR)

Inputs:
  - output/regressions/country_effects_trailing_full_pe.parquet
  - output/regressions/country_effects_trailing_pe.parquet
  - output/regressions/country_effects_xrd_at.parquet
  - OECD SDMX API (CIT, dividend income, WHT datasets)
  - World Bank API (savings/GNI, real interest rate)

Outputs:
  - output/regressions/tax_country_data.parquet
  - output/regressions/scatter_cit_vs_pe.{pdf,png}
  - output/regressions/scatter_divtax_vs_pe.{pdf,png}
  - output/regressions/scatter_realrate_vs_pe.{pdf,png}
  - output/regressions/scatter_tax_variables_vs_pe.{pdf,png}
  - output/regressions/tax_univariate_regressions.csv

Author: Augustin Landier, HEC Paris
"""

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import wbgapi as wb

np.random.seed(20260216)

OUT_DIR = Path("output/regressions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#2c3e50",
    "secondary": "#7f8c8d",
    "accent": "#2980b9",
    "highlight": "#8e44ad",
    "alert": "#c0392b",
}

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

FOCUS = ["USA", "JPN", "CHN", "GBR", "IND", "DEU", "FRA", "KOR", "BRA", "AUS"]

# Offshore jurisdictions to exclude
OFFSHORE = {"BMU", "CYM", "VGB", "GGY", "JEY", "IMN", "LIE", "MCO", "AND",
            "BHS", "BRB", "BLZ", "VUT", "MHL", "PLW", "TCA", "AIA", "MSR"}


def _sig_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# 1. Tax data: OECD SDMX API
# ---------------------------------------------------------------------------
def fetch_oecd_csv(dataset_id: str, start: int = 2023, end: int = 2023,
                   label: str = "") -> pd.DataFrame:
    """Fetch CSV from OECD SDMX REST API.

    Parameters
    ----------
    dataset_id : str
        Full OECD dataset identifier (e.g. 'OECD.CTP.TPS,DSD_TAX_CIT@DF_CIT').
    start, end : int
        Year range.
    label : str
        Human-readable label for logging.

    Returns
    -------
    pd.DataFrame or empty DataFrame on failure.
    """
    url = (f"https://sdmx.oecd.org/public/rest/data/{dataset_id}/all"
           f"?format=csvfilewithlabels&startPeriod={start}&endPeriod={end}")
    print(f"  Fetching {label or dataset_id}...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        print(f"    -> {len(df):,} rows, {df.shape[1]} cols")
        return df
    except (requests.RequestException, pd.errors.ParserError) as e:
        print(f"    -> FAILED: {e}")
        return pd.DataFrame()


def parse_oecd_cit(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract combined CIT rate (central + sub-national) from OECD CIT data.

    OECD DF_CIT columns: REF_AREA, MEASURE, SECTOR, OBS_VALUE.
    MEASURE=CIT_C is the combined (central + sub-national) CIT rate.
    """
    if raw.empty:
        return pd.DataFrame(columns=["iso3", "cit_rate"])

    # Filter to combined CIT rate (MEASURE=CIT_C)
    combined = raw[raw["MEASURE"] == "CIT_C"].copy()
    if len(combined) == 0:
        # Fallback: central government CIT
        combined = raw[(raw["MEASURE"] == "CIT") & (raw["SECTOR"] == "S1311")].copy()

    out = (combined[["REF_AREA", "OBS_VALUE"]]
           .drop_duplicates("REF_AREA")
           .rename(columns={"REF_AREA": "iso3", "OBS_VALUE": "cit_rate"}))
    out["cit_rate"] = pd.to_numeric(out["cit_rate"], errors="coerce")
    return out.dropna(subset=["cit_rate"])


def parse_oecd_dividend(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract combined personal+corporate income tax on dividends from OECD data.

    OECD DF_CIT_DIVD_INCOME columns: REF_AREA, MEASURE, OBS_VALUE.
    MEASURE=CPITCIT is the combined personal and corporate income tax rate.
    """
    if raw.empty:
        return pd.DataFrame(columns=["iso3", "div_tax_combined"])

    # CPITCIT = combined personal + corporate tax on dividends
    cpitcit = raw[raw["MEASURE"] == "CPITCIT"].copy()
    if len(cpitcit) == 0:
        # Fallback: FWHT (final withholding tax)
        cpitcit = raw[raw["MEASURE"] == "FWHT"].copy()

    out = (cpitcit[["REF_AREA", "OBS_VALUE"]]
           .drop_duplicates("REF_AREA")
           .rename(columns={"REF_AREA": "iso3", "OBS_VALUE": "div_tax_combined"}))
    out["div_tax_combined"] = pd.to_numeric(out["div_tax_combined"], errors="coerce")
    return out.dropna(subset=["div_tax_combined"])


def parse_oecd_wht(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract dividend withholding tax from OECD WHT data.

    OECD DF_WHT_STANDARD columns: REF_AREA, MEASURE, OBS_VALUE.
    MEASURE=WHT_DIV is dividend withholding tax.
    """
    if raw.empty:
        return pd.DataFrame(columns=["iso3", "div_wht"])

    # Filter to dividend WHT (MEASURE=WHT_DIV)
    div_wht = raw[raw["MEASURE"] == "WHT_DIV"].copy()

    out = (div_wht[["REF_AREA", "OBS_VALUE"]]
           .drop_duplicates("REF_AREA")
           .rename(columns={"REF_AREA": "iso3", "OBS_VALUE": "div_wht"}))
    out["div_wht"] = pd.to_numeric(out["div_wht"], errors="coerce")
    return out.dropna(subset=["div_wht"])


# ---------------------------------------------------------------------------
# 2. Tax Foundation (GitHub CSV) -- corporate tax rates
# ---------------------------------------------------------------------------
def fetch_tax_foundation() -> pd.DataFrame:
    """Fetch corporate tax rates from Tax Foundation GitHub repo."""
    url = ("https://raw.githubusercontent.com/TaxFoundation/"
           "international-tax-competitiveness-index/master/source-data/"
           "corporate_rate.csv")
    print("  Fetching Tax Foundation CIT data...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        print(f"    -> {len(df):,} rows")
        # Typically has columns like iso_3, year, rate
        year_cols = [c for c in df.columns if c.isdigit()]
        iso_candidates = [c for c in df.columns if "iso" in c.lower()]
        if not iso_candidates:
            print("    -> FAILED: no ISO column found")
            return pd.DataFrame(columns=["iso3", "cit_rate_tf"])
        iso_col = iso_candidates[0]
        if year_cols:
            latest = max(year_cols)
            out = df[[iso_col, latest]].rename(
                columns={iso_col: "iso3", latest: "cit_rate_tf"})
            out["cit_rate_tf"] = pd.to_numeric(out["cit_rate_tf"], errors="coerce") * 100
        else:
            rate_candidates = [c for c in df.columns if "rate" in c.lower()]
            year_candidates = [c for c in df.columns if "year" in c.lower()]
            if not rate_candidates or not year_candidates:
                print("    -> FAILED: could not identify rate/year columns")
                return pd.DataFrame(columns=["iso3", "cit_rate_tf"])
            rate_col, year_col = rate_candidates[0], year_candidates[0]
            recent = df[df[year_col] == df[year_col].max()]
            out = recent[[iso_col, rate_col]].rename(
                columns={iso_col: "iso3", rate_col: "cit_rate_tf"})
            out["cit_rate_tf"] = pd.to_numeric(out["cit_rate_tf"], errors="coerce")
            if out["cit_rate_tf"].median() < 1:
                out["cit_rate_tf"] *= 100
        return out.dropna(subset=["cit_rate_tf"])
    except (requests.RequestException, pd.errors.ParserError) as e:
        print(f"    -> FAILED: {e}")
        return pd.DataFrame(columns=["iso3", "cit_rate_tf"])


# ---------------------------------------------------------------------------
# 3. Manual tax data (non-OECD countries, EY/PwC/KPMG 2023)
# ---------------------------------------------------------------------------
def get_manual_tax_data() -> pd.DataFrame:
    """Tax rates for non-OECD countries not covered by OECD API.

    Sources: EY Worldwide Corporate Tax Guide 2023, PwC Worldwide Tax
    Summaries 2023, KPMG Corporate Tax Rates Table 2023.

    Returns DataFrame with iso3, cit_rate, personal_div_rate, cgt_rate.
    Combined dividend tax computed as: CIT + (1 - CIT/100) * personal_div_rate.
    """
    manual = {
        #           CIT%  PIT-div%  CGT%   WHT-div%
        "CHN":     (25.0,  20.0,    20.0,  10.0),
        "IND":     (25.2,  20.0,    12.5,  20.0),  # DDT abolished 2020; top marginal
        "HKG":     (16.5,   0.0,     0.0,   0.0),
        "SGP":     (17.0,   0.0,     0.0,   0.0),
        "MYS":     (24.0,   0.0,     0.0,   0.0),  # Single-tier: no div tax
        "THA":     (20.0,  10.0,     0.0,  10.0),
        "IDN":     (22.0,  10.0,     0.0,  10.0),
        "PHL":     (25.0,  10.0,    15.0,  25.0),
        "PAK":     (29.0,  15.0,    15.0,  15.0),
        "TWN":     (20.0,  28.0,    20.0,  21.0),
        "BRA":     (34.0,   0.0,    15.0,   0.0),  # Dividends tax-exempt (pre-reform)
        "ARG":     (35.0,   7.0,    15.0,   7.0),
        "PER":     (29.5,   5.0,     0.0,   5.0),
        "ARE":     ( 9.0,   0.0,     0.0,   0.0),
        "SAU":     (20.0,   0.0,    20.0,   5.0),  # CIT for foreign; zakat for domestic
        "QAT":     (10.0,   0.0,     0.0,   0.0),
        "EGY":     (22.5,  10.0,    10.0,  10.0),
        "ZAF":     (27.0,  20.0,    18.0,  20.0),
        "RUS":     (20.0,  13.0,    13.0,  15.0),
        "ROU":     (16.0,   8.0,    10.0,   8.0),
        "CYP":     (12.5,   0.0,     0.0,   0.0),  # Dividend income exempt
        "BGD":     (27.5,  20.0,    15.0,  20.0),
        "VNM":     (20.0,   5.0,    20.0,   0.0),
        "NGA":     (30.0,  10.0,    10.0,  10.0),
        "KEN":     (30.0,   5.0,     5.0,  15.0),
    }
    rows = []
    for iso3, (cit, pit_div, cgt, wht) in manual.items():
        combined = cit + (1 - cit / 100) * pit_div
        rows.append({
            "iso3": iso3,
            "cit_rate_manual": cit,
            "personal_div_rate": pit_div,
            "cgt_rate": cgt,
            "div_wht_manual": wht,
            "div_tax_combined_manual": combined,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Assemble unified tax dataset
# ---------------------------------------------------------------------------
def assemble_tax_data() -> pd.DataFrame:
    """Fetch from all sources and merge into a single country-level dataset.

    Priority: OECD > Tax Foundation > Manual for CIT.
    OECD for combined dividend tax; manual for non-OECD countries.
    """
    print("\n" + "=" * 70)
    print("FETCHING TAX DATA")
    print("=" * 70)

    # --- OECD CIT ---
    oecd_cit_raw = fetch_oecd_csv(
        "OECD.CTP.TPS,DSD_TAX_CIT@DF_CIT", 2023, 2023, "OECD CIT rates")
    oecd_cit = parse_oecd_cit(oecd_cit_raw)
    print(f"  OECD CIT: {len(oecd_cit)} countries")

    # --- OECD dividend income tax ---
    oecd_div_raw = fetch_oecd_csv(
        "OECD.CTP.TPS,DSD_TAX_CIT@DF_CIT_DIVD_INCOME", 2023, 2023,
        "OECD dividend income tax")
    oecd_div = parse_oecd_dividend(oecd_div_raw)
    print(f"  OECD dividend tax: {len(oecd_div)} countries")

    # --- OECD WHT ---
    oecd_wht_raw = fetch_oecd_csv(
        "OECD.CTP.TPS,DSD_WHT@DF_WHT_STANDARD", 2023, 2023,
        "OECD withholding tax")
    oecd_wht = parse_oecd_wht(oecd_wht_raw)
    print(f"  OECD WHT: {len(oecd_wht)} countries")

    # --- Tax Foundation ---
    tf = fetch_tax_foundation()
    print(f"  Tax Foundation: {len(tf)} countries")

    # --- Manual ---
    manual = get_manual_tax_data()
    print(f"  Manual: {len(manual)} countries")

    # --- Merge ---
    # Start with OECD CIT
    tax = oecd_cit[["iso3", "cit_rate"]].copy()

    # Add Tax Foundation as fallback for CIT
    tax = tax.merge(tf[["iso3", "cit_rate_tf"]], on="iso3", how="outer")
    tax["cit_rate"] = tax["cit_rate"].fillna(tax["cit_rate_tf"])
    tax = tax.drop(columns=["cit_rate_tf"])

    # Add manual CIT for remaining gaps
    tax = tax.merge(manual[["iso3", "cit_rate_manual"]], on="iso3", how="outer")
    tax["cit_rate"] = tax["cit_rate"].fillna(tax["cit_rate_manual"])
    tax = tax.drop(columns=["cit_rate_manual"])

    # OECD combined dividend tax
    tax = tax.merge(oecd_div[["iso3", "div_tax_combined"]], on="iso3", how="outer")

    # Manual combined dividend tax for non-OECD
    tax = tax.merge(manual[["iso3", "div_tax_combined_manual"]],
                    on="iso3", how="outer")
    tax["div_tax_combined"] = tax["div_tax_combined"].fillna(
        tax["div_tax_combined_manual"])
    tax = tax.drop(columns=["div_tax_combined_manual"])

    # Capital gains tax (manual only for now)
    tax = tax.merge(manual[["iso3", "cgt_rate"]], on="iso3", how="outer")

    # WHT: OECD preferred, manual fallback
    tax = tax.merge(oecd_wht[["iso3", "div_wht"]], on="iso3", how="outer")
    tax = tax.merge(manual[["iso3", "div_wht_manual"]], on="iso3", how="outer")
    tax["div_wht"] = tax["div_wht"].fillna(tax["div_wht_manual"])
    tax = tax.drop(columns=["div_wht_manual"])

    # Deduplicate and clean
    n_before = len(tax)
    tax = tax.drop_duplicates("iso3")
    if len(tax) < n_before:
        print(f"  WARNING: {n_before - len(tax)} duplicate iso3 removed")
    tax = tax[~tax["iso3"].isin(OFFSHORE)]

    print(f"\n  Assembled tax dataset: {len(tax)} jurisdictions")
    print(f"    CIT available:        {tax['cit_rate'].notna().sum()}")
    print(f"    Combined div tax:     {tax['div_tax_combined'].notna().sum()}")
    print(f"    CGT available:        {tax['cgt_rate'].notna().sum()}")
    print(f"    WHT available:        {tax['div_wht'].notna().sum()}")

    return tax


# ---------------------------------------------------------------------------
# 5. Load country P/E effects and macro variables
# ---------------------------------------------------------------------------
def load_country_effects() -> dict[str, pd.Series]:
    """Load average country effects from parquet files."""
    effects = {}
    for name in ["trailing_full", "trailing"]:
        path = OUT_DIR / f"country_effects_{name}_pe.parquet"
        if path.exists():
            ce = pd.read_parquet(path)
            avg = ce.groupby("fic")["country_effect"].mean()
            effects[name] = avg
            print(f"  {name}: {len(avg)} countries, σ = {avg.std():.3f}")
        else:
            print(f"  WARNING: {path} not found")
    return effects


def load_country_controls() -> pd.DataFrame:
    """Load country-level R&D, savings, and real interest rate as controls."""
    controls = pd.DataFrame()

    # R&D country effects
    xrd_path = OUT_DIR / "country_effects_xrd_at.parquet"
    if xrd_path.exists():
        xrd_ce = pd.read_parquet(xrd_path)
        avg_xrd = xrd_ce.groupby("fic")["country_effect"].mean().rename("xrd_effect")
        controls = pd.DataFrame(avg_xrd)

    # Savings/GNI from World Bank
    print("  Loading savings/GNI from World Bank...")
    try:
        sav = wb.data.DataFrame("NY.GNS.ICTR.ZS", time=range(2000, 2024),
                                labels=False)
        sav = sav.T.mean().rename("savings_gni")
        if controls.empty:
            controls = pd.DataFrame(sav)
        else:
            controls = controls.join(sav, how="outer")
    except Exception as e:
        print(f"    World Bank savings API failed: {e}")

    # Real interest rate from World Bank (FR.INR.RINR)
    # Motivated by Gormsen & Lazarus (2026): cross-country interest rate
    # variation is a key driver of equity valuations
    print("  Loading real interest rate from World Bank...")
    try:
        rir = wb.data.DataFrame("FR.INR.RINR", time=range(2000, 2024),
                                labels=False)
        rir = rir.T.mean().rename("real_interest_rate")
        controls = controls.join(rir, how="outer")
        n_rir = controls["real_interest_rate"].notna().sum()
        print(f"    Real interest rate: {n_rir} countries")
    except Exception as e:
        print(f"    World Bank interest rate API failed: {e}")

    # Lending interest rate (FR.INR.LEND) -- broader coverage
    print("  Loading lending rate from World Bank...")
    try:
        lend = wb.data.DataFrame("FR.INR.LEND", time=range(2000, 2024),
                                 labels=False)
        lend = lend.T.mean().rename("lending_rate")
        controls = controls.join(lend, how="outer")
        n_lend = controls["lending_rate"].notna().sum()
        print(f"    Lending rate: {n_lend} countries")
    except Exception as e:
        print(f"    World Bank lending rate API failed: {e}")

    return controls


# ---------------------------------------------------------------------------
# 6. Analysis
# ---------------------------------------------------------------------------
def run_univariate(country: pd.DataFrame, pe_col: str, tax_vars: list[str],
                   label: str = "") -> pd.DataFrame:
    """Run univariate OLS of P/E effect on each tax variable.

    Parameters
    ----------
    country : pd.DataFrame
        Country-level data with pe_col and tax variables.
    pe_col : str
        Name of the P/E effect column (log scale).
    tax_vars : list[str]
        Variable names to regress on.
    label : str
        Label for output header.

    Returns
    -------
    pd.DataFrame
        One row per variable with corr, beta, t_stat, r_squared, p_value.
    """
    print(f"\n  --- Univariate regressions ({label}) ---")
    results = []
    for var in tax_vars:
        sub = country[[pe_col, var]].dropna().astype(float)
        if len(sub) < 10:
            print(f"    {var}: too few obs ({len(sub)})")
            continue
        X = sm.add_constant(sub[var])
        m = sm.OLS(sub[pe_col], X).fit()
        r = sub[pe_col].corr(sub[var])
        t_val = m.tvalues.iloc[1]
        results.append({
            "variable": var,
            "n": len(sub),
            "corr": r,
            "beta": m.params.iloc[1],
            "t_stat": t_val,
            "r_squared": m.rsquared,
            "p_value": m.pvalues.iloc[1],
        })
        sig = _sig_stars(m.pvalues.iloc[1])
        print(f"    {var:>20s}: r={r:+.3f}, β={m.params.iloc[1]:+.5f}, "
              f"t={t_val:+.2f}{sig}, R²={m.rsquared:.3f}, n={len(sub)}")
    return pd.DataFrame(results)


def run_horse_race(country: pd.DataFrame, pe_col: str) -> pd.DataFrame:
    """Run multivariate regressions: tax + interest rates + R&D + savings.

    Parameters
    ----------
    country : pd.DataFrame
        Country-level data.
    pe_col : str
        Name of the P/E effect column (log scale).

    Returns
    -------
    pd.DataFrame
        One row per specification with R², coefficients, t-stats.
    """
    print(f"\n  --- Horse race regressions ---")

    specs = [
        ("CIT only", ["cit_rate"]),
        ("Combined div tax only", ["div_tax_combined"]),
        ("Real interest rate only", ["real_interest_rate"]),
        ("CIT + div tax", ["cit_rate", "div_tax_combined"]),
        ("CIT + R&D", ["cit_rate", "xrd_effect"]),
        ("Div tax + R&D", ["div_tax_combined", "xrd_effect"]),
        ("Real rate + R&D", ["real_interest_rate", "xrd_effect"]),
        ("Div tax + Real rate",
         ["div_tax_combined", "real_interest_rate"]),
        ("Div tax + Real rate + R&D",
         ["div_tax_combined", "real_interest_rate", "xrd_effect"]),
        ("Div tax + R&D + Savings",
         ["div_tax_combined", "xrd_effect", "savings_gni"]),
        ("Div tax + Real rate + R&D + Savings",
         ["div_tax_combined", "real_interest_rate", "xrd_effect",
          "savings_gni"]),
        ("Kitchen sink",
         ["cit_rate", "div_tax_combined", "real_interest_rate",
          "xrd_effect", "savings_gni"]),
    ]

    all_results = []
    for label, rhs in specs:
        cols = [pe_col] + rhs
        sub = country[cols].dropna().astype(float)
        if len(sub) < len(rhs) + 5:
            print(f"\n    {label}: too few obs ({len(sub)})")
            continue
        X = sm.add_constant(sub[rhs])
        m = sm.OLS(sub[pe_col], X).fit()
        row = {"specification": label, "n": len(sub),
               "r_squared": m.rsquared, "adj_r_squared": m.rsquared_adj}
        print(f"\n    {label} (n={len(sub)}): R²={m.rsquared:.3f}, "
              f"adj-R²={m.rsquared_adj:.3f}")
        for i, var in enumerate(rhs):
            p = m.pvalues.iloc[i + 1]
            sig = _sig_stars(p)
            row[f"beta_{var}"] = m.params.iloc[i + 1]
            row[f"tstat_{var}"] = m.tvalues.iloc[i + 1]
            print(f"      {var:>25s}: {m.params.iloc[i+1]:+.5f} "
                  f"(t={m.tvalues.iloc[i+1]:+.2f}){sig}")
        all_results.append(row)

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def plot_scatter(x: np.ndarray, y: np.ndarray, labels: pd.Index,
                 x_label: str, y_label: str, title: str,
                 filename: str, fit_line: bool = True) -> None:
    """Plot scatter with FOCUS annotations and optional OLS fit line.

    Parameters
    ----------
    x : np.ndarray
        Horizontal axis values.
    y : np.ndarray
        Vertical axis values (e.g., P/E ratio relative to USA).
    labels : pd.Index
        ISO3 country codes for annotation.
    x_label, y_label : str
        Axis labels.
    title : str
        Figure title (should include r and n).
    filename : str
        Output file stem (without extension).
    fit_line : bool
        If True, overlay OLS fit line.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color=PALETTE["accent"], s=40, alpha=0.7,
               edgecolors="white", linewidth=0.5)

    for iso, xi, yi in zip(labels, x, y):
        if iso in FOCUS:
            ax.annotate(iso, (xi, yi), fontsize=8, fontweight="bold",
                        ha="left", va="bottom", xytext=(4, 2),
                        textcoords="offset points", color=PALETTE["primary"])

    if fit_line and len(x) >= 3:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3:
            z = np.polyfit(x[mask], y[mask], 1)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
                    linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def make_2x2_panel(country: pd.DataFrame, pe_col: str) -> None:
    """2x2 panel: CIT, combined div tax, real interest rate, WHT vs P/E."""
    tax_info = [
        ("cit_rate", "Statutory CIT Rate (%)", "CIT"),
        ("div_tax_combined", "Combined Dividend Tax (%)", "Div Tax"),
        ("real_interest_rate", "Real Interest Rate (%)", "Real Rate"),
        ("div_wht", "Dividend WHT (%)", "WHT"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (var, xlabel, short) in enumerate(tax_info):
        ax = axes[i]
        sub = country[[pe_col, var]].dropna()
        if len(sub) < 5:
            ax.text(0.5, 0.5, f"Insufficient data\n({len(sub)} obs)",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(short)
            continue

        pe_ratio = np.exp(sub[pe_col])
        ax.scatter(sub[var], pe_ratio, color=PALETTE["accent"], s=30,
                   alpha=0.7, edgecolors="white", linewidth=0.5)

        # Annotate FOCUS
        for iso in sub.index:
            if iso in FOCUS:
                ax.annotate(iso, (sub.loc[iso, var], pe_ratio[iso]),
                            fontsize=7, fontweight="bold", ha="left",
                            va="bottom", xytext=(3, 1),
                            textcoords="offset points",
                            color=PALETTE["primary"])

        # Fit line
        mask = np.isfinite(sub[var].values) & np.isfinite(pe_ratio.values)
        if mask.sum() >= 3:
            z = np.polyfit(sub[var].values[mask], pe_ratio.values[mask], 1)
            xl = np.linspace(sub[var].min(), sub[var].max(), 100)
            ax.plot(xl, np.polyval(z, xl), color=PALETTE["alert"],
                    linewidth=1.5, linestyle="--", alpha=0.8)

        r = pe_ratio.corr(sub[var])  # correlation in levels (matches y-axis)
        n = len(sub)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("P/E relative to USA", fontsize=10)
        ax.set_title(f"{short} vs P/E (r={r:+.2f}, n={n})", fontsize=11)
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("Tax/Rate Variables and Country P/E Effects", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_tax_variables_vs_pe.pdf", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUT_DIR / "scatter_tax_variables_vs_pe.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_tax_variables_vs_pe")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TAX & INTEREST RATE EXPLORATION")
    print("Do taxes and interest rates explain country P/E differences?")
    print("=" * 70)

    # Step 1: Fetch tax data
    tax = assemble_tax_data()

    # Step 2: Load country P/E effects
    print("\nLoading country P/E effects...")
    effects = load_country_effects()

    if "trailing_full" not in effects:
        raise FileNotFoundError(
            "country_effects_trailing_full_pe.parquet not found. "
            "Run ibes_forward_pe.py first.")

    # Step 3: Load controls (R&D, savings, real interest rate)
    print("\nLoading country-level controls...")
    controls = load_country_controls()

    # Step 4: Merge everything
    print("\nMerging tax + P/E + controls...")
    for spec_name, pe_series in effects.items():
        pe_df = pe_series.rename(f"pe_{spec_name}").to_frame()
        pe_df.index.name = "iso3"
        tax = tax.merge(pe_df, on="iso3", how="outer")

    # Merge controls
    controls.index.name = "iso3"
    tax = tax.merge(controls, on="iso3", how="outer")

    # Deduplicate after merges
    n_before = len(tax)
    tax = tax.drop_duplicates("iso3")
    if len(tax) < n_before:
        print(f"  WARNING: {n_before - len(tax)} duplicate rows removed")

    # Set index
    tax = tax.set_index("iso3")

    # P/E ratio (for scatter y-axis)
    for spec in effects:
        col = f"pe_{spec}"
        if col in tax.columns:
            tax[f"pe_ratio_{spec}"] = np.exp(tax[col])

    # Filter to countries with at least some tax/rate data
    rate_tax_cols = ["cit_rate", "div_tax_combined", "cgt_rate", "div_wht",
                     "real_interest_rate"]
    available_cols = [c for c in rate_tax_cols if c in tax.columns]
    has_any_data = tax[available_cols].notna().any(axis=1)
    has_pe = (tax["pe_trailing_full"].notna()
              if "pe_trailing_full" in tax.columns
              else pd.Series(False, index=tax.index))

    country = tax[has_any_data & has_pe].copy()
    print(f"\n  Countries with data + P/E: {len(country)}")

    # Save compiled dataset
    tax.to_parquet(OUT_DIR / "tax_country_data.parquet")
    print(f"  Saved tax_country_data.parquet ({len(tax)} rows)")

    # Step 5: Summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    all_vars = ["cit_rate", "div_tax_combined", "cgt_rate", "div_wht",
                "real_interest_rate", "lending_rate"]
    for var in all_vars:
        if var not in country.columns:
            continue
        valid = country[var].dropna()
        if len(valid) > 0:
            print(f"  {var:>22s}: n={len(valid):>3d}, "
                  f"mean={valid.mean():.1f}, sd={valid.std():.1f}, "
                  f"[{valid.min():.1f}, {valid.max():.1f}]")

    # Focus countries
    print(f"\n  {'Ctry':>5s}  {'CIT':>6s}  {'DivTax':>7s}  {'RealR':>6s}  "
          f"{'WHT':>6s}  {'PE_eff':>7s}  {'PE_ratio':>8s}")
    for fic in FOCUS:
        if fic in country.index:
            r = country.loc[fic]
            cit = f"{r['cit_rate']:.1f}" if pd.notna(r.get("cit_rate")) else "n/a"
            dtx = f"{r['div_tax_combined']:.1f}" if pd.notna(r.get("div_tax_combined")) else "n/a"
            rir = f"{r['real_interest_rate']:.1f}" if pd.notna(r.get("real_interest_rate")) else "n/a"
            wht = f"{r['div_wht']:.1f}" if pd.notna(r.get("div_wht")) else "n/a"
            pe = f"{r['pe_trailing_full']:+.3f}" if pd.notna(r.get("pe_trailing_full")) else "n/a"
            per = f"{r['pe_ratio_trailing_full']:.2f}" if pd.notna(r.get("pe_ratio_trailing_full")) else "n/a"
            print(f"  {fic:>5s}  {cit:>6s}  {dtx:>7s}  {rir:>6s}  "
                  f"{wht:>6s}  {pe:>7s}  {per:>8s}")

    # Step 6: Regressions
    print("\n" + "=" * 70)
    print("REGRESSIONS: TAX & INTEREST RATE VARIABLES vs P/E")
    print("=" * 70)

    # Include interest rate variables in univariate tests
    tax_vars = ["cit_rate", "div_tax_combined", "cgt_rate", "div_wht"]
    rate_vars = ["real_interest_rate", "lending_rate"]
    test_vars = tax_vars + [v for v in rate_vars if v in country.columns]

    # Primary: trailing_full
    uni_results = run_univariate(
        country, "pe_trailing_full", test_vars, "trailing_full P/E")

    # Robustness: trailing (broader coverage)
    if "pe_trailing" in country.columns:
        country_broad = tax[has_any_data & tax["pe_trailing"].notna()].copy()
        if len(country_broad) > len(country) + 5:
            print(f"\n  Broader sample (trailing, no LTG control): "
                  f"{len(country_broad)} countries")
            run_univariate(
                country_broad, "pe_trailing", test_vars,
                "trailing P/E (broad)")

    # Horse race
    horse_results = run_horse_race(country, "pe_trailing_full")

    # Export regression tables
    if len(uni_results) > 0:
        uni_results.to_csv(OUT_DIR / "tax_univariate_regressions.csv",
                           index=False)
        print("\n  Saved tax_univariate_regressions.csv")
    if len(horse_results) > 0:
        horse_results.to_csv(OUT_DIR / "tax_horse_race_regressions.csv",
                             index=False)
        print("  Saved tax_horse_race_regressions.csv")

    # Step 7: Figures
    print("\n" + "=" * 70)
    print("FIGURES")
    print("=" * 70)

    # Fig 1: CIT vs P/E
    sub = country[["cit_rate", "pe_trailing_full"]].dropna()
    if len(sub) >= 10:
        pe_ratio = np.exp(sub["pe_trailing_full"])
        r = pe_ratio.corr(sub["cit_rate"])
        plot_scatter(
            sub["cit_rate"].values, pe_ratio.values, sub.index,
            "Statutory CIT Rate (%)", "P/E relative to USA",
            f"Corporate Tax Rate and P/E (r = {r:+.2f}, n = {len(sub)})",
            "scatter_cit_vs_pe")

    # Fig 2: Combined dividend tax vs P/E
    sub = country[["div_tax_combined", "pe_trailing_full"]].dropna()
    if len(sub) >= 10:
        pe_ratio = np.exp(sub["pe_trailing_full"])
        r = pe_ratio.corr(sub["div_tax_combined"])
        plot_scatter(
            sub["div_tax_combined"].values, pe_ratio.values, sub.index,
            "Combined Dividend Tax Rate (%)", "P/E relative to USA",
            f"Combined Dividend Tax and P/E (r = {r:+.2f}, n = {len(sub)})",
            "scatter_divtax_vs_pe")

    # Fig 3: Real interest rate vs P/E (motivated by Gormsen & Lazarus 2026)
    if "real_interest_rate" in country.columns:
        sub = country[["real_interest_rate", "pe_trailing_full"]].dropna()
        if len(sub) >= 10:
            pe_ratio = np.exp(sub["pe_trailing_full"])
            r = pe_ratio.corr(sub["real_interest_rate"])
            plot_scatter(
                sub["real_interest_rate"].values, pe_ratio.values, sub.index,
                "Real Interest Rate (%)", "P/E relative to USA",
                f"Real Interest Rate and P/E "
                f"(r = {r:+.2f}, n = {len(sub)})",
                "scatter_realrate_vs_pe")

    # Fig 4: 2x2 panel
    make_2x2_panel(country, "pe_trailing_full")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if len(uni_results) > 0:
        best = uni_results.sort_values("r_squared", ascending=False).iloc[0]
        print(f"  Best univariate predictor: {best['variable']} "
              f"(R²={best['r_squared']:.3f}, r={best['corr']:+.3f})")

        print("\n  All univariate R² (sorted):")
        for _, row in uni_results.sort_values("r_squared", ascending=False).iterrows():
            sig = _sig_stars(row["p_value"])
            print(f"    {row['variable']:>22s}: R²={row['r_squared']:.3f}, "
                  f"r={row['corr']:+.3f}{sig}")

    # Coverage check
    n_cit = country["cit_rate"].notna().sum()
    n_div = country["div_tax_combined"].notna().sum()
    n_rir = country["real_interest_rate"].notna().sum() if "real_interest_rate" in country.columns else 0
    print(f"\n  CIT coverage:     {n_cit} countries (target >= 40)")
    print(f"  Div tax coverage: {n_div} countries (target >= 35)")
    print(f"  Real rate coverage: {n_rir} countries")

    print("\nDone.")
