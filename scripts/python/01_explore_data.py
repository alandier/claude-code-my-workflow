"""
01_explore_data.py — Initial exploration of Global Valuation datasets.

Purpose: Schema inspection, coverage diagnostics, missingness analysis,
         and preliminary valuation distribution plots for the four raw
         data files (Compustat Global/America, Returns Global/America).

Inputs:
  - compustat_global.csv (2.3 GB — Global annual fundamentals)
  - compustat_america.csv (0.8 GB — NA fundamentals + CRSP link)
  - returns_global.csv (1.9 GB — Global monthly prices)
  - returns_america.csv (0.9 GB — CRSP US monthly returns)
  All from DATA_DIR (see constants below).

Outputs:
  - output/diagnostics/data_exploration_report.txt (console log)
  - output/diagnostics/coverage_by_country.pdf/png
  - output/diagnostics/coverage_by_year.pdf/png
  - output/diagnostics/valuation_distributions.pdf/png (P/B raw vs winsorized)

Author: Augustin Landier, HEC Paris
"""

# stdlib
import gc
from io import StringIO
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    "~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/"
).expanduser()

OUT_DIR = Path("output/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "compustat_global": DATA_DIR / "compustat_global.csv",
    "compustat_america": DATA_DIR / "compustat_america.csv",
    "returns_global": DATA_DIR / "returns_global.csv",
    "returns_america": DATA_DIR / "returns_america.csv",
}

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

# Capture all printed output for the report file
_report_buf = StringIO()


def rprint(*args, **kwargs):
    """Print to both console and report buffer."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=_report_buf)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def peek_schema(filepath: Path, nrows: int = 5) -> pd.DataFrame:
    """Load a small sample to inspect column names and dtypes.

    Parameters
    ----------
    filepath : Path
        CSV file to peek.
    nrows : int
        Number of rows to read for dtype inference.

    Returns
    -------
    pd.DataFrame
        Small sample of the data.
    """
    return pd.read_csv(filepath, nrows=nrows, low_memory=False)


def count_rows(filepath: Path) -> int:
    """Count data rows in a CSV without loading into memory.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.

    Returns
    -------
    int
        Number of data rows (excluding header).
    """
    with open(filepath, "rb") as f:
        n = sum(1 for _ in f) - 1  # binary mode is faster; subtract header
    return n


def summarize_schema(name: str, filepath: Path) -> dict:
    """Print schema info for a single file.

    Returns
    -------
    dict
        Columns list and row count for downstream use.
    """
    rprint(f"\n{'='*70}")
    rprint(f"  {name}: {filepath.name}")
    rprint(f"  Size: {filepath.stat().st_size / 1e9:.2f} GB")
    rprint(f"{'='*70}")

    sample = peek_schema(filepath)
    cols = list(sample.columns)
    rprint(f"  Columns ({len(cols)}): {', '.join(cols[:15])}")
    if len(cols) > 15:
        rprint(f"    ... and {len(cols) - 15} more")

    rprint(f"\n  Dtypes (from {len(sample)}-row sample):")
    for col in cols:
        rprint(f"    {col:<25s} {str(sample[col].dtype):<15s}")

    rprint("\n  Counting rows (may take a moment)...")
    nrows = count_rows(filepath)
    rprint(f"  Total rows: {nrows:,}")

    del sample
    gc.collect()
    return {"columns": cols, "nrows": nrows}


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Winsorize a Series at given quantiles."""
    lower = s.quantile(lo)
    upper = s.quantile(hi)
    return s.clip(lower, upper)


# ---------------------------------------------------------------------------
# Compustat diagnostics
# ---------------------------------------------------------------------------


def explore_compustat_global() -> pd.DataFrame:
    """Explore compustat_global.csv: coverage, missingness, quality flags.

    Returns
    -------
    pd.DataFrame
        Loaded subset for downstream use (coverage figures).
    """
    rprint("\n" + "#" * 70)
    rprint("# COMPUSTAT GLOBAL — Detailed Exploration")
    rprint("#" * 70)

    # Key columns we care about
    want_cols = [
        "gvkey", "fyear", "at", "ceq", "sale", "ebitda",
        "mkvalt", "curcd", "fic",
    ]

    # First check which columns actually exist
    sample = peek_schema(FILES["compustat_global"])
    available = [c for c in want_cols if c in sample.columns]
    missing_cols = [c for c in want_cols if c not in sample.columns]
    if missing_cols:
        rprint(f"  [WARN] Columns not found: {missing_cols}")
    del sample

    dtype_map = {"gvkey": str}
    df = pd.read_csv(
        FILES["compustat_global"],
        usecols=available,
        dtype=dtype_map,
        low_memory=False,
    )
    rprint(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")

    # Year range
    if "fyear" in df.columns:
        rprint(f"  Fiscal year range: {df['fyear'].min():.0f} – {df['fyear'].max():.0f}")

    # Country coverage
    if "fic" in df.columns:
        country_counts = df.groupby("fic")["gvkey"].nunique().sort_values(ascending=False)
        rprint(f"\n  Countries represented: {country_counts.shape[0]}")
        rprint("  Top 20 countries by unique firms:")
        for country, cnt in country_counts.head(20).items():
            rprint(f"    {country:<6s} {cnt:>7,}")

    # Missingness for key financial variables
    fin_vars = [c for c in ["at", "ceq", "sale", "ebitda", "mkvalt"] if c in df.columns]
    rprint("\n  Missingness (% NaN):")
    for v in fin_vars:
        pct = df[v].isna().mean() * 100
        rprint(f"    {v:<12s} {pct:6.1f}%")

    # Data quality flags
    rprint("\n  Data quality flags:")
    if "at" in df.columns:
        n_zero_at = (df["at"] == 0).sum()
        n_neg_at = (df["at"] < 0).sum()
        rprint(f"    Zero total assets:     {n_zero_at:>8,}")
        rprint(f"    Negative total assets: {n_neg_at:>8,}")
    if "ceq" in df.columns:
        n_neg_ceq = (df["ceq"] < 0).sum()
        rprint(f"    Negative equity:       {n_neg_ceq:>8,}")

    return df


def explore_compustat_america() -> pd.DataFrame:
    """Explore compustat_america.csv: coverage, missingness, quality flags.

    Returns
    -------
    pd.DataFrame
        Loaded subset for downstream use.
    """
    rprint("\n" + "#" * 70)
    rprint("# COMPUSTAT AMERICA — Detailed Exploration")
    rprint("#" * 70)

    want_cols = [
        "GVKEY", "gvkey", "fyear", "at", "ceq", "sale", "ebitda",
        "prcc_f", "csho", "LPERMNO", "LPERMCO",
    ]

    sample = peek_schema(FILES["compustat_america"])
    available = [c for c in want_cols if c in sample.columns]
    missing_cols = [c for c in want_cols if c not in sample.columns and c != "gvkey"]
    if missing_cols:
        rprint(f"  [WARN] Columns not found: {missing_cols}")
    del sample

    # GVKEY is uppercase in compustat_america
    dtype_map = {"GVKEY": str, "gvkey": str}
    df = pd.read_csv(
        FILES["compustat_america"],
        usecols=available,
        dtype=dtype_map,
        low_memory=False,
    )
    rprint(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")

    if "fyear" in df.columns:
        rprint(f"  Fiscal year range: {df['fyear'].min():.0f} – {df['fyear'].max():.0f}")

    # CRSP link availability
    for lnk in ["LPERMNO", "LPERMCO"]:
        if lnk in df.columns:
            pct = df[lnk].notna().mean() * 100
            rprint(f"  {lnk} available: {pct:.1f}%")

    # Missingness
    fin_vars = [c for c in ["at", "ceq", "sale", "ebitda", "prcc_f", "csho"] if c in df.columns]
    rprint("\n  Missingness (% NaN):")
    for v in fin_vars:
        pct = df[v].isna().mean() * 100
        rprint(f"    {v:<12s} {pct:6.1f}%")

    # Data quality flags
    rprint("\n  Data quality flags:")
    if "at" in df.columns:
        rprint(f"    Zero total assets:     {(df['at'] == 0).sum():>8,}")
        rprint(f"    Negative total assets: {(df['at'] < 0).sum():>8,}")
    if "ceq" in df.columns:
        rprint(f"    Negative equity:       {(df['ceq'] < 0).sum():>8,}")

    return df


# ---------------------------------------------------------------------------
# Returns diagnostics
# ---------------------------------------------------------------------------


def explore_returns(name: str, key: str) -> None:
    """Explore a returns file (global or america).

    Parameters
    ----------
    name : str
        Display name for the dataset.
    key : str
        Key in FILES dict.
    """
    rprint("\n" + "#" * 70)
    rprint(f"# {name} — Detailed Exploration")
    rprint("#" * 70)

    sample = peek_schema(FILES[key])
    cols = list(sample.columns)
    rprint(f"  Columns ({len(cols)}): {', '.join(cols[:15])}")
    if len(cols) > 15:
        rprint(f"    ... and {len(cols) - 15} more")

    # Identify date and return columns
    date_col = None
    for candidate in ["date", "datadate", "MthCalDt", "mthcaldt"]:
        if candidate in cols:
            date_col = candidate
            break

    ret_col = None
    for candidate in ["ret", "RET", "trt1m", "MthRet"]:
        if candidate in cols:
            ret_col = candidate
            break

    id_col = None
    for candidate in ["PERMNO", "permno", "gvkey", "GVKEY"]:
        if candidate in cols:
            id_col = candidate
            break

    load_cols = [c for c in [id_col, date_col, ret_col] if c is not None]
    if not load_cols:
        rprint("  [WARN] Could not identify key columns. Skipping detailed exploration.")
        del sample
        return

    rprint(f"  Identified: id={id_col}, date={date_col}, ret={ret_col}")

    del sample
    gc.collect()

    dtype_map = {}
    if id_col and id_col.lower() == "gvkey":
        dtype_map[id_col] = str
    df = pd.read_csv(FILES[key], usecols=load_cols, dtype=dtype_map, low_memory=False)
    rprint(f"  Loaded {len(df):,} rows")

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        rprint(f"  Date range: {df[date_col].min()} – {df[date_col].max()}")

    if id_col:
        rprint(f"  Unique securities: {df[id_col].nunique():,}")

    if ret_col:
        rprint(f"\n  Return ({ret_col}) summary:")
        n_valid = df[ret_col].notna().sum()
        pct_valid = df[ret_col].notna().mean() * 100
        rprint(f"    Non-missing: {n_valid:,} ({pct_valid:.1f}%)")
        numeric_ret = pd.to_numeric(df[ret_col], errors="coerce")
        rprint(f"    Mean:   {numeric_ret.mean():.4f}")
        rprint(f"    Median: {numeric_ret.median():.4f}")
        rprint(f"    Std:    {numeric_ret.std():.4f}")
        rprint(f"    Min:    {numeric_ret.min():.4f}")
        rprint(f"    Max:    {numeric_ret.max():.4f}")

    del df
    gc.collect()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_coverage_by_country(df_global: pd.DataFrame) -> None:
    """Bar chart: firm-years by country (top 20) for Compustat Global."""
    if "fic" not in df_global.columns:
        rprint("  [SKIP] No 'fic' column for country coverage plot.")
        return

    counts = df_global.groupby("fic").size().sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots()
    ax.barh(counts.index, counts.values, color=PALETTE["accent"])
    ax.set_xlabel("Firm-Years")
    ax.set_title("Compustat Global: Top 20 Countries by Firm-Years")
    ax.tick_params(axis="y", labelsize=9)

    fig.savefig(OUT_DIR / "coverage_by_country.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "coverage_by_country.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    rprint("  Saved: coverage_by_country.pdf/png")


def plot_coverage_by_year(
    df_global: pd.DataFrame, df_america: pd.DataFrame
) -> None:
    """Line chart: unique firms per year for Global vs US."""
    fig, ax = plt.subplots()

    # Global coverage — check if US firms are included
    gvkey_col_g = "gvkey" if "gvkey" in df_global.columns else "GVKEY"
    if "fyear" in df_global.columns and gvkey_col_g in df_global.columns:
        if "fic" in df_global.columns and (df_global["fic"] == "USA").any():
            label_global = "Global (incl. US)"
        else:
            label_global = "Global (ex-US)"
        g = df_global.groupby("fyear")[gvkey_col_g].nunique()
        ax.plot(g.index, g.values, color=PALETTE["accent"], label=label_global)

    # US coverage — handle uppercase GVKEY in compustat_america
    gvkey_col_a = "gvkey" if "gvkey" in df_america.columns else "GVKEY"
    if "fyear" in df_america.columns and gvkey_col_a in df_america.columns:
        a = df_america.groupby("fyear")[gvkey_col_a].nunique()
        ax.plot(a.index, a.values, color=PALETTE["primary"], label="US (Compustat NA)")

    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Unique Firms")
    ax.set_title("Coverage Over Time")
    ax.legend()

    fig.savefig(OUT_DIR / "coverage_by_year.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "coverage_by_year.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    rprint("  Saved: coverage_by_year.pdf/png")


def plot_valuation_distributions(
    df_global: pd.DataFrame, df_america: pd.DataFrame
) -> None:
    """Histograms of P/B, raw vs winsorized, for Global and US samples.

    P/B = mkvalt / ceq  (Global) or (prcc_f * csho) / ceq  (US).
    Negative or zero equity observations are excluded.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # --- Global P/B ---
    pb_global = None
    if all(c in df_global.columns for c in ["mkvalt", "ceq"]):
        # P/B is currency-neutral (ratio of same-currency values per firm-year)
        mask = (df_global["ceq"] > 0) & (df_global["mkvalt"] > 0)
        pb_global = df_global.loc[mask, "mkvalt"] / df_global.loc[mask, "ceq"]
        pb_global = pb_global.replace([np.inf, -np.inf], np.nan).dropna()

    if pb_global is not None and len(pb_global) > 0:
        pb_win = winsorize(pb_global)
        axes[0, 0].hist(pb_global.clip(upper=pb_global.quantile(0.99)), bins=100,
                        color=PALETTE["secondary"], alpha=0.7)
        axes[0, 0].set_title("Global P/B (raw, clipped 99th)")
        axes[0, 0].set_xlabel("P/B")
        axes[0, 0].set_ylabel("Frequency")

        axes[0, 1].hist(pb_win, bins=100, color=PALETTE["accent"], alpha=0.7)
        axes[0, 1].set_title("Global P/B (winsorized 1/99)")
        axes[0, 1].set_xlabel("P/B")
    else:
        axes[0, 0].text(0.5, 0.5, "P/B not available", ha="center", va="center",
                        transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, "P/B not available", ha="center", va="center",
                        transform=axes[0, 1].transAxes)

    # --- US P/B ---
    pb_us = None
    if all(c in df_america.columns for c in ["prcc_f", "csho", "ceq"]):
        mask = (df_america["ceq"] > 0) & (df_america["prcc_f"] > 0) & (df_america["csho"] > 0)
        mkcap = df_america.loc[mask, "prcc_f"] * df_america.loc[mask, "csho"]
        pb_us = mkcap / df_america.loc[mask, "ceq"]
        pb_us = pb_us.replace([np.inf, -np.inf], np.nan).dropna()

    if pb_us is not None and len(pb_us) > 0:
        pb_us_win = winsorize(pb_us)
        axes[1, 0].hist(pb_us.clip(upper=pb_us.quantile(0.99)), bins=100,
                        color=PALETTE["secondary"], alpha=0.7)
        axes[1, 0].set_title("US P/B (raw, clipped 99th)")
        axes[1, 0].set_xlabel("P/B")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].hist(pb_us_win, bins=100, color=PALETTE["accent"], alpha=0.7)
        axes[1, 1].set_title("US P/B (winsorized 1/99)")
        axes[1, 1].set_xlabel("P/B")
    else:
        axes[1, 0].text(0.5, 0.5, "US P/B not available", ha="center", va="center",
                        transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, "US P/B not available", ha="center", va="center",
                        transform=axes[1, 1].transAxes)

    fig.suptitle("P/B Distributions: Raw vs Winsorized (Global and US)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "valuation_distributions.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "valuation_distributions.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    rprint("  Saved: valuation_distributions.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full data exploration pipeline."""
    rprint("=" * 70)
    rprint("  GLOBAL VALUATION — Data Exploration Report")
    rprint(f"  Date: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    rprint(f"  Data directory: {DATA_DIR}")
    rprint("=" * 70)

    # Verify all files exist
    all_exist = True
    for name, path in FILES.items():
        exists = path.exists()
        size = f"{path.stat().st_size / 1e9:.2f} GB" if exists else "MISSING"
        rprint(f"  {name:<25s} {size:<10s} {'OK' if exists else 'MISSING!'}")
        if not exists:
            all_exist = False

    if not all_exist:
        rprint("\n  [ERROR] One or more data files are missing. Cannot proceed.")
        rprint(f"  Expected directory: {DATA_DIR}")
        report_path = OUT_DIR / "data_exploration_report.txt"
        report_path.write_text(_report_buf.getvalue(), encoding="utf-8")
        raise FileNotFoundError(f"Missing data files in {DATA_DIR}")
    rprint()

    # --- Step 1: Schema summaries for all 4 files ---
    rprint("\n" + "=" * 70)
    rprint("  STEP 1: Schema Summaries")
    rprint("=" * 70)
    schemas = {}
    for name, path in FILES.items():
        schemas[name] = summarize_schema(name, path)

    # --- Step 2: Detailed Compustat exploration ---
    rprint("\n" + "=" * 70)
    rprint("  STEP 2: Detailed Compustat Exploration")
    rprint("=" * 70)
    df_global = explore_compustat_global()
    df_america = explore_compustat_america()

    # --- Step 3: Returns exploration ---
    rprint("\n" + "=" * 70)
    rprint("  STEP 3: Returns Exploration")
    rprint("=" * 70)
    explore_returns("RETURNS GLOBAL", "returns_global")
    explore_returns("RETURNS AMERICA", "returns_america")

    # --- Step 4: Figures ---
    rprint("\n" + "=" * 70)
    rprint("  STEP 4: Diagnostic Figures")
    rprint("=" * 70)
    plot_coverage_by_country(df_global)
    plot_coverage_by_year(df_global, df_america)
    plot_valuation_distributions(df_global, df_america)

    # Free memory
    del df_global, df_america
    gc.collect()

    # --- Save report ---
    report_path = OUT_DIR / "data_exploration_report.txt"
    report_path.write_text(_report_buf.getvalue(), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
