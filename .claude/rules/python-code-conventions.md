---
paths:
  - "**/*.py"
  - "scripts/**/*.py"
  - "explorations/**/*.py"
---

# Python Code Standards

**Standard:** Senior data engineer + PhD financial economist quality

---

## 1. Reproducibility

- `np.random.seed(YYYYMMDD)` called ONCE at top for stochastic code
- All imports at top, grouped: stdlib / third-party / local
- All paths via `pathlib.Path` -- never string concatenation for paths
- `Path(...).mkdir(parents=True, exist_ok=True)` for output directories

### DATA_DIR Constant

```python
from pathlib import Path

DATA_DIR = Path("~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/").expanduser()
OUT_DIR = Path("output")
FIG_DIR = Path("Figures")
```

---

## 2. Data Handling for Large Files

Raw data is ~5.5 GB across 4 CSVs. Always be memory-conscious.

```python
# GOOD: specify columns and dtypes
df = pd.read_csv(DATA_DIR / "compustat_global.csv",
                 usecols=["gvkey", "fyear", "at", "ceq", "mkvalt"],
                 dtype={"gvkey": str, "fyear": int})

# GOOD: parquet pattern -- load CSV once, save parquet, reload from parquet
parquet_path = OUT_DIR / "compustat_global.parquet"
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
else:
    df = pd.read_csv(DATA_DIR / "compustat_global.csv", usecols=[...])
    df.to_parquet(parquet_path, index=False)
```

- **Never** `pd.read_csv()` without `usecols` on raw data files
- **Never** load all 4 files simultaneously unless absolutely necessary
- Use `del df; gc.collect()` after large intermediate DataFrames

---

## 3. Function Design

- `snake_case` naming, verb-noun pattern (`compute_pe_ratio`, `filter_sample`)
- Docstrings on all non-trivial functions (NumPy style)
- Default parameters, no magic numbers
- Type hints for function signatures

---

## 4. Visual Identity

### Project Palette (Muted Blues/Grays)

```python
PALETTE = {
    "primary":   "#2c3e50",  # dark blue-gray
    "secondary": "#7f8c8d",  # medium gray
    "accent":    "#2980b9",  # blue
    "highlight": "#8e44ad",  # purple
    "alert":     "#c0392b",  # red
}
```

### Matplotlib Style

```python
import matplotlib.pyplot as plt

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
```

### Figure Export (Always Dual Format)

```python
fig.savefig(FIG_DIR / "name.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "name.png", bbox_inches="tight", dpi=300)
plt.close(fig)
```

---

## 5. Financial Data Pitfalls

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| Negative book equity | Undefined P/B, Tobin's Q | Filter or flag; never silently divide |
| Currency mixing | Apples-to-oranges valuation | Check `curcd`; convert or analyze within-currency |
| Survivorship bias | Overestimate returns | Include delisted firms; use CRSP delist returns |
| Outlier ratios | Skewed means | Winsorize at 1st/99th percentiles |
| Stock splits | Inflated share counts | Use `ajex` adjustment factor or CRSP adjusted prices |
| Fiscal year alignment | Mismatch across countries | Use `fyear` not calendar year for Compustat |
| Missing values coded as 0 | Fake zeros in financials | Check for suspicious zeros in `at`, `sale`, `ceq` |

### Compustat Variable Quick Reference

| Variable | Meaning | Dataset |
|----------|---------|---------|
| `gvkey` | Firm identifier | Both |
| `fyear` | Fiscal year | Both |
| `at` | Total assets | Both |
| `ceq` | Common equity | Both |
| `sale` | Net sales | Both |
| `ebitda` | EBITDA | Both |
| `mkvalt` | Market value (Global) | Global |
| `prcc_f` | Price close fiscal year (NA) | NA |
| `csho` | Shares outstanding (NA) | NA |
| `curcd` | Currency code | Global |

---

## 6. Line Length & Exceptions

**Standard:** Keep lines <= 100 characters.

**Exception:** Long pandas chains or financial formulas may exceed 100 chars if:
1. Breaking the line harms readability of the financial logic
2. An inline comment explains the computation
3. The line implements a formula matching the paper

---

## 7. Code Quality Checklist

```
[ ] Imports at top, grouped (stdlib / third-party / local)
[ ] DATA_DIR constant, pathlib.Path for all paths
[ ] np.random.seed() once at top (if stochastic)
[ ] usecols/dtype for all large CSV loads
[ ] Functions documented (docstrings)
[ ] Figures: dual export (PDF + PNG), 300 DPI, project palette
[ ] Data exports: .parquet for DataFrames, pickle for models
[ ] Winsorize valuation ratios before computing means
[ ] Comments explain WHY not WHAT
```
