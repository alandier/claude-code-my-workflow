# Project Memory

Corrections and learned facts that persist across sessions.
When a mistake is corrected, append a `[LEARN:category]` entry below.

---

## Project Bootstrap

- **Project:** Global Valuation -- how company valuation varies across countries, sectors, and time
- **Author:** Augustin Landier, HEC Paris
- **Target journals:** JF / RFS / JFE
- **Primary language:** Python (pandas, matplotlib/seaborn, statsmodels)
- **Paper format:** LaTeX (XeLaTeX)
- **Presentations:** Beamer + Quarto (secondary, infrastructure ready)

## Data Files (External -- NOT in repo)

Path: `~/Dropbox/Research_Data/GlobalValuation/`

| File | Source | Approx Size | Key Variables |
|------|--------|-------------|---------------|
| `compustat_global.csv` | Compustat Global | ~2 GB | gvkey, fyear, at, ceq, sale, ebitda, mkvalt, curcd |
| `compustat_na.csv` | Compustat North America | ~1.5 GB | gvkey, fyear, at, ceq, sale, ebitda, prcc_f, csho |
| `crsp_monthly.csv` | CRSP Monthly | ~1.5 GB | permno, date, ret, prc, shrout, vwretd |
| `crsp_daily.csv` | CRSP Daily | ~500 MB | permno, date, ret, prc, vol |

## Key Conventions

- **DATA_DIR:** `pathlib.Path("~/Dropbox/Research_Data/GlobalValuation/").expanduser()`
- **Seed:** `np.random.seed(20260215)` (YYYYMMDD format)
- **Figures:** 300 DPI, PDF + PNG, 6x4 inches default, muted blues/grays palette
- **Parquet pattern:** Load CSV once, save as `.parquet`, load parquet thereafter
- **Winsorization:** 1st/99th percentiles by default for valuation ratios

---

<!-- Append new [LEARN] entries below. Most recent at bottom. -->
