---
name: data-analysis-python
description: End-to-end Python data analysis workflow for financial research — from data loading through analysis to publication-ready tables and figures
disable-model-invocation: true
argument-hint: "[dataset path or description of analysis goal]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Edit", "Bash", "Task"]
---

# Python Data Analysis Workflow

Run an end-to-end data analysis in Python: load, explore, analyze, and produce publication-ready output.

**Input:** `$ARGUMENTS` -- a dataset path (e.g., `compustat_global`) or a description of the analysis goal (e.g., "compute median P/E by country and year using Compustat Global").

---

## Constraints

- **Follow Python code conventions** in `.claude/rules/python-code-conventions.md`
- **Save all scripts** to `scripts/python/` with descriptive names
- **Save all outputs** (figures, tables, parquet) to `output/`
- **Save figures** to `Figures/` (PDF + PNG dual export)
- **Use `DATA_DIR` constant** for external data paths
- **Use `usecols`/`dtype`** for all large CSV reads -- data files are multi-GB
- **Run python-reviewer** on the generated script before presenting results

---

## Workflow Phases

### Phase 1: Setup and Data Loading

1. Read `.claude/rules/python-code-conventions.md` for project standards
2. Create Python script with proper header (title, author, purpose, inputs, outputs)
3. Import packages at top, grouped (stdlib / third-party / local)
4. Set seed if stochastic: `np.random.seed(YYYYMMDD)`
5. Define `DATA_DIR`, `OUT_DIR`, `FIG_DIR` constants
6. Load data with `usecols` and `dtype` -- use parquet if available

### Phase 2: Clean and Explore

Generate diagnostic outputs:
- **Summary statistics:** `.describe()`, missingness rates, variable types
- **Sample coverage:** year range, country count, firm count by year
- **Distributions:** Histograms for key continuous variables
- **Outliers:** Flag extreme valuation ratios before winsorization
- **Data quality:** Check for suspicious zeros, negative equity, currency mixing

Save diagnostic figures to `output/diagnostics/`.

### Phase 3: Main Analysis

Based on the research question:
- **Cross-sectional:** Valuation ratios by country, sector, or time
- **Panel regression:** Use `linearmodels` or `statsmodels` for fixed effects
- **Standard errors:** Cluster at firm or country-year level (document why)
- **Multiple specifications:** Start simple, progressively add controls
- **Winsorize** valuation ratios at 1st/99th percentiles before regressions

### Phase 4: Publication-Ready Output

**Tables:**
- Use `pandas` `.to_latex()` or `stargazer` for regression tables
- Include all standard elements: coefficients, SEs, significance stars, N, R-squared
- Export as `.tex` for paper inclusion and `.csv` for inspection

**Figures:**
- Use project palette (muted blues/grays)
- Set matplotlib rcParams at top
- Dual export: PDF (for paper) + PNG (for inspection) at 300 DPI
- Include proper axis labels (sentence case, units)
- `plt.close(fig)` after every save

### Phase 5: Save and Review

1. Save processed DataFrames as `.parquet`
2. Save summary tables as `.csv` and `.tex`
3. Save figures to `Figures/` (PDF + PNG)
4. Create `output/` subdirectories as needed
5. Run the python-reviewer agent on the generated script:

```
Delegate to the python-reviewer agent:
"Review the script at scripts/python/[script_name].py"
```

6. Address any Critical or High issues from the review.

---

## Script Template

```python
# ============================================================
# [Descriptive Title]
# Author: Augustin Landier, HEC Paris
# Purpose: [What this script does]
# Inputs: [Data files]
# Outputs: [Figures, tables, parquet files]
# ============================================================

# 0. Setup ----
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(20260215)

DATA_DIR = Path("~/Dropbox/Research_Data/GlobalValuation/").expanduser()
OUT_DIR = Path("output")
FIG_DIR = Path("Figures")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Project palette
PALETTE = {
    "primary":   "#2c3e50",
    "secondary": "#7f8c8d",
    "accent":    "#2980b9",
    "highlight": "#8e44ad",
    "alert":     "#c0392b",
}

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 1. Data Loading ----
# [Load with usecols/dtype, use parquet if available]

# 2. Data Cleaning ----
# [Handle missing values, filter sample, winsorize]

# 3. Analysis ----
# [Summary statistics, regressions, decompositions]

# 4. Figures ----
# [Publication-quality plots with project palette]

# 5. Export ----
# [Save parquet, CSV, LaTeX tables, figures as PDF+PNG]
```

---

## Important

- **Be memory-conscious.** Raw data is ~5.5 GB. Always use `usecols`/`dtype`.
- **Reproduce, don't guess.** If the user specifies an analysis, run exactly that.
- **Show your work.** Print summary statistics before jumping to regression.
- **Check for issues.** Look for negative equity, currency mixing, survivorship bias.
- **Use relative paths.** Only `DATA_DIR` points outside the repo.
- **No hardcoded values.** Use variables for sample restrictions, date ranges, etc.
