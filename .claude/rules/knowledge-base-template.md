---
paths:
  - "Paper/**/*.tex"
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
  - "scripts/**/*.py"
  - "scripts/**/*.R"
---

# Domain Knowledge Base: Corporate Finance & Valuation

## Notation Registry

| Rule | Convention | Example | Anti-Pattern |
|------|-----------|---------|-------------|
| Valuation multiples | Numerator / Denominator | P/E = Price / Earnings | Inverting the ratio |
| Returns | Lowercase r with subscripts | r_{i,t} | Using R for returns (conflicts with language) |
| Time subscripts | t for year, m for month | r_{i,t} | Mixing t and T inconsistently |
| Firm subscripts | i for firm, j for country | MCap_{i,t} | Using f for firm (conflicts with function) |
| Currency | 3-letter ISO code | USD, EUR, JPY | Ambiguous symbols ($, etc.) |
| Significance | Stars: * p<0.10, ** p<0.05, *** p<0.01 | 2.34*** | Non-standard star conventions |

## Symbol Reference

| Symbol | Meaning | Context |
|--------|---------|---------|
| P/E | Price-to-Earnings ratio | Equity valuation |
| P/B | Price-to-Book ratio | Equity valuation |
| EV/EBITDA | Enterprise Value / EBITDA | Firm valuation |
| Tobin's Q | (MVE + Debt) / Total Assets | Firm valuation |
| MCap | Market capitalization | prcc_f * csho (NA) or mkvalt (Global) |
| BV | Book value of equity | ceq in Compustat |
| ROE | Return on equity | ni / ceq |
| ROA | Return on assets | ni / at |

## Paper Sections (Planned)

| # | Section | Core Question | Key Data | Key Method |
|---|---------|--------------|----------|------------|
| 1 | Introduction | Why do valuations differ across countries? | Aggregate stats | Descriptive |
| 2 | Data & Sample | How is the sample constructed? | Compustat Global + NA | Filters, merges |
| 3 | Cross-Country Facts | What are the valuation patterns? | Country-year panels | Summary stats, time series |
| 4 | Decomposition | Fundamentals vs. discount rates? | Firm-level panel | Panel regressions |
| 5 | Robustness | Do results survive alternative specs? | Various | Sensitivity analysis |

## Data Sources

| Source | Coverage | Key Variables | Access |
|--------|----------|--------------|--------|
| Compustat Global | Non-US firms, 1980s-present | Financials, market value | WRDS |
| Compustat NA | US/Canada firms, 1960s-present | Financials, price, shares | WRDS |
| CRSP Monthly | US equities, 1926-present | Returns, prices, shares | WRDS |
| CRSP Daily | US equities, 1926-present | Daily returns, volume | WRDS |

## Data Cleaning Principles

| Principle | Implementation | Rationale |
|-----------|---------------|-----------|
| Positive assets | Filter `at > 0` | Undefined ratios otherwise |
| Positive equity | Filter `ceq > 0` for P/B | Negative book value is economically meaningful but breaks ratios |
| Winsorize ratios | 1st/99th percentiles | Extreme outliers dominate means |
| Fiscal year alignment | Use `fyear`, not calendar year | Fiscal years vary across firms/countries |
| Currency consistency | Analyze within-currency or convert | Cross-currency comparisons require PPP or exchange rates |
| Minimum observations | Require N >= 30 per country-year | Small samples produce unstable estimates |

## Anti-Patterns (Don't Do This)

| Anti-Pattern | What Happened | Correction |
|-------------|---------------|-----------|
| Computing mean P/E without winsorizing | Means dominated by outliers (P/E > 1000) | Winsorize first, or use median |
| Mixing Compustat Global mkvalt with NA prcc_f*csho | Inconsistent market cap definition | Use dataset-specific formulas |
| Ignoring currency in cross-country comparison | Comparing JPY-denominated to USD-denominated | Convert or use ratios (which are currency-neutral) |
| Using calendar year instead of fiscal year | Misaligned financial data | Always use fyear for Compustat |
| Dropping all firms with any missing variable | Survivorship bias, tiny sample | Drop per-analysis, document missingness |

## Python/Finance Pitfalls

| Bug | Impact | Fix |
|-----|--------|-----|
| `pd.read_csv` without `dtype={"gvkey": str}` | gvkey parsed as int, loses leading zeros | Always specify `dtype` |
| Division without checking denominator | NaN or Inf in valuation ratios | Filter denominator > 0 first |
| Merging without checking duplicates | Silent row explosion | `validate="1:1"` or `validate="m:1"` |
| `groupby().mean()` including NaN | Silent exclusion changes sample | Use `min_count` or explicit dropna |
| Winsorizing after taking logs | Log of extreme values already compressed | Winsorize in levels, then take log |
