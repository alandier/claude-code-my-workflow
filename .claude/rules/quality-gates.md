---
paths:
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
  - "scripts/**/*.R"
  - "scripts/**/*.py"
  - "Paper/**/*.tex"
---

# Quality Gates & Scoring Rubrics

## Thresholds

- **80/100 = Commit** -- good enough to save
- **90/100 = PR** -- ready for deployment
- **95/100 = Excellence** -- aspirational

## Quarto Slides (.qmd)

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Compilation failure | -100 |
| Critical | Equation overflow | -20 |
| Critical | Broken citation | -15 |
| Critical | Typo in equation | -10 |
| Major | Text overflow | -5 |
| Major | TikZ label overlap | -5 |
| Major | Notation inconsistency | -3 |
| Minor | Font size reduction | -1 per slide |
| Minor | Long lines (>100 chars) | -1 (EXCEPT documented math formulas) |

## Python Scripts (.py)

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Syntax errors | -100 |
| Critical | Domain-specific bugs (wrong ratio, currency mixing) | -30 |
| Critical | Hardcoded absolute paths (except DATA_DIR) | -20 |
| Major | Memory-inefficient loading (no usecols on large CSV) | -15 |
| Major | Missing np.random.seed() (when stochastic) | -10 |
| Major | Missing figure export (no savefig) | -5 |
| Major | Missing data export (no to_parquet/to_csv) | -5 |
| Minor | Style violation | -1 |
| Minor | Missing docstring | -1 |

## R Scripts (.R)

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | Syntax errors | -100 |
| Critical | Domain-specific bugs | -30 |
| Critical | Hardcoded absolute paths | -20 |
| Major | Missing set.seed() | -10 |
| Major | Missing figure generation | -5 |

## Beamer Slides (.tex)

| Severity | Issue | Deduction |
|----------|-------|-----------|
| Critical | XeLaTeX compilation failure | -100 |
| Critical | Undefined citation | -15 |
| Critical | Overfull hbox > 10pt | -10 |

## Enforcement

- **Score < 80:** Block commit. List blocking issues.
- **Score < 90:** Allow commit, warn. List recommendations.
- User can override with justification.

## Quality Reports

Generated **only at merge time**. Use `templates/quality-report.md` for format.
Save to `quality_reports/merges/YYYY-MM-DD_[branch-name].md`.

## Tolerance Thresholds (Research)

| Quantity | Tolerance | Rationale |
|----------|-----------|-----------|
| Valuation ratios (P/E, P/B, EV/EBITDA) | +/- 0.001 | 3 decimal places for ratios |
| Returns | +/- 0.0001 (1 bp) | Basis point precision |
| Regression coefficients | +/- 1e-6 | Numerical precision |
| Standard errors | +/- 1e-4 | Clustering variability |
| Sample sizes (N) | Exact | No reason for any difference |
| R-squared | +/- 0.001 | Display rounding |
