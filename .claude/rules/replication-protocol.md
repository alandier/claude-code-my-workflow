---
paths:
  - "scripts/**/*.R"
  - "scripts/**/*.py"
  - "Figures/**/*.R"
---

# Replication-First Protocol

**Core principle:** Replicate original results to the dot BEFORE extending.

---

## Phase 1: Inventory & Baseline

Before writing any R code:

- [ ] Read the paper's replication README
- [ ] Inventory replication package: language, data files, scripts, outputs
- [ ] Record gold standard numbers from the paper:

```markdown
## Replication Targets: [Paper Author (Year)]

| Target | Table/Figure | Value | SE/CI | Notes |
|--------|-------------|-------|-------|-------|
| Main ATT | Table 2, Col 3 | -1.632 | (0.584) | Primary specification |
```

- [ ] Store targets in `quality_reports/LectureNN_replication_targets.md` or as RDS

---

## Phase 2: Translate & Execute

- [ ] Follow `r-code-conventions.md` for all R coding standards
- [ ] Translate line-by-line initially -- don't "improve" during replication
- [ ] Match original specification exactly (covariates, sample, clustering, SE computation)
- [ ] Save all intermediate results as RDS

### Stata to R Translation Pitfalls

| Stata | R | Trap |
|-------|---|------|
| `reg y x, cluster(id)` | `feols(y ~ x, cluster = ~id)` | Stata clusters df-adjust differently from some R packages |
| `areg y x, absorb(id)` | `feols(y ~ x \| id)` | Check demeaning method matches |
| `probit` for PS | `glm(family=binomial(link="probit"))` | R default logit != Stata default in some commands |
| `bootstrap, reps(999)` | Depends on method | Match seed, reps, and bootstrap type exactly |

### Stata to Python Translation Pitfalls

| Stata | Python | Trap |
|-------|--------|------|
| `reg y x, cluster(id)` | `sm.OLS(...).fit(cov_type='cluster', cov_kwds={'groups': df['id']})` | statsmodels clustering differs from Stata's small-sample adjustment |
| `areg y x, absorb(id)` | `linearmodels.PanelOLS(..., entity_effects=True)` | Demeaning method and df correction may differ |
| `winsor2 var, cuts(1 99)` | `scipy.stats.mstats.winsorize(var, limits=[0.01, 0.01])` | Stata winsor2 uses percentiles; scipy uses proportions |
| `collapse (mean) y, by(id)` | `df.groupby('id')['y'].mean()` | pandas excludes NaN by default; Stata includes |
| `merge 1:1 id using ...` | `pd.merge(..., validate='1:1')` | pandas merge without validate silently creates duplicates |
| `xtset id t` | `df = df.set_index(['id', 't'])` | Panel structure must be explicit in Python |

---

## Phase 3: Verify Match

### Tolerance Thresholds

| Type | Tolerance | Rationale |
|------|-----------|-----------|
| Integers (N, counts) | Exact match | No reason for any difference |
| Point estimates | < 0.01 | Rounding in paper display |
| Standard errors | < 0.05 | Bootstrap/clustering variation |
| P-values | Same significance level | Exact p may differ slightly |
| Percentages | < 0.1pp | Display rounding |

### If Mismatch

**Do NOT proceed to extensions.** Isolate which step introduces the difference, check common causes (sample size, SE computation, default options, variable definitions), and document the investigation even if unresolved.

### Replication Report

Save to `quality_reports/LectureNN_replication_report.md`:

```markdown
# Replication Report: [Paper Author (Year)]
**Date:** [YYYY-MM-DD]
**Original language:** [Stata/R/etc.]
**R translation:** [script path]

## Summary
- **Targets checked / Passed / Failed:** N / M / K
- **Overall:** [REPLICATED / PARTIAL / FAILED]

## Results Comparison

| Target | Paper | Ours | Diff | Status |
|--------|-------|------|------|--------|

## Discrepancies (if any)
- **Target:** X | **Investigation:** ... | **Resolution:** ...

## Environment
- R version, key packages (with versions), data source
```

---

## Phase 4: Only Then Extend

After replication is verified (all targets PASS):

- [ ] Commit replication script: "Replicate [Paper] Table X -- all targets match"
- [ ] Now extend with course-specific modifications (different estimators, new figures, etc.)
- [ ] Each extension builds on the verified baseline
