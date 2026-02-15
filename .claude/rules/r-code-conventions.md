---
paths:
  - "**/*.R"
  - "Figures/**/*.R"
  - "scripts/**/*.R"
---

# R Code Standards

**Standard:** Senior Principal Data Engineer + PhD researcher quality

---

## 1. Reproducibility

- `set.seed()` called ONCE at top (YYYYMMDD format)
- All packages loaded at top via `library()` (not `require()`)
- All paths relative to repository root
- `dir.create(..., recursive = TRUE)` for output directories

## 2. Function Design

- `snake_case` naming, verb-noun pattern
- Roxygen-style documentation
- Default parameters, no magic numbers
- Named return values (lists or tibbles)

## 3. Domain Correctness

- Verify valuation ratio computations (P/E, P/B, EV/EBITDA, Tobin's Q)
- Use adjusted prices for return calculations (handle stock splits)
- Winsorize valuation ratios at 1st/99th percentiles before computing means
- Filter negative equity before computing P/B ratios
- Check currency consistency in cross-country comparisons
- Document and justify all sample filters (exclusion of financials, utilities, etc.)
- Check known package bugs (document below in Common Pitfalls)

## 4. Visual Identity

```r
# --- Project palette (muted blues/grays) ---
primary_dark  <- "#2c3e50"
secondary_gray <- "#7f8c8d"
accent_blue   <- "#2980b9"
highlight_purple <- "#8e44ad"
alert_red     <- "#c0392b"
```

### Custom Theme
```r
theme_custom <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", color = primary_dark),
      legend.position = "bottom"
    )
}
```

### Figure Dimensions for Beamer
```r
ggsave(filepath, width = 12, height = 5, bg = "transparent")
```

## 5. RDS Data Pattern

**Heavy computations saved as RDS; slide rendering loads pre-computed data.**

```r
saveRDS(result, file.path(out_dir, "descriptive_name.rds"))
```

## 6. Common Pitfalls

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| Missing `bg = "transparent"` | White boxes on slides | Always include in ggsave() |
| Hardcoded paths | Breaks on other machines | Use relative paths |
| Computing mean P/E without winsorizing | Outliers dominate (P/E > 1000) | Winsorize first or use median |
| Dividing by zero equity | Inf/NaN in P/B ratios | Filter ceq > 0 before division |
| Mixing fiscal and calendar year | Misaligned financial data | Use fyear for Compustat |
| Ignoring currency in cross-country data | Apples-to-oranges comparisons | Use ratios or convert currencies |

## 7. Line Length & Mathematical Exceptions

**Standard:** Keep lines <= 100 characters.

**Exception: Mathematical Formulas** -- lines may exceed 100 chars **if and only if:**

1. Breaking the line would harm readability of the math (influence functions, matrix ops, finite-difference approximations, formula implementations matching paper equations)
2. An inline comment explains the mathematical operation:
   ```r
   # Sieve projection: inner product of residuals onto basis functions P_k
   alpha_k <- sum(r_i * basis[, k]) / sum(basis[, k]^2)
   ```
3. The line is in a numerically intensive section (simulation loops, estimation routines, inference calculations)

**Quality Gate Impact:**
- Long lines in non-mathematical code: minor penalty (-1 to -2 per line)
- Long lines in documented mathematical sections: no penalty

## 8. Code Quality Checklist

```
[ ] Packages at top via library()
[ ] set.seed() once at top
[ ] All paths relative
[ ] Functions documented (Roxygen)
[ ] Figures: transparent bg, explicit dimensions
[ ] RDS: every computed object saved
[ ] Comments explain WHY not WHAT
```
