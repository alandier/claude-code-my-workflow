# Workflow Quick Reference

**Model:** Contractor (you direct, Claude orchestrates)

---

## The Loop

```
Your instruction
    |
[PLAN] (if multi-file or unclear) -> Show plan -> Your approval
    |
[EXECUTE] Implement, verify, done
    |
[REPORT] Summary + what's ready
    |
Repeat
```

---

## I Ask You When

- **Design forks:** "Option A (fast) vs. Option B (robust). Which?"
- **Code ambiguity:** "Spec unclear on X. Assume Y?"
- **Replication edge case:** "Just missed tolerance. Investigate?"
- **Scope question:** "Also refactor Y while here, or focus on X?"

---

## I Just Execute When

- Code fix is obvious (bug, pattern application)
- Verification (tolerance checks, tests, compilation)
- Documentation (logs, commits)
- Plotting (per established standards)
- Deployment (after you approve, I ship automatically)

---

## Quality Gates (No Exceptions)

| Score | Action |
|-------|--------|
| >= 80 | Ready to commit |
| < 80  | Fix blocking issues |

---

## Non-Negotiables

- **Paths:** `pathlib.Path` everywhere; `DATA_DIR = Path("~/Dropbox/Research_Data/GlobalValuation/").expanduser()`
- **Seed:** `np.random.seed(YYYYMMDD)` once at top for any stochastic code
- **Figure standards:** white background, 300 DPI, PDF + PNG dual export, 6x4 inches default
- **Color palette:** muted blues/grays (`#2c3e50`, `#7f8c8d`, `#2980b9`, `#8e44ad`, `#c0392b`)
- **Data precision:** valuation ratios to 3 decimals, returns to 4 decimals (basis points)
- **Memory:** always use `usecols`/`dtype` when loading large CSVs; prefer parquet after first load

---

## Preferences

**Visual:** Publication-quality figures (JF/RFS/JFE standard); PDF for paper, PNG for quick inspection
**Reporting:** Concise bullets; details on request
**Session logs:** Always (post-plan, incremental, end-of-session)
**Data handling:** Memory-conscious -- 5.5 GB across 4 raw files; chunk or subset when possible

---

## Exploration Mode

For experimental work, use the **Fast-Track** workflow:
- Work in `explorations/` folder
- 60/100 quality threshold (vs. 80/100 for production)
- No plan needed -- just a research value check (2 min)
- See `.claude/rules/exploration-fast-track.md`

---

## Next Step

You provide task -> I plan (if needed) -> Your approval -> Execute -> Done.
