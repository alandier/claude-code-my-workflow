# CLAUDE.MD -- Academic Project Development with Claude Code

**Project:** Global Valuation
**Author:** Augustin Landier, HEC Paris
**Branch:** main

---

## Core Principles

- **Plan first** -- enter plan mode before non-trivial tasks; save plans to `quality_reports/plans/`
- **Verify after** -- compile/render and confirm output at the end of every task
- **Single source of truth** -- Paper `.tex` is authoritative; presentations derive from it
- **Quality gates** -- nothing ships below 80/100
- **[LEARN] tags** -- when corrected, save `[LEARN:category] wrong -> right` to MEMORY.md

---

## Folder Structure

```
global-valuation/
├── CLAUDE.MD                    # This file
├── .claude/                     # Rules, skills, agents, hooks
├── Paper/                       # Main paper LaTeX source
├── Bibliography_base.bib        # Centralized bibliography
├── Figures/                     # Generated figures (PDF + PNG)
├── Preambles/header.tex         # LaTeX headers
├── Slides/                      # Beamer .tex (presentations, secondary)
├── Quarto/                      # RevealJS .qmd (presentations, secondary)
├── docs/                        # GitHub Pages (auto-generated)
├── scripts/python/              # Python analysis scripts
├── scripts/                     # Utility scripts
├── quality_reports/             # Plans, session logs, merge reports
├── explorations/                # Research sandbox (see rules)
├── templates/                   # Session log, quality report templates
└── master_supporting_docs/      # Papers and existing slides
```

**External data (not in repo):** `~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/`
- `compustat_global.csv` (2.3 GB, 981K rows, 468 cols — Global annual fundamentals, `gvkey` lowercase)
- `compustat_america.csv` (0.8 GB, 254K rows, 984 cols — NA fundamentals + CRSP link, `GVKEY` uppercase, has `mkvalt`/`prcc_f`)
- `returns_global.csv` (1.9 GB, 10.4M rows, 25 cols — Global monthly prices, no computed returns)
- `returns_america.csv` (0.9 GB, 3.6M rows, 55 cols — CRSP monthly, `PERMNO`/`RET`/`PRC`/`SHROUT`)
- ~5.9 GB total -- always use `usecols`/`dtype` to limit memory

---

## Commands

```bash
# LaTeX paper compilation (3-pass, XeLaTeX)
cd Paper && TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode paper.tex
BIBINPUTS=..:$BIBINPUTS bibtex paper
TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode paper.tex
TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode paper.tex

# Python analysis
python3 scripts/python/script_name.py

# Quality score
python scripts/quality_score.py Paper/paper.tex
python scripts/quality_score.py scripts/python/script.py

# Beamer slides (secondary)
cd Slides && TEXINPUTS=../Preambles:$TEXINPUTS xelatex -interaction=nonstopmode file.tex
```

---

## Quality Thresholds

| Score | Gate | Meaning |
|-------|------|---------|
| 80 | Commit | Good enough to save |
| 90 | PR | Ready for deployment |
| 95 | Excellence | Aspirational |

---

## Skills Quick Reference

| Command | What It Does |
|---------|-------------|
| `/compile-latex [file]` | 3-pass XeLaTeX + bibtex |
| `/data-analysis [dataset]` | End-to-end R analysis |
| `/data-analysis-python [dataset]` | End-to-end Python analysis |
| `/review-r [file]` | R code quality review |
| `/proofread [file]` | Grammar/typo/overflow review |
| `/validate-bib` | Cross-reference citations |
| `/commit [msg]` | Stage, commit, PR, merge |
| `/lit-review [topic]` | Literature search + synthesis |
| `/research-ideation [topic]` | Research questions + strategies |
| `/interview-me [topic]` | Interactive research interview |
| `/review-paper [file]` | Manuscript review |
| `/deploy [LectureN]` | Render Quarto + sync to docs/ |
| `/translate-to-quarto [file]` | Beamer -> Quarto translation |
| `/slide-excellence [file]` | Combined multi-agent review |

---

## LaTeX Custom Environments

To be defined as paper develops.

## Quarto CSS Classes

To be defined if presentations are created.

---

## Current Project State

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Paper | `Paper/` | Not started | Cross-country valuation multiples |
| Data exploration | `scripts/python/01_explore_data.py` | Done | Schema, coverage, missingness, P/B diagnostics |
| Data pipeline | `scripts/python/` | Not started | Compustat + CRSP processing |
