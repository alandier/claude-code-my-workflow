"""
Trajectory plot: each country's path through (P/E, Expected Return) space over time.
Uses country-year MEDIANS of forward P/E and expected return (no regression).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

OUT = Path("output/regressions")

# --- Load firm-level panel ---
panel = pd.read_parquet(OUT / "forward_pe_country_data.parquet")
# Check columns — we need the firm-level data, not the country aggregates
print(f"Country-level parquet: {panel.columns.tolist()}")

# The firm-level panel is built inside 04_forward_pe_baseline.py but not saved
# as a separate file. Let's rebuild medians from the raw IBES data.
# Actually, let's check if we can load the panel from a cached intermediate.

# Better approach: reload the panel from the baseline script's intermediate output.
# The script saves country_effects_fwd_baseline.parquet (year-by-year FE) but not
# the firm-level panel. Let's compute medians directly from the IBES-merged data.

# --- Rebuild firm-level panel (lightweight version) ---
print("\nRebuilding firm-level panel for medians...")

import sys
sys.path.insert(0, "scripts/python")

DATA_DIR = Path("~/Augustin Landier Dropbox/augustin landier/Valuation_Global/Data_Global/").expanduser()
CACHE = OUT / "wrds_cache"

# Load cached IBES data
def load_cached(name):
    f = CACHE / f"{name}.parquet"
    return pd.read_parquet(f)

link = pd.concat([load_cached("iclink_ccm_link"), load_cached("ibtic_gvkey_link")],
                 ignore_index=True).drop_duplicates("ticker")

# FY1 consensus (columns: ticker, statpers, fpedats, fy1_eps, numest)
fy1 = pd.concat([load_cached("ibes_us_fy1"), load_cached("ibes_int_fy1")], ignore_index=True)
fy1 = fy1.merge(link[["ticker", "gvkey"]], on="ticker", how="inner")
fy1["statpers"] = pd.to_datetime(fy1["statpers"])
fy1["fyear"] = fy1["statpers"].dt.year
fy1 = fy1.sort_values(["gvkey", "fyear", "statpers"]).drop_duplicates(
    ["gvkey", "fyear"], keep="last")

# Prices from pansum (columns: ticker, statpers, price, shout, curr_price, usfirm)
pansum = load_cached("ibes_pansum")
pansum["statpers"] = pd.to_datetime(pansum["statpers"])
pansum = pansum.merge(link[["ticker", "gvkey"]], on="ticker", how="inner")
pansum = pansum.sort_values(["gvkey", "statpers"]).drop_duplicates(
    ["gvkey", "statpers"], keep="last")

# Price targets from ptgsum (columns: ticker, statpers, meanptg, medptg, numest, curr, usfirm)
ptgsum = load_cached("ibes_ptgsum")
ptgsum["statpers"] = pd.to_datetime(ptgsum["statpers"])
ptgsum = ptgsum.merge(link[["ticker", "gvkey"]], on="ticker", how="inner")
ptgsum = ptgsum.sort_values(["gvkey", "statpers"]).drop_duplicates(
    ["gvkey", "statpers"], keep="last")

# Merge FY1 with pansum price at same statpers
fy1 = fy1.merge(pansum[["gvkey", "statpers", "price"]].rename(columns={"price": "ibes_price"}),
                on=["gvkey", "statpers"], how="inner")

# Forward P/E = price / EPS forecast
fy1["fwd_pe"] = fy1["ibes_price"] / fy1["fy1_eps"]
fy1 = fy1[(fy1["fy1_eps"] > 0) & (fy1["fwd_pe"] > 0) & (fy1["numest"] >= 3)]

# Expected return from price targets
fy1 = fy1.merge(ptgsum[["gvkey", "statpers", "medptg"]],
                on=["gvkey", "statpers"], how="left")
fy1["exp_ret"] = fy1["medptg"] / fy1["ibes_price"] - 1

# Winsorize
lo_pe, hi_pe = fy1["fwd_pe"].quantile([0.01, 0.99])
fy1["fwd_pe"] = fy1["fwd_pe"].clip(lo_pe, hi_pe)

vals_er = fy1["exp_ret"].dropna()
lo_er, hi_er = vals_er.quantile([0.025, 0.975])
fy1["exp_ret"] = fy1["exp_ret"].clip(lo_er, hi_er)

# Get country from Compustat
us_fund = pd.read_csv(DATA_DIR / "compustat_america.csv",
                       usecols=["GVKEY", "fyear", "fic"],
                       dtype={"GVKEY": str}).rename(columns={"GVKEY": "gvkey"})
us_fund["fic"] = us_fund["fic"].fillna("USA")
gl_fund = pd.read_csv(DATA_DIR / "compustat_global.csv",
                       usecols=["gvkey", "fyear", "fic"],
                       dtype={"gvkey": str, "fyear": "Int64"})
fund = pd.concat([us_fund, gl_fund], ignore_index=True)
fund["fyear"] = pd.to_numeric(fund["fyear"], errors="coerce")
fund = fund.drop_duplicates(["gvkey", "fyear"], keep="first")

fy1 = fy1.merge(fund[["gvkey", "fyear", "fic"]], on=["gvkey", "fyear"], how="inner")
fy1 = fy1[fy1["fic"].notna() & (fy1["fic"] != "nan")]

print(f"Firm-level panel: {len(fy1):,} obs, {fy1['fic'].nunique()} countries, "
      f"years {fy1['fyear'].min()}-{fy1['fyear'].max()}")

# --- Compute country-year medians ---
MIN_FIRMS = 20

medians = fy1.groupby(["fic", "fyear"]).agg(
    med_pe=("fwd_pe", "median"),
    med_er=("exp_ret", "median"),
    n_firms=("fwd_pe", "size"),
).reset_index()

# Filter: enough firms per country-year
medians = medians[medians["n_firms"] >= MIN_FIRMS]

# Subtract USA median each year
usa = medians[medians["fic"] == "USA"][["fyear", "med_pe", "med_er"]].rename(
    columns={"med_pe": "usa_pe", "med_er": "usa_er"})
medians = medians.merge(usa, on="fyear", how="inner")
medians["pe_diff"] = medians["med_pe"] - medians["usa_pe"]
medians["er_diff"] = medians["med_er"] - medians["usa_er"]

print(f"Country-year medians: {len(medians):,} obs, {medians['fic'].nunique()} countries")

df = medians.copy()

# --- Focus countries ---
focus = {
    "JPN": {"color": "#e74c3c", "label": "Japan"},
    "KOR": {"color": "#3498db", "label": "Korea"},
    "GBR": {"color": "#2ecc71", "label": "UK"},
    "DEU": {"color": "#f39c12", "label": "Germany"},
    "FRA": {"color": "#9b59b6", "label": "France"},
    "CHN": {"color": "#e67e22", "label": "China"},
    "IND": {"color": "#1abc9c", "label": "India"},
    "BRA": {"color": "#34495e", "label": "Brazil"},
    "AUS": {"color": "#95a5a6", "label": "Australia"},
}

# Print summary for focus countries
print("\n--- Focus Country Medians vs USA (time average) ---")
print(f"{'Country':>8s}  {'P/E−USA':>8s}  {'ER−USA':>8s}  {'N years':>7s}")
for fic in focus:
    sub = df[df["fic"] == fic]
    if len(sub) > 0:
        print(f"{fic:>8s}  {sub['pe_diff'].mean():>+8.1f}  "
              f"{sub['er_diff'].mean():>+8.1%}  {len(sub):>7d}")

# ================================================================
# Figure 1: Individual country trajectories (3x3 small multiples)
# ================================================================
fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle("Country Trajectories vs USA: P/E and Expected Return",
             fontsize=11, y=0.98)

for ax, (fic, info) in zip(axes.flat, focus.items()):
    sub = df[df["fic"] == fic].dropna(subset=["pe_diff", "er_diff"]).sort_values("fyear").copy()
    if len(sub) < 5:
        ax.set_title(info["label"], fontsize=9)
        continue

    x = sub["pe_diff"].values.astype(float)
    y = sub["er_diff"].values.astype(float)
    years = sub["fyear"].values.astype(int)

    n = len(x)
    colors = cm.viridis(np.linspace(0.15, 0.95, n))

    # Reference lines at USA = (0, 0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    for i in range(n - 1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                color=colors[i], linewidth=1.2, alpha=0.7)

    # Start and end markers
    ax.scatter(x[0], y[0], color=colors[0], s=40, zorder=5,
               edgecolors="black", linewidths=0.5)
    ax.scatter(x[-1], y[-1], color=colors[-1], s=40, zorder=5,
               edgecolors="black", linewidths=0.5, marker="s")

    ax.annotate(f"{years[0]}", (x[0], y[0]), fontsize=6,
                xytext=(4, 4), textcoords="offset points", color=colors[0])
    ax.annotate(f"{years[-1]}", (x[-1], y[-1]), fontsize=6,
                xytext=(4, -8), textcoords="offset points", color=colors[-1])

    ax.set_title(info["label"], fontsize=9, fontweight="bold", color=info["color"])

for ax in axes[-1, :]:
    ax.set_xlabel("Median Forward P/E − USA", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("Median Expected Return − USA", fontsize=8)
for ax in axes.flat:
    ax.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"trajectory_pe_er_panels.{fmt}", dpi=300, bbox_inches="tight")
print("\nSaved trajectory_pe_er_panels")
plt.close()


# ================================================================
# Figure 2: All focus countries on one plot
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

# USA = origin reference lines
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.scatter(0, 0, color="black", s=60, zorder=6, marker="+", linewidths=1.5)
ax.annotate("USA", (0, 0), fontsize=7.5, fontweight="bold", color="black",
            xytext=(5, -10), textcoords="offset points")

for fic, info in focus.items():
    sub = df[df["fic"] == fic].dropna(subset=["pe_diff", "er_diff"]).sort_values("fyear").copy()
    if len(sub) < 5:
        continue

    x = sub["pe_diff"].values.astype(float)
    y = sub["er_diff"].values.astype(float)

    ax.plot(x, y, color=info["color"], linewidth=1.0, alpha=0.5)

    ax.scatter(x[0], y[0], color=info["color"], s=25, zorder=5,
               edgecolors="black", linewidths=0.3, alpha=0.7)
    ax.scatter(x[-1], y[-1], color=info["color"], s=35, zorder=5,
               edgecolors="black", linewidths=0.3, marker="s")

    offsets = {
        "JPN": (5, 3), "KOR": (5, -2), "GBR": (5, 3), "DEU": (-30, 5),
        "FRA": (5, -8), "CHN": (5, 3), "IND": (5, -5),
        "BRA": (5, 3), "AUS": (-25, -8),
    }
    ox, oy = offsets.get(fic, (5, 0))
    ax.annotate(info["label"], (x[-1], y[-1]), fontsize=7.5,
                fontweight="bold", color=info["color"],
                xytext=(ox, oy), textcoords="offset points")

ax.set_xlabel("Median Forward P/E − USA", fontsize=10)
ax.set_ylabel("Median Expected Return − USA", fontsize=10)
ax.set_title("Country Trajectories vs USA\n"
             "Circle = earliest year, Square = latest year", fontsize=10)
ax.tick_params(labelsize=8)

plt.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"trajectory_pe_er_combined.{fmt}", dpi=300, bbox_inches="tight")
print("Saved trajectory_pe_er_combined")
plt.close()


# ================================================================
# Figure 3: Clean arrows — centroid drift
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5))

# USA = origin reference lines
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.scatter(0, 0, color="black", s=60, zorder=6, marker="+", linewidths=1.5)
ax.annotate("USA", (0, 0), fontsize=8, fontweight="bold",
            color="black", xytext=(5, -10), textcoords="offset points", zorder=5)

for fic, info in focus.items():
    sub = df[df["fic"] == fic].dropna(subset=["pe_diff", "er_diff"]).sort_values("fyear").copy()
    if len(sub) < 5:
        continue

    x = sub["pe_diff"].values.astype(float)
    y = sub["er_diff"].values.astype(float)

    early = sub[sub["fyear"] <= 2013]
    late = sub[sub["fyear"] > 2013]
    if len(early) < 3 or len(late) < 3:
        continue

    x1, y1 = early["pe_diff"].mean(), early["er_diff"].mean()
    x2, y2 = late["pe_diff"].mean(), late["er_diff"].mean()

    # Scatter all points faintly
    ax.scatter(x, y, color=info["color"], s=8, alpha=0.15, zorder=2)

    # Arrow
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=info["color"],
                                lw=2.0, mutation_scale=14),
                zorder=4)

    offsets = {
        "JPN": (5, 5), "KOR": (5, 5), "GBR": (5, 5), "DEU": (-40, 5),
        "FRA": (-35, -8), "CHN": (5, -8), "IND": (5, -8),
        "BRA": (5, 5), "AUS": (-42, -5),
    }
    ox, oy = offsets.get(fic, (5, 0))
    ax.annotate(info["label"], (x2, y2), fontsize=8,
                fontweight="bold", color=info["color"],
                xytext=(ox, oy), textcoords="offset points", zorder=5)

ax.set_xlabel("Median Forward P/E − USA", fontsize=10)
ax.set_ylabel("Median Expected Return − USA", fontsize=10)
ax.set_title("Country Drift vs USA\n"
             "Arrow: 2002--2013 centroid → 2014--2025 centroid",
             fontsize=10)
ax.tick_params(labelsize=8)

plt.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"trajectory_pe_er_arrows.{fmt}", dpi=300, bbox_inches="tight")
print("Saved trajectory_pe_er_arrows")
plt.close()

print("\nDone — 3 figures saved.")
