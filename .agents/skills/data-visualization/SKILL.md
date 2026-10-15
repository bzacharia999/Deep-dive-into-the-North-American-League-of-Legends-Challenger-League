---
name: data-visualization
description: >-
  Use this skill whenever creating plots or charts with matplotlib/seaborn in this
  project (EDA, player/champion stats, model results). Defines the project color
  palette, default figure sizes, and a reusable plotting template.
---

# Data Visualization Best Practices

## Project Preset (always apply first)

Put this in `src/viz.py` and call `set_theme()` at the top of every notebook/script.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# League-inspired palette (Hextech gold / blue / Noxus red)
PALETTE = {
    "gold":   "#C89B3C",
    "blue":   "#0AC8B9",
    "navy":   "#0A1428",
    "red":    "#D13639",
    "purple": "#7B5EA7",
    "grey":   "#A09B8C",
    "light":  "#F0E6D2",
}
CATEGORICAL = [PALETTE[c] for c in ("gold", "blue", "red", "purple", "grey")]
SEQUENTIAL = sns.light_palette(PALETTE["gold"], as_cmap=True)         # low -> high
DIVERGING = sns.blend_palette([PALETTE["red"], PALETTE["light"], PALETTE["blue"]], as_cmap=True)
TEAM_COLORS = {"Blue": PALETTE["blue"], "Red": PALETTE["red"]}         # side-of-map
OUTCOME_COLORS = {True: PALETTE["blue"], False: PALETTE["red"]}        # win / loss

FIGSIZE = {"single": (10, 6), "wide": (14, 6), "square": (8, 8), "grid": (16, 10)}

def set_theme():
    sns.set_theme(style="whitegrid", context="notebook", palette=CATEGORICAL)
    plt.rcParams.update({
        "figure.figsize": FIGSIZE["single"],
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
```

## Reusable Plot Template

```python
from pathlib import Path

FIG_DIR = Path("figures")

def finalize(ax, title, xlabel=None, ylabel=None, source="Riot Games API", save_as=None):
    """Apply consistent titles/labels/source note and optionally save."""
    ax.set_title(title, loc="left")
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.figure.text(0.99, 0.01, f"Source: {source}", ha="right", fontsize=8, color=PALETTE["grey"])
    if save_as:
        FIG_DIR.mkdir(exist_ok=True)
        ax.figure.savefig(FIG_DIR / f"{save_as}.png")
    return ax

# Usage
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
sns.histplot(df, x="win_rate", bins=20, color=PALETTE["gold"], ax=ax)
ax.axvline(df["win_rate"].median(), color=PALETTE["red"], ls="--", label="Median")
ax.legend()
finalize(ax, "Win Rate Distribution – NA Top 100", "Win rate", "Players", save_as="winrate_hist")
plt.show()
```

## Rules of Thumb

1. **Pick the chart for the question**
   | Question | Chart |
   |---|---|
   | Distribution | `histplot` / `kdeplot` / `boxplot` / `violinplot` |
   | Compare categories | sorted horizontal `barplot` (long labels like champion names) |
   | Relationship | `scatterplot` / `regplot`, add `hue` sparingly |
   | Trend over time/patch | `lineplot` |
   | Correlation matrix | `heatmap` with `DIVERGING`, `center=0`, mask upper triangle |
2. **Always use the object-oriented API** (`fig, ax = plt.subplots()`), never bare `plt.plot` in reusable code. Functions should accept and return `ax`.
3. **Sort bars** by value; cap categories (e.g. top 15 champions) and group the rest as "Other".
4. **Bars start at zero**; don't truncate axes for bar charts.
5. **Titles state the insight** ("Junglers have the highest KDA"), not just the variables.
6. **Label units** (%, gold/min, LP) and format ticks: `ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))`.
7. **Colorblind-safe & consistent**: same entity = same color across all figures (use `TEAM_COLORS`, `OUTCOME_COLORS`). Never encode info by color alone for critical comparisons — add labels/markers.
8. **Reduce clutter**: no 3D, no pie charts with >4 slices, remove top/right spines, use `alpha` for overplotting (`alpha=0.4`).
9. **Annotate key points** with `ax.annotate` / `ax.bar_label(ax.containers[0], fmt="%.1f")`.
10. **Show sample size** (`n=100 players`) in subtitle or legend — small samples mislead.
11. **Close figures in loops** (`plt.close(fig)`) to avoid memory leaks.

## Correlation Heatmap Template

```python
import numpy as np

def corr_heatmap(df, cols, title="Feature Correlations"):
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=FIGSIZE["square"])
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    return finalize(ax, title)
```

## Verification
- Figure renders with gold/blue/red palette and no top/right spines.
- Saved files appear in `figures/` at 300 dpi.
