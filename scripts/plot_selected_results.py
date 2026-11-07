"""Render two selected scientific figures from published, rounded tables."""
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures"
NAMES = {"GAN v3": "GAN", "DDPM v3": "DDPM", "Interpolação": "Interpolation",
         "Centro da caixa": "Box center", "Jitter": "Jitter", "Cópias": "Copies"}
COLORS = ["#485fc7", "#9761c5", "#168679", "#d69a42", "#7e8994", "#c9ced5"]


def number(value):
    return float(value.replace("−", "-").replace(",", "."))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    with (ROOT / "selected_results/gains.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for dataset, ax in enumerate(axes):
        values = [number(row[dataset + 1]) for row in rows]
        ax.bar(np.arange(len(rows)), values, color=COLORS)
        ax.axhline(0, color="#354052", linewidth=.8)
        ax.set_title(f"H{dataset+1}", fontweight="bold")
        ax.set_xticks(np.arange(len(rows)), [NAMES[row[0]] for row in rows], rotation=45, ha="right")
        ax.set_ylim(-.28, 2.35)
        for i, value in enumerate(values):
            ax.text(i, value + .05 if value >= 0 else value - .06, f"{value:+.2f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    axes[0].set_ylabel("Macro-F1 gain over real-only (pp)")
    fig.suptitle("Simple interpolation exceeds mean neural augmentation gains", fontsize=14, fontweight="bold")
    fig.text(.5, .015, "Source: published gains.csv; v3, 100 seeds × 5 training sizes per dataset. Rounded aggregates.",
             ha="center", fontsize=9, color="#526074")
    fig.tight_layout(rect=(0, .055, 1, .92))
    fig.savefig(OUTPUT / "gains.png", dpi=160, facecolor="white")
    plt.close(fig)
    with (ROOT / "selected_results/contrasts.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, row in enumerate(rows):
        mean, _, _, lo, hi = re.match(r"([−+\d,]+) \((\d)/(\d)\) \[([−+\d,]+); ([−+\d,]+)\]", row[3]).groups()
        mean, lo, hi = map(number, (mean, lo, hi))
        ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", capsize=4,
                    color=COLORS[0 if row[0].startswith("GAN") else 1], markersize=7)
    ax.axvline(0, color="#354052", linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{NAMES[row[0]]} / {row[1]}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Network − interpolation macro-F1 gain (pp), 95% bootstrap interval")
    ax.set_title("Neither network establishes superiority to interpolation", fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=.15)
    fig.text(.5, .025, "Source: contrasts.csv; 10,000 seed resamples within training sizes; descriptive, no multiplicity correction.",
             ha="center", fontsize=9, color="#526074")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUTPUT / "contrasts.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
