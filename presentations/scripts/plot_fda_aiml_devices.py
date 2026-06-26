from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lymphnode")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "assets" / "rendered_figures" / "aiml-devices-csv.csv"
FIG_DIR = ROOT / "figures"

BLUE = "#b7d7f0"
RED = "#c4332b"
GRAY = "#7f7f7f"
GREEN = "#1d9163"
ORANGE = "#e17e3a"
INK = "#222222"
MUTED = "#333333"
GRID = "#d6d6d6"


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["_date"] = datetime.strptime(row["Date of Final Decision"], "%m/%d/%Y")
            rows.append(row)
    return rows


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=240)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9, linestyle="--")
    ax.set_axisbelow(True)


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.965, title, fontsize=14, fontweight="bold",
             color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.06, 0.915, subtitle, fontsize=9.5, color=MUTED,
                 ha="left", va="top")


def add_source(fig: plt.Figure, source: str) -> None:
    fig.text(0.06, 0.055, source, fontsize=8.5, color=MUTED,
             ha="left", va="bottom")


def plot_yearly(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["_date"].year for row in rows)
    years = list(range(2015, max(counts) + 1))
    annual = [counts.get(year, 0) for year in years]
    cumulative = []
    total = sum(value for year, value in counts.items() if year < years[0])
    for value in annual:
        total += value
        cumulative.append(total)

    fig, ax = plt.subplots(figsize=(4.75, 3.25))
    colors = [BLUE if year < 2026 else "#b8b8b8" for year in years]
    ax.bar(years, annual, color=colors, edgecolor="none", linewidth=0, width=0.48)

    ax2 = ax.twinx()
    ax2.plot(years, cumulative, color=RED, linewidth=2.4, marker="o", markersize=3.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.yaxis.set_major_locator(MaxNLocator(5, integer=True))

    style_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(6, integer=True))
    add_title(
        fig,
        "FDA AI/ML Medical Devices by Year",
        None,
    )
    ax.set_ylabel("Decisions / year", color=MUTED, fontsize=8.5)
    ax2.set_ylabel("Cumulative", color=MUTED, fontsize=8.5)
    ax.set_xlim(min(years) - 0.45, max(years) + 0.45)
    ax.set_xticks([2015, 2018, 2021, 2024, 2026])
    add_source(fig, "Source: FDA Artificial Intelligence-Enabled Medical Devices list.")
    fig.subplots_adjust(left=0.16, right=0.86, top=0.80, bottom=0.24)
    save_all(fig, "fda_aiml_devices_by_year")
    plt.close(fig)


def plot_panel(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["Panel (Lead)"] or "Unknown" for row in rows)
    top = counts.most_common(8)
    top_total = sum(value for _, value in top)
    other = len(rows) - top_total
    labels = [name for name, _ in top] + ["Other panels"]
    values = [value for _, value in top] + [other]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    y = list(range(len(labels)))
    colors = [BLUE] + [GREEN] * 2 + [ORANGE] * (len(labels) - 3)
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Number of FDA-listed AI/ML-enabled devices", color=MUTED, fontsize=10)
    add_title(
        fig,
        "AI-Enabled Medical Devices Span Multiple Clinical Panels",
        "The FDA list is dominated by radiology, but cardiovascular and neurology devices are already substantial.",
    )

    for yi, value in zip(y, values):
        pct = value / len(rows) * 100
        ax.text(value + max(values) * 0.012, yi, f"{value:,} ({pct:.1f}%)",
                va="center", fontsize=9, color=INK)

    ax.set_xlim(0, max(values) * 1.18)
    add_source(fig, "Source: FDA Artificial Intelligence-Enabled Medical Devices list.")
    fig.subplots_adjust(left=0.23, right=0.95, top=0.82, bottom=0.18)
    save_all(fig, "fda_aiml_devices_by_panel")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    plot_yearly(rows)
    plot_panel(rows)
    print(f"Loaded {len(rows):,} FDA device rows from {CSV_PATH}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
