from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lymphnode")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

INK = "#1f2937"
MUTED = "#667085"
GRID = "#d0d5dd"
BLUE = "#275eb0"
GREEN = "#1d9163"
ORANGE = "#e17e3a"
RED = "#b84545"
PURPLE = "#6e58b0"
PANEL = "#f6f8fb"
WHITE = "#ffffff"


def rounded_box(ax, xy, width, height, label, color=BLUE, face=WHITE,
                fontsize=11, weight="bold", text_color=INK, lw=1.8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=lw, edgecolor=color, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, label,
        ha="center", va="center", fontsize=fontsize,
        color=text_color, fontweight=weight, linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, color=INK, lw=1.8, rad=0.0):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)
    return patch


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=240)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.055, 0.955,
        "Edge-Deployed DNNs Face Many Potential Threats",
        ha="left", va="top", fontsize=17, fontweight="bold", color=INK,
    )
    fig.text(
        0.055, 0.905,
        "Once the model runs outside the cloud, attackers can reach more surfaces around the model and device.",
        ha="left", va="top", fontsize=10.5, color=MUTED,
    )

    ax.add_patch(Circle((0.50, 0.48), 0.315, linewidth=1.2,
                        edgecolor=GRID, facecolor="#f9fafb", alpha=0.95))

    rounded_box(
        ax, (0.365, 0.35), 0.27, 0.25,
        "",
        color=BLUE, face="#eef4ff", fontsize=13, text_color=INK, weight="bold", lw=2.0,
    )
    ax.text(0.50, 0.535, "Edge DNN\nmodel + device",
            ha="center", va="center", fontsize=13, color=INK,
            fontweight="bold", linespacing=1.15)
    ax.add_patch(Rectangle((0.45, 0.405), 0.10, 0.065, linewidth=1.6,
                           edgecolor=BLUE, facecolor=WHITE))
    for i in range(6):
        x = 0.437 + i * 0.022
        ax.plot([x, x], [0.395, 0.405], color=BLUE, lw=1)
        ax.plot([x, x], [0.470, 0.480], color=BLUE, lw=1)

    threats = [
        ("Model\nextraction", (0.125, 0.63), RED, (0.365, 0.525)),
        ("Model\ninversion", (0.125, 0.35), PURPLE, (0.365, 0.445)),
        ("Adversarial\nqueries", (0.395, 0.13), ORANGE, (0.470, 0.35)),
        ("Artifact\ntheft", (0.755, 0.35), BLUE, (0.635, 0.445)),
        ("Tampering /\nfine-tuning", (0.755, 0.63), RED, (0.635, 0.525)),
        ("Side-channel\nleakage", (0.395, 0.70), GREEN, (0.470, 0.60)),
    ]
    for label, (x, y), color, target in threats:
        rounded_box(ax, (x, y), 0.17, 0.085, label, color=color,
                    face=WHITE, fontsize=10.6, text_color=INK, weight="bold", lw=1.6)
        arrow(ax, (x + 0.085, y + 0.042), target, color="#98a2b3", lw=1.35)

    fig.text(
        0.055, 0.035,
        "Motivation figure: local deployment creates multiple attack opportunities around the model.",
        ha="left", va="bottom", fontsize=8.5, color=MUTED,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_all(fig, "dnn_edge_attack_surface")
    plt.close(fig)
    print(f"Wrote attack surface figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
