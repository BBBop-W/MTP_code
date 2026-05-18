from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mtp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.dynamic_segmentation import CarriageGeometry


PALETTE = [
    "#F59E0B",  # amber
    "#14B8A6",  # teal
    "#3B82F6",  # blue
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#22C55E",  # green
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#A855F7",  # purple
    "#84CC16",  # lime
]


def _clearance_curve(geom: CarriageGeometry, layer: str, mode: str, x: np.ndarray) -> np.ndarray:
    return np.array([geom.get_clearance(float(v), layer, mode) for v in x])


def _segment_ranges(geom: CarriageGeometry, layer: str, side_blocks: int):
    segments = geom.generate_even_side_segments(side_blocks, layer)
    central = segments[0]
    blocks = segments[1:]

    central_len = central["len"]
    central_start = geom.center_x - central_len / 2.0
    central_end = geom.center_x + central_len / 2.0
    ranges = [("central", 0, central_start, central_end, central)]

    cursor = central_start
    for block_idx, block in enumerate(blocks, start=1):
        left_end = cursor
        left_start = left_end - block["len"]
        right_start = geom.carriage_length - left_end
        right_end = geom.carriage_length - left_start
        ranges.append((f"b{block_idx} left", block_idx, left_start, left_end, block))
        ranges.append((f"b{block_idx} right", block_idx, right_start, right_end, block))
        cursor = left_start
    return ranges


def _draw_cell(
    ax,
    geom: CarriageGeometry,
    layer: str,
    mode: str,
    side_blocks: int,
    show_xlabel: bool,
) -> None:
    x = np.linspace(0.0, geom.carriage_length, 1600)
    h = _clearance_curve(geom, layer, mode, x)
    ymax = max(2350.0, float(np.max(h)) + 120.0)

    ax.set_facecolor("#FBFCFE")
    ax.fill_between(x, 0.0, h, color="#CBD5E1", alpha=0.36, lw=0)

    for name, block_idx, x0, x1, block in _segment_ranges(geom, layer, side_blocks):
        height = block["h_m"] if mode == "m" else block["h_h"]
        color = "#FBBF24" if name == "central" else PALETTE[(block_idx - 1) % len(PALETTE)]
        alpha = 0.72 if name == "central" else 0.63
        rect = Rectangle(
            (x0, 0.0),
            x1 - x0,
            height,
            facecolor=color,
            edgecolor="white",
            linewidth=1.4,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.plot([x0, x1], [height, height], color="#1E293B", lw=1.0, alpha=0.52, zorder=3)
        ax.axvline(x0, color="white", lw=0.9, alpha=0.82, zorder=4)
        ax.axvline(x1, color="white", lw=0.9, alpha=0.82, zorder=4)

        width = x1 - x0
        if (name == "central" and width > 1600) or (side_blocks <= 2 and width > 1050):
            label = "central" if name == "central" else name.replace(" ", "\n")
            ax.text(
                (x0 + x1) / 2.0,
                min(height * 0.52, ymax - 260),
                f"{label}\n{height:.0f} mm",
                ha="center",
                va="center",
                color="#0F172A",
                fontsize=8,
                weight="bold",
                zorder=5,
            )

    ax.plot(x, h, color="#111827", lw=2.2, zorder=6)
    ax.plot(x, h, color="#FFFFFF", lw=0.8, alpha=0.75, zorder=7)

    ax.text(
        0.018,
        0.90,
        f"{side_blocks} side block{'s' if side_blocks > 1 else ''}\n{2 * side_blocks + 1} rectangles",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.92,
        },
        zorder=10,
        clip_on=False,
    )

    ax.set_xlim(0, geom.carriage_length)
    ax.set_ylim(0, ymax)
    ax.grid(True, color="#E2E8F0", lw=0.7, zorder=1)
    ax.tick_params(axis="both", labelsize=8, colors="#334155")
    if show_xlabel:
        ax.set_xlabel("x position along wagon (mm)", fontsize=9, color="#334155")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("clearance H(x) (mm)", fontsize=9, color="#334155")


def draw_progression(side_blocks: Iterable[int], output_dir: Path) -> tuple[Path, Path]:
    geom = CarriageGeometry()
    side_blocks = list(side_blocks)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "segmentation_progression_rectangles"
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"

    fig, axes = plt.subplots(
        len(side_blocks),
        2,
        figsize=(14.5, 2.65 * len(side_blocks) + 1.0),
        sharex=True,
        constrained_layout=False,
    )
    if len(side_blocks) == 1:
        axes = np.array([axes])

    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        "Dynamic segmentation as a rectangle approximation of irregular clearance",
        fontsize=16,
        weight="bold",
        color="#0F172A",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        "Gray area = true continuous clearance; colored blocks = rectangular capacities used by the model.",
        ha="center",
        va="top",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.27,
        0.912,
        "Lower deck under h deck: floor slope",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#0F172A",
    )
    fig.text(
        0.74,
        0.912,
        "Upper deck over m deck: tilted deck",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#0F172A",
    )

    for row, n_blocks in enumerate(side_blocks):
        show_xlabel = row == len(side_blocks) - 1
        _draw_cell(
            axes[row, 0],
            geom,
            layer="lower",
            mode="h",
            side_blocks=n_blocks,
            show_xlabel=show_xlabel,
        )
        _draw_cell(
            axes[row, 1],
            geom,
            layer="upper",
            mode="m",
            side_blocks=n_blocks,
            show_xlabel=show_xlabel,
        )

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.06, top=0.885, hspace=0.25, wspace=0.12)

    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-blocks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "result" / "geometry"))
    args = parser.parse_args()

    png_path, svg_path = draw_progression(args.side_blocks, Path(args.output_dir))
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
