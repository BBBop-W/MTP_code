from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mtp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.dynamic_segmentation import CarriageGeometry, get_model_segments


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _load_car_info(instance_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "Instance" / instance_name / "cars.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cars.csv for instance {instance_name}: {path}")
    df = _clean_columns(pd.read_csv(path))
    if "Height" in df.columns and "height" not in df.columns:
        df = df.rename(columns={"Height": "height"})
    if "height" not in df.columns:
        raise ValueError(f"{path} must contain a height or Height column")
    df["height"] = pd.to_numeric(df["height"])
    return df


def _mirror_x(geom: CarriageGeometry, x: np.ndarray) -> np.ndarray:
    return np.where(x > geom.center_x, geom.carriage_length - x, x)


def _deck_profile(geom: CarriageGeometry, x: np.ndarray, mode: str) -> np.ndarray:
    xm = _mirror_x(geom, x)
    return np.array([geom._deck_height(float(v), mode) for v in xm])


def _floor_profile(geom: CarriageGeometry, x: np.ndarray) -> np.ndarray:
    xm = _mirror_x(geom, x)
    return np.array([geom._floor_height(float(v)) for v in xm])


def _segment_ranges(geom: CarriageGeometry, layer: str, layer_segments: dict):
    central_len = layer_segments["central"]["len"]
    central_start = geom.center_x - central_len / 2.0
    central_end = geom.center_x + central_len / 2.0

    ranges = [("central", central_start, central_end, layer_segments["central"])]
    cursor = central_start
    for block in layer_segments["blocks"]:
        left_end = cursor
        left_start = left_end - block["len"]
        right_start = geom.carriage_length - left_end
        right_end = geom.carriage_length - left_start
        ranges.append((f"{block['name']}_left", left_start, left_end, block))
        ranges.append((f"{block['name']}_right", right_start, right_end, block))
        cursor = left_start
    return ranges


def _annotate_bracket(ax, x0, x1, y, text, color):
    ax.annotate(
        "",
        xy=(x0, y),
        xytext=(x1, y),
        arrowprops={"arrowstyle": "|-|", "lw": 1.2, "color": color},
    )
    ax.text((x0 + x1) / 2.0, y + 55, text, ha="center", va="bottom", fontsize=9, color=color)


def _draw_profile(ax, geom: CarriageGeometry):
    x = np.linspace(0.0, geom.carriage_length, 1200)
    floor = _floor_profile(geom, x)
    deck_h = _deck_profile(geom, x, "h")
    deck_m = _deck_profile(geom, x, "m")
    roof = np.full_like(x, geom.roof_height)

    ax.fill_between(x, floor, deck_h, color="#9dd9d2", alpha=0.25, label="lower clearance under h deck")
    ax.fill_between(x, deck_h, roof, color="#b8c0ff", alpha=0.18, label="upper clearance over h deck")
    ax.plot(x, roof, color="#2f2f2f", lw=2.2, label="roof y=4340")
    ax.plot(x, deck_h, color="#006d77", lw=2.0, label="deck h y=2270")
    ax.plot(x, deck_m, color="#e76f51", lw=2.2, label="deck m profile")
    ax.plot(x, floor, color="#6b4f2a", lw=2.2, label="floor profile")

    deck_left = geom.deck_m_groove_start
    deck_right = geom.carriage_length - geom.deck_m_groove_start
    floor_left = geom.floor_groove_start
    floor_right = geom.carriage_length - geom.floor_groove_start
    slope_left = geom.floor_slope_start
    slope_right = geom.carriage_length - geom.floor_slope_start

    for xpos, label, color, ytext in [
        (slope_left, "floor slope start\n2773.2", "#6b4f2a", 310),
        (deck_left, "deck m flat start\n6800", "#e76f51", 3740),
        (floor_left, "floor flat start\n7066.5", "#6b4f2a", 680),
        (geom.center_x, "center\n12500", "#555555", 4280),
        (floor_right, "floor flat end\n17933.5", "#6b4f2a", 680),
        (deck_right, "deck m flat end\n18200", "#e76f51", 3740),
        (slope_right, "floor slope end\n22226.8", "#6b4f2a", 310),
    ]:
        ax.axvline(xpos, color=color, lw=0.8, ls="--", alpha=0.55)
        ax.text(xpos, ytext, label, ha="center", va="bottom", fontsize=8, color=color)

    _annotate_bracket(ax, 0, geom.carriage_length, -390, "carriage length = 25000 mm", "#333333")
    _annotate_bracket(ax, deck_left, deck_right, 3220, "deck m central flat = 11400 mm", "#e76f51")
    _annotate_bracket(ax, floor_left, floor_right, 145, "floor central flat = 10867 mm", "#6b4f2a")

    text = "\n".join(
        [
            "Clearance values from code:",
            "lower h: end 1590, center 2270",
            "lower m: end 1880, center 2270",
            "upper h: 2070 everywhere",
            "upper m: end 1780, center 2070",
            "floor slope angle = 9 deg",
        ]
    )
    ax.text(
        420,
        4050,
        text,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    ax.set_title("A. Physical cross-section profile encoded in CarriageGeometry", loc="left", fontsize=12, weight="bold")
    ax.set_xlim(-250, geom.carriage_length + 250)
    ax.set_ylim(-520, 4620)
    ax.set_ylabel("absolute height y (mm)")
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(loc="lower right", ncol=2, fontsize=8, frameon=True)


def _draw_segments(ax, geom: CarriageGeometry, segments: dict, instance_name: str, num_splits: int, indep: bool):
    ax.set_title(
        f"B. Model dynamic segments from {instance_name} "
        f"(num_splits={num_splits}, independent_mode_split={indep})",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlim(-250, geom.carriage_length + 250)
    ax.set_ylim(-0.25, 2.4)
    ax.set_yticks([])
    ax.set_xlabel("wagon length x (mm)")
    ax.grid(True, axis="x", color="#e8e8e8", lw=0.6)

    cmap = plt.get_cmap("viridis")
    min_h, max_h = 1550.0, 2300.0

    def color_for(limit):
        t = min(1.0, max(0.0, (float(limit) - min_h) / (max_h - min_h)))
        return cmap(t)

    for row, layer in [(1.35, "lower"), (0.45, "upper")]:
        ranges = _segment_ranges(geom, layer, segments[layer])
        ax.text(-190, row + 0.17, layer, ha="right", va="center", fontsize=11, weight="bold")
        for name, x0, x1, block in ranges:
            width = x1 - x0
            rect = Rectangle(
                (x0, row),
                width,
                0.35,
                facecolor=color_for(block["h_h"]),
                edgecolor="#333333",
                lw=0.7,
                alpha=0.82,
            )
            ax.add_patch(rect)
            if width >= 700:
                short = "C" if name == "central" else name.replace("block_", "b").replace("_left", "L").replace("_right", "R")
                label = f"{short}\nL={block['len']:.0f}\nh/m={block['h_h']:.0f}/{block['h_m']:.0f}"
                fs = 7.5 if width < 1350 else 8.5
                ax.text(x0 + width / 2.0, row + 0.175, label, ha="center", va="center", fontsize=fs, color="white")

    ax.text(
        0,
        2.13,
        "Each side block is mirrored left/right. Color follows h-deck height limit; labels show h/m limits.",
        ha="left",
        va="center",
        fontsize=9,
        color="#333333",
    )


def draw(instance_name: str, num_splits: int, independent_mode_split: bool, output_dir: Path) -> tuple[Path, Path]:
    geom = CarriageGeometry()
    car_info = _load_car_info(instance_name)
    segments = get_model_segments(car_info[["height"]], num_splits, independent_mode_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"carriage_geometry_{instance_name}_splits{num_splits}_indep{int(independent_mode_split)}"
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10.5),
        gridspec_kw={"height_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )
    fig.suptitle("JSQ6 carriage internal geometry used by the current code", fontsize=16, weight="bold")
    _draw_profile(axes[0], geom)
    _draw_segments(axes[1], geom, segments, instance_name, num_splits, independent_mode_split)

    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="m11c11")
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--independent-mode-split", type=int, choices=[0, 1], default=1)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "result" / "geometry"))
    args = parser.parse_args()

    png_path, svg_path = draw(
        instance_name=args.instance,
        num_splits=args.num_splits,
        independent_mode_split=bool(args.independent_mode_split),
        output_dir=Path(args.output_dir),
    )
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
