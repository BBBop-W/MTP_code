from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    ROOT / "backup" / "sensitivity_small_results_2026-05-15_v2.csv",
    ROOT / "backup" / "sensitivity_medium_results_2026-05-15_v2.csv",
]
OUTPUT_DIR = ROOT / "result" / "backup_figures_2026-05-18_v2"


def load_data() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in INPUT_FILES]
    data = pd.concat(frames, ignore_index=True)

    numeric_columns = [
        "runtime_sec",
        "optional_acceptance_rate",
        "nominal_utilization",
        "node_count",
        "varied_value",
        "obj_val",
        "obj_bound",
        "loaded_length_m",
        "avg_len_per_wagon_mm",
        "avg_len_per_vehicle_mm",
        "mip_gap",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "runtime_sec" in data.columns and "obj_val" in data.columns and "obj_bound" in data.columns:
        data["relative_gap_pct"] = 100.0 * (data["obj_bound"] - data["obj_val"]).abs() / data["obj_val"].abs().clip(lower=1.0)
    else:
        data["relative_gap_pct"] = pd.NA

    data["timeout_flag"] = data["runner_status"].ne("worker_completed")
    return data


def min_max_scale(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - minimum) / (maximum - minimum)


def prepare_family_summary(data: pd.DataFrame, family: str, parameter: str) -> pd.DataFrame:
    subset = data[data["experiment_family"].eq(family)].copy()
    subset = subset[subset["objective_type"].eq("length")].copy()
    subset[parameter] = pd.to_numeric(subset[parameter], errors="coerce")

    grouped = (
        subset.groupby(["problem_scale", parameter], as_index=False)
        .agg(
            runtime_median=("runtime_sec", "median"),
            utilization_mean=("nominal_utilization", "mean"),
            loaded_length_median=("loaded_length_m", "median"),
            avg_vehicle_length_median=("avg_len_per_vehicle_mm", "median"),
            avg_wagon_length_median=("avg_len_per_wagon_mm", "median"),
            acceptance_mean=("optional_acceptance_rate", "mean"),
            timeout_rate=("timeout_flag", "mean"),
            runs=("instance_id", "count"),
        )
        .sort_values(["problem_scale", parameter])
    )

    grouped["loaded_length_norm"] = grouped.groupby("problem_scale")["loaded_length_median"].transform(min_max_scale)
    grouped["avg_vehicle_length_norm"] = grouped.groupby("problem_scale")["avg_vehicle_length_median"].transform(min_max_scale)
    grouped["avg_wagon_length_norm"] = grouped.groupby("problem_scale")["avg_wagon_length_median"].transform(min_max_scale)
    return grouped


def prepare_chunking_summary(data: pd.DataFrame) -> pd.DataFrame:
    subset = data[(data["experiment_family"].eq("chunking")) & (data["objective_type"].eq("length"))].copy()
    subset["num_splits"] = pd.to_numeric(subset["num_splits"], errors="coerce")

    grouped = (
        subset.groupby(["problem_scale", "num_splits"], as_index=False)
        .agg(
            runtime_median=("runtime_sec", "median"),
            relative_gap_median=("relative_gap_pct", "median"),
            obj_val_median=("obj_val", "median"),
            node_count_median=("node_count", "median"),
            timeout_rate=("timeout_flag", "mean"),
            runs=("instance_id", "count"),
        )
        .sort_values(["problem_scale", "num_splits"])
    )
    return grouped


def prepare_objective_summary(data: pd.DataFrame) -> pd.DataFrame:
    subset = data[data["experiment_family"].eq("objective")].copy()
    subset["num_types_I"] = pd.to_numeric(subset["num_types_I"], errors="coerce")

    grouped = (
        subset.groupby(["problem_scale", "objective_type"], as_index=False)
        .agg(
            obj_val_median=("obj_val", "median"),
            loaded_length_median=("loaded_length_m", "median"),
            avg_vehicle_length_median=("avg_len_per_vehicle_mm", "median"),
            avg_wagon_length_median=("avg_len_per_wagon_mm", "median"),
            utilization_mean=("nominal_utilization", "mean"),
            runs=("instance_id", "count"),
        )
        .sort_values(["problem_scale", "objective_type"])
    )
    return grouped


def apply_axis_style(ax: plt.Axes, ylabel: str, logy: bool = False, ylim: tuple[float, float] | None = None) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_density_family(grouped: pd.DataFrame, parameter: str, title: str, xlabel: str, filename: str) -> None:
    palette = {"small": "#0f766e", "medium": "#b45309"}
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)

    for scale, scale_frame in grouped.groupby("problem_scale"):
        color = palette.get(scale, "#2563eb")
        axes[0].plot(scale_frame[parameter], scale_frame["utilization_mean"], marker="o", linewidth=2.2, markersize=5.5, color=color, label=scale)
        axes[1].plot(scale_frame[parameter], scale_frame["loaded_length_norm"], marker="o", linewidth=2.2, markersize=5.5, color=color, label=scale)
        axes[2].plot(scale_frame[parameter], scale_frame["avg_vehicle_length_norm"], marker="o", linewidth=2.2, markersize=5.5, color=color, label=scale)

    apply_axis_style(axes[0], "nominal utilization", ylim=(0.0, 1.0))
    apply_axis_style(axes[1], "loaded length, min-max normalized", ylim=(0.0, 1.0))
    apply_axis_style(axes[2], "avg vehicle load length, min-max normalized", ylim=(0.0, 1.0))
    axes[2].set_xlabel(xlabel)
    axes[0].set_title(title)
    axes[0].legend(title="scale", frameon=False, loc="best")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_chunking(grouped: pd.DataFrame) -> None:
    palette = {"small": "#0f766e", "medium": "#b45309"}
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)

    for scale, scale_frame in grouped.groupby("problem_scale"):
        color = palette.get(scale, "#2563eb")
        axes[0].plot(scale_frame["num_splits"], scale_frame["runtime_median"], marker="o", linewidth=2.2, markersize=5.5, color=color, label=scale)
        axes[1].plot(scale_frame["num_splits"], scale_frame["relative_gap_median"], marker="o", linewidth=2.2, markersize=5.5, color=color, label=scale)
        axes[2].plot(scale_frame["num_splits"], scale_frame["obj_val_median"], marker="o", linewidth=2.2, markersize=5.5, color=color, label=scale)

    apply_axis_style(axes[0], "median runtime (s)", logy=True)
    apply_axis_style(axes[1], "median relative gap (%)")
    apply_axis_style(axes[2], "median objective value")
    axes[2].set_xlabel("number of splits")
    axes[0].set_title("Chunking experiment: runtime, gap, and objective vs split count")
    axes[0].legend(title="scale", frameon=False, loc="best")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_DIR / "chunking_runtime_gap_objective.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_objective_comparison(grouped: pd.DataFrame) -> None:
    palette = {"small": "#0f766e", "medium": "#b45309", "large": "#7c3aed"}
    metrics = [
        ("obj_val_median", "median objective value"),
        ("loaded_length_median", "median loaded length (m)"),
        ("avg_vehicle_length_median", "median avg vehicle load length (mm)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)

    for metric, ylabel in metrics:
        axis = axes[metrics.index((metric, ylabel))]
        for scale, scale_frame in grouped.groupby("problem_scale"):
            axis.plot(
                scale_frame["problem_scale"],
                scale_frame[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=palette.get(scale, "#2563eb"),
                label=scale,
            )
        apply_axis_style(axis, ylabel)

    axes[0].set_title("Objective family: current backup only contains quantity objective rows")
    axes[2].set_xlabel("problem scale")
    axes[0].legend(title="scale", frameon=False, loc="best")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_DIR / "objective_comparison_quantity_only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    plot_density_family(
        prepare_family_summary(data, "optional_ratio", "rho_optional"),
        parameter="rho_optional",
        title="Optional ratio: density metrics vs rho_optional",
        xlabel="rho_optional",
        filename="optional_ratio_density_metrics.png",
    )
    plot_density_family(
        prepare_family_summary(data, "proportion", "p_small"),
        parameter="p_small",
        title="Proportion: density metrics vs p_small",
        xlabel="p_small",
        filename="proportion_density_metrics.png",
    )
    plot_chunking(prepare_chunking_summary(data))
    plot_objective_comparison(prepare_objective_summary(data))

    summary = (
        data.groupby(["experiment_family", "problem_scale", "objective_type"], as_index=False)
        .agg(
            runs=("instance_id", "count"),
            completed=("timeout_flag", lambda s: (~s).sum()),
            timeout_runs=("timeout_flag", "sum"),
            median_runtime=("runtime_sec", "median"),
            median_obj=("obj_val", "median"),
            median_utilization=("nominal_utilization", "median"),
        )
        .sort_values(["experiment_family", "problem_scale", "objective_type"])
    )
    summary.to_csv(OUTPUT_DIR / "summary_by_family_scale_objective.csv", index=False)
    print(f"Saved figures and summary to {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()