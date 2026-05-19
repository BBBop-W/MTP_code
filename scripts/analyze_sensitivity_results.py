#!/usr/bin/env python3
"""Summarize and plot sensitivity experiment results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

RESULT_ROOT = PROJECT_ROOT / "result"
SCALES = ["small", "medium", "large"]
SCALE_LABELS = {"small": "Small-scale", "medium": "Medium-scale", "large": "Large-scale"}
SCALE_COLORS = {"small": "#377eb8", "medium": "#4daf4a", "large": "#e41a1c"}
FAMILY_LABELS = {
    "optional_ratio": "Optional ratio",
    "proportion": "Composition",
    "objective": "Objective",
    "chunking": "Chunking",
}


def load_results(date_tag: str) -> pd.DataFrame:
    frames = []
    for scale in SCALES:
        path = RESULT_ROOT / f"sensitivity_{date_tag}" / scale / f"sensitivity_{scale}_results_{date_tag}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No result CSVs found for date tag {date_tag}")
    out = pd.concat(frames, ignore_index=True)
    out["scale_label"] = out["problem_scale"].map(SCALE_LABELS)
    out["gap_pct"] = out["mip_gap"].fillna(0.0) * 100.0
    out["runtime_min"] = out["runtime_sec"] / 60.0
    out["avg_len_per_wagon_m"] = out["avg_len_per_wagon_mm"] / 1000.0
    out["avg_len_per_vehicle_m"] = out["avg_len_per_vehicle_mm"] / 1000.0
    out["nominal_util_pct"] = out["nominal_utilization"] * 100.0
    out["avg_idle_m"] = out["nominal_idle_mm_per_wagon"] / 1000.0
    out["time_limit_hit"] = out["status"].eq(9)
    out["optimal"] = out["status"].eq(2)
    return out


def mean_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_latex_table(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> Path:
    path = out_dir / f"{name}.tex"
    latex = df.to_latex(
        index=False,
        longtable=True,
        escape=True,
        caption=caption,
        label=label,
        column_format=column_format,
    )
    path.write_text(latex, encoding="utf-8")
    return path


def add_common_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["optimal_rate_pct"] = out["optimal"].astype(float) * 100.0
    out["time_limit_rate_pct"] = out["time_limit_hit"].astype(float) * 100.0
    return out


def format_appendix_table(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, fmt in formats.items():
        if col in out.columns:
            def _format_value(x: object) -> str:
                if pd.isna(x):
                    return ""
                try:
                    return fmt.format(float(x))
                except (TypeError, ValueError):
                    return str(x)

            out[col] = out[col].map(_format_value)
    return out


def write_appendix_outputs(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write appendix-ready longtables and selected run-level records."""
    df = add_common_rates(df)
    df["problem_scale"] = pd.Categorical(df["problem_scale"], categories=SCALES, ordered=True)
    df["experiment_family"] = pd.Categorical(
        df["experiment_family"],
        categories=["objective", "proportion", "optional_ratio", "chunking"],
        ordered=True,
    )
    out_paths: list[Path] = []

    def mean_agg(frame: pd.DataFrame, group_cols: list[str], agg_spec: dict[str, tuple[str, str]]) -> pd.DataFrame:
        return frame.groupby(group_cols, dropna=False, observed=True).agg(**agg_spec).reset_index()

    overview = mean_agg(
        df,
        ["problem_scale", "experiment_family"],
        {
            "Runs": ("run_id", "count"),
            "Opt. %": ("optimal_rate_pct", "mean"),
            "TL %": ("time_limit_rate_pct", "mean"),
            "Time (min)": ("runtime_min", "mean"),
            "Gap (%)": ("gap_pct", "mean"),
            "Len/w (m)": ("avg_len_per_wagon_m", "mean"),
            "Qty/w": ("avg_qty_per_wagon", "mean"),
            "Vars": ("num_vars", "mean"),
            "Constrs": ("num_constrs", "mean"),
        },
    )
    overview["Scale"] = overview["problem_scale"].map(SCALE_LABELS)
    overview["Experiment"] = overview["experiment_family"].map(FAMILY_LABELS)
    overview = overview[
        ["Scale", "Experiment", "Runs", "Opt. %", "TL %", "Time (min)", "Gap (%)", "Len/w (m)", "Qty/w", "Vars", "Constrs"]
    ]
    overview_fmt = format_appendix_table(
        overview,
        {
            "Opt. %": "{:.1f}",
            "TL %": "{:.1f}",
            "Time (min)": "{:.2f}",
            "Gap (%)": "{:.3f}",
            "Len/w (m)": "{:.3f}",
            "Qty/w": "{:.3f}",
            "Vars": "{:.0f}",
            "Constrs": "{:.0f}",
        },
    )
    out_paths.append(save_table(overview, out_dir, "appendix_table_overview"))
    out_paths.append(
        save_latex_table(
            overview_fmt,
            out_dir,
            "appendix_table_overview",
            "Summary of the sensitivity experiment families.",
            "tab:app_sensitivity_overview",
        )
    )

    length_baseline = df[
        (df["experiment_family"].eq("chunking"))
        & (df["objective_type"].eq("length"))
        & (df["num_splits"].eq(3))
    ].copy()
    length_baseline["objective_label"] = "Length"
    quantity_runs = df[df["experiment_family"].eq("objective")].copy()
    quantity_runs["objective_label"] = "Quantity"
    objective_rows = pd.concat([length_baseline, quantity_runs], ignore_index=True)
    objective = mean_agg(
        objective_rows,
        ["problem_scale", "objective_label"],
        {
            "Runs": ("run_id", "count"),
            "Opt. %": ("optimal_rate_pct", "mean"),
            "TL %": ("time_limit_rate_pct", "mean"),
            "Time (min)": ("runtime_min", "mean"),
            "Gap (%)": ("gap_pct", "mean"),
            "TotalLen (m)": ("loaded_length_m", "mean"),
            "Len/w (m)": ("avg_len_per_wagon_m", "mean"),
            "TotalQty": ("loaded_quantity", "mean"),
            "Qty/w": ("avg_qty_per_wagon", "mean"),
            "Idle/w (m)": ("avg_idle_m", "mean"),
        },
    )
    objective["Scale"] = objective["problem_scale"].map(SCALE_LABELS)
    objective["Objective"] = objective["objective_label"]
    objective = objective[
        ["Scale", "Objective", "Runs", "Opt. %", "TL %", "Time (min)", "Gap (%)", "TotalLen (m)", "Len/w (m)", "TotalQty", "Qty/w", "Idle/w (m)"]
    ]
    objective_fmt = format_appendix_table(
        objective,
        {
            "Opt. %": "{:.1f}",
            "TL %": "{:.1f}",
            "Time (min)": "{:.2f}",
            "Gap (%)": "{:.3f}",
            "TotalLen (m)": "{:.3f}",
            "Len/w (m)": "{:.3f}",
            "TotalQty": "{:.1f}",
            "Qty/w": "{:.3f}",
            "Idle/w (m)": "{:.3f}",
        },
    )
    out_paths.append(save_table(objective, out_dir, "appendix_table_objective"))
    out_paths.append(
        save_latex_table(
            objective_fmt,
            out_dir,
            "appendix_table_objective",
            "Comparison between the length-maximization and quantity-maximization objectives.",
            "tab:app_objective_comparison",
        )
    )

    optional = mean_agg(
        df[df["experiment_family"].eq("optional_ratio")],
        ["problem_scale", "rho_optional"],
        {
            "Runs": ("run_id", "count"),
            "D": ("mandatory_total", "mean"),
            "C": ("optional_total", "mean"),
            "Opt. %": ("optimal_rate_pct", "mean"),
            "TL %": ("time_limit_rate_pct", "mean"),
            "Time (min)": ("runtime_min", "mean"),
            "Gap (%)": ("gap_pct", "mean"),
            "Len/w (m)": ("avg_len_per_wagon_m", "mean"),
            "Qty/w": ("avg_qty_per_wagon", "mean"),
            "Util. %": ("nominal_util_pct", "mean"),
            "OptLoaded": ("optional_loaded", "mean"),
            "OptAccept %": ("optional_acceptance_rate", "mean"),
        },
    )
    optional["OptAccept %"] *= 100.0
    optional["Scale"] = optional["problem_scale"].map(SCALE_LABELS)
    optional = optional[
        ["Scale", "rho_optional", "Runs", "D", "C", "Opt. %", "TL %", "Time (min)", "Gap (%)", "Len/w (m)", "Qty/w", "Util. %", "OptLoaded", "OptAccept %"]
    ].rename(columns={"rho_optional": "rho"})
    optional_fmt = format_appendix_table(
        optional,
        {
            "rho": "{:.2f}",
            "D": "{:.1f}",
            "C": "{:.1f}",
            "Opt. %": "{:.1f}",
            "TL %": "{:.1f}",
            "Time (min)": "{:.2f}",
            "Gap (%)": "{:.3f}",
            "Len/w (m)": "{:.3f}",
            "Qty/w": "{:.3f}",
            "Util. %": "{:.1f}",
            "OptLoaded": "{:.1f}",
            "OptAccept %": "{:.1f}",
        },
    )
    out_paths.append(save_table(optional, out_dir, "appendix_table_optional_ratio"))
    out_paths.append(
        save_latex_table(
            optional_fmt,
            out_dir,
            "appendix_table_optional_ratio",
            "Sensitivity results for the optional-to-mandatory ratio.",
            "tab:app_optional_ratio",
        )
    )

    composition = mean_agg(
        df[df["experiment_family"].eq("proportion")],
        ["problem_scale", "p_small"],
        {
            "Runs": ("run_id", "count"),
            "Opt. %": ("optimal_rate_pct", "mean"),
            "TL %": ("time_limit_rate_pct", "mean"),
            "Time (min)": ("runtime_min", "mean"),
            "Gap (%)": ("gap_pct", "mean"),
            "Len/w (m)": ("avg_len_per_wagon_m", "mean"),
            "Qty/w": ("avg_qty_per_wagon", "mean"),
            "Len/v (m)": ("avg_len_per_vehicle_m", "mean"),
            "Idle/w (m)": ("avg_idle_m", "mean"),
            "SmallQty": ("loaded_small_qty", "mean"),
            "MediumQty": ("loaded_medium_qty", "mean"),
            "LargeQty": ("loaded_large_qty", "mean"),
        },
    )
    composition["Scale"] = composition["problem_scale"].map(SCALE_LABELS)
    composition = composition[
        ["Scale", "p_small", "Runs", "Opt. %", "TL %", "Time (min)", "Gap (%)", "Len/w (m)", "Qty/w", "Len/v (m)", "Idle/w (m)", "SmallQty", "MediumQty", "LargeQty"]
    ].rename(columns={"p_small": "p_s"})
    composition_fmt = format_appendix_table(
        composition,
        {
            "p_s": "{:.1f}",
            "Opt. %": "{:.1f}",
            "TL %": "{:.1f}",
            "Time (min)": "{:.2f}",
            "Gap (%)": "{:.3f}",
            "Len/w (m)": "{:.3f}",
            "Qty/w": "{:.3f}",
            "Len/v (m)": "{:.3f}",
            "Idle/w (m)": "{:.3f}",
            "SmallQty": "{:.1f}",
            "MediumQty": "{:.1f}",
            "LargeQty": "{:.1f}",
        },
    )
    out_paths.append(save_table(composition, out_dir, "appendix_table_composition"))
    out_paths.append(
        save_latex_table(
            composition_fmt,
            out_dir,
            "appendix_table_composition",
            "Sensitivity results for the automobile-size composition.",
            "tab:app_composition",
        )
    )

    chunking = mean_agg(
        df[df["experiment_family"].eq("chunking")],
        ["problem_scale", "num_splits"],
        {
            "Runs": ("run_id", "count"),
            "Opt. %": ("optimal_rate_pct", "mean"),
            "TL %": ("time_limit_rate_pct", "mean"),
            "Time (min)": ("runtime_min", "mean"),
            "Gap (%)": ("gap_pct", "mean"),
            "Lower blocks": ("num_lower_side_blocks", "mean"),
            "Upper blocks": ("num_upper_side_blocks", "mean"),
            "Components": ("num_components_total", "mean"),
            "Regions": ("num_lower_length_regions", "mean"),
            "Vars": ("num_vars", "mean"),
            "Constrs": ("num_constrs", "mean"),
            "Len/w (m)": ("avg_len_per_wagon_m", "mean"),
            "Qty/w": ("avg_qty_per_wagon", "mean"),
        },
    )
    chunking["Regions"] = chunking["Regions"] * 2.0
    chunking["Scale"] = chunking["problem_scale"].map(SCALE_LABELS)
    chunking = chunking[
        ["Scale", "num_splits", "Runs", "Opt. %", "TL %", "Time (min)", "Gap (%)", "Lower blocks", "Upper blocks", "Components", "Regions", "Vars", "Constrs", "Len/w (m)", "Qty/w"]
    ].rename(columns={"num_splits": "k"})
    chunking_fmt = format_appendix_table(
        chunking,
        {
            "k": "{:.0f}",
            "Opt. %": "{:.1f}",
            "TL %": "{:.1f}",
            "Time (min)": "{:.2f}",
            "Gap (%)": "{:.3f}",
            "Lower blocks": "{:.0f}",
            "Upper blocks": "{:.0f}",
            "Components": "{:.0f}",
            "Regions": "{:.0f}",
            "Vars": "{:.0f}",
            "Constrs": "{:.0f}",
            "Len/w (m)": "{:.3f}",
            "Qty/w": "{:.3f}",
        },
    )
    out_paths.append(save_table(chunking, out_dir, "appendix_table_chunking"))
    out_paths.append(
        save_latex_table(
            chunking_fmt,
            out_dir,
            "appendix_table_chunking",
            "Sensitivity results for the chunking granularity.",
            "tab:app_chunking",
        )
    )

    selected = df[
        [
            "problem_scale",
            "experiment_family",
            "base_id",
            "num_types_I",
            "num_wagons_J",
            "varied_parameter",
            "varied_value",
            "num_splits",
            "objective_type",
            "status",
            "runtime_min",
            "gap_pct",
            "avg_len_per_wagon_m",
            "avg_qty_per_wagon",
            "nominal_util_pct",
            "optional_acceptance_rate",
            "num_vars",
            "num_constrs",
        ]
    ].copy()
    selected["Scale"] = selected["problem_scale"].map(SCALE_LABELS)
    selected["Experiment"] = selected["experiment_family"].map(FAMILY_LABELS)
    selected["OptAccept %"] = selected["optional_acceptance_rate"] * 100.0
    selected = selected.rename(
        columns={
            "base_id": "Base",
            "num_types_I": "I",
            "num_wagons_J": "J",
            "varied_parameter": "Parameter",
            "varied_value": "Value",
            "num_splits": "k",
            "objective_type": "Objective",
            "status": "Status",
            "runtime_min": "Time (min)",
            "gap_pct": "Gap (%)",
            "avg_len_per_wagon_m": "Len/w (m)",
            "avg_qty_per_wagon": "Qty/w",
            "nominal_util_pct": "Util. %",
            "num_vars": "Vars",
            "num_constrs": "Constrs",
        }
    )
    selected = selected[
        ["Scale", "Experiment", "Base", "I", "J", "Parameter", "Value", "k", "Objective", "Status", "Time (min)", "Gap (%)", "Len/w (m)", "Qty/w", "Util. %", "OptAccept %", "Vars", "Constrs"]
    ]
    selected_path = out_dir / "appendix_run_level_selected.csv"
    selected.to_csv(selected_path, index=False)
    out_paths.append(selected_path)

    selected_fmt = format_appendix_table(
        selected,
        {
            "Value": "{:.2f}",
            "k": "{:.0f}",
            "Time (min)": "{:.2f}",
            "Gap (%)": "{:.3f}",
            "Len/w (m)": "{:.3f}",
            "Qty/w": "{:.3f}",
            "Util. %": "{:.1f}",
            "OptAccept %": "{:.1f}",
            "Vars": "{:.0f}",
            "Constrs": "{:.0f}",
        },
    )
    out_paths.append(
        save_latex_table(
            selected_fmt,
            out_dir,
            "appendix_run_level_selected",
            "Selected run-level records for all sensitivity solves.",
            "tab:app_run_level_selected",
        )
    )

    combined = out_dir / "appendix_tables.tex"
    with open(combined, "w", encoding="utf-8") as f:
        f.write("% Requires: \\usepackage{booktabs,longtable}\n")
        f.write("% Optional for wide tables: \\usepackage{pdflscape} and wrap tables in landscape.\n\n")
        for name in [
            "appendix_table_overview",
            "appendix_table_objective",
            "appendix_table_optional_ratio",
            "appendix_table_composition",
            "appendix_table_chunking",
        ]:
            f.write((out_dir / f"{name}.tex").read_text(encoding="utf-8"))
            f.write("\n\n")
    out_paths.append(combined)

    md_path = out_dir / "appendix_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Appendix Tables\n\n")
        for title, table in [
            ("Sensitivity experiment overview", overview_fmt),
            ("Objective comparison", objective_fmt),
            ("Optional-ratio sensitivity", optional_fmt),
            ("Automobile-size composition sensitivity", composition_fmt),
            ("Chunking granularity sensitivity", chunking_fmt),
        ]:
            f.write(f"## {title}\n\n")
            f.write(table.to_markdown(index=False))
            f.write("\n\n")
        f.write(f"Run-level selected records: `{selected_path}`\n")
    out_paths.append(md_path)
    return out_paths


def plot_objective(df: pd.DataFrame, out_dir: Path) -> Path:
    obj_len = df[
        (df["experiment_family"].eq("chunking"))
        & (df["objective_type"].eq("length"))
        & (df["num_splits"].eq(3))
    ].copy()
    obj_len["objective_label"] = "Length"
    obj_qty = df[df["experiment_family"].eq("objective")].copy()
    obj_qty["objective_label"] = "Quantity"
    plot_df = pd.concat([obj_len, obj_qty], ignore_index=True)
    agg = mean_table(
        plot_df,
        ["problem_scale", "scale_label", "objective_label"],
        ["avg_len_per_wagon_m", "avg_qty_per_wagon", "runtime_min", "gap_pct"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    objectives = ["Length", "Quantity"]
    width = 0.34
    x = range(len(SCALES))
    all_len_vals = []
    all_qty_vals = []
    for offset, objective in zip([-width / 2, width / 2], objectives):
        vals_len = []
        vals_qty = []
        for scale in SCALES:
            row = agg[(agg["problem_scale"].eq(scale)) & (agg["objective_label"].eq(objective))]
            vals_len.append(float(row["avg_len_per_wagon_m"].iloc[0]) if not row.empty else 0)
            vals_qty.append(float(row["avg_qty_per_wagon"].iloc[0]) if not row.empty else 0)
        all_len_vals.extend(vals_len)
        all_qty_vals.extend(vals_qty)
        axes[0].bar([i + offset for i in x], vals_len, width=width, label=objective)
        axes[1].bar([i + offset for i in x], vals_qty, width=width, label=objective)

    for ax, ylabel in zip(axes, ["Avg. loaded length per wagon (m)", "Avg. loaded automobiles per wagon"]):
        ax.set_xticks(list(x), [SCALE_LABELS[s] for s in SCALES])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylim(min(all_len_vals) - 0.8, max(all_len_vals) + 0.8)
    axes[1].set_ylim(min(all_qty_vals) - 0.25, max(all_qty_vals) + 0.25)
    axes[0].set_title("Space utilization by objective")
    axes[1].set_title("Loaded quantity by objective")
    path = out_dir / "fig_objective_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    save_table(agg, out_dir, "table_objective_comparison")
    return path


def plot_optional_ratio(df: pd.DataFrame, out_dir: Path) -> Path:
    plot_df = df[df["experiment_family"].eq("optional_ratio")].copy()
    agg = mean_table(
        plot_df,
        ["problem_scale", "scale_label", "rho_optional"],
        ["avg_len_per_wagon_m", "loaded_quantity", "optional_acceptance_rate", "runtime_min", "gap_pct"],
    )
    agg["optional_acceptance_pct"] = agg["optional_acceptance_rate"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for scale in SCALES:
        sub = agg[agg["problem_scale"].eq(scale)].sort_values("rho_optional")
        axes[0].plot(sub["rho_optional"], sub["avg_len_per_wagon_m"], marker="o", color=SCALE_COLORS[scale], label=SCALE_LABELS[scale])
        axes[1].plot(sub["rho_optional"], sub["optional_acceptance_pct"], marker="o", color=SCALE_COLORS[scale], label=SCALE_LABELS[scale])
    axes[0].set_title("Optional pool size vs. utilization")
    axes[0].set_xlabel("Optional-to-mandatory ratio")
    axes[0].set_ylabel("Avg. loaded length per wagon (m)")
    axes[1].set_title("Optional acceptance")
    axes[1].set_xlabel("Optional-to-mandatory ratio")
    axes[1].set_ylabel("Accepted optional automobiles (%)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    path = out_dir / "fig_optional_ratio_sensitivity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    save_table(agg, out_dir, "table_optional_ratio")
    return path


def plot_composition(df: pd.DataFrame, out_dir: Path) -> Path:
    plot_df = df[df["experiment_family"].eq("proportion")].copy()
    agg = mean_table(
        plot_df,
        ["problem_scale", "scale_label", "p_small"],
        [
            "avg_len_per_wagon_m",
            "avg_qty_per_wagon",
            "avg_len_per_vehicle_m",
            "avg_idle_m",
            "loaded_small_qty",
            "loaded_medium_qty",
            "loaded_large_qty",
            "runtime_min",
            "gap_pct",
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for scale in SCALES:
        sub = agg[agg["problem_scale"].eq(scale)].sort_values("p_small")
        axes[0].plot(sub["p_small"], sub["avg_len_per_wagon_m"], marker="o", color=SCALE_COLORS[scale], label=SCALE_LABELS[scale])
        axes[1].plot(sub["p_small"], sub["avg_qty_per_wagon"], marker="o", color=SCALE_COLORS[scale], label=SCALE_LABELS[scale])
    axes[0].set_title("Composition effect on wagon utilization")
    axes[0].set_xlabel("Small-automobile proportion")
    axes[0].set_ylabel("Avg. loaded length per wagon (m)")
    axes[1].set_title("Composition effect on loaded quantity")
    axes[1].set_xlabel("Small-automobile proportion")
    axes[1].set_ylabel("Avg. loaded automobiles per wagon")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    path = out_dir / "fig_composition_sensitivity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    save_table(agg, out_dir, "table_composition")
    return path


def plot_chunking(df: pd.DataFrame, out_dir: Path) -> Path:
    plot_df = df[df["experiment_family"].eq("chunking")].copy()
    agg = mean_table(
        plot_df,
        ["problem_scale", "scale_label", "num_splits"],
        ["avg_len_per_wagon_m", "runtime_min", "gap_pct", "num_vars", "num_constrs"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for scale in SCALES:
        sub = agg[agg["problem_scale"].eq(scale)].sort_values("num_splits")
        axes[0].plot(sub["num_splits"], sub["avg_len_per_wagon_m"], marker="o", color=SCALE_COLORS[scale], label=SCALE_LABELS[scale])
    axes[0].set_title("Chunking granularity and utilization")
    axes[0].set_xlabel("Side components per side")
    axes[0].set_ylabel("Avg. loaded length per wagon (m)")
    for scale in SCALES:
        sub = agg[agg["problem_scale"].eq(scale)].sort_values("num_splits")
        axes[1].plot(sub["num_splits"], sub["num_constrs"], marker="s", color=SCALE_COLORS[scale], label=SCALE_LABELS[scale])
    axes[1].set_title("Model size")
    axes[1].set_xlabel("Side components per side")
    axes[1].set_ylabel("Average number of constraints")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    path = out_dir / "fig_chunking_sensitivity.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    save_table(agg, out_dir, "table_chunking")
    return path


def plot_status(df: pd.DataFrame, out_dir: Path) -> Path:
    status = (
        df.groupby(["problem_scale", "experiment_family"])
        .agg(
            runs=("run_id", "count"),
            optimal_rate=("optimal", "mean"),
            time_limit_rate=("time_limit_hit", "mean"),
            avg_runtime_min=("runtime_min", "mean"),
            avg_gap_pct=("gap_pct", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    labels = []
    opt = []
    tl = []
    for family in ["optional_ratio", "proportion", "objective", "chunking"]:
        for scale in SCALES:
            row = status[(status["experiment_family"].eq(family)) & (status["problem_scale"].eq(scale))]
            if row.empty:
                continue
            labels.append(f"{family}\n{SCALE_LABELS[scale]}")
            opt.append(float(row["optimal_rate"].iloc[0]) * 100)
            tl.append(float(row["time_limit_rate"].iloc[0]) * 100)
    x = range(len(labels))
    ax.bar(x, opt, label="Optimal", color="#4daf4a")
    ax.bar(x, tl, bottom=opt, label="Time limit", color="#ff7f00")
    ax.set_xticks(list(x), labels, rotation=45, ha="right")
    ax.set_ylabel("Run share (%)")
    ax.set_title("Solver status by experiment family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = out_dir / "fig_status_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    save_table(status, out_dir, "table_status_summary")
    return path


def write_markdown(df: pd.DataFrame, out_dir: Path, figure_paths: list[Path]) -> Path:
    status = pd.read_csv(out_dir / "table_status_summary.csv")
    composition = pd.read_csv(out_dir / "table_composition.csv")
    optional = pd.read_csv(out_dir / "table_optional_ratio.csv")
    chunking = pd.read_csv(out_dir / "table_chunking.csv")
    objective = pd.read_csv(out_dir / "table_objective_comparison.csv")

    best_comp = composition.loc[
        composition.groupby("problem_scale")["avg_len_per_wagon_m"].idxmax(),
        ["problem_scale", "p_small", "avg_len_per_wagon_m", "avg_qty_per_wagon"],
    ]
    opt_gain = []
    for scale in SCALES:
        sub = objective[objective["problem_scale"].eq(scale)]
        if {"Length", "Quantity"}.issubset(set(sub["objective_label"])):
            length = sub[sub["objective_label"].eq("Length")].iloc[0]
            quantity = sub[sub["objective_label"].eq("Quantity")].iloc[0]
            opt_gain.append(
                {
                    "problem_scale": scale,
                    "len_gain_m_per_wagon": length["avg_len_per_wagon_m"] - quantity["avg_len_per_wagon_m"],
                    "qty_change_per_wagon": length["avg_qty_per_wagon"] - quantity["avg_qty_per_wagon"],
                }
            )
    opt_gain_df = pd.DataFrame(opt_gain)

    path = out_dir / "sensitivity_summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Sensitivity Experiment Summary\n\n")
        f.write(f"Total runs: {len(df)}. Scales: {', '.join(SCALES)}.\n\n")
        f.write("## Solver Status\n\n")
        f.write(status.to_markdown(index=False))
        f.write("\n\n## Objective Comparison\n\n")
        f.write(opt_gain_df.to_markdown(index=False))
        f.write("\n\n## Best Composition by Scale\n\n")
        f.write(best_comp.to_markdown(index=False))
        f.write("\n\n## Figure Files\n\n")
        for fig_path in figure_paths:
            f.write(f"- {fig_path}\n")
    return path


def analyze(date_tag: str) -> None:
    df = load_results(date_tag)
    out_dir = RESULT_ROOT / f"sensitivity_{date_tag}" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(df, out_dir, "all_results_clean")
    overview = (
        df.groupby(["problem_scale", "experiment_family"])
        .agg(
            runs=("run_id", "count"),
            optimal_rate=("optimal", "mean"),
            time_limit_rate=("time_limit_hit", "mean"),
            avg_runtime_min=("runtime_min", "mean"),
            avg_gap_pct=("gap_pct", "mean"),
            avg_len_per_wagon_m=("avg_len_per_wagon_m", "mean"),
            avg_qty_per_wagon=("avg_qty_per_wagon", "mean"),
        )
        .reset_index()
    )
    save_table(overview, out_dir, "table_experiment_overview")

    figures = [
        plot_objective(df, out_dir),
        plot_composition(df, out_dir),
        plot_optional_ratio(df, out_dir),
        plot_chunking(df, out_dir),
        plot_status(df, out_dir),
    ]
    appendix_paths = write_appendix_outputs(df, out_dir)
    summary_path = write_markdown(df, out_dir, figures)
    print(f"Analysis directory: {out_dir}")
    print(f"Summary: {summary_path}")
    for fig in figures:
        print(f"Figure: {fig}")
    for path in appendix_paths:
        print(f"Appendix: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sensitivity experiment results.")
    parser.add_argument("--date", required=True, help="Date tag of the sensitivity run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.date)


if __name__ == "__main__":
    main()
