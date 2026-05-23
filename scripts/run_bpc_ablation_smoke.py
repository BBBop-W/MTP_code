#!/usr/bin/env python3
"""Smoke runner for compartment-label BPC ablation methods.

This is intentionally separate from the older numerical-batch drivers. It runs a
small set of instances through the ablation configurations needed before the
formal experiment script is designed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gurobi import build_and_solve  # noqa: E402


METHODS: List[Dict[str, Any]] = [
    {
        "method": "clabel_ex_plain",
        "family": "bpc_compartment",
        "profile_generator_mode": "ex",
        "use_warmstart": False,
        "use_cuts": False,
        "use_dominance": False,
        "use_local_d1_pruning": False,
        "use_rc_bound": True,
        "dominance_rule": "none",
    },
    {
        "method": "clabel_ex_ws_cut_no_dom",
        "family": "bpc_compartment",
        "profile_generator_mode": "ex",
        "use_warmstart": True,
        "use_cuts": True,
        "use_dominance": False,
        "use_local_d1_pruning": False,
        "use_rc_bound": True,
        "dominance_rule": "none",
    },
    {
        "method": "clabel_ex_ws_cut_d1",
        "family": "bpc_compartment",
        "profile_generator_mode": "ex",
        "use_warmstart": True,
        "use_cuts": True,
        "use_dominance": True,
        "use_local_d1_pruning": True,
        "use_rc_bound": True,
        "dominance_rule": "D1",
    },
    {
        "method": "clabel_hyb_ws_cut_d3",
        "family": "bpc_compartment",
        "profile_generator_mode": "hyb",
        "use_warmstart": True,
        "use_cuts": True,
        "use_dominance": True,
        "use_local_d1_pruning": True,
        "use_rc_bound": True,
        "dominance_rule": "D3",
    },
]

METHOD_BY_NAME = {str(config["method"]): config for config in METHODS}
ALL_METHOD_NAMES = ["compact_gurobi", *METHOD_BY_NAME]

GUROBI_STATUS_NAMES = {
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    9: "TIME_LIMIT",
    13: "SUBOPTIMAL",
}


CSV_COLUMNS = [
    "instance",
    "method",
    "family",
    "status",
    "loaded_length_mm",
    "objective",
    "best_bound",
    "gap",
    "runtime_sec",
    "nodes",
    "threads",
    "num_splits",
    "independent_mode_split",
    "time_limit",
    "mip_gap_target",
    "profile_generator_mode",
    "use_warmstart",
    "warmstart_source",
    "warmstart_incumbent",
    "warmstart_loaded_length_mm",
    "warmstart_added",
    "use_cuts",
    "use_dominance",
    "use_local_d1_pruning",
    "use_rc_bound",
    "dominance_rule",
    "generated_columns",
    "total_columns",
    "generated_subpatterns",
    "labels_generated_raw",
    "labels_feasible",
    "labels_pruned_by_bound",
    "labels_pruned_by_dominance",
    "labels_pruned_total",
    "labels_after_dominance",
    "labels_avoided_by_d2",
    "hybrid_calls",
    "hybrid_ordered_type_sum",
    "hybrid_ordered_type_max",
    "hybrid_ordered_quantity_sum",
    "hybrid_total_quantity_sum",
    "master_time",
    "pricing_time",
    "result_dir",
    "runner_status",
    "stderr",
]


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def run_process(
    cmd: List[str],
    *,
    cwd: Path,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def default_bpc_exe() -> Path:
    name = "bpc_label_solver.exe" if sys.platform.startswith("win") else "bpc_label_solver"
    return PROJECT_ROOT / "BPC_label_cpp" / name


def default_vns_exe() -> Path:
    name = "vns_solver.exe" if sys.platform.startswith("win") else "vns_solver"
    return PROJECT_ROOT / "VNS_cpp" / name


def status_from_bpc(payload: Dict[str, Any], mip_gap: float) -> str:
    if not payload.get("has_incumbent", False):
        return "NO_SOLUTION"
    gap = payload.get("gap")
    if gap is None:
        return "INCUMBENT"
    return "OK" if float(gap) <= mip_gap else "GAP"


def status_from_compact(summary: Dict[str, Any]) -> str | None:
    status = summary.get("status")
    if status is None:
        return None
    try:
        return GUROBI_STATUS_NAMES.get(int(status), str(status))
    except (TypeError, ValueError):
        return str(status)


def ensure_warmstart(instance: str, args: argparse.Namespace) -> Dict[str, Path]:
    bi_json = PROJECT_ROOT / "result" / instance / "BI" / "carriage_info.json"
    vns_json = PROJECT_ROOT / "result" / instance / "VNS" / "carriage_info.json"
    if not args.refresh_warmstart and bi_json.exists() and vns_json.exists():
        return {"BI": bi_json, "VNS": vns_json}

    exe = Path(args.vns_exe)
    if not exe.is_absolute():
        exe = PROJECT_ROOT / exe
    if not exe.exists():
        raise FileNotFoundError(f"VNS executable not found: {exe}")

    cmd = [
        str(exe),
        instance,
        str(args.num_splits),
        "1" if args.independent_mode_split else "0",
        str(args.warmstart_time_limit),
    ]
    run_process(
        cmd,
        cwd=exe.parent,
        timeout=args.warmstart_time_limit + args.grace_sec,
    )
    if not bi_json.exists() or not vns_json.exists():
        raise FileNotFoundError(
            f"Warmstart files missing after VNS run for {instance}: {bi_json}, {vns_json}"
        )
    return {"BI": bi_json, "VNS": vns_json}


def run_compact(instance: str, result_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    instance_dir = PROJECT_ROOT / "data" / "Instance" / instance
    output_dir = result_root / instance / "compact_gurobi"
    started = time.perf_counter()
    try:
        summary = build_and_solve(
            instance_dir=instance_dir,
            output_dir=output_dir,
            log_to_console=False,
            num_splits=args.num_splits,
            independent_mode_split=args.independent_mode_split,
            objective_type="length",
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
        )
        runtime = float(summary.get("runtime_sec") or (time.perf_counter() - started))
        status = status_from_compact(summary)
        runner_status = "completed"
        stderr = ""
    except BaseException as exc:
        summary = {}
        runtime = time.perf_counter() - started
        status = "ERROR"
        runner_status = "exception"
        stderr = repr(exc)

    return {
        "instance": instance,
        "method": "compact_gurobi",
        "family": "compact",
        "status": status,
        "loaded_length_mm": summary.get("loaded_length_mm"),
        "objective": summary.get("obj_val"),
        "best_bound": summary.get("obj_bound"),
        "gap": summary.get("mip_gap"),
        "runtime_sec": runtime,
        "nodes": summary.get("node_count"),
        "threads": args.threads,
        "num_splits": args.num_splits,
        "independent_mode_split": args.independent_mode_split,
        "time_limit": args.time_limit,
        "mip_gap_target": args.mip_gap,
        "profile_generator_mode": "",
        "use_warmstart": False,
        "warmstart_source": "",
        "use_cuts": False,
        "use_dominance": False,
        "use_local_d1_pruning": False,
        "use_rc_bound": False,
        "dominance_rule": "",
        "result_dir": str(output_dir),
        "runner_status": runner_status,
        "stderr": stderr,
    }


def parse_solver_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("solver stdout did not contain JSON")
    return json.loads(stdout[start:])


def run_bpc_method(
    instance: str,
    config: Dict[str, Any],
    warmstarts: Dict[str, Path] | None,
    result_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    exe = Path(args.bpc_exe)
    if not exe.is_absolute():
        exe = PROJECT_ROOT / exe
    if not exe.exists():
        raise FileNotFoundError(f"BPC executable not found: {exe}")

    instance_dir = PROJECT_ROOT / "data" / "Instance" / instance
    cmd = [
        str(exe),
        "--instance",
        str(instance_dir),
        "--method",
        "compartment",
        "--pricing-backend",
        "label",
        "--max-nodes",
        str(args.max_nodes),
        "--max-cg-iters",
        str(args.max_cg_iters),
        "--max-columns-per-pricing",
        str(args.label_columns),
        "--max-columns-per-subproblem",
        str(args.label_columns),
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(args.threads),
        "--num-splits",
        str(args.num_splits),
        "--independent-mode-split",
        bool_arg(args.independent_mode_split),
        "--use-cuts",
        bool_arg(bool(config["use_cuts"])),
        "--use-dominance",
        bool_arg(bool(config["use_dominance"])),
        "--use-local-d1-pruning",
        bool_arg(bool(config["use_local_d1_pruning"])),
        "--use-rc-bound",
        bool_arg(bool(config["use_rc_bound"])),
        "--profile-generator-mode",
        str(config["profile_generator_mode"]),
        "--residual-profile-mode",
        args.residual_profile_mode,
        "--quiet",
    ]
    warmstart_source = ""
    if config["use_warmstart"]:
        if warmstarts is None:
            raise ValueError(f"{config['method']} requires warmstarts")
        warmstart_source = "BI+VNS"
        cmd.extend(["--warmstart-json", f"BI:{warmstarts['BI']}"])
        cmd.extend(["--warmstart-json", f"VNS:{warmstarts['VNS']}"])

    started = time.perf_counter()
    result_dir = result_root / instance / str(config["method"])
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = run_process(
            cmd,
            cwd=exe.parent,
            timeout=args.time_limit + args.grace_sec,
        )
        runtime = time.perf_counter() - started
        if completed.returncode not in (0, 1):
            payload: Dict[str, Any] = {}
            status = "ERROR"
            runner_status = f"error:{completed.returncode}"
        else:
            payload = parse_solver_json(completed.stdout)
            status = status_from_bpc(payload, args.mip_gap)
            runner_status = "completed"
        stderr = completed.stderr.strip()
        (result_dir / "solver_stdout.json").write_text(completed.stdout, encoding="utf-8")
        if stderr:
            (result_dir / "solver_stderr.txt").write_text(stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        payload = {}
        runtime = time.perf_counter() - started
        status = "TIME_LIMIT"
        runner_status = "timeout"
        stderr = str(exc)
    except BaseException as exc:
        payload = {}
        runtime = time.perf_counter() - started
        status = "ERROR"
        runner_status = "exception"
        stderr = repr(exc)

    return {
        "instance": instance,
        "method": config["method"],
        "family": config["family"],
        "status": status,
        "loaded_length_mm": payload.get("loaded_length_mm"),
        "objective": payload.get("best_objective"),
        "best_bound": payload.get("best_bound"),
        "gap": payload.get("gap"),
        "runtime_sec": payload.get("wall_time", runtime),
        "nodes": payload.get("explored_nodes"),
        "threads": args.threads,
        "num_splits": args.num_splits,
        "independent_mode_split": args.independent_mode_split,
        "time_limit": args.time_limit,
        "mip_gap_target": args.mip_gap,
        "profile_generator_mode": config["profile_generator_mode"],
        "use_warmstart": config["use_warmstart"],
        "warmstart_source": warmstart_source,
        "warmstart_incumbent": payload.get("warmstart_incumbent", False),
        "warmstart_loaded_length_mm": payload.get("warmstart_loaded_length_mm", ""),
        "warmstart_added": payload.get("warmstart_added", 0),
        "use_cuts": config["use_cuts"],
        "use_dominance": config["use_dominance"],
        "use_local_d1_pruning": config["use_local_d1_pruning"],
        "use_rc_bound": config["use_rc_bound"],
        "dominance_rule": config["dominance_rule"],
        "generated_columns": payload.get("generated_columns"),
        "total_columns": payload.get("total_columns"),
        "generated_subpatterns": payload.get("generated_subpatterns"),
        "labels_generated_raw": payload.get("labels_generated_raw"),
        "labels_feasible": payload.get("labels_feasible"),
        "labels_pruned_by_bound": payload.get("labels_pruned_by_bound"),
        "labels_pruned_by_dominance": payload.get("labels_pruned_by_dominance"),
        "labels_pruned_total": payload.get("labels_pruned_total"),
        "labels_after_dominance": payload.get("labels_after_dominance"),
        "labels_avoided_by_d2": payload.get("labels_avoided_by_d2"),
        "hybrid_calls": payload.get("hybrid_calls"),
        "hybrid_ordered_type_sum": payload.get("hybrid_ordered_type_sum"),
        "hybrid_ordered_type_max": payload.get("hybrid_ordered_type_max"),
        "hybrid_ordered_quantity_sum": payload.get("hybrid_ordered_quantity_sum"),
        "hybrid_total_quantity_sum": payload.get("hybrid_total_quantity_sum"),
        "master_time": payload.get("master_time"),
        "pricing_time": payload.get("pricing_time"),
        "result_dir": str(result_dir),
        "runner_status": runner_status,
        "stderr": stderr,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small BPC ablation smoke test.")
    parser.add_argument("--instances", nargs="+", default=["m5c5"])
    parser.add_argument("--methods", nargs="+", choices=ALL_METHOD_NAMES, default=ALL_METHOD_NAMES)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--warmstart-time-limit", type=float, default=30.0)
    parser.add_argument("--grace-sec", type=float, default=20.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual-profile-mode", choices=["full", "fans_diag"], default="full")
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-cg-iters", type=int, default=3000)
    parser.add_argument("--label-columns", type=int, default=0)
    parser.add_argument("--skip-compact", action="store_true")
    parser.add_argument("--skip-warmstart-refresh", dest="refresh_warmstart", action="store_false")
    parser.set_defaults(refresh_warmstart=True)
    parser.add_argument("--bpc-exe", type=Path, default=default_bpc_exe())
    parser.add_argument("--vns-exe", type=Path, default=default_vns_exe())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"_{args.result_suffix}" if args.result_suffix else ""
    result_root = PROJECT_ROOT / "result" / f"ablation_smoke_{args.date}{suffix}"
    rows: List[Dict[str, Any]] = []

    for instance in args.instances:
        selected_configs = [METHOD_BY_NAME[name] for name in args.methods if name in METHOD_BY_NAME]
        if not args.skip_compact and "compact_gurobi" in args.methods:
            rows.append(run_compact(instance, result_root, args))

        needs_warmstart = any(config["use_warmstart"] for config in selected_configs)
        warmstarts = ensure_warmstart(instance, args) if needs_warmstart else None
        for config in selected_configs:
            rows.append(run_bpc_method(instance, config, warmstarts, result_root, args))

        results_path = result_root / f"ablation_smoke_results_{args.date}{suffix}.csv"
        write_csv(results_path, rows)
        print(f"Wrote {len(rows)} rows to {results_path}")


if __name__ == "__main__":
    main()
