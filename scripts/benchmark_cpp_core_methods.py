"""Benchmark the four C++ BPC variants plus compact Gurobi."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHODS = [
    ("wagon_label", "wagon", "label"),
    ("wagon_solver", "wagon", "solver"),
    ("compartment_label", "compartment", "label"),
    ("compartment_solver", "compartment", "solver"),
]


def run_cpp_method(
    exe: Path,
    instance_dir: Path,
    method_name: str,
    bpc_method: str,
    backend: str,
    warmstart_json: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cmd = [
        str(exe),
        "--instance",
        str(instance_dir),
        "--method",
        bpc_method,
        "--pricing-backend",
        backend,
        "--max-nodes",
        str(args.max_nodes),
        "--max-cg-iters",
        str(args.max_cg_iters),
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--num-splits",
        str(args.num_splits),
        "--independent-mode-split",
        "true" if args.independent_mode_split else "false",
        "--use-cuts",
        "true" if args.use_cuts else "false",
        "--profile-generator-mode",
        args.profile_generator_mode,
        "--residual-profile-mode",
        args.residual_profile_mode,
        "--quiet",
    ]
    for item in warmstart_json:
        cmd.extend(["--warmstart-json", item])
    if backend == "solver":
        cmd.extend(["--max-columns-per-pricing", str(args.solver_columns)])
        cmd.extend(["--max-columns-per-subproblem", str(args.solver_columns)])
    else:
        cmd.extend(["--max-columns-per-pricing", str(args.label_columns)])
        cmd.extend(["--max-columns-per-subproblem", str(args.label_columns)])

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT / "BPC_label_cpp",
        text=True,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - started
    if completed.returncode not in (0, 1):
        return {
            "method": method_name,
            "status": f"error:{completed.returncode}",
            "obj": "",
            "gap": "",
            "time_sec": wall,
            "best_bound": "",
            "nodes": "",
            "generated_columns": "",
            "use_cuts": args.use_cuts,
            "warmstart_added": "",
            "warmstart_incumbent": "",
            "warmstart_loaded_length_mm": "",
            "warmstart_skipped_infeasible": "",
            "warmstart_skipped_duplicate": "",
            "warmstart_skipped_empty": "",
            "warmstart_skipped_unknown": "",
            "warmstart_skipped_over_limit": "",
            "stderr": completed.stderr.strip(),
        }

    payload = json.loads(completed.stdout)
    status = "ok" if payload.get("has_incumbent") else "no_incumbent"
    if args.require_warmstart_incumbent:
        if int(payload.get("warmstart_added", 0)) <= 0:
            status = "warmstart_not_loaded"
        elif not payload.get("warmstart_incumbent", False):
            status = "warmstart_no_incumbent"
    return {
        "method": method_name,
        "status": status,
        "obj": payload.get("loaded_length_mm", ""),
        "gap": payload.get("gap", ""),
        "time_sec": payload.get("wall_time", wall),
        "best_bound": payload.get("best_bound", ""),
        "nodes": payload.get("explored_nodes", ""),
        "generated_columns": payload.get("generated_columns", ""),
        "use_cuts": args.use_cuts,
        "warmstart_added": payload.get("warmstart_added", ""),
        "warmstart_incumbent": payload.get("warmstart_incumbent", ""),
        "warmstart_loaded_length_mm": payload.get("warmstart_loaded_length_mm", ""),
        "warmstart_skipped_infeasible": payload.get("warmstart_skipped_infeasible", ""),
        "warmstart_skipped_duplicate": payload.get("warmstart_skipped_duplicate", ""),
        "warmstart_skipped_empty": payload.get("warmstart_skipped_empty", ""),
        "warmstart_skipped_unknown": payload.get("warmstart_skipped_unknown", ""),
        "warmstart_skipped_over_limit": payload.get("warmstart_skipped_over_limit", ""),
        "stderr": "",
    }


def run_compact(instance_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from src.model.gurobi import build_and_solve

    started = time.perf_counter()
    summary = build_and_solve(
        instance_dir=instance_dir,
        output_dir=PROJECT_ROOT / "result" / "tmp_cpp_core_benchmark" / instance_dir.name / "compact",
        log_to_console=False,
        num_splits=args.num_splits,
        independent_mode_split=args.independent_mode_split,
        objective_type="length",
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=1,
    )
    wall = time.perf_counter() - started
    return {
        "method": "compact_gurobi",
        "status": "ok" if summary.get("obj_val") is not None else f"status:{summary.get('status')}",
        "obj": summary.get("loaded_length_mm", ""),
        "gap": summary.get("mip_gap", ""),
        "time_sec": wall,
        "best_bound": summary.get("obj_bound", ""),
        "nodes": summary.get("node_count", ""),
        "generated_columns": "",
        "use_cuts": "",
        "warmstart_added": "",
        "warmstart_incumbent": "",
        "warmstart_loaded_length_mm": "",
        "warmstart_skipped_infeasible": "",
        "warmstart_skipped_duplicate": "",
        "warmstart_skipped_empty": "",
        "warmstart_skipped_unknown": "",
        "warmstart_skipped_over_limit": "",
        "stderr": "",
    }


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def run_heuristic_warmstart(instance_name: str, args: argparse.Namespace) -> None:
    exe = resolve_repo_path(args.vns_exe)
    if not exe.exists():
        raise FileNotFoundError(f"VNS executable not found: {exe}")

    cmd = [
        str(exe),
        instance_name,
        str(args.num_splits),
        "1" if args.independent_mode_split else "0",
    ]
    timeout = args.warmstart_time_limit if args.warmstart_time_limit > 0 else None
    try:
        completed = subprocess.run(
            cmd,
            cwd=exe.parent,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"warmstart heuristic timed out for {instance_name} after {timeout}s"
        ) from exc
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            f"warmstart heuristic failed for {instance_name} with code {completed.returncode}:\n{tail}"
        )


def warmstart_specs(instance_name: str, args: argparse.Namespace) -> List[str]:
    if args.warmstart_source == "none":
        if args.require_warmstart_incumbent:
            raise ValueError("--warmstart-source none conflicts with --require-warmstart-incumbent")
        return []
    if args.warmstart_source != "heuristic":
        raise ValueError(f"unsupported warmstart source: {args.warmstart_source}")

    bi_json = RESULT_ROOT / instance_name / "BI" / "carriage_info.json"
    vns_json = RESULT_ROOT / instance_name / "VNS" / "carriage_info.json"
    if args.refresh_warmstart or not (bi_json.exists() and vns_json.exists()):
        run_heuristic_warmstart(instance_name, args)

    missing = [str(path) for path in (bi_json, vns_json) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{instance_name}: warmstart files missing; refusing to run BPC without warmstart: {missing}"
        )
    return [f"BI:{bi_json}", f"VNS:{vns_json}"]


FIELDS = [
    "instance",
    "method",
    "status",
    "obj",
    "gap",
    "time_sec",
    "best_bound",
    "nodes",
    "generated_columns",
    "use_cuts",
    "warmstart_added",
    "warmstart_incumbent",
    "warmstart_loaded_length_mm",
    "warmstart_skipped_infeasible",
    "warmstart_skipped_duplicate",
    "warmstart_skipped_empty",
    "warmstart_skipped_unknown",
    "warmstart_skipped_over_limit",
    "stderr",
]


def emit_row(writer: csv.DictWriter, row: Dict[str, Any]) -> None:
    writer.writerow(row)
    sys.stdout.flush()


def emit(rows: Iterable[Dict[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=["m5c5", "m6c6", "m7c7"])
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--max-cg-iters", type=int, default=1000)
    parser.add_argument("--label-columns", type=int, default=20)
    parser.add_argument("--solver-columns", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmstart-source", default="heuristic", choices=["heuristic", "none"])
    parser.add_argument("--refresh-warmstart", action="store_true")
    parser.add_argument("--warmstart-time-limit", type=float, default=300.0)
    parser.add_argument("--require-warmstart-incumbent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vns-exe", type=Path, default=PROJECT_ROOT / "VNS_cpp" / "vns_solver")
    parser.add_argument("--profile-generator-mode", default="hyb", choices=["ex", "gr", "hyb"])
    parser.add_argument("--residual-profile-mode", default="full", choices=["full", "fans_diag"])
    parser.add_argument("--cpp-exe", type=Path, default=PROJECT_ROOT / "BPC_label_cpp" / "bpc_label_solver")
    parser.add_argument("--skip-compact", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for name in args.instances:
        instance_dir = PROJECT_ROOT / "data" / "Instance" / name
        warmstarts = warmstart_specs(name, args)
        for method_name, bpc_method, backend in METHODS:
            row = run_cpp_method(
                args.cpp_exe,
                instance_dir,
                method_name,
                bpc_method,
                backend,
                warmstarts,
                args,
            )
            row["instance"] = name
            rows.append(row)
            emit_row(writer, row)
        if not args.skip_compact:
            row = run_compact(instance_dir, args)
            row["instance"] = name
            rows.append(row)
            emit_row(writer, row)


if __name__ == "__main__":
    main()
