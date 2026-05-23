#!/usr/bin/env python3
"""Run selected batch instances with compartment-label D1 and D3 settings.

This script is intentionally separate from the older 96-case batch driver. It
targets the selected ablation cases requested for the D1-vs-D3 comparison on a
local macOS checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_ROOT = PROJECT_ROOT / "data" / "Instance"
RESULT_ROOT = PROJECT_ROOT / "result"
POST_SOLVE_GRACE_SEC = 600.0

CASES = [
    "large_I13_J13_R1",
    "large_I13_J16_R2",
    "large_I14_J13_R1",
    "large_I14_J13_R2",
    "large_I14_J15_R1",
    "large_I14_J16_R1",
    "large_I15_J13_R1",
    "large_I15_J14_R2",
    "large_I15_J15_R1",
    "large_I15_J16_R2",
    "large_I16_J13_R2",
    "large_I16_J14_R1",
    "large_I16_J14_R2",
    "large_I16_J15_R2",
    "large_I16_J16_R1",
    "large_I16_J16_R2",
    "medium_I10_J09_R2",
]

METHODS: Dict[str, Dict[str, Any]] = {
    "clabel_ex_ws_cut_d1": {
        "family": "bpc_compartment",
        "pricing_method": "label_d1",
        "profile_generator_mode": "ex",
        "dominance_rule": "D1",
    },
    "clabel_hyb_ws_cut_d3": {
        "family": "bpc_compartment",
        "pricing_method": "label_d3_hyb",
        "profile_generator_mode": "hyb",
        "dominance_rule": "D3",
    },
}

RESULT_COLUMNS = [
    "run_id",
    "method",
    "family",
    "instance_id",
    "instance_name",
    "problem_scale",
    "num_types_I",
    "num_wagons_J",
    "seed_id",
    "replicate_id",
    "num_splits",
    "independent_mode_split",
    "threads",
    "time_limit",
    "mip_gap_target",
    "status",
    "loaded_length_mm",
    "runtime_sec",
    "gap",
    "nodes",
    "best_bound",
    "objective",
    "pricing_method",
    "use_warmstart",
    "warmstart_source",
    "use_cuts",
    "use_dominance",
    "use_local_d1_pruning",
    "use_rc_bound",
    "dominance_rule",
    "profile_generator_mode",
    "residual_profile_mode",
    "label_columns",
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
    "warmstart_incumbent",
    "warmstart_loaded_length_mm",
    "warmstart_added",
    "warmstart_skipped_infeasible",
    "warmstart_skipped_duplicate",
    "warmstart_skipped_empty",
    "warmstart_skipped_unknown",
    "warmstart_skipped_over_limit",
    "bi_runtime_sec",
    "vns_runtime_sec",
    "bi_loaded_length_mm",
    "vns_loaded_length_mm",
    "master_time",
    "pricing_time",
    "result_dir",
    "runner_status",
    "stderr",
]


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def append_result(path: Path, record: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_id") != record.get("run_id"):
                    rows.append(row)
    rows.append({key: json_safe(value) for key, value in record.items()})

    ordered = list(RESULT_COLUMNS)
    extra = sorted({key for row in rows for key in row if key not in ordered})
    fieldnames = ordered + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("run_id")}


def resolve_path(path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else PROJECT_ROOT / out


def default_bpc_exe() -> Path:
    exe = "bpc_label_solver.exe" if os.name == "nt" else "bpc_label_solver"
    return PROJECT_ROOT / "BPC_label_cpp" / exe


def default_vns_exe() -> Path:
    exe = "vns_solver.exe" if os.name == "nt" else "vns_solver"
    return PROJECT_ROOT / "VNS_cpp" / exe


def case_metadata(case_id: str, batch_tag: str) -> Dict[str, Any]:
    match = re.match(r"(?P<scale>[^_]+)_I(?P<I>\d+)_J(?P<J>\d+)_R(?P<R>\d+)$", case_id)
    if not match:
        raise ValueError(f"Unexpected case id: {case_id}")
    return {
        "instance_id": case_id,
        "instance_name": f"{batch_tag}/{case_id}",
        "problem_scale": match.group("scale"),
        "num_types_I": int(match.group("I")),
        "num_wagons_J": int(match.group("J")),
        "seed_id": int(match.group("R")),
        "replicate_id": int(match.group("R")),
    }


def parse_vns_output(stdout: str) -> Dict[str, Any]:
    patterns = {
        "bi_loaded_length_mm": r"BI LOADED LENGTH:\s*([-0-9.eE]+)",
        "bi_runtime_sec": r"BI Execution Time:\s*([-0-9.eE]+)\s*s",
        "vns_loaded_length_mm": r"OPTIMAL MAXIMUM LOADED LENGTH:\s*([-0-9.eE]+)",
        "vns_runtime_sec": r"VNS Execution Time:\s*([-0-9.eE]+)\s*s",
    }
    out: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        out[key] = float(match.group(1)) if match else ""
    return out


def ensure_warmstart(case_id: str, args: argparse.Namespace, result_root: Path) -> Dict[str, Any]:
    batch_tag = args.batch_tag
    bi_json = result_root / case_id / "BI" / "carriage_info.json"
    vns_json = result_root / case_id / "VNS" / "carriage_info.json"
    if not args.refresh_warmstart and bi_json.exists() and vns_json.exists():
        return {
            "bi_json": bi_json,
            "vns_json": vns_json,
            "warmstart_runner_status": "existing",
        }

    exe = resolve_path(args.vns_exe)
    if not exe.exists():
        raise FileNotFoundError(f"VNS executable not found: {exe}")

    cmd = [
        str(exe),
        f"{batch_tag}/{case_id}",
        str(args.num_splits),
        "1" if args.independent_mode_split else "0",
        str(args.warmstart_time_limit),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=exe.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.warmstart_time_limit + args.grace_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        return {
            "bi_json": bi_json,
            "vns_json": vns_json,
            "warmstart_runner_status": "timeout",
            "warmstart_stderr": str(exc),
            **parse_vns_output(stdout),
        }

    payload = {
        "bi_json": bi_json,
        "vns_json": vns_json,
        "warmstart_runner_status": "completed" if completed.returncode == 0 else f"error:{completed.returncode}",
        "warmstart_wall_time_sec": time.perf_counter() - started,
        "warmstart_stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
        **parse_vns_output(completed.stdout),
    }
    if not bi_json.exists() or not vns_json.exists():
        missing = [str(path) for path in [bi_json, vns_json] if not path.exists()]
        payload["warmstart_runner_status"] = "missing_json"
        payload["warmstart_stderr"] = f"missing warmstart JSON: {missing}"
    return payload


def parse_solver_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("solver stdout did not contain JSON")
    return json.loads(stdout[start:])


def bpc_status(payload: Dict[str, Any], mip_gap: float) -> str:
    if not payload.get("has_incumbent", False):
        return "NO_SOLUTION"
    gap = payload.get("gap")
    if gap is None:
        return "INCUMBENT"
    return "OK" if float(gap) <= float(mip_gap) else "GAP"


def run_method(
    case_id: str,
    method_name: str,
    warmstart: Dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    config = METHODS[method_name]
    exe = resolve_path(args.bpc_exe)
    if not exe.exists():
        raise FileNotFoundError(f"BPC executable not found: {exe}")

    instance_dir = INSTANCE_ROOT / args.batch_tag / case_id
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    result_dir = output_root / case_id / method_name
    result_dir.mkdir(parents=True, exist_ok=True)

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
        "true",
        "--use-dominance",
        "true",
        "--use-local-d1-pruning",
        "true",
        "--use-rc-bound",
        "true",
        "--profile-generator-mode",
        str(config["profile_generator_mode"]),
        "--residual-profile-mode",
        args.residual_profile_mode,
        "--warmstart-json",
        f"BI:{warmstart['bi_json']}",
        "--warmstart-json",
        f"VNS:{warmstart['vns_json']}",
        "--quiet",
    ]

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=exe.parent,
            text=True,
            capture_output=True,
            timeout=args.time_limit + args.grace_sec,
            check=False,
        )
        wall = time.perf_counter() - started
        stdout_path = result_dir / "solver_stdout.json"
        stderr_path = result_dir / "solver_stderr.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        if completed.stderr.strip():
            stderr_path.write_text(completed.stderr, encoding="utf-8")

        if completed.returncode not in (0, 1):
            payload: Dict[str, Any] = {}
            status = "ERROR"
            runner_status = f"error:{completed.returncode}"
        else:
            payload = parse_solver_json(completed.stdout)
            status = bpc_status(payload, args.mip_gap)
            runner_status = "completed"
        stderr = completed.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        payload = {}
        wall = time.perf_counter() - started
        status = "TIME_LIMIT"
        runner_status = "worker_timeout"
        stderr = str(exc)
    except BaseException as exc:
        payload = {}
        wall = time.perf_counter() - started
        status = "ERROR"
        runner_status = "exception"
        stderr = repr(exc)

    meta = case_metadata(case_id, args.batch_tag)
    record: Dict[str, Any] = {
        **meta,
        "run_id": f"{method_name}_{case_id}",
        "method": method_name,
        "family": config["family"],
        "num_splits": args.num_splits,
        "independent_mode_split": args.independent_mode_split,
        "threads": args.threads,
        "time_limit": args.time_limit,
        "mip_gap_target": args.mip_gap,
        "status": status,
        "loaded_length_mm": payload.get("loaded_length_mm"),
        "runtime_sec": payload.get("wall_time", wall),
        "gap": payload.get("gap"),
        "nodes": payload.get("explored_nodes"),
        "best_bound": payload.get("best_bound"),
        "objective": payload.get("best_objective"),
        "pricing_method": config["pricing_method"],
        "use_warmstart": True,
        "warmstart_source": "BI+VNS",
        "use_cuts": True,
        "use_dominance": True,
        "use_local_d1_pruning": True,
        "use_rc_bound": True,
        "dominance_rule": config["dominance_rule"],
        "profile_generator_mode": config["profile_generator_mode"],
        "residual_profile_mode": args.residual_profile_mode,
        "label_columns": args.label_columns,
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
        "warmstart_incumbent": payload.get("warmstart_incumbent"),
        "warmstart_loaded_length_mm": payload.get("warmstart_loaded_length_mm"),
        "warmstart_added": payload.get("warmstart_added"),
        "warmstart_skipped_infeasible": payload.get("warmstart_skipped_infeasible"),
        "warmstart_skipped_duplicate": payload.get("warmstart_skipped_duplicate"),
        "warmstart_skipped_empty": payload.get("warmstart_skipped_empty"),
        "warmstart_skipped_unknown": payload.get("warmstart_skipped_unknown"),
        "warmstart_skipped_over_limit": payload.get("warmstart_skipped_over_limit"),
        "bi_runtime_sec": warmstart.get("bi_runtime_sec", ""),
        "vns_runtime_sec": warmstart.get("vns_runtime_sec", ""),
        "bi_loaded_length_mm": warmstart.get("bi_loaded_length_mm", ""),
        "vns_loaded_length_mm": warmstart.get("vns_loaded_length_mm", ""),
        "master_time": payload.get("master_time"),
        "pricing_time": payload.get("pricing_time"),
        "result_dir": result_dir,
        "runner_status": runner_status,
        "stderr": stderr,
    }
    return record


def run_case(
    case_id: str,
    method_names: List[str],
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    messages = [f"=== {case_id}: warmstart ==="]
    records: List[Dict[str, Any]] = []
    warmstart = ensure_warmstart(case_id, args, RESULT_ROOT / args.batch_tag)
    if warmstart.get("warmstart_runner_status") in {"timeout", "missing_json"}:
        messages.append(f"{case_id}: warmstart issue: {warmstart.get('warmstart_stderr', '')}")

    for method_name in method_names:
        messages.append(f"=== {case_id}: {method_name} ===")
        record = run_method(case_id, method_name, warmstart, args, output_root)
        records.append(record)
        messages.append(
            f"{record['run_id']}: status={record['status']} loaded={record.get('loaded_length_mm')} "
            f"gap={record.get('gap')} time={record.get('runtime_sec')}"
        )

    return {"case_id": case_id, "records": records, "messages": messages}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected D1/D3 compartment-label BPC cases.")
    parser.add_argument("--batch-tag", default="batch_2026-05-19")
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--result-tag", default=date.today().isoformat())
    parser.add_argument("--result-suffix", default="selected_d1_d3")
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0001)
    parser.add_argument("--warmstart-time-limit", type=float, default=3600.0)
    parser.add_argument("--grace-sec", type=float, default=POST_SOLVE_GRACE_SEC)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-cg-iters", type=int, default=3000)
    parser.add_argument("--label-columns", type=int, default=0)
    parser.add_argument("--residual-profile-mode", choices=["full", "fans_diag"], default="full")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-warmstart", action="store_true")
    parser.add_argument("--bpc-exe", type=Path, default=default_bpc_exe())
    parser.add_argument("--vns-exe", type=Path, default=default_vns_exe())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = RESULT_ROOT / f"{args.batch_tag}_{args.result_suffix}_{args.result_tag}"
    results_path = output_root / f"d1_d3_results_{args.result_tag}.csv"
    seen = load_existing_run_ids(results_path)
    pending: Dict[str, List[str]] = {}

    for case_id in args.cases:
        pending[case_id] = []
        for method_name in args.methods:
            run_id = f"{method_name}_{case_id}"
            if args.skip_existing and run_id in seen:
                print(f"skip existing {run_id}", flush=True)
                continue
            pending[case_id].append(method_name)

    active_cases = [case_id for case_id in args.cases if pending[case_id]]
    if not active_cases:
        print(f"Results CSV: {results_path}")
        return

    parallel_jobs = max(1, args.parallel_jobs)
    if parallel_jobs == 1:
        for case_id in active_cases:
            result = run_case(case_id, pending[case_id], args, output_root)
            for message in result["messages"]:
                print(message, flush=True)
            for record in result["records"]:
                append_result(results_path, record)
                seen.add(record["run_id"])
        print(f"Results CSV: {results_path}")
        return

    print(f"Running {len(active_cases)} cases with parallel_jobs={parallel_jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        futures = {
            pool.submit(run_case, case_id, pending[case_id], args, output_root): case_id
            for case_id in active_cases
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                print(f"{case_id}: runner exception: {exc!r}", flush=True)
                continue

            for message in result["messages"]:
                print(message, flush=True)
            for record in result["records"]:
                append_result(results_path, record)
                seen.add(record["run_id"])

    print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
