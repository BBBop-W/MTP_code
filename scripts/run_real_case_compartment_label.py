#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CASE_TAG = "real_case_2026-05-21"
REAL_CASE_ROOT = PROJECT_ROOT / "data" / "Instance" / REAL_CASE_TAG
RESULT_ROOT = PROJECT_ROOT / "result" / REAL_CASE_TAG

DEFAULT_MIP_GAP = 0.0001


def parse_vns_output(stdout: str) -> dict[str, float | None]:
    patterns = {
        "bi_objective": r"BI Objective:\s*([-0-9.eE]+)",
        "bi_loaded_length": r"BI LOADED LENGTH:\s*([-0-9.eE]+)",
        "bi_runtime_sec": r"BI Execution Time:\s*([-0-9.eE]+)\s*s",
        "vns_objective": r"Best VNS Objective:\s*([-0-9.eE]+)",
        "vns_loaded_length": r"OPTIMAL MAXIMUM LOADED LENGTH:\s*([-0-9.eE]+)",
        "vns_runtime_sec": r"VNS Execution Time:\s*([-0-9.eE]+)\s*s",
    }
    out: dict[str, float | None] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        out[key] = float(match.group(1)) if match else None
    return out


SUMMARY_COLUMNS = [
    "算例编号",
    "抽取车厢数",
    "总车辆数",
    "平均每厢车辆数",
    "总长度_mm",
    "涉及车型数",
    "method",
    "status",
    "loaded_length_mm",
    "runtime_sec",
    "gap",
    "best_bound",
    "explored_nodes",
    "generated_columns",
    "total_columns",
    "warmstart_time_limit_sec",
    "bpc_time_limit_sec",
    "bi_runtime_sec",
    "vns_runtime_sec",
    "bi_loaded_length_mm",
    "vns_loaded_length_mm",
    "warmstart_added",
    "warmstart_incumbent",
    "warmstart_loaded_length_mm",
    "result_dir",
    "stderr",
]


def default_bpc_exe() -> Path:
    exe_name = "bpc_label_solver.exe" if os.name == "nt" else "bpc_label_solver"
    return PROJECT_ROOT / "BPC_label_cpp" / exe_name


def default_vns_exe() -> Path:
    exe_name = "vns_solver.exe" if os.name == "nt" else "vns_solver"
    return PROJECT_ROOT / "VNS_cpp" / exe_name


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: float | None,
    merge_stderr: bool = False,
) -> tuple[int | None, str, str, bool]:
    stderr_target = subprocess.STDOUT if merge_stderr else subprocess.PIPE
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=stderr_target,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout or "", "" if stderr is None else stderr, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return None, stdout or "", "" if stderr is None else stderr, True


def instance_metrics(case_id: str) -> dict[str, Any]:
    instance_dir = REAL_CASE_ROOT / case_id
    cars = pd.read_csv(instance_dir / "cars.csv", encoding="utf-8-sig")
    carriage = pd.read_csv(instance_dir / "carriage.csv", encoding="utf-8-sig")
    carriage_num = int(carriage.loc[0, "carriage_num"])
    total_vehicles = int(cars["mandatory"].sum())
    return {
        "算例编号": case_id,
        "抽取车厢数": carriage_num,
        "总车辆数": total_vehicles,
        "平均每厢车辆数": round(total_vehicles / carriage_num, 2) if carriage_num else "",
        "总长度_mm": int((cars["length"] * cars["mandatory"]).sum()),
        "涉及车型数": int(len(cars)),
    }


def write_summary_row(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_frame = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing = existing[existing["算例编号"].astype(str) != str(row["算例编号"])]
        combined = pd.concat([existing, row_frame], ignore_index=True, sort=False)
    else:
        combined = row_frame
    ordered = [col for col in SUMMARY_COLUMNS if col in combined.columns]
    ordered.extend(col for col in combined.columns if col not in ordered)
    combined = combined.reindex(columns=ordered)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_warmstart(case_id: str, args: argparse.Namespace) -> dict[str, Any]:
    vns_exe = resolve_repo_path(args.vns_exe)
    if not vns_exe.exists():
        raise FileNotFoundError(f"VNS executable not found: {vns_exe}")

    instance_name = f"{REAL_CASE_TAG}/{case_id}"
    cmd = [
        str(vns_exe),
        instance_name,
        str(args.num_splits),
        "1" if args.independent_mode_split else "0",
    ]
    if args.vns_time_limit is not None:
        cmd.append(str(args.vns_time_limit))
    started = time.perf_counter()
    return_code, stdout, _stderr, timed_out = run_command(
        cmd,
        cwd=vns_exe.parent,
        timeout=args.warmstart_timeout,
        merge_stderr=True,
    )
    wall = time.perf_counter() - started
    if timed_out:
        raise TimeoutError(f"{case_id}: VNS timed out after {args.warmstart_timeout} seconds")
    parsed = parse_vns_output(stdout)
    parsed["status"] = "OK" if return_code == 0 else f"ERROR:{return_code}"
    parsed["wall_time_sec"] = wall
    parsed["stdout"] = stdout

    result_dir = RESULT_ROOT / case_id
    bi_json = result_dir / "BI" / "carriage_info.json"
    vns_json = result_dir / "VNS" / "carriage_info.json"
    missing = [str(path) for path in (bi_json, vns_json) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{case_id}: warmstart JSON missing after heuristic run: {missing}")
    parsed["bi_json"] = bi_json
    parsed["vns_json"] = vns_json
    return parsed


def run_bpc(case_id: str, warmstart: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    bpc_exe = resolve_repo_path(args.bpc_exe)
    if not bpc_exe.exists():
        raise FileNotFoundError(f"C++ BPC executable not found: {bpc_exe}")

    instance_dir = REAL_CASE_ROOT / case_id
    cmd = [
        str(bpc_exe),
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
        str(args.bpc_time_limit),
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
        "--warmstart-json",
        f"BI:{warmstart['bi_json']}",
        "--warmstart-json",
        f"VNS:{warmstart['vns_json']}",
        "--quiet",
    ]
    started = time.perf_counter()
    return_code, stdout, stderr, timed_out = run_command(
        cmd,
        cwd=PROJECT_ROOT / "BPC_label_cpp",
        timeout=args.bpc_time_limit + args.bpc_grace,
    )
    wall = time.perf_counter() - started
    if timed_out:
        return {
            "return_code": None,
            "timed_out": True,
            "subprocess_wall_time_sec": wall,
        }, stderr.strip()

    payload: dict[str, Any]
    if stdout.strip():
        payload = json.loads(stdout)
    else:
        payload = {}
    payload["return_code"] = return_code
    payload["subprocess_wall_time_sec"] = wall
    return payload, stderr.strip()


def build_result_row(case_id: str, metrics: dict[str, Any], warmstart: dict[str, Any], payload: dict[str, Any], stderr: str, args: argparse.Namespace) -> dict[str, Any]:
    if payload.get("timed_out"):
        status = "TIMEOUT"
    elif payload.get("return_code") not in (0, 1):
        status = f"ERROR:{payload.get('return_code')}"
    elif not payload.get("has_incumbent", False):
        status = "NO_SOLUTION"
    else:
        gap = payload.get("gap")
        status = "OK" if gap is not None and float(gap) <= args.mip_gap else "GAP"

    row = dict(metrics)
    row.update(
        {
            "method": "compartment_label",
            "status": status,
            "loaded_length_mm": payload.get("loaded_length_mm", ""),
            "runtime_sec": payload.get("wall_time", payload.get("subprocess_wall_time_sec", "")),
            "gap": payload.get("gap", ""),
            "best_bound": payload.get("best_bound", ""),
            "explored_nodes": payload.get("explored_nodes", ""),
            "generated_columns": payload.get("generated_columns", ""),
            "total_columns": payload.get("total_columns", ""),
            "warmstart_time_limit_sec": "unlimited" if args.vns_time_limit is None else args.vns_time_limit,
            "bpc_time_limit_sec": args.bpc_time_limit,
            "bi_runtime_sec": warmstart.get("bi_runtime_sec", ""),
            "vns_runtime_sec": warmstart.get("vns_runtime_sec", warmstart.get("wall_time_sec", "")),
            "bi_loaded_length_mm": warmstart.get("bi_loaded_length", ""),
            "vns_loaded_length_mm": warmstart.get("vns_loaded_length", ""),
            "warmstart_added": payload.get("warmstart_added", ""),
            "warmstart_incumbent": payload.get("warmstart_incumbent", ""),
            "warmstart_loaded_length_mm": payload.get("warmstart_loaded_length_mm", ""),
            "result_dir": str(RESULT_ROOT / case_id),
            "stderr": stderr,
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real cases with C++ compartment-label BPC.")
    parser.add_argument("--instances", nargs="+", default=[f"case_{idx:02d}" for idx in range(1, 11)])
    parser.add_argument(
        "--vns-time-limit",
        type=float,
        default=None,
        help="Optional time limit passed into the C++ VNS solver. Omit to let VNS use its own termination rule.",
    )
    parser.add_argument(
        "--warmstart-timeout",
        type=float,
        default=None,
        help="Optional Python-side hard timeout for BI/VNS. Omit for no Python-side warmstart timeout.",
    )
    parser.add_argument("--bpc-time-limit", type=float, default=120.0)
    parser.add_argument("--bpc-grace", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-generator-mode", choices=["gr", "ex", "hyb"], default="hyb")
    parser.add_argument("--residual-profile-mode", choices=["full", "fans_diag"], default="full")
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-cg-iters", type=int, default=3000)
    parser.add_argument("--label-columns", type=int, default=20)
    parser.add_argument("--bpc-exe", type=Path, default=default_bpc_exe())
    parser.add_argument("--vns-exe", type=Path, default=default_vns_exe())
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "real_case_compartment_label_summary.csv")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_repo_path(args.output)

    existing_done: set[str] = set()
    if args.skip_existing and output_path.exists():
        existing = pd.read_csv(output_path)
        existing_done = set(existing["算例编号"].astype(str).tolist())

    for case_id in args.instances:
        if args.skip_existing and case_id in existing_done:
            continue
        metrics = instance_metrics(case_id)
        try:
            warmstart = run_warmstart(case_id, args)
            payload, stderr = run_bpc(case_id, warmstart, args)
            row = build_result_row(case_id, metrics, warmstart, payload, stderr, args)
        except Exception as exc:
            row = dict(metrics)
            row.update(
                {
                    "method": "compartment_label",
                    "status": "ERROR",
                    "warmstart_time_limit_sec": "unlimited" if args.vns_time_limit is None else args.vns_time_limit,
                    "bpc_time_limit_sec": args.bpc_time_limit,
                    "result_dir": str(RESULT_ROOT / case_id),
                    "stderr": str(exc),
                }
            )
        write_summary_row(output_path, row)
        print(f"{case_id}: {row['status']}")

    print(f"Summary CSV: {output_path}")


if __name__ == "__main__":
    main()
