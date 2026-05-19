#!/usr/bin/env python3
"""Generate and run batch experiments for BI/VNS, Gurobi, and BPC variants."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gurobi import build_and_solve
from src.model.BPC_wagon.BBtree import BBTree as WagonBBTree
from src.model.BPC_compartment.BBtree import BBTree as CompartmentBBTree
from src.model.warmstart import load_wagon_warmstart_columns, load_compartment_warmstart_columns
from src.utility.config import config as Config
from src.utility.sensitivity_experiments import (
    BaseCase,
    load_scaled_candidates,
    sample_vehicle_types,
    write_instance,
)
import src.model.BPC_compartment.feasibility_check as compartment_feasibility


INSTANCE_ROOT = PROJECT_ROOT / "data" / "Instance"
RESULT_ROOT = PROJECT_ROOT / "result"

BATCH_PREFIX = "batch"
SCALE_ORDER = ["small", "medium", "large"]
SCALE_RANGES = {
    "small": range(5, 9),
    "medium": range(9, 13),
    "large": range(13, 17),
}
SEED_IDS = [1, 2]

DEFAULT_NUM_SPLITS = 3
DEFAULT_INDEP_MODE = False
DEFAULT_TIME_LIMIT = 3600.0
DEFAULT_MIP_GAP = 0.0001
POST_SOLVE_GRACE_SEC = 600.0
DEFAULT_PROFILE_GENERATOR_MODE = "gr"
DEFAULT_MAX_NODES = 5000
DEFAULT_MAX_CG_ITERS = 3000


@dataclass(frozen=True)
class PlanRow:
    base: BaseCase
    instance_id: str
    instance_name: str


def batch_tag(date_tag: str) -> str:
    return f"{BATCH_PREFIX}_{date_tag}"


def build_cases() -> List[BaseCase]:
    cases: List[BaseCase] = []
    for seed_id in SEED_IDS:
        for scale in SCALE_ORDER:
            for num_types in SCALE_RANGES[scale]:
                for num_wagons in SCALE_RANGES[scale]:
                    cases.append(BaseCase(scale=scale, num_types=num_types, num_wagons=num_wagons, seed_id=seed_id))
    return cases


def build_plan(date_tag: str, cases: Iterable[BaseCase]) -> List[PlanRow]:
    rows: List[PlanRow] = []
    prefix = batch_tag(date_tag)
    for base in cases:
        instance_id = base.base_id
        instance_name = f"{prefix}/{instance_id}"
        rows.append(PlanRow(base=base, instance_id=instance_id, instance_name=instance_name))
    return rows


def generate_instances(date_tag: str, cases: Iterable[BaseCase]) -> Tuple[pd.DataFrame, Path, Path]:
    candidates = load_scaled_candidates()
    instance_root = INSTANCE_ROOT / batch_tag(date_tag)
    result_root = RESULT_ROOT / batch_tag(date_tag)
    plan_dir = result_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    instance_root.mkdir(parents=True, exist_ok=True)

    balanced = {"small": 1 / 3, "medium": 1 / 3, "large": 1 / 3}
    plan_rows: List[Dict[str, object]] = []

    for base in cases:
        base_rng = random.Random(base.seed)
        base_sample = sample_vehicle_types(candidates, base.num_types, balanced, base_rng)
        instance_id = base.base_id
        write_instance(
            instance_root,
            instance_id,
            base,
            "batch",
            base_sample,
            mandatory_total=5 * base.num_wagons,
            optional_total=10 * base.num_wagons,
            rng=random.Random(base.seed + 11),
            extra_meta={
                "vehicle_mix_target": balanced,
                "rho_optional": 2.0,
                "replicate_id": base.seed_id,
                "batch_tag": batch_tag(date_tag),
            },
        )
        plan_rows.append(
            {
                "date_tag": date_tag,
                "batch_tag": batch_tag(date_tag),
                "instance_id": instance_id,
                "instance_name": f"{batch_tag(date_tag)}/{instance_id}",
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
            }
        )

    plan = pd.DataFrame(plan_rows)
    plan_path = plan_dir / f"batch_plan_{date_tag}.csv"
    plan.to_csv(plan_path, index=False)
    return plan, instance_root, result_root


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def append_result(path: Path, record: Dict[str, object]) -> None:
    row = pd.DataFrame([json_safe(record)])
    if path.exists():
        existing = pd.read_csv(path)
        if "run_id" in existing.columns and "run_id" in record:
            existing = existing[existing["run_id"] != record["run_id"]]
        combined = pd.concat([existing, row], ignore_index=True, sort=False) if not existing.empty else row
        combined.to_csv(path, index=False)
    else:
        row.to_csv(path, index=False)


def load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "run_id" not in df.columns:
        return set()
    return set(df["run_id"].astype(str).tolist())


def vns_executable() -> Path:
    exe_name = "vns_solver.exe" if os.name == "nt" else "vns_solver"
    return PROJECT_ROOT / "VNS_cpp" / exe_name


def parse_vns_output(stdout: str) -> Dict[str, float | None]:
    patterns = {
        "bi_objective": r"BI Objective:\s*([-0-9.eE]+)",
        "bi_loaded_length": r"BI LOADED LENGTH:\s*([-0-9.eE]+)",
        "bi_runtime_sec": r"BI Execution Time:\s*([-0-9.eE]+)\s*s",
        "vns_objective": r"Best VNS Objective:\s*([-0-9.eE]+)",
        "vns_loaded_length": r"OPTIMAL MAXIMUM LOADED LENGTH:\s*([-0-9.eE]+)",
        "vns_runtime_sec": r"VNS Execution Time:\s*([-0-9.eE]+)\s*s",
    }
    out: Dict[str, float | None] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        out[key] = float(match.group(1)) if match else None
    return out


def run_vns_solver(
    instance_name: str,
    num_splits: int,
    independent_mode_split: bool,
    time_limit: float | None,
) -> Dict[str, object]:
    exe = vns_executable()
    if not exe.exists():
        raise FileNotFoundError(f"VNS executable not found: {exe}")

    # Pre-create output folders for cross-platform safety.
    bi_dir = RESULT_ROOT / instance_name / "BI"
    vns_dir = RESULT_ROOT / instance_name / "VNS"
    bi_dir.mkdir(parents=True, exist_ok=True)
    vns_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(exe), instance_name, str(int(num_splits)), str(int(independent_mode_split))]
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT / "VNS_cpp"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=time_limit,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - t0
        stdout = exc.stdout or ""
        return {
            "status": "TIME_LIMIT",
            "return_code": None,
            "runtime_sec": float(runtime),
            "bi_runtime_sec": float(runtime),
            "bi_objective": None,
            "bi_loaded_length": None,
            "vns_objective": None,
            "vns_loaded_length": None,
            "stdout": stdout,
        }

    runtime = time.perf_counter() - t0
    parsed = parse_vns_output(completed.stdout)
    return {
        "status": "OK" if completed.returncode == 0 else "ERROR",
        "return_code": int(completed.returncode),
        "runtime_sec": float(parsed.get("vns_runtime_sec") or runtime),
        "bi_runtime_sec": float(parsed.get("bi_runtime_sec") or runtime),
        "bi_objective": parsed.get("bi_objective"),
        "bi_loaded_length": parsed.get("bi_loaded_length"),
        "vns_objective": parsed.get("vns_objective"),
        "vns_loaded_length": parsed.get("vns_loaded_length"),
        "stdout": completed.stdout,
    }


def solve_with_watchdog(
    worker,
    payload: Dict[str, object],
    time_limit: float,
    grace_sec: float,
) -> Dict[str, object]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=worker, args=(payload, queue), daemon=False)
    process.start()
    process.join(timeout=time_limit + grace_sec)

    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=30)
        return {
            "status": "TIME_LIMIT",
            "runner_status": "worker_timeout",
            "worker_timeout_sec": float(time_limit + grace_sec),
        }

    if not queue.empty():
        payload_out = queue.get()
        summary = payload_out.get("summary", {})
        if payload_out.get("ok"):
            summary["runner_status"] = "worker_completed"
            return summary
        summary["runner_status"] = "worker_exception"
        return summary

    return {
        "status": "ERROR",
        "runner_status": "worker_no_result",
    }


def gurobi_worker(payload: Dict[str, object], queue: mp.Queue) -> None:
    try:
        summary = build_and_solve(
            instance_dir=Path(str(payload["instance_dir"])),
            output_dir=Path(str(payload["output_dir"])),
            log_to_console=False,
            num_splits=int(payload["num_splits"]),
            independent_mode_split=bool(payload["independent_mode_split"]),
            objective_type=str(payload["objective_type"]),
            time_limit=float(payload["time_limit"]),
            mip_gap=float(payload["mip_gap"]),
            threads=int(payload["threads"]) if payload["threads"] is not None else None,
        )
        queue.put({"ok": True, "summary": summary})
    except BaseException as exc:
        queue.put({"ok": False, "summary": {"status": "ERROR", "error_message": repr(exc)}})


def bpc_wagon_worker(payload: Dict[str, object], queue: mp.Queue) -> None:
    try:
        old_timelimit = Config.timelimit
        Config.timelimit = float(payload["time_limit"]) if payload["time_limit"] is not None else old_timelimit
        try:
            compartment_feasibility.GLOBAL_NUM_SPLITS = int(payload["num_splits"])
            compartment_feasibility.GLOBAL_INDEP_MODE = bool(payload["independent_mode_split"])
            compartment_feasibility._SEGMENTS_CACHE.clear()

            tree = WagonBBTree(
                instance_dir=Path(str(payload["instance_dir"])),
                output_root=Path(str(payload["output_root"])),
                max_nodes=int(payload["max_nodes"]),
                max_cg_iters=int(payload["max_cg_iters"]),
                log_to_console=False,
                use_dominance=True,
                use_cuts=bool(payload["use_cuts"]),
                pricing_method=str(payload["pricing_method"]),
                num_splits=int(payload["num_splits"]),
                independent_mode_split=bool(payload["independent_mode_split"]),
                max_columns_per_pricing=int(payload["max_columns_per_pricing"]),
                profile_generator_mode=str(payload["profile_generator_mode"]),
                print_bb_progress=False,
                print_subproblem_progress=False,
                time_limit=float(payload["time_limit"]),
                mip_gap_tol=float(payload["mip_gap_tol"]),
            )

            if bool(payload["use_warmstart"]):
                bi_json = Path(str(payload["bi_json"]))
                vns_json = Path(str(payload["vns_json"]))
                load_wagon_warmstart_columns(tree.master, [(bi_json, "BI"), (vns_json, "VNS")])

            t0 = time.perf_counter()
            result = tree.solve()
            runtime_sec = time.perf_counter() - t0
        finally:
            Config.timelimit = old_timelimit

        obj = result.best_objective
        summary = {
            "status": "OK" if obj is not None else "NO_SOLUTION",
            "objective": obj,
            "loaded_length_mm": None if obj is None else -float(obj),
            "best_bound": result.best_bound,
            "bpc_gap": result.gap,
            "explored_nodes": result.explored_nodes,
            "generated_columns": result.generated_columns,
            "runtime_sec": runtime_sec,
            "master_time": tree.cg_engine.stats.master_time,
            "pricing_time": tree.cg_engine.stats.pricing_time,
            "labeling_time": tree.cg_engine.stats.labeling_time,
            "feasibility_time": tree.cg_engine.stats.feasibility_time,
            "merge_time": tree.cg_engine.stats.merge_time,
            "subproblem_solver_time": tree.cg_engine.stats.solver_time,
            "subproblem_solver_nodes": tree.cg_engine.stats.solver_nodes,
        }
        queue.put({"ok": True, "summary": summary})
    except BaseException as exc:
        queue.put({"ok": False, "summary": {"status": "ERROR", "error_message": repr(exc)}})


def bpc_compartment_worker(payload: Dict[str, object], queue: mp.Queue) -> None:
    try:
        old_timelimit = Config.timelimit
        Config.timelimit = float(payload["time_limit"]) if payload["time_limit"] is not None else old_timelimit
        try:
            compartment_feasibility.GLOBAL_NUM_SPLITS = int(payload["num_splits"])
            compartment_feasibility.GLOBAL_INDEP_MODE = bool(payload["independent_mode_split"])
            compartment_feasibility._SEGMENTS_CACHE.clear()

            tree = CompartmentBBTree(
                instance_dir=Path(str(payload["instance_dir"])),
                output_root=Path(str(payload["output_root"])),
                max_nodes=int(payload["max_nodes"]),
                max_cg_iters=int(payload["max_cg_iters"]),
                log_to_console=False,
                use_dominance=True,
                use_cuts=bool(payload["use_cuts"]),
                pricing_method=str(payload["pricing_method"]),
                use_rc_bound=True,
                use_height_order=True,
                use_local_residual_skyline=True,
                residual_profile_mode="full",
                profile_generator_mode=str(payload["profile_generator_mode"]),
                max_columns_per_subproblem=int(payload["max_columns_per_subproblem"]),
                mip_gap_tol=float(payload["mip_gap_tol"]),
                print_bb_progress=False,
                print_subproblem_progress=False,
                time_limit=float(payload["time_limit"]),
            )

            if bool(payload["use_warmstart"]):
                bi_json = Path(str(payload["bi_json"]))
                vns_json = Path(str(payload["vns_json"]))
                load_compartment_warmstart_columns(tree.master, [(bi_json, "BI"), (vns_json, "VNS")])

            t0 = time.perf_counter()
            result = tree.solve()
            runtime_sec = time.perf_counter() - t0
        finally:
            Config.timelimit = old_timelimit

        obj = result.best_objective
        summary = {
            "status": "OK" if obj is not None else "NO_SOLUTION",
            "objective": obj,
            "loaded_length_mm": None if obj is None else -float(obj),
            "best_bound": result.best_bound,
            "bpc_gap": result.gap,
            "explored_nodes": result.explored_nodes,
            "generated_columns": result.generated_columns,
            "runtime_sec": runtime_sec,
            "master_time": tree.cg_engine.stats.master_time,
            "pricing_time": tree.cg_engine.stats.pricing_time,
            "labeling_time": tree.cg_engine.stats.labeling_time,
            "feasibility_time": tree.cg_engine.stats.feasibility_time,
            "subproblem_solver_time": tree.cg_engine.stats.solver_time,
            "subproblem_solver_nodes": tree.cg_engine.stats.solver_nodes,
            "labels_feasible": tree.cg_engine.stats.labels_feasible,
            "labels_pruned_by_bound": tree.cg_engine.stats.labels_pruned_by_bound,
            "labels_pruned_by_dominance": tree.cg_engine.stats.labels_pruned_by_dominance,
            "labels_pruned_by_local_skyline": tree.cg_engine.stats.labels_pruned_by_local_skyline,
            "labels_after_dominance": tree.cg_engine.stats.labels_after_dominance,
            "reachability_probes": tree.cg_engine.stats.reachability_probes,
        }
        queue.put({"ok": True, "summary": summary})
    except BaseException as exc:
        queue.put({"ok": False, "summary": {"status": "ERROR", "error_message": repr(exc)}})


def run_plan(
    plan: List[PlanRow],
    date_tag: str,
    num_splits: int,
    independent_mode_split: bool,
    time_limit: float,
    mip_gap: float,
    max_nodes: int,
    max_cg_iters: int,
    profile_generator_mode: str,
    skip_existing: bool = True,
    max_runs: int | None = None,
    skip_heuristics: bool = False,
    use_warmstart: bool = True,
    result_suffix: str = "",
) -> Path:
    result_root = RESULT_ROOT / batch_tag(date_tag)
    result_root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{result_suffix}" if result_suffix else ""
    results_path = result_root / f"batch_results_{date_tag}{suffix}.csv"
    seen_run_ids = load_existing_run_ids(results_path)

    completed = 0
    for row in plan:
        base = row.base
        instance_dir = INSTANCE_ROOT / batch_tag(date_tag) / row.instance_id
        bi_json = result_root / row.instance_id / "BI" / "carriage_info.json"
        vns_json = result_root / row.instance_id / "VNS" / "carriage_info.json"

        def should_run(run_id: str) -> bool:
            return not (skip_existing and run_id in seen_run_ids)

        warmstart_requested = bool(use_warmstart)
        warmstart_available = bi_json.exists() and vns_json.exists()
        warmstart_needed = warmstart_requested and any(
            should_run(run_id)
            for run_id in [
                f"bpc_wagon_label_{row.instance_id}",
                f"bpc_wagon_solver_{row.instance_id}",
                f"bpc_compartment_label_{row.instance_id}",
                f"bpc_compartment_solver_{row.instance_id}",
            ]
        )
        need_vns = False
        if not skip_heuristics:
            need_vns = any(
                should_run(run_id)
                for run_id in [f"bi_{row.instance_id}", f"vns_{row.instance_id}"]
            )
            if warmstart_needed and not warmstart_available:
                need_vns = True

        if not skip_heuristics and need_vns:
            vns_summary = run_vns_solver(
                row.instance_name,
                num_splits=num_splits,
                independent_mode_split=independent_mode_split,
                time_limit=time_limit + POST_SOLVE_GRACE_SEC,
            )
            bi_run_id = f"bi_{row.instance_id}"
            if should_run(bi_run_id):
                record = {
                    "run_id": bi_run_id,
                    "method": "bi",
                    "family": "heuristic",
                    "instance_id": row.instance_id,
                    "instance_name": row.instance_name,
                    "problem_scale": base.scale,
                    "num_types_I": base.num_types,
                    "num_wagons_J": base.num_wagons,
                    "seed_id": base.seed_id,
                    "seed": base.seed,
                    "replicate_id": base.seed_id,
                    "num_splits": num_splits,
                    "independent_mode_split": independent_mode_split,
                    "time_limit": time_limit,
                    "mip_gap_target": mip_gap,
                    "status": vns_summary.get("status"),
                    "objective": vns_summary.get("bi_objective"),
                    "loaded_length_mm": vns_summary.get("bi_loaded_length"),
                    "runtime_sec": vns_summary.get("bi_runtime_sec"),
                    "result_dir": str(result_root / row.instance_id / "BI"),
                    "runner_status": "completed" if vns_summary.get("status") == "OK" else "error",
                }
                append_result(results_path, record)
                seen_run_ids.add(bi_run_id)
                completed += 1
                if max_runs is not None and completed >= max_runs:
                    break

            vns_run_id = f"vns_{row.instance_id}"
            if should_run(vns_run_id):
                record = {
                    "run_id": vns_run_id,
                    "method": "vns",
                    "family": "heuristic",
                    "instance_id": row.instance_id,
                    "instance_name": row.instance_name,
                    "problem_scale": base.scale,
                    "num_types_I": base.num_types,
                    "num_wagons_J": base.num_wagons,
                    "seed_id": base.seed_id,
                    "seed": base.seed,
                    "replicate_id": base.seed_id,
                    "num_splits": num_splits,
                    "independent_mode_split": independent_mode_split,
                    "time_limit": time_limit,
                    "mip_gap_target": mip_gap,
                    "status": vns_summary.get("status"),
                    "objective": vns_summary.get("vns_objective"),
                    "loaded_length_mm": vns_summary.get("vns_loaded_length"),
                    "runtime_sec": vns_summary.get("runtime_sec"),
                    "result_dir": str(result_root / row.instance_id / "VNS"),
                    "runner_status": "completed" if vns_summary.get("status") == "OK" else "error",
                }
                append_result(results_path, record)
                seen_run_ids.add(vns_run_id)
                completed += 1
                if max_runs is not None and completed >= max_runs:
                    break

        if skip_heuristics and warmstart_requested and not warmstart_available:
            print(
                f"[skip] {row.instance_id} warmstart files missing; running without warmstart",
                flush=True,
            )

        warmstart_available = bi_json.exists() and vns_json.exists()
        warmstart_enabled = warmstart_requested and warmstart_available

        gurobi_run_id = f"gurobi_{row.instance_id}"
        if should_run(gurobi_run_id):
            output_dir = result_root / "compact_gurobi" / row.instance_id
            payload = {
                "instance_dir": str(instance_dir),
                "output_dir": str(output_dir),
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "objective_type": "length",
                "time_limit": time_limit,
                "mip_gap": mip_gap,
                "threads": 1,
            }
            summary = solve_with_watchdog(gurobi_worker, payload, time_limit, POST_SOLVE_GRACE_SEC)
            record = {
                "run_id": gurobi_run_id,
                "method": "gurobi",
                "family": "compact",
                "instance_id": row.instance_id,
                "instance_name": row.instance_name,
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_target": mip_gap,
                "status": summary.get("status"),
                "objective": summary.get("obj_val"),
                "loaded_length_mm": summary.get("loaded_length_mm"),
                "runtime_sec": summary.get("runtime_sec"),
                "node_count": summary.get("node_count"),
                "mip_gap": summary.get("mip_gap"),
                "result_dir": str(output_dir),
                "runner_status": summary.get("runner_status"),
            }
            append_result(results_path, record)
            seen_run_ids.add(gurobi_run_id)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break

        bpc_wagon_label_id = f"bpc_wagon_label_{row.instance_id}"
        if should_run(bpc_wagon_label_id):
            payload = {
                "instance_dir": str(instance_dir),
                "output_root": str(result_root / "bpc_wagon_label"),
                "pricing_method": "merging",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "bi_json": str(bi_json),
                "vns_json": str(vns_json),
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_tol": mip_gap,
                "profile_generator_mode": profile_generator_mode,
                "max_nodes": max_nodes,
                "max_cg_iters": max_cg_iters,
                "max_columns_per_pricing": Config.max_wagon_pricing_columns,
            }
            summary = solve_with_watchdog(bpc_wagon_worker, payload, time_limit, POST_SOLVE_GRACE_SEC)
            record = {
                "run_id": bpc_wagon_label_id,
                "method": "bpc_wagon_label",
                "family": "bpc_wagon",
                "pricing_method": "merging",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "profile_generator_mode": profile_generator_mode,
                "instance_id": row.instance_id,
                "instance_name": row.instance_name,
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_target": mip_gap,
                "status": summary.get("status"),
                "objective": summary.get("objective"),
                "loaded_length_mm": summary.get("loaded_length_mm"),
                "runtime_sec": summary.get("runtime_sec"),
                "explored_nodes": summary.get("explored_nodes"),
                "generated_columns": summary.get("generated_columns"),
                "bpc_gap": summary.get("bpc_gap"),
                "master_time": summary.get("master_time"),
                "pricing_time": summary.get("pricing_time"),
                "labeling_time": summary.get("labeling_time"),
                "feasibility_time": summary.get("feasibility_time"),
                "merge_time": summary.get("merge_time"),
                "subproblem_solver_time": summary.get("subproblem_solver_time"),
                "subproblem_solver_nodes": summary.get("subproblem_solver_nodes"),
                "result_dir": str(result_root / "bpc_wagon_label" / row.instance_id),
                "runner_status": summary.get("runner_status"),
            }
            append_result(results_path, record)
            seen_run_ids.add(bpc_wagon_label_id)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break

        bpc_wagon_solver_id = f"bpc_wagon_solver_{row.instance_id}"
        if should_run(bpc_wagon_solver_id):
            payload = {
                "instance_dir": str(instance_dir),
                "output_root": str(result_root / "bpc_wagon_solver"),
                "pricing_method": "solver",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "bi_json": str(bi_json),
                "vns_json": str(vns_json),
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_tol": mip_gap,
                "profile_generator_mode": profile_generator_mode,
                "max_nodes": max_nodes,
                "max_cg_iters": max_cg_iters,
                "max_columns_per_pricing": Config.max_wagon_pricing_columns,
            }
            summary = solve_with_watchdog(bpc_wagon_worker, payload, time_limit, POST_SOLVE_GRACE_SEC)
            record = {
                "run_id": bpc_wagon_solver_id,
                "method": "bpc_wagon_solver",
                "family": "bpc_wagon",
                "pricing_method": "solver",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "profile_generator_mode": profile_generator_mode,
                "instance_id": row.instance_id,
                "instance_name": row.instance_name,
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_target": mip_gap,
                "status": summary.get("status"),
                "objective": summary.get("objective"),
                "loaded_length_mm": summary.get("loaded_length_mm"),
                "runtime_sec": summary.get("runtime_sec"),
                "explored_nodes": summary.get("explored_nodes"),
                "generated_columns": summary.get("generated_columns"),
                "bpc_gap": summary.get("bpc_gap"),
                "master_time": summary.get("master_time"),
                "pricing_time": summary.get("pricing_time"),
                "labeling_time": summary.get("labeling_time"),
                "feasibility_time": summary.get("feasibility_time"),
                "merge_time": summary.get("merge_time"),
                "subproblem_solver_time": summary.get("subproblem_solver_time"),
                "subproblem_solver_nodes": summary.get("subproblem_solver_nodes"),
                "result_dir": str(result_root / "bpc_wagon_solver" / row.instance_id),
                "runner_status": summary.get("runner_status"),
            }
            append_result(results_path, record)
            seen_run_ids.add(bpc_wagon_solver_id)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break

        bpc_comp_label_id = f"bpc_compartment_label_{row.instance_id}"
        if should_run(bpc_comp_label_id):
            payload = {
                "instance_dir": str(instance_dir),
                "output_root": str(result_root / "bpc_compartment_label"),
                "pricing_method": "labeling",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "bi_json": str(bi_json),
                "vns_json": str(vns_json),
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_tol": mip_gap,
                "profile_generator_mode": profile_generator_mode,
                "max_nodes": max_nodes,
                "max_cg_iters": max_cg_iters,
                "max_columns_per_subproblem": Config.max_compartment_pricing_columns_per_subproblem,
            }
            summary = solve_with_watchdog(bpc_compartment_worker, payload, time_limit, POST_SOLVE_GRACE_SEC)
            record = {
                "run_id": bpc_comp_label_id,
                "method": "bpc_compartment_label",
                "family": "bpc_compartment",
                "pricing_method": "labeling",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "profile_generator_mode": profile_generator_mode,
                "instance_id": row.instance_id,
                "instance_name": row.instance_name,
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_target": mip_gap,
                "status": summary.get("status"),
                "objective": summary.get("objective"),
                "loaded_length_mm": summary.get("loaded_length_mm"),
                "runtime_sec": summary.get("runtime_sec"),
                "explored_nodes": summary.get("explored_nodes"),
                "generated_columns": summary.get("generated_columns"),
                "bpc_gap": summary.get("bpc_gap"),
                "master_time": summary.get("master_time"),
                "pricing_time": summary.get("pricing_time"),
                "labeling_time": summary.get("labeling_time"),
                "feasibility_time": summary.get("feasibility_time"),
                "subproblem_solver_time": summary.get("subproblem_solver_time"),
                "subproblem_solver_nodes": summary.get("subproblem_solver_nodes"),
                "labels_feasible": summary.get("labels_feasible"),
                "labels_pruned_by_bound": summary.get("labels_pruned_by_bound"),
                "labels_pruned_by_dominance": summary.get("labels_pruned_by_dominance"),
                "labels_pruned_by_local_skyline": summary.get("labels_pruned_by_local_skyline"),
                "labels_after_dominance": summary.get("labels_after_dominance"),
                "reachability_probes": summary.get("reachability_probes"),
                "result_dir": str(result_root / "bpc_compartment_label" / row.instance_id),
                "runner_status": summary.get("runner_status"),
            }
            append_result(results_path, record)
            seen_run_ids.add(bpc_comp_label_id)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break

        bpc_comp_solver_id = f"bpc_compartment_solver_{row.instance_id}"
        if should_run(bpc_comp_solver_id):
            payload = {
                "instance_dir": str(instance_dir),
                "output_root": str(result_root / "bpc_compartment_solver"),
                "pricing_method": "solver",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "bi_json": str(bi_json),
                "vns_json": str(vns_json),
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_tol": mip_gap,
                "profile_generator_mode": profile_generator_mode,
                "max_nodes": max_nodes,
                "max_cg_iters": max_cg_iters,
                "max_columns_per_subproblem": Config.max_compartment_pricing_columns_per_subproblem,
            }
            summary = solve_with_watchdog(bpc_compartment_worker, payload, time_limit, POST_SOLVE_GRACE_SEC)
            record = {
                "run_id": bpc_comp_solver_id,
                "method": "bpc_compartment_solver",
                "family": "bpc_compartment",
                "pricing_method": "solver",
                "use_warmstart": warmstart_enabled,
                "use_cuts": True,
                "profile_generator_mode": profile_generator_mode,
                "instance_id": row.instance_id,
                "instance_name": row.instance_name,
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_target": mip_gap,
                "status": summary.get("status"),
                "objective": summary.get("objective"),
                "loaded_length_mm": summary.get("loaded_length_mm"),
                "runtime_sec": summary.get("runtime_sec"),
                "explored_nodes": summary.get("explored_nodes"),
                "generated_columns": summary.get("generated_columns"),
                "bpc_gap": summary.get("bpc_gap"),
                "master_time": summary.get("master_time"),
                "pricing_time": summary.get("pricing_time"),
                "labeling_time": summary.get("labeling_time"),
                "feasibility_time": summary.get("feasibility_time"),
                "subproblem_solver_time": summary.get("subproblem_solver_time"),
                "subproblem_solver_nodes": summary.get("subproblem_solver_nodes"),
                "labels_feasible": summary.get("labels_feasible"),
                "labels_pruned_by_bound": summary.get("labels_pruned_by_bound"),
                "labels_pruned_by_dominance": summary.get("labels_pruned_by_dominance"),
                "labels_pruned_by_local_skyline": summary.get("labels_pruned_by_local_skyline"),
                "labels_after_dominance": summary.get("labels_after_dominance"),
                "reachability_probes": summary.get("reachability_probes"),
                "result_dir": str(result_root / "bpc_compartment_solver" / row.instance_id),
                "runner_status": summary.get("runner_status"),
            }
            append_result(results_path, record)
            seen_run_ids.add(bpc_comp_solver_id)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break

        bpc_comp_label_no_ws_id = f"bpc_compartment_label_no_ws_no_cut_{row.instance_id}"
        if should_run(bpc_comp_label_no_ws_id):
            payload = {
                "instance_dir": str(instance_dir),
                "output_root": str(result_root / "bpc_compartment_label_no_ws_no_cut"),
                "pricing_method": "labeling",
                "use_warmstart": False,
                "use_cuts": False,
                "bi_json": str(bi_json),
                "vns_json": str(vns_json),
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_tol": mip_gap,
                "profile_generator_mode": profile_generator_mode,
                "max_nodes": max_nodes,
                "max_cg_iters": max_cg_iters,
                "max_columns_per_subproblem": Config.max_compartment_pricing_columns_per_subproblem,
            }
            summary = solve_with_watchdog(bpc_compartment_worker, payload, time_limit, POST_SOLVE_GRACE_SEC)
            record = {
                "run_id": bpc_comp_label_no_ws_id,
                "method": "bpc_compartment_label_no_ws_no_cut",
                "family": "bpc_compartment",
                "pricing_method": "labeling",
                "use_warmstart": False,
                "use_cuts": False,
                "profile_generator_mode": profile_generator_mode,
                "instance_id": row.instance_id,
                "instance_name": row.instance_name,
                "problem_scale": base.scale,
                "num_types_I": base.num_types,
                "num_wagons_J": base.num_wagons,
                "seed_id": base.seed_id,
                "seed": base.seed,
                "replicate_id": base.seed_id,
                "num_splits": num_splits,
                "independent_mode_split": independent_mode_split,
                "time_limit": time_limit,
                "mip_gap_target": mip_gap,
                "status": summary.get("status"),
                "objective": summary.get("objective"),
                "loaded_length_mm": summary.get("loaded_length_mm"),
                "runtime_sec": summary.get("runtime_sec"),
                "explored_nodes": summary.get("explored_nodes"),
                "generated_columns": summary.get("generated_columns"),
                "bpc_gap": summary.get("bpc_gap"),
                "master_time": summary.get("master_time"),
                "pricing_time": summary.get("pricing_time"),
                "labeling_time": summary.get("labeling_time"),
                "feasibility_time": summary.get("feasibility_time"),
                "subproblem_solver_time": summary.get("subproblem_solver_time"),
                "subproblem_solver_nodes": summary.get("subproblem_solver_nodes"),
                "labels_feasible": summary.get("labels_feasible"),
                "labels_pruned_by_bound": summary.get("labels_pruned_by_bound"),
                "labels_pruned_by_dominance": summary.get("labels_pruned_by_dominance"),
                "labels_pruned_by_local_skyline": summary.get("labels_pruned_by_local_skyline"),
                "labels_after_dominance": summary.get("labels_after_dominance"),
                "reachability_probes": summary.get("reachability_probes"),
                "result_dir": str(result_root / "bpc_compartment_label_no_ws_no_cut" / row.instance_id),
                "runner_status": summary.get("runner_status"),
            }
            append_result(results_path, record)
            seen_run_ids.add(bpc_comp_label_no_ws_id)
            completed += 1
            if max_runs is not None and completed >= max_runs:
                break

        if max_runs is not None and completed >= max_runs:
            break

    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiments for BI/VNS, Gurobi, and BPC.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date tag for instances/results.")
    parser.add_argument("--generate", action="store_true", help="Generate instances and plan.")
    parser.add_argument("--run", action="store_true", help="Run all methods on the plan.")
    parser.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=DEFAULT_INDEP_MODE)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--max-cg-iters", type=int, default=DEFAULT_MAX_CG_ITERS)
    parser.add_argument("--profile-generator-mode", choices=["gr", "ex", "hyb"], default=DEFAULT_PROFILE_GENERATOR_MODE)
    parser.add_argument("--skip-heuristics", action="store_true", help="Skip BI/VNS runs.")
    parser.add_argument("--no-warmstart", action="store_true", help="Disable warmstart columns for BPC runs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for this trial run.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Rerun rows even if results exist.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of parallel shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index, from 0 to num-shards-1.")
    parser.add_argument("--result-suffix", type=str, default="", help="Suffix for shard-specific result CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.generate and not args.run:
        args.generate = True

    cases = build_cases()
    plan = build_plan(args.date, cases)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards - 1]")

    if args.run and args.num_shards > 1:
        plan = [row for idx, row in enumerate(plan) if idx % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(plan)} instances")

    if args.generate:
        plan_df, instance_root, result_root = generate_instances(args.date, cases)
        print(f"Generated {len(plan_df)} instances")
        print(f"Instances: {instance_root}")
        print(f"Results: {result_root}")

    if args.run:
        results_path = run_plan(
            plan,
            date_tag=args.date,
            num_splits=args.num_splits,
            independent_mode_split=args.independent_mode_split,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            max_nodes=args.max_nodes,
            max_cg_iters=args.max_cg_iters,
            profile_generator_mode=args.profile_generator_mode,
            skip_existing=not args.no_skip_existing,
            max_runs=args.max_runs,
            use_warmstart=not args.no_warmstart,
            result_suffix=args.result_suffix,
        )
        print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
