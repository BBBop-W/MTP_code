#!/usr/bin/env python3
"""Run the 96-case numerical batch for the C++ core methods.

Default method set:
- BI construction heuristic
- VNS heuristic
- compact Gurobi
- C++ BPC wagon label pricing
- C++ BPC wagon solver pricing
- C++ BPC compartment label pricing
- C++ BPC compartment solver pricing
- C++ BPC compartment label pricing with EX, BI warm start only, and no cuts

The four accelerated BPC methods always run with cuts and BI/VNS warm-start
columns. The explicit EX baseline uses only BI warm-start columns and no cuts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.batch_experiments import (  # noqa: E402
    DEFAULT_INDEP_MODE,
    DEFAULT_MIP_GAP,
    DEFAULT_NUM_SPLITS,
    DEFAULT_PROFILE_GENERATOR_MODE,
    DEFAULT_TIME_LIMIT,
    INSTANCE_ROOT,
    POST_SOLVE_GRACE_SEC,
    RESULT_ROOT,
    batch_tag,
    build_cases,
    build_plan,
    generate_instances,
    gurobi_worker,
    json_safe,
    load_existing_run_ids,
    run_vns_solver,
    solve_with_watchdog,
)


ALL_METHODS = [
    "bi",
    "vns",
    "gurobi",
    "bpc_wagon_label",
    "bpc_wagon_solver",
    "bpc_compartment_label",
    "bpc_compartment_solver",
    "bpc_compartment_label_ex_bi_ws_no_cut",
]

BPC_METHODS = {
    "bpc_wagon_label": {
        "method": "wagon",
        "pricing_backend": "label",
        "family": "bpc_wagon",
        "pricing_method": "label",
        "use_warmstart": True,
        "warmstart_sources": ["BI", "VNS"],
        "use_cuts": True,
        "profile_generator_mode": None,
    },
    "bpc_wagon_solver": {
        "method": "wagon",
        "pricing_backend": "solver",
        "family": "bpc_wagon",
        "pricing_method": "solver",
        "use_warmstart": True,
        "warmstart_sources": ["BI", "VNS"],
        "use_cuts": True,
        "profile_generator_mode": None,
    },
    "bpc_compartment_label": {
        "method": "compartment",
        "pricing_backend": "label",
        "family": "bpc_compartment",
        "pricing_method": "label",
        "use_warmstart": True,
        "warmstart_sources": ["BI", "VNS"],
        "use_cuts": True,
        "profile_generator_mode": None,
    },
    "bpc_compartment_solver": {
        "method": "compartment",
        "pricing_backend": "solver",
        "family": "bpc_compartment",
        "pricing_method": "solver",
        "use_warmstart": True,
        "warmstart_sources": ["BI", "VNS"],
        "use_cuts": True,
        "profile_generator_mode": None,
    },
    "bpc_compartment_label_ex_bi_ws_no_cut": {
        "method": "compartment",
        "pricing_backend": "label",
        "family": "bpc_compartment",
        "pricing_method": "label_ex_bi_ws_no_cut",
        "use_warmstart": True,
        "warmstart_sources": ["BI"],
        "use_cuts": False,
        "profile_generator_mode": "ex",
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
    "seed",
    "replicate_id",
    "num_splits",
    "independent_mode_split",
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
    "profile_generator_mode",
    "generated_columns",
    "total_columns",
    "warmstart_incumbent",
    "warmstart_loaded_length_mm",
    "warmstart_added",
    "warmstart_skipped_infeasible",
    "warmstart_skipped_duplicate",
    "warmstart_skipped_empty",
    "warmstart_skipped_unknown",
    "warmstart_skipped_over_limit",
    "bi_loaded_length_mm",
    "bi_runtime_sec",
    "master_time",
    "pricing_time",
    "result_dir",
    "runner_status",
    "stderr",
]


def append_result(path: Path, record: Dict[str, object]) -> None:
    row = pd.DataFrame([json_safe(record)])
    if path.exists():
        existing = pd.read_csv(path)
        if "run_id" in existing.columns and "run_id" in record:
            existing = existing[existing["run_id"] != record["run_id"]]
        combined = pd.concat([existing, row], ignore_index=True, sort=False) if not existing.empty else row
    else:
        combined = row
    ordered = [col for col in RESULT_COLUMNS if col in combined.columns]
    ordered.extend(col for col in combined.columns if col not in ordered)
    combined = combined.reindex(columns=ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def default_bpc_exe() -> Path:
    exe_name = "bpc_label_solver.exe" if os.name == "nt" else "bpc_label_solver"
    return PROJECT_ROOT / "BPC_label_cpp" / exe_name


def default_vns_exe() -> Path:
    exe_name = "vns_solver.exe" if os.name == "nt" else "vns_solver"
    return PROJECT_ROOT / "VNS_cpp" / exe_name


def bpc_status(payload: Dict[str, object], mip_gap: float, require_warmstart: bool) -> str:
    if require_warmstart and not payload.get("warmstart_incumbent", False):
        return "WARMSTART_NO_INCUMBENT"
    if not payload.get("has_incumbent", False):
        return "NO_SOLUTION"
    gap = payload.get("gap")
    if gap is None:
        return "INCUMBENT"
    return "OK" if float(gap) <= float(mip_gap) else "GAP"


def run_cpp_bpc(
    *,
    method_name: str,
    instance_dir: Path,
    bi_json: Path | None,
    vns_json: Path | None,
    args: argparse.Namespace,
) -> Dict[str, object]:
    config = BPC_METHODS[method_name]
    bpc_method = str(config["method"])
    pricing_backend = str(config["pricing_backend"])
    family = str(config["family"])
    pricing_method = str(config["pricing_method"])
    warmstart_sources = list(config.get("warmstart_sources", []))
    use_warmstart = bool(config["use_warmstart"]) and bool(warmstart_sources)
    use_cuts = bool(config["use_cuts"])
    profile_mode = str(config["profile_generator_mode"] or args.profile_generator_mode)
    exe = resolve_repo_path(args.bpc_exe)
    if not exe.exists():
        raise FileNotFoundError(f"C++ BPC executable not found: {exe}")

    max_columns = args.solver_columns if pricing_backend == "solver" else args.label_columns
    cmd = [
        str(exe),
        "--instance",
        str(instance_dir),
        "--method",
        bpc_method,
        "--pricing-backend",
        pricing_backend,
        "--max-nodes",
        str(args.max_nodes),
        "--max-cg-iters",
        str(args.max_cg_iters),
        "--max-columns-per-pricing",
        str(max_columns),
        "--max-columns-per-subproblem",
        str(max_columns),
        "--time-limit",
        str(args.time_limit),
        "--mip-gap",
        str(args.mip_gap),
        "--num-splits",
        str(args.num_splits),
        "--independent-mode-split",
        "true" if args.independent_mode_split else "false",
        "--use-cuts",
        "true" if use_cuts else "false",
        "--profile-generator-mode",
        profile_mode,
        "--residual-profile-mode",
        args.residual_profile_mode,
        "--quiet",
    ]
    if use_warmstart:
        source_paths = {"BI": bi_json, "VNS": vns_json}
        for source in warmstart_sources:
            path = source_paths.get(source)
            if path is None:
                raise ValueError(f"{method_name} requires {source} warm-start JSON")
            cmd.extend(["--warmstart-json", f"{source}:{path}"])

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT / "BPC_label_cpp",
            text=True,
            capture_output=True,
            timeout=args.time_limit + POST_SOLVE_GRACE_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - started
        return {
            "method": method_name,
            "family": family,
            "pricing_method": pricing_method,
            "use_warmstart": use_warmstart,
            "warmstart_source": "+".join(warmstart_sources) if use_warmstart else "",
            "use_cuts": use_cuts,
            "profile_generator_mode": profile_mode,
            "status": "TIME_LIMIT",
            "runtime_sec": runtime,
            "runner_status": "worker_timeout",
            "stderr": str(exc),
        }

    runtime = time.perf_counter() - started
    if completed.returncode not in (0, 1):
        return {
            "method": method_name,
            "family": family,
            "pricing_method": pricing_method,
            "use_warmstart": use_warmstart,
            "warmstart_source": "+".join(warmstart_sources) if use_warmstart else "",
            "use_cuts": use_cuts,
            "profile_generator_mode": profile_mode,
            "status": "ERROR",
            "runtime_sec": runtime,
            "runner_status": f"error:{completed.returncode}",
            "stderr": completed.stderr.strip(),
        }

    payload = json.loads(completed.stdout)
    status = bpc_status(payload, args.mip_gap, require_warmstart=use_warmstart)
    if use_warmstart and args.strict_warmstart and status == "WARMSTART_NO_INCUMBENT":
        raise RuntimeError(
            f"{method_name} on {instance_dir.name} did not obtain an incumbent from warmstart; "
            f"warmstart_added={payload.get('warmstart_added')}"
        )

    return {
        "method": method_name,
        "family": family,
        "pricing_method": pricing_method,
        "use_warmstart": use_warmstart,
        "warmstart_source": "+".join(warmstart_sources) if use_warmstart else "",
        "use_cuts": use_cuts,
        "profile_generator_mode": profile_mode,
        "status": status,
        "objective": payload.get("best_objective"),
        "loaded_length_mm": payload.get("loaded_length_mm"),
        "runtime_sec": payload.get("wall_time", runtime),
        "gap": payload.get("gap"),
        "nodes": payload.get("explored_nodes"),
        "best_bound": payload.get("best_bound"),
        "generated_columns": payload.get("generated_columns"),
        "total_columns": payload.get("total_columns"),
        "warmstart_incumbent": payload.get("warmstart_incumbent") if use_warmstart else False,
        "warmstart_loaded_length_mm": payload.get("warmstart_loaded_length_mm") if use_warmstart else "",
        "warmstart_added": payload.get("warmstart_added") if use_warmstart else 0,
        "warmstart_skipped_infeasible": payload.get("warmstart_skipped_infeasible") if use_warmstart else 0,
        "warmstart_skipped_duplicate": payload.get("warmstart_skipped_duplicate") if use_warmstart else 0,
        "warmstart_skipped_empty": payload.get("warmstart_skipped_empty") if use_warmstart else 0,
        "warmstart_skipped_unknown": payload.get("warmstart_skipped_unknown") if use_warmstart else 0,
        "warmstart_skipped_over_limit": payload.get("warmstart_skipped_over_limit") if use_warmstart else 0,
        "master_time": payload.get("master_time"),
        "pricing_time": payload.get("pricing_time"),
        "runner_status": "completed",
        "stderr": completed.stderr.strip(),
    }


def run_compact_gurobi(row, instance_dir: Path, result_root: Path, args: argparse.Namespace) -> Dict[str, object]:
    output_dir = result_root / "cpp_core_compact_gurobi" / row.instance_id
    payload = {
        "instance_dir": str(instance_dir),
        "output_dir": str(output_dir),
        "num_splits": args.num_splits,
        "independent_mode_split": args.independent_mode_split,
        "objective_type": "length",
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "threads": 1,
    }
    summary = solve_with_watchdog(gurobi_worker, payload, args.time_limit, POST_SOLVE_GRACE_SEC)
    return {
        "method": "gurobi",
        "family": "compact",
        "status": summary.get("status"),
        "objective": summary.get("obj_val"),
        "loaded_length_mm": summary.get("loaded_length_mm"),
        "runtime_sec": summary.get("runtime_sec"),
        "gap": summary.get("mip_gap"),
        "nodes": summary.get("node_count"),
        "best_bound": summary.get("obj_bound"),
        "result_dir": str(output_dir),
        "runner_status": summary.get("runner_status"),
        "stderr": summary.get("error_message", ""),
    }


def base_record(row, args: argparse.Namespace) -> Dict[str, object]:
    base = row.base
    return {
        "instance_id": row.instance_id,
        "instance_name": row.instance_name,
        "problem_scale": base.scale,
        "num_types_I": base.num_types,
        "num_wagons_J": base.num_wagons,
        "seed_id": base.seed_id,
        "seed": base.seed,
        "replicate_id": base.seed_id,
        "num_splits": args.num_splits,
        "independent_mode_split": args.independent_mode_split,
        "time_limit": args.time_limit,
        "mip_gap_target": args.mip_gap,
    }


def finalize_record(row, method: str, payload: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    record = base_record(row, args)
    record.update(payload)
    record["run_id"] = f"{method}_{row.instance_id}"
    record["method"] = method
    return record


def run_plan(plan, args: argparse.Namespace) -> Path:
    result_root = RESULT_ROOT / batch_tag(args.date)
    result_root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.result_suffix}" if args.result_suffix else ""
    results_path = result_root / f"cpp_core_results_{args.date}{suffix}.csv"
    seen_run_ids = load_existing_run_ids(results_path)
    selected = set(args.methods)

    completed = 0
    for row in plan:
        instance_dir = INSTANCE_ROOT / batch_tag(args.date) / row.instance_id
        if not instance_dir.exists():
            raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

        bi_json = result_root / row.instance_id / "BI" / "carriage_info.json"
        vns_json = result_root / row.instance_id / "VNS" / "carriage_info.json"

        def should_run(method: str) -> bool:
            return not (args.skip_existing and f"{method}_{row.instance_id}" in seen_run_ids)

        pending_warm_bpc = [
            method
            for method, config in BPC_METHODS.items()
            if method in selected and bool(config["use_warmstart"]) and should_run(method)
        ]
        required_warmstart_sources = {
            source
            for method in pending_warm_bpc
            for source in BPC_METHODS[method].get("warmstart_sources", [])
        }
        warmstart_paths = {"BI": bi_json, "VNS": vns_json}
        missing_warmstart_sources = [
            source for source in sorted(required_warmstart_sources) if not warmstart_paths[source].exists()
        ]
        need_bi_for_method = "bi" in selected and should_run("bi")
        need_vns_for_method = "vns" in selected and should_run("vns")
        need_vns_for_warmstart = bool(pending_warm_bpc) and (
            args.refresh_warmstart or bool(missing_warmstart_sources)
        )

        vns_summary: Dict[str, object] | None = None
        if need_bi_for_method or need_vns_for_method or need_vns_for_warmstart:
            vns_summary = run_vns_solver(
                row.instance_name,
                num_splits=args.num_splits,
                independent_mode_split=args.independent_mode_split,
                time_limit=args.warmstart_time_limit,
                vns_exe=args.vns_exe,
            )

        if need_bi_for_method:
            assert vns_summary is not None
            record = finalize_record(
                row,
                "bi",
                {
                    "family": "heuristic",
                    "status": vns_summary.get("status"),
                    "loaded_length_mm": vns_summary.get("bi_loaded_length"),
                    "runtime_sec": vns_summary.get("bi_runtime_sec"),
                    "result_dir": str(result_root / row.instance_id / "BI"),
                    "runner_status": "completed" if vns_summary.get("status") == "OK" else "error",
                },
                args,
            )
            append_result(results_path, record)
            seen_run_ids.add(record["run_id"])
            completed += 1
            if args.max_runs is not None and completed >= args.max_runs:
                break

        if need_vns_for_method:
            assert vns_summary is not None
            record = finalize_record(
                row,
                "vns",
                {
                    "family": "heuristic",
                    "status": vns_summary.get("status"),
                    "loaded_length_mm": vns_summary.get("vns_loaded_length"),
                    "runtime_sec": vns_summary.get("runtime_sec"),
                    "bi_loaded_length_mm": vns_summary.get("bi_loaded_length"),
                    "bi_runtime_sec": vns_summary.get("bi_runtime_sec"),
                    "result_dir": str(result_root / row.instance_id / "VNS"),
                    "runner_status": "completed" if vns_summary.get("status") == "OK" else "error",
                },
                args,
            )
            append_result(results_path, record)
            seen_run_ids.add(record["run_id"])
            completed += 1
            if args.max_runs is not None and completed >= args.max_runs:
                break

        if "gurobi" in selected and should_run("gurobi"):
            record = finalize_record(
                row,
                "gurobi",
                run_compact_gurobi(row, instance_dir, result_root, args),
                args,
            )
            append_result(results_path, record)
            seen_run_ids.add(record["run_id"])
            completed += 1
            if args.max_runs is not None and completed >= args.max_runs:
                break

        if pending_warm_bpc:
            missing = [
                str(warmstart_paths[source])
                for source in sorted(required_warmstart_sources)
                if not warmstart_paths[source].exists()
            ]
            if missing:
                raise RuntimeError(
                    f"{row.instance_id}: warmstart files missing; refusing to run BPC without warmstart. "
                    f"Missing: {missing}"
                )

        for method in ALL_METHODS:
            if method not in BPC_METHODS or method not in selected or not should_run(method):
                continue
            record = finalize_record(
                row,
                method,
                run_cpp_bpc(
                    method_name=method,
                    instance_dir=instance_dir,
                    bi_json=bi_json if "BI" in BPC_METHODS[method].get("warmstart_sources", []) else None,
                    vns_json=vns_json if "VNS" in BPC_METHODS[method].get("warmstart_sources", []) else None,
                    args=args,
                ),
                args,
            )
            append_result(results_path, record)
            seen_run_ids.add(record["run_id"])
            completed += 1
            if args.max_runs is not None and completed >= args.max_runs:
                break

        if args.max_runs is not None and completed >= args.max_runs:
            break

    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C++ core numerical batch: 96 cases x 8 methods.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=DEFAULT_INDEP_MODE)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--profile-generator-mode", choices=["gr", "ex", "hyb"], default=DEFAULT_PROFILE_GENERATOR_MODE)
    parser.add_argument("--residual-profile-mode", choices=["full", "fans_diag"], default="full")
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-cg-iters", type=int, default=3000)
    parser.add_argument("--label-columns", type=int, default=20)
    parser.add_argument("--solver-columns", type=int, default=1)
    parser.add_argument("--warmstart-time-limit", type=float, default=DEFAULT_TIME_LIMIT + POST_SOLVE_GRACE_SEC)
    parser.add_argument("--refresh-warmstart", action="store_true")
    parser.add_argument("--strict-warmstart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--bpc-exe", type=Path, default=default_bpc_exe())
    parser.add_argument("--vns-exe", type=Path, default=default_vns_exe())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.generate and not args.run:
        args.generate = True

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards - 1]")

    cases = build_cases()
    plan = build_plan(args.date, cases)

    if args.generate:
        plan_df, instance_root, result_root = generate_instances(args.date, cases)
        print(f"Generated {len(plan_df)} instances")
        print(f"Instances: {instance_root}")
        print(f"Results: {result_root}")

    if args.run:
        if args.num_shards > 1:
            plan = [row for idx, row in enumerate(plan) if idx % args.num_shards == args.shard_id]
            print(f"Shard {args.shard_id}/{args.num_shards}: {len(plan)} instances")
        if args.max_instances is not None:
            plan = plan[: args.max_instances]
        results_path = run_plan(plan, args)
        print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
