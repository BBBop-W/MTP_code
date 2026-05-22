"""Compare C++ label BPC against Python BPC and compact Gurobi."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def _run_heuristic_warmstart(instance_name: str, args: argparse.Namespace) -> None:
    exe = _resolve_repo_path(args.vns_exe)
    if not exe.exists():
        raise FileNotFoundError(f"VNS executable not found: {exe}")
    cmd = [
        str(exe),
        instance_name,
        str(args.num_splits),
        "1" if args.independent_mode_split else "0",
    ]
    timeout = args.warmstart_time_limit if args.warmstart_time_limit > 0 else None
    completed = subprocess.run(
        cmd,
        cwd=exe.parent,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            f"warmstart heuristic failed for {instance_name} with code {completed.returncode}:\n{tail}"
        )


def _warmstart_specs(instance_name: str, args: argparse.Namespace) -> List[str]:
    items = list(args.warmstart_json)
    if items:
        return items
    if args.warmstart_source == "none":
        if args.require_warmstart_incumbent:
            raise ValueError("--warmstart-source none conflicts with --require-warmstart-incumbent")
        return []

    bi_json = RESULT_ROOT / instance_name / "BI" / "carriage_info.json"
    vns_json = RESULT_ROOT / instance_name / "VNS" / "carriage_info.json"
    if args.refresh_warmstart or not (bi_json.exists() and vns_json.exists()):
        _run_heuristic_warmstart(instance_name, args)
    missing = [str(path) for path in (bi_json, vns_json) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{instance_name}: warmstart files missing; refusing to run BPC without warmstart: {missing}"
        )
    return [f"BI:{bi_json}", f"VNS:{vns_json}"]


def _parse_warmstart_items(items: List[str]) -> List[tuple[Path, str]]:
    out: List[tuple[Path, str]] = []
    for item in items:
        sep = item.find(":")
        if sep < 0:
            out.append((_resolve_repo_path(item), "WS"))
        else:
            out.append((_resolve_repo_path(item[sep + 1 :]), item[:sep]))
    return out


def _run_cpp(
    exe: Path,
    instance_dir: Path,
    method: str,
    warmstart_json: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    max_columns_per_pricing = args.max_columns_per_pricing
    max_columns_per_subproblem = args.max_columns_per_subproblem
    if max_columns_per_pricing is None:
        max_columns_per_pricing = 1 if args.pricing_backend == "solver" else 20
    if max_columns_per_subproblem is None:
        max_columns_per_subproblem = 1 if args.pricing_backend == "solver" else 20
    cmd = [
        str(exe),
        "--instance",
        str(instance_dir),
        "--method",
        method,
        "--max-nodes",
        str(args.max_nodes),
        "--max-cg-iters",
        str(args.max_cg_iters),
        "--max-columns-per-pricing",
        str(max_columns_per_pricing),
        "--max-columns-per-subproblem",
        str(max_columns_per_subproblem),
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
        "--pricing-backend",
        args.pricing_backend,
        "--profile-generator-mode",
        args.profile_generator_mode,
        "--residual-profile-mode",
        args.residual_profile_mode,
        "--quiet",
    ]
    for item in warmstart_json:
        cmd.extend(["--warmstart-json", item])
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT / "BPC_label_cpp", text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    if args.require_warmstart_incumbent and not payload.get("warmstart_incumbent", False):
        raise RuntimeError(
            f"cpp {method} did not obtain an incumbent from warmstart on {instance_dir.name}; "
            f"warmstart_added={payload.get('warmstart_added')}"
        )
    payload["runner"] = f"cpp_{method}"
    return payload


def _run_python_bpc(
    instance_dir: Path,
    method: str,
    warmstart_json: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if method == "wagon":
        from src.model.BPC_wagon.BBtree import BBTree
        from src.model.warmstart import load_wagon_warmstart_columns

        kwargs = {"pricing_method": "solver" if args.pricing_backend == "solver" else "merging"}
        if args.pricing_backend == "label":
            kwargs.update(
                {
                    "max_columns_per_pricing": args.max_columns_per_pricing or 20,
                    "profile_generator_mode": args.profile_generator_mode,
                    "num_splits": args.num_splits,
                    "independent_mode_split": args.independent_mode_split,
                }
            )
    else:
        from src.model.BPC_compartment.BBtree import BBTree
        from src.model.warmstart import load_compartment_warmstart_columns

        kwargs = {"pricing_method": "solver" if args.pricing_backend == "solver" else "labeling"}
        if args.pricing_backend == "label":
            kwargs.update(
                {
                    "max_columns_per_subproblem": args.max_columns_per_subproblem or 20,
                    "profile_generator_mode": args.profile_generator_mode,
                    "residual_profile_mode": args.residual_profile_mode,
                }
            )

    t0 = time.perf_counter()
    tree = BBTree(
        instance_dir=instance_dir,
        output_root=PROJECT_ROOT / "result" / "tmp_cpp_compare" / f"python_{method}_{instance_dir.name}",
        max_nodes=args.max_nodes,
        max_cg_iters=args.max_cg_iters,
        log_to_console=False,
        use_dominance=True,
        use_cuts=args.use_cuts,
        print_bb_progress=False,
        print_subproblem_progress=False,
        time_limit=args.time_limit,
        mip_gap_tol=args.mip_gap,
        **kwargs,
    )
    warmstart_added = ""
    if warmstart_json:
        pairs = _parse_warmstart_items(warmstart_json)
        with contextlib.redirect_stdout(sys.stderr):
            if method == "wagon":
                warmstart = load_wagon_warmstart_columns(tree.master, pairs)
            else:
                warmstart = load_compartment_warmstart_columns(tree.master, pairs)
        warmstart_added = warmstart.added
        if args.require_warmstart_incumbent and warmstart.added <= 0:
            raise RuntimeError(f"python {method} did not load any warmstart columns on {instance_dir.name}")
    result = tree.solve()
    wall = time.perf_counter() - t0
    return {
        "runner": f"python_{method}",
        "pricing_backend": args.pricing_backend,
        "method": method,
        "has_incumbent": result.best_objective is not None,
        "best_objective": result.best_objective,
        "loaded_length_mm": None if result.best_objective is None else -float(result.best_objective),
        "best_bound": result.best_bound,
        "gap": result.gap,
        "explored_nodes": result.explored_nodes,
        "generated_columns": result.generated_columns,
        "wall_time": wall,
        "master_time": tree.cg_engine.stats.master_time,
        "pricing_time": tree.cg_engine.stats.pricing_time,
        "warmstart_added": warmstart_added,
        "threads": 1,
    }


def _run_compact(instance_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from src.model.gurobi import build_and_solve

    t0 = time.perf_counter()
    summary = build_and_solve(
        instance_dir=instance_dir,
        output_dir=PROJECT_ROOT / "result" / "tmp_cpp_compare" / f"compact_{instance_dir.name}",
        log_to_console=False,
        num_splits=args.num_splits,
        independent_mode_split=args.independent_mode_split,
        objective_type="length",
        time_limit=args.compact_time_limit,
        mip_gap=args.compact_mip_gap,
        threads=1,
    )
    wall = time.perf_counter() - t0
    return {
        "runner": "compact_gurobi",
        "pricing_backend": "",
        "method": "compact",
        "has_incumbent": summary.get("obj_val") is not None,
        "best_objective": summary.get("obj_val"),
        "loaded_length_mm": summary.get("loaded_length_mm"),
        "best_bound": summary.get("obj_bound"),
        "gap": summary.get("mip_gap"),
        "explored_nodes": summary.get("node_count"),
        "generated_columns": "",
        "wall_time": wall,
        "master_time": summary.get("runtime_sec"),
        "pricing_time": "",
        "threads": summary.get("threads"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=["m5c5"])
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--max-cg-iters", type=int, default=300)
    parser.add_argument("--max-columns-per-pricing", type=int, default=None)
    parser.add_argument("--max-columns-per-subproblem", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pricing-backend", default="label", choices=["label", "solver"])
    parser.add_argument("--profile-generator-mode", default="hyb", choices=["ex", "gr", "hyb"])
    parser.add_argument("--residual-profile-mode", default="full", choices=["full", "fans_diag"])
    parser.add_argument("--warmstart-json", action="append", default=[])
    parser.add_argument("--warmstart-source", default="heuristic", choices=["heuristic", "none"])
    parser.add_argument("--refresh-warmstart", action="store_true")
    parser.add_argument("--warmstart-time-limit", type=float, default=300.0)
    parser.add_argument("--require-warmstart-incumbent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vns-exe", type=Path, default=PROJECT_ROOT / "VNS_cpp" / "vns_solver")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-compact", action="store_true")
    parser.add_argument("--compact-time-limit", type=float, default=120.0)
    parser.add_argument("--compact-mip-gap", type=float, default=1e-4)
    parser.add_argument("--cpp-exe", type=Path, default=PROJECT_ROOT / "BPC_label_cpp" / "bpc_label_solver")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for name in args.instances:
        instance_dir = PROJECT_ROOT / "data" / "Instance" / name
        warmstarts = _warmstart_specs(name, args)
        for method in ["wagon", "compartment"]:
            row = _run_cpp(args.cpp_exe, instance_dir, method, warmstarts, args)
            row["instance"] = name
            rows.append(row)
            if not args.skip_python:
                py_row = _run_python_bpc(instance_dir, method, warmstarts, args)
                py_row["instance"] = name
                rows.append(py_row)
        if not args.skip_compact:
            compact = _run_compact(instance_dir, args)
            compact["instance"] = name
            rows.append(compact)

    fields = [
        "instance",
        "runner",
        "pricing_backend",
        "loaded_length_mm",
        "best_bound",
        "gap",
        "wall_time",
        "explored_nodes",
        "generated_columns",
        "warmstart_added",
        "threads",
    ]
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))


if __name__ == "__main__":
    main()
