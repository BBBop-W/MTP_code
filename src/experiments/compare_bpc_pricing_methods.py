from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.model.BPC_compartment.feasibility_check as compartment_feasibility
from src.model.BPC_compartment.BBtree import BBTree as CompartmentBBTree
from src.model.BPC_wagon.BBtree import BBTree as WagonBBTree
from src.model.gurobi import build_and_solve
from src.utility.config import config as Config


def _set_segmentation(num_splits: int, independent_mode_split: bool) -> None:
    compartment_feasibility.GLOBAL_NUM_SPLITS = int(num_splits)
    compartment_feasibility.GLOBAL_INDEP_MODE = bool(independent_mode_split)
    compartment_feasibility._SEGMENTS_CACHE.clear()


def run_method(
    family: str,
    pricing_method: str,
    instance_dir: Path,
    output_root: Path,
    max_nodes: int,
    max_cg_iters: int,
    num_splits: int,
    independent_mode_split: bool,
    mip_gap_tol: float,
    wagon_pricing_columns: int,
    compartment_columns_per_subproblem: int,
    use_cuts: bool,
    profile_generator_mode: str,
) -> Dict[str, object]:
    _set_segmentation(num_splits, independent_mode_split)
    t0 = time.perf_counter()
    if family == "wagon":
        tree = WagonBBTree(
            instance_dir=instance_dir,
            output_root=output_root / f"wagon_{pricing_method}",
            max_nodes=max_nodes,
            max_cg_iters=max_cg_iters,
            log_to_console=False,
            use_dominance=True,
            use_cuts=use_cuts,
            pricing_method=pricing_method,
            num_splits=num_splits,
            independent_mode_split=independent_mode_split,
            max_columns_per_pricing=wagon_pricing_columns,
            profile_generator_mode=profile_generator_mode,
            print_bb_progress=False,
            print_subproblem_progress=False,
        )
    elif family == "compartment":
        tree = CompartmentBBTree(
            instance_dir=instance_dir,
            output_root=output_root / f"compartment_{pricing_method}",
            max_nodes=max_nodes,
            max_cg_iters=max_cg_iters,
            log_to_console=False,
            use_dominance=True,
            use_cuts=use_cuts,
            pricing_method=pricing_method,
            residual_profile_mode="full",
            profile_generator_mode=profile_generator_mode,
            max_columns_per_subproblem=compartment_columns_per_subproblem,
            mip_gap_tol=mip_gap_tol,
            print_bb_progress=False,
            print_subproblem_progress=False,
        )
    else:
        raise ValueError(f"Unknown family: {family}")

    result = tree.solve()
    elapsed = time.perf_counter() - t0
    objective = result.best_objective
    return {
        "method": f"{family}_{pricing_method}",
        "family": family,
        "pricing_method": pricing_method,
        "use_cuts": use_cuts,
        "objective": objective,
        "loaded_length": -objective if objective is not None else None,
        "best_bound": result.best_bound,
        "bpc_gap": result.gap,
        "explored_nodes": result.explored_nodes,
        "generated_columns": result.generated_columns,
        "active_sr_cuts": len(getattr(tree.cg_engine, "active_sr_cuts", set())),
        "total_time": elapsed,
        "master_time": tree.cg_engine.stats.master_time,
        "pricing_time": tree.cg_engine.stats.pricing_time,
        "labeling_time": tree.cg_engine.stats.labeling_time,
        "feasibility_time": tree.cg_engine.stats.feasibility_time,
        "merge_time": tree.cg_engine.stats.merge_time,
        "subproblem_solver_time": tree.cg_engine.stats.solver_time,
        "subproblem_solver_nodes": tree.cg_engine.stats.solver_nodes,
        "gurobi_status": None,
    }


def run_gurobi_method(
    instance_dir: Path,
    output_root: Path,
    num_splits: int,
    independent_mode_split: bool,
    time_limit: float | None,
    mip_gap: float | None,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    summary = build_and_solve(
        instance_dir=instance_dir,
        output_dir=output_root / "compact_gurobi" / instance_dir.name,
        log_to_console=False,
        num_splits=num_splits,
        independent_mode_split=independent_mode_split,
        objective_type="length",
        time_limit=time_limit,
        mip_gap=mip_gap,
    )
    elapsed = time.perf_counter() - t0
    loaded_length = summary.get("loaded_length_mm")
    objective = -float(loaded_length) if loaded_length is not None else None
    obj_bound = summary.get("obj_bound")
    best_bound = -float(obj_bound) if obj_bound is not None else None
    return {
        "method": "compact_gurobi",
        "family": "compact",
        "pricing_method": "solver",
        "use_cuts": False,
        "objective": objective,
        "loaded_length": loaded_length,
        "best_bound": best_bound,
        "bpc_gap": summary.get("mip_gap"),
        "explored_nodes": summary.get("node_count"),
        "generated_columns": None,
        "active_sr_cuts": None,
        "total_time": elapsed,
        "master_time": None,
        "pricing_time": None,
        "labeling_time": None,
        "feasibility_time": None,
        "merge_time": None,
        "subproblem_solver_time": summary.get("runtime_sec"),
        "subproblem_solver_nodes": summary.get("node_count"),
        "gurobi_status": summary.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="m5c5")
    parser.add_argument("--instance-dir", type=Path, default=None)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-cg-iters", type=int, default=3000)
    parser.add_argument("--mip-gap-tol", type=float, default=1e-8)
    parser.add_argument("--gurobi-threads", type=int, default=None)
    parser.add_argument("--wagon-pricing-columns", type=int, default=Config.max_wagon_pricing_columns)
    parser.add_argument(
        "--compartment-columns-per-subproblem",
        dest="compartment_columns_per_subproblem",
        type=int,
        default=Config.max_compartment_pricing_columns_per_subproblem,
    )
    parser.add_argument("--use-cuts", action="store_true")
    parser.add_argument("--profile-generator-mode", choices=["ex", "gr", "hyb"], default="hyb")
    parser.add_argument("--include-gurobi", action="store_true")
    parser.add_argument("--gurobi-time-limit", type=float, default=300.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=Config.gap)
    parser.add_argument("--output-root", type=Path, default=Path("/private/tmp/mtp_bpc_pricing_compare"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    old_threads = Config.gurobi_threads
    Config.gurobi_threads = args.gurobi_threads
    try:
        instance_dir = args.instance_dir if args.instance_dir is not None else PROJECT_ROOT / "data" / "Instance" / args.instance
        rows: List[Dict[str, object]] = []
        for family, method in [
            ("wagon", "merging"),
            ("wagon", "solver"),
            ("compartment", "labeling"),
            ("compartment", "solver"),
        ]:
            row = run_method(
                family=family,
                pricing_method=method,
                instance_dir=instance_dir,
                output_root=args.output_root,
                max_nodes=args.max_nodes,
                max_cg_iters=args.max_cg_iters,
                num_splits=args.num_splits,
                independent_mode_split=args.independent_mode_split,
                mip_gap_tol=args.mip_gap_tol,
                wagon_pricing_columns=args.wagon_pricing_columns,
                compartment_columns_per_subproblem=args.compartment_columns_per_subproblem,
                use_cuts=args.use_cuts,
                profile_generator_mode=args.profile_generator_mode,
            )
            rows.append(row)
        if args.include_gurobi:
            rows.append(
                run_gurobi_method(
                    instance_dir=instance_dir,
                    output_root=args.output_root,
                    num_splits=args.num_splits,
                    independent_mode_split=args.independent_mode_split,
                    time_limit=args.gurobi_time_limit,
                    mip_gap=args.gurobi_mip_gap,
                )
            )
    finally:
        Config.gurobi_threads = old_threads

    finite_objectives = [float(row["objective"]) for row in rows if row["objective"] is not None]
    reference = min(finite_objectives) if finite_objectives else None
    for row in rows:
        row["aligned_with_best"] = (
            reference is not None
            and row["objective"] is not None
            and abs(float(row["objective"]) - float(reference)) <= 1e-5
        )
        row["abs_gap_to_best"] = None if reference is None or row["objective"] is None else abs(float(row["objective"]) - float(reference))
        row["rel_gap_to_best"] = (
            None
            if reference is None or row["objective"] is None or abs(float(reference)) <= 1e-10
            else row["abs_gap_to_best"] / abs(float(reference))
        )

    print(
        "method,use_cuts,objective,loaded_length,best_bound,bpc_gap,generated_columns,active_sr_cuts,"
        "total_time,master_time,pricing_time,labeling_time,feasibility_time,merge_time,"
        "subproblem_solver_time,subproblem_solver_nodes,abs_gap_to_best,rel_gap_to_best,aligned_with_best"
    )
    for row in rows:
        def fmt(value, digits=3):
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        print(
            f"{row['method']},{row.get('use_cuts')},{row['objective']},{row['loaded_length']},{row['best_bound']},{row['bpc_gap']},"
            f"{row['generated_columns']},{row.get('active_sr_cuts')},{fmt(row['total_time'])},{fmt(row['master_time'])},"
            f"{fmt(row['pricing_time'])},{fmt(row['labeling_time'])},{fmt(row['feasibility_time'])},"
            f"{fmt(row['merge_time'])},{fmt(row['subproblem_solver_time'])},{fmt(row['subproblem_solver_nodes'], 0)},"
            f"{row['abs_gap_to_best']},{row['rel_gap_to_best']},{row['aligned_with_best']}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
