#!/usr/bin/env python3
"""Generate and run type-share D3 ablation experiments.

The experiment varies the number of car types that the C++ HYB/D3 structural
certificate recognizes as an ordered chain. Candidate instances are validated by
calling the same C++ chain diagnostic used by the solver before they are kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
INSTANCE_ROOT = PROJECT_ROOT / "data" / "Instance"
RESULT_ROOT = PROJECT_ROOT / "result"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_bpc_ablation_smoke import (  # noqa: E402
    CSV_COLUMNS as BASE_RESULT_COLUMNS,
    bool_arg,
    default_bpc_exe,
    default_vns_exe,
    ensure_warmstart,
    run_bpc_method,
    run_compact,
)


METHODS: Dict[str, Dict[str, Any]] = {
    "clabel_ex_plain": {
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
    "clabel_ex_ws_cut_no_dom": {
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
    "clabel_ex_ws_cut_d1": {
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
    "clabel_hyb_ws_cut_d3": {
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
    "clabel_d2_ws_cut_full": {
        "method": "clabel_d2_ws_cut_full",
        "family": "bpc_compartment",
        "profile_generator_mode": "d2",
        "use_warmstart": True,
        "use_cuts": True,
        "use_dominance": True,
        "use_local_d1_pruning": True,
        "use_rc_bound": True,
        "dominance_rule": "D2",
    },
}

DEFAULT_METHODS = [
    "compact_gurobi",
    "clabel_ex_plain",
    "clabel_ex_ws_cut_no_dom",
    "clabel_ex_ws_cut_d1",
    "clabel_hyb_ws_cut_d3",
]

META_COLUMNS = [
    "run_id",
    "case_id",
    "experiment_family",
    "instance_group",
    "problem_scale",
    "num_types_I",
    "num_wagons_J",
    "replicate_id",
    "safe_type_count",
    "conflict_type_count",
    "safe_type_share",
    "validation_expected_ordered_type_count",
    "validation_min_ordered_type_count",
    "validation_max_ordered_type_count",
    "validation_ordered_quantity_sum",
    "validation_total_quantity_sum",
    "validation_attempt",
    "optional_each_type",
    "total_optional",
    "mean_length",
    "mean_height",
    "length_range",
    "height_range",
]

RESULT_COLUMNS = META_COLUMNS + [col for col in BASE_RESULT_COLUMNS if col not in META_COLUMNS]


def scale_name(type_count: int) -> str:
    if type_count <= 8:
        return "small"
    if type_count <= 12:
        return "medium"
    return "large"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def default_case_id(type_count: int, safe_count: int, rep: int) -> str:
    scale = scale_name(type_count)
    return f"{scale}_I{type_count:02d}_J{type_count:02d}_S{safe_count:02d}_R{rep}"


def safe_geometry(slot: int, rep: int) -> Dict[str, Any]:
    rep_shift = 3 * (rep - 1)
    if slot == 0:
        return {
            "program": "safe_chain",
            "model": "OC0",
            "length": 3500 + rep_shift,
            "height": 1500,
        }
    if slot == 1:
        return {
            "program": "safe_chain",
            "model": "OC1",
            "length": 3500 + rep_shift,
            "height": 2100,
        }
    return {
        "program": "safe_chain",
        "model": f"OC{slot}",
        "length": 3500 + 100 * (slot - 1) + rep_shift,
        "height": max(2080, 2100 - 5 * (slot - 1)),
    }


def conflict_geometry(slot: int, rep: int, attempt: int) -> Dict[str, Any]:
    safe = safe_geometry(slot, rep)
    profile = attempt % 4
    if profile == 0:
        length_delta = 20 if slot % 2 == 0 else -20
        height = 1500 + 10 * slot
    elif profile == 1:
        length_delta = 35 if slot % 2 == 0 else -35
        height = 1510 + 8 * slot
    elif profile == 2:
        length_delta = 15
        height = 1740 - 18 * slot
    else:
        length_delta = -15
        height = 1530 + 6 * slot
    return {
        "program": "background",
        "model": f"BG{slot}",
        "length": max(3000, int(safe["length"]) + length_delta),
        "height": max(1450, height),
    }


def build_car_rows(type_count: int, safe_count: int, rep: int, attempt: int, optional_each: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for slot in range(type_count):
        if slot < safe_count:
            row = safe_geometry(slot, rep)
            group = "safe"
            safe_type = 1
        else:
            row = conflict_geometry(slot, rep, attempt)
            group = "conflict"
            safe_type = 0
        rows.append(
            {
                **row,
                "optional": optional_each,
                "mandatory": 0,
                "slot_id": slot,
                "structural_group": group,
                "safe_type": safe_type,
            }
        )
    return rows


def weighted_mean(rows: Iterable[Dict[str, Any]], field: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        demand = float(row["mandatory"]) + float(row["optional"])
        numerator += demand * float(row[field])
        denominator += demand
    return numerator / denominator if denominator else 0.0


def write_instance_files(instance_dir: Path, rows: List[Dict[str, Any]], carriages: int, meta: Dict[str, Any]) -> None:
    instance_dir.mkdir(parents=True, exist_ok=True)
    with (instance_dir / "cars.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "program",
                "model",
                "length",
                "height",
                "optional",
                "mandatory",
                "slot_id",
                "structural_group",
                "safe_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (instance_dir / "carriage.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["carriage_num"])
        writer.writeheader()
        writer.writerow({"carriage_num": carriages})
    (instance_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_solver_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("solver stdout did not contain JSON")
    return json.loads(stdout[start:])


def run_chain_diagnostic(instance_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    exe = Path(args.bpc_exe)
    if not exe.is_absolute():
        exe = PROJECT_ROOT / exe
    instance_dir = INSTANCE_ROOT / instance_name
    cmd = [
        str(exe),
        "--instance",
        str(instance_dir),
        "--method",
        "compartment",
        "--pricing-backend",
        "label",
        "--diagnose-structural-chain",
        "--profile-generator-mode",
        "hyb",
        "--residual-profile-mode",
        args.residual_profile_mode,
        "--num-splits",
        str(args.num_splits),
        "--independent-mode-split",
        bool_arg(args.independent_mode_split),
        "--quiet",
    ]
    completed = subprocess.run(
        cmd,
        cwd=exe.parent,
        text=True,
        capture_output=True,
        timeout=args.diagnostic_timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"structural diagnostic failed for {instance_name}: "
            f"returncode={completed.returncode}, stderr={completed.stderr.strip()}"
        )
    return parse_solver_json(completed.stdout)


def expected_ordered_count(safe_count: int, args: argparse.Namespace) -> int:
    if safe_count == 0 and args.allow_trivial_singleton:
        return 1
    return safe_count


def diagnostic_passes(diagnostic: Dict[str, Any], safe_count: int, args: argparse.Namespace) -> bool:
    expected = expected_ordered_count(safe_count, args)
    min_count = int(diagnostic.get("min_ordered_type_count", -1))
    max_count = int(diagnostic.get("max_ordered_type_count", -1))
    if args.validation_mode == "max":
        return max_count == expected
    return min_count == expected and max_count == expected


def summarize_geometry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [float(row["length"]) for row in rows]
    heights = [float(row["height"]) for row in rows]
    return {
        "mean_length": weighted_mean(rows, "length"),
        "mean_height": weighted_mean(rows, "height"),
        "length_range": max(lengths) - min(lengths) if lengths else 0.0,
        "height_range": max(heights) - min(heights) if heights else 0.0,
    }


def generate_one_instance(type_count: int, safe_count: int, rep: int, args: argparse.Namespace) -> Dict[str, Any]:
    case_id = default_case_id(type_count, safe_count, rep)
    instance_name = f"{args.instance_group}/{case_id}"
    instance_dir = INSTANCE_ROOT / instance_name
    carriages = type_count

    if args.skip_existing_instances and (instance_dir / "meta.json").exists():
        meta = json.loads((instance_dir / "meta.json").read_text(encoding="utf-8"))
        if (instance_dir / "structural_chain_diagnostic.json").exists():
            return meta

    last_diagnostic: Dict[str, Any] | None = None
    for attempt in range(args.max_attempts):
        rows = build_car_rows(type_count, safe_count, rep, attempt, args.optional_each_type)
        geometry = summarize_geometry(rows)
        meta = {
            "case_id": case_id,
            "instance": instance_name,
            "experiment_family": "type_share_d3_ablation",
            "instance_group": args.instance_group,
            "problem_scale": scale_name(type_count),
            "num_types_I": type_count,
            "num_wagons_J": carriages,
            "replicate_id": rep,
            "safe_type_count": safe_count,
            "conflict_type_count": type_count - safe_count,
            "safe_type_share": safe_count / type_count if type_count else 0.0,
            "validation_expected_ordered_type_count": expected_ordered_count(safe_count, args),
            "optional_each_type": args.optional_each_type,
            "total_optional": args.optional_each_type * type_count,
            "generation_attempt": attempt,
            **geometry,
        }
        write_instance_files(instance_dir, rows, carriages, meta)
        diagnostic = run_chain_diagnostic(instance_name, args)
        last_diagnostic = diagnostic
        (instance_dir / "structural_chain_diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2),
            encoding="utf-8",
        )
        if diagnostic_passes(diagnostic, safe_count, args):
            meta.update(
                {
                    "validation_attempt": attempt,
                    "validation_min_ordered_type_count": diagnostic.get("min_ordered_type_count"),
                    "validation_max_ordered_type_count": diagnostic.get("max_ordered_type_count"),
                    "validation_ordered_quantity_sum": diagnostic.get("ordered_quantity_sum"),
                    "validation_total_quantity_sum": diagnostic.get("total_quantity_sum"),
                }
            )
            (instance_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return meta

    if instance_dir.exists() and not args.keep_failed_candidates:
        shutil.rmtree(instance_dir)
    raise RuntimeError(
        f"could not validate {instance_name} after {args.max_attempts} attempts; "
        f"last diagnostic={last_diagnostic}"
    )


def generate_instances(args: argparse.Namespace) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for type_count in args.type_counts:
        for rep in range(1, args.reps + 1):
            for safe_count in range(0, type_count + 1, args.safe_step):
                meta = generate_one_instance(type_count, safe_count, rep, args)
                metas.append(meta)
                print(
                    f"generated {meta['instance']} safe={safe_count}/{type_count} "
                    f"chain={meta.get('validation_min_ordered_type_count')}-"
                    f"{meta.get('validation_max_ordered_type_count')}",
                    flush=True,
                )
    manifest_path = INSTANCE_ROOT / args.instance_group / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metas, indent=2), encoding="utf-8")
    return metas


def load_manifest(args: argparse.Namespace) -> List[Dict[str, Any]]:
    manifest_path = INSTANCE_ROOT / args.instance_group / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("run_id")}


def append_rows(path: Path, new_rows: Iterable[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    new_rows = list(new_rows)
    new_ids = {row.get("run_id") for row in new_rows}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("run_id") not in new_ids:
                    rows.append(row)
    rows.extend({key: json_safe(value) for key, value in row.items()} for row in new_rows)
    ordered = list(RESULT_COLUMNS)
    extra = sorted({key for row in rows for key in row if key not in ordered})
    fieldnames = ordered + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def methods_for_case(meta: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    methods = list(args.methods)
    if (
        not args.no_full_d2
        and int(meta["safe_type_count"]) == int(meta["num_types_I"])
        and "clabel_d2_ws_cut_full" not in methods
    ):
        methods.append("clabel_d2_ws_cut_full")
    if int(meta["safe_type_count"]) != int(meta["num_types_I"]):
        methods = [method for method in methods if method != "clabel_d2_ws_cut_full"]
    return methods


def add_meta_to_row(row: Dict[str, Any], meta: Dict[str, Any], method: str) -> Dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "run_id": f"{method}_{meta['case_id']}",
            "case_id": meta["case_id"],
            "experiment_family": meta["experiment_family"],
            "instance_group": meta["instance_group"],
            "problem_scale": meta["problem_scale"],
            "num_types_I": meta["num_types_I"],
            "num_wagons_J": meta["num_wagons_J"],
            "replicate_id": meta["replicate_id"],
            "safe_type_count": meta["safe_type_count"],
            "conflict_type_count": meta["conflict_type_count"],
            "safe_type_share": meta["safe_type_share"],
            "validation_expected_ordered_type_count": meta.get("validation_expected_ordered_type_count"),
            "validation_min_ordered_type_count": meta.get("validation_min_ordered_type_count"),
            "validation_max_ordered_type_count": meta.get("validation_max_ordered_type_count"),
            "validation_ordered_quantity_sum": meta.get("validation_ordered_quantity_sum"),
            "validation_total_quantity_sum": meta.get("validation_total_quantity_sum"),
            "validation_attempt": meta.get("validation_attempt"),
            "optional_each_type": meta.get("optional_each_type"),
            "total_optional": meta.get("total_optional"),
            "mean_length": meta.get("mean_length"),
            "mean_height": meta.get("mean_height"),
            "length_range": meta.get("length_range"),
            "height_range": meta.get("height_range"),
        }
    )
    return out


def run_case(meta: Dict[str, Any], methods: List[str], args: argparse.Namespace, result_root: Path) -> Dict[str, Any]:
    instance = str(meta["instance"])
    selected_configs = [METHODS[name] for name in methods if name in METHODS]
    messages = [f"=== {instance} ==="]
    rows: List[Dict[str, Any]] = []

    if "compact_gurobi" in methods:
        row = run_compact(instance, result_root, args)
        rows.append(add_meta_to_row(row, meta, "compact_gurobi"))
        messages.append(
            f"compact_gurobi: status={row.get('status')} loaded={row.get('loaded_length_mm')} "
            f"gap={row.get('gap')} time={row.get('runtime_sec')}"
        )

    needs_warmstart = any(config["use_warmstart"] for config in selected_configs)
    warmstarts = ensure_warmstart(instance, args) if needs_warmstart else None
    for config in selected_configs:
        row = run_bpc_method(instance, config, warmstarts, result_root, args)
        rows.append(add_meta_to_row(row, meta, str(config["method"])))
        messages.append(
            f"{config['method']}: status={row.get('status')} loaded={row.get('loaded_length_mm')} "
            f"gap={row.get('gap')} time={row.get('runtime_sec')} "
            f"d2_avoided={row.get('labels_avoided_by_d2')}"
        )
    return {"rows": rows, "messages": messages}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run type-share D3 ablation experiments.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--instance-group", default=None)
    parser.add_argument("--result-suffix", default="type_share_ablation")
    parser.add_argument("--type-counts", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--safe-step", type=int, default=2)
    parser.add_argument("--optional-each-type", type=int, default=15)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--validation-mode", choices=["all", "max"], default="all")
    parser.add_argument("--allow-trivial-singleton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-failed-candidates", action="store_true")
    parser.add_argument("--skip-existing-instances", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=["compact_gurobi", *sorted(METHODS)], default=DEFAULT_METHODS)
    parser.add_argument("--no-full-d2", action="store_true")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--warmstart-time-limit", type=float, default=3600.0)
    parser.add_argument("--grace-sec", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--independent-mode-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual-profile-mode", choices=["full", "fans_diag"], default="full")
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-cg-iters", type=int, default=3000)
    parser.add_argument("--label-columns", type=int, default=0)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-warmstart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic-timeout", type=float, default=30.0)
    parser.add_argument("--bpc-exe", type=Path, default=default_bpc_exe())
    parser.add_argument("--vns-exe", type=Path, default=default_vns_exe())
    args = parser.parse_args()
    if args.instance_group is None:
        args.instance_group = f"type_share_ablation_{args.date}"
    if args.threads != 1:
        raise ValueError("Formal ablation experiments should use --threads 1.")
    if args.optional_each_type < 14:
        raise ValueError("--optional-each-type must be at least 14 because J=I.")
    if args.safe_step <= 0:
        raise ValueError("--safe-step must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if args.skip_generation:
        metas = load_manifest(args)
    else:
        metas = generate_instances(args)

    if args.generate_only:
        print(f"Manifest: {INSTANCE_ROOT / args.instance_group / 'manifest.json'}")
        return

    result_root = RESULT_ROOT / f"{args.instance_group}_{args.result_suffix}_{args.date}"
    results_path = result_root / f"type_share_ablation_results_{args.date}.csv"
    seen = load_existing_run_ids(results_path)

    pending: List[tuple[Dict[str, Any], List[str]]] = []
    for meta in metas:
        methods = []
        for method in methods_for_case(meta, args):
            run_id = f"{method}_{meta['case_id']}"
            if args.skip_existing and run_id in seen:
                print(f"skip existing {run_id}", flush=True)
                continue
            methods.append(method)
        if methods:
            pending.append((meta, methods))

    if not pending:
        print(f"Results CSV: {results_path}")
        return

    parallel_jobs = max(1, args.parallel_jobs)
    if parallel_jobs == 1:
        for meta, methods in pending:
            result = run_case(meta, methods, args, result_root)
            for message in result["messages"]:
                print(message, flush=True)
            append_rows(results_path, result["rows"])
        print(f"Results CSV: {results_path}")
        return

    print(f"Running {len(pending)} cases with parallel_jobs={parallel_jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        futures = {
            pool.submit(run_case, meta, methods, args, result_root): meta["case_id"]
            for meta, methods in pending
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
            append_rows(results_path, result["rows"])

    print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
