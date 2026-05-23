#!/usr/bin/env python3
"""Generate D3 ordered-share smoke instances and run the ablation suite.

The generated instances keep the same vehicle geometry across share levels.
Only the demand assigned within each ordered/conflict matched pair changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_bpc_ablation_smoke import (  # noqa: E402
    ALL_METHOD_NAMES,
    CSV_COLUMNS,
    METHOD_BY_NAME,
    default_bpc_exe,
    default_vns_exe,
    ensure_warmstart,
    run_bpc_method,
    run_compact,
)


INSTANCE_ROOT = PROJECT_ROOT / "data" / "Instance"
RESULT_ROOT = PROJECT_ROOT / "result"

ORDERED_TYPES = [
    {"program": "ordered_chain", "model": "OC0", "length": 3500, "height": 1500},
    {"program": "ordered_chain", "model": "OC1", "length": 3600, "height": 2100},
    {"program": "ordered_chain", "model": "OC2", "length": 3700, "height": 2095},
    {"program": "ordered_chain", "model": "OC3", "length": 3800, "height": 2090},
    {"program": "ordered_chain", "model": "OC4", "length": 3900, "height": 2085},
]

CONFLICT_TYPES = [
    {"program": "background", "model": "BG0", "length": 3500, "height": 1500},
    {"program": "background", "model": "BG1", "length": 3580, "height": 1510},
    {"program": "background", "model": "BG2", "length": 3660, "height": 1520},
    {"program": "background", "model": "BG3", "length": 3740, "height": 1530},
    {"program": "background", "model": "BG4", "length": 3820, "height": 1540},
]

META_COLUMNS = [
    "experiment_family",
    "ordered_share_target",
    "ordered_share_actual",
    "pair_total",
    "demand_total",
    "ordered_demand_total",
    "conflict_demand_total",
    "weighted_mean_length",
    "weighted_mean_height",
    "length_mean_delta_vs_mid",
    "height_mean_delta_vs_mid",
]


def weighted_mean(rows: Iterable[Dict[str, Any]], field: str) -> float:
    num = 0.0
    den = 0.0
    for row in rows:
        demand = float(row["mandatory"]) + float(row["optional"])
        num += demand * float(row[field])
        den += demand
    return num / den if den else 0.0


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(CSV_COLUMNS)
    for col in META_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)
    extra = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cars_for_share(share: float, pair_total: int) -> List[Dict[str, Any]]:
    ordered_q = int(round(pair_total * share))
    ordered_q = max(0, min(pair_total, ordered_q))
    conflict_q = pair_total - ordered_q

    rows: List[Dict[str, Any]] = []
    for idx, (ordered, conflict) in enumerate(zip(ORDERED_TYPES, CONFLICT_TYPES), start=1):
        rows.append(
            {
                **ordered,
                "optional": ordered_q,
                "mandatory": 0,
                "pair_id": idx,
                "d3_group": "ordered",
            }
        )
        rows.append(
            {
                **conflict,
                "optional": conflict_q,
                "mandatory": 0,
                "pair_id": idx,
                "d3_group": "conflict",
            }
        )
    return rows


def write_instance(
    instance_name: str,
    share: float,
    pair_total: int,
    carriages: int,
    mid_length: float,
    mid_height: float,
) -> Dict[str, Any]:
    rows = cars_for_share(share, pair_total)
    instance_dir = INSTANCE_ROOT / instance_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    demand_total = sum(int(row["optional"]) + int(row["mandatory"]) for row in rows)
    ordered_total = sum(int(row["optional"]) for row in rows if row["d3_group"] == "ordered")
    conflict_total = demand_total - ordered_total
    mean_length = weighted_mean(rows, "length")
    mean_height = weighted_mean(rows, "height")

    cars_path = instance_dir / "cars.csv"
    with cars_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "program",
                "model",
                "length",
                "height",
                "optional",
                "mandatory",
                "pair_id",
                "d3_group",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (instance_dir / "carriage.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["carriage_num"])
        writer.writeheader()
        writer.writerow({"carriage_num": carriages})

    meta = {
        "instance": instance_name,
        "experiment_family": "d3_ordered_share_smoke",
        "ordered_share_target": float(share),
        "ordered_share_actual": float(ordered_total / demand_total) if demand_total else 0.0,
        "pair_total": int(pair_total),
        "demand_total": int(demand_total),
        "ordered_demand_total": int(ordered_total),
        "conflict_demand_total": int(conflict_total),
        "weighted_mean_length": float(mean_length),
        "weighted_mean_height": float(mean_height),
        "length_mean_delta_vs_mid": float(mean_length - mid_length),
        "height_mean_delta_vs_mid": float(mean_height - mid_height),
    }
    with (instance_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def generate_instances(args: argparse.Namespace) -> List[Dict[str, Any]]:
    mid_rows = cars_for_share(args.mid_share, args.pair_total)
    mid_length = weighted_mean(mid_rows, "length")
    mid_height = weighted_mean(mid_rows, "height")

    metas: List[Dict[str, Any]] = []
    for share in args.shares:
        suffix = int(round(share * 100))
        type_count = len(ORDERED_TYPES) + len(CONFLICT_TYPES)
        instance_name = f"{args.instance_group}/d3share_I{type_count:02d}_J{args.carriages:02d}_p{suffix:03d}"
        metas.append(
            write_instance(
                instance_name=instance_name,
                share=share,
                pair_total=args.pair_total,
                carriages=args.carriages,
                mid_length=mid_length,
                mid_height=mid_height,
            )
        )
    return metas


def run_suite(metas: List[Dict[str, Any]], args: argparse.Namespace) -> Path:
    result_root = RESULT_ROOT / f"{args.instance_group}_results"
    results_path = result_root / f"d3_share_smoke_results_{args.date}.csv"
    rows: List[Dict[str, Any]] = []

    selected_configs = [METHOD_BY_NAME[name] for name in args.methods if name in METHOD_BY_NAME]
    for meta in metas:
        instance = str(meta["instance"])
        print(f"=== {instance} share={meta['ordered_share_actual']:.2f} ===", flush=True)
        if not args.skip_compact and "compact_gurobi" in args.methods:
            row = run_compact(instance, result_root, args)
            row.update(meta)
            rows.append(row)
            print(
                f"compact_gurobi: status={row.get('status')} gap={row.get('gap')} "
                f"time={row.get('runtime_sec')}",
                flush=True,
            )

        needs_warmstart = any(config["use_warmstart"] for config in selected_configs)
        warmstarts = ensure_warmstart(instance, args) if needs_warmstart else None
        for config in selected_configs:
            row = run_bpc_method(instance, config, warmstarts, result_root, args)
            hybrid_total = row.get("hybrid_total_quantity_sum")
            hybrid_ordered = row.get("hybrid_ordered_quantity_sum")
            if hybrid_total:
                row["hybrid_ordered_quantity_share"] = float(hybrid_ordered or 0) / float(hybrid_total)
            row.update(meta)
            rows.append(row)
            print(
                f"{config['method']}: status={row.get('status')} gap={row.get('gap')} "
                f"time={row.get('runtime_sec')} d2_avoided={row.get('labels_avoided_by_d2')}",
                flush=True,
            )
        write_csv(results_path, rows)
    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run D3 ordered-share smoke instances.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--instance-group", default=None)
    parser.add_argument("--shares", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--mid-share", type=float, default=0.50)
    parser.add_argument("--pair-total", type=int, default=8)
    parser.add_argument("--carriages", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHOD_NAMES, default=ALL_METHOD_NAMES)
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--warmstart-time-limit", type=float, default=20.0)
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
    parser.add_argument("--refresh-warmstart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bpc-exe", type=Path, default=default_bpc_exe())
    parser.add_argument("--vns-exe", type=Path, default=default_vns_exe())
    args = parser.parse_args()
    if args.instance_group is None:
        args.instance_group = f"d3_share_smoke_{args.date}"
    if args.threads != 1:
        raise ValueError("This smoke script is intended to run with --threads 1.")
    return args


def main() -> None:
    args = parse_args()
    metas = generate_instances(args)
    results_path = run_suite(metas, args)
    print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
