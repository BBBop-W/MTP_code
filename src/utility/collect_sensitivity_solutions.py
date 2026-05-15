#!/usr/bin/env python3
"""Collect completed sensitivity runs into a Python pickle archive."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ROOT = PROJECT_ROOT / "data" / "Instance"
RESULT_ROOT = PROJECT_ROOT / "result"
SCALE_ORDER = ["small", "medium", "large"]


def json_safe(value: Any) -> Any:
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


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json_safe(pd.read_csv(path).to_dict(orient="records"))


def load_native_detail(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_scale(date_tag: str, scale: str) -> dict[str, Any]:
    result_root = RESULT_ROOT / f"sensitivity_{date_tag}" / scale
    instance_root = INSTANCE_ROOT / f"sensitivity_{date_tag}" / scale
    results_path = result_root / f"sensitivity_{scale}_results_{date_tag}.csv"
    plan_path = result_root / "plans" / f"sensitivity_{scale}_plan_{date_tag}.csv"

    scale_archive: dict[str, Any] = {
        "scale": scale,
        "results_path": str(results_path),
        "plan_path": str(plan_path),
        "runs": {},
    }
    if not results_path.exists():
        scale_archive["missing_results"] = True
        return scale_archive

    results = pd.read_csv(results_path)
    for row in results.to_dict(orient="records"):
        run_id = str(row.get("run_id"))
        instance_id = str(row.get("instance_id"))
        result_dir = Path(str(row.get("result_dir")))
        solver_dir = result_dir / "solver"
        instance_dir = instance_root / instance_id

        native_detail = load_native_detail(solver_dir / "solution_detail.pkl")
        assembled_detail = {
            "summary": read_json(solver_dir / "solve_summary.json"),
            "car_solution": read_csv_records(solver_dir / "car_sol.csv"),
            "carriage": read_json(solver_dir / "carriage_info.json"),
            "instance_meta": read_json(instance_dir / "meta.json"),
            "cars": read_csv_records(instance_dir / "cars.csv"),
            "result_record": json_safe(row),
            "files": {
                "solver_dir": str(solver_dir),
                "instance_dir": str(instance_dir),
                "summary_json": str(solver_dir / "solve_summary.json"),
                "car_solution_csv": str(solver_dir / "car_sol.csv"),
                "carriage_info_json": str(solver_dir / "carriage_info.json"),
                "native_solution_detail_pkl": str(solver_dir / "solution_detail.pkl"),
            },
        }
        scale_archive["runs"][run_id] = native_detail if native_detail is not None else assembled_detail
    return scale_archive


def collect_archive(date_tag: str, scales: list[str]) -> dict[str, Any]:
    archive = {
        "date_tag": date_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scales": {},
        "run_count": 0,
        "status_counts": {},
    }
    for scale in scales:
        scale_archive = collect_scale(date_tag, scale)
        archive["scales"][scale] = scale_archive
        runs = scale_archive.get("runs", {})
        archive["run_count"] += len(runs)
        for detail in runs.values():
            summary = detail.get("summary") if isinstance(detail, dict) else None
            status = "missing_summary"
            if isinstance(summary, dict):
                status = str(summary.get("status", "missing_status"))
            archive["status_counts"][status] = archive["status_counts"].get(status, 0) + 1
    return archive


def write_archive(date_tag: str, archive: dict[str, Any]) -> tuple[Path, Path]:
    output_root = RESULT_ROOT / f"sensitivity_{date_tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    pickle_path = output_root / f"solution_archive_{date_tag}.pkl"
    index_path = output_root / f"solution_archive_index_{date_tag}.json"

    with open(pickle_path, "wb") as f:
        pickle.dump(archive, f, protocol=pickle.HIGHEST_PROTOCOL)

    index = {
        "date_tag": archive["date_tag"],
        "generated_at": archive["generated_at"],
        "run_count": archive["run_count"],
        "status_counts": archive["status_counts"],
        "scales": {
            scale: {
                "run_count": len(scale_data.get("runs", {})),
                "results_path": scale_data.get("results_path"),
                "plan_path": scale_data.get("plan_path"),
            }
            for scale, scale_data in archive["scales"].items()
        },
        "pickle_path": str(pickle_path),
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return pickle_path, index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sensitivity solution details into a pickle archive.")
    parser.add_argument("--date", required=True, help="Sensitivity date tag, e.g. 2026-05-15_formal.")
    parser.add_argument("--scales", nargs="*", default=SCALE_ORDER, choices=SCALE_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = collect_archive(args.date, args.scales)
    pickle_path, index_path = write_archive(args.date, archive)
    print(f"Collected {archive['run_count']} completed runs")
    print(f"Status counts: {archive['status_counts']}")
    print(f"Pickle archive: {pickle_path}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
