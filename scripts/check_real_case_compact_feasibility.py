"""Check real-case instances with the compact Gurobi model.

Each instance is solved in a child process so a solve that reaches its time
limit but hangs during cleanup can still be recovered from solve_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model.gurobi import build_and_solve


GRB_OPTIMAL = 2
GRB_INFEASIBLE = 3
GRB_TIME_LIMIT = 9


def _solve_case(
    instance_dir: str,
    output_dir: str,
    num_splits: int,
    independent_mode_split: bool,
    objective_type: str,
    time_limit: float,
    mip_gap: float,
    threads: int,
) -> None:
    build_and_solve(
        Path(instance_dir),
        Path(output_dir),
        log_to_console=False,
        num_splits=num_splits,
        independent_mode_split=independent_mode_split,
        objective_type=objective_type,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
    )


def _load_summary(output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "solver" / "solve_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _classify(summary: dict[str, Any] | None, watchdog_timeout: bool) -> str:
    if summary is None:
        return "WATCHDOG_TIMEOUT" if watchdog_timeout else "NO_SUMMARY"

    status = int(summary.get("status", -1))
    sol_count = int(summary.get("sol_count", 0))

    if sol_count > 0:
        return "FEASIBLE_OPTIMAL" if status == GRB_OPTIMAL else "FEASIBLE_INCUMBENT"
    if status == GRB_INFEASIBLE:
        return "INFEASIBLE"
    if status == GRB_TIME_LIMIT:
        return "NO_INCUMBENT_TIME_LIMIT"
    return f"NO_INCUMBENT_STATUS_{status}"


def _case_row(case: str, output_dir: Path, watchdog_timeout: bool) -> dict[str, Any]:
    summary = _load_summary(output_dir)
    verdict = _classify(summary, watchdog_timeout)
    row: dict[str, Any] = {
        "case": case,
        "verdict": verdict,
        "watchdog_timeout": watchdog_timeout,
    }
    if summary:
        for key in [
            "status",
            "sol_count",
            "obj_val",
            "obj_bound",
            "mip_gap",
            "runtime_sec",
            "node_count",
            "loaded_length_mm",
            "mandatory_loaded",
            "mandatory_total",
            "optional_loaded",
            "num_types",
            "num_wagons",
            "num_vars",
            "num_constrs",
        ]:
            row[key] = summary.get(key)
    return row


def _run_one(args: argparse.Namespace, case: str) -> dict[str, Any]:
    instance_dir = args.instance_root / case
    output_dir = args.result_root / case / args.output_name
    process = mp.Process(
        target=_solve_case,
        args=(
            str(instance_dir),
            str(output_dir),
            args.num_splits,
            not args.linked_side_blocks,
            args.objective_type,
            args.time_limit,
            args.mip_gap,
            args.threads,
        ),
    )
    process.start()
    process.join(args.time_limit + args.grace)
    watchdog_timeout = process.is_alive()
    if watchdog_timeout:
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
    return _case_row(case, output_dir, watchdog_timeout)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-root",
        type=Path,
        default=Path("data/Instance/real_case_2026-05-21"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("result/real_case_2026-05-21"),
    )
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--output-name", default="compact_gurobi_check")
    parser.add_argument("--summary-name", default="compact_feasibility_summary.csv")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--grace", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--objective-type", choices=["length", "quantity"], default="length")
    parser.add_argument(
        "--linked-side-blocks",
        action="store_true",
        default=True,
        help="Use independent_mode_split=False, matching the current compartment checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = args.cases
    if cases is None:
        cases = sorted(path.name for path in args.instance_root.glob("case_*") if path.is_dir())

    rows: list[dict[str, Any]] = []
    for case in cases:
        row = _run_one(args, case)
        rows.append(row)
        print(
            f"{case}: {row['verdict']} "
            f"status={row.get('status')} sol_count={row.get('sol_count')} "
            f"loaded_length={row.get('loaded_length_mm')}"
        )

    summary_path = args.result_root / args.summary_name
    _write_csv(summary_path, rows)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
