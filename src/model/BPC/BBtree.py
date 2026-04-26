from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.CG import MasterLPSolution, MasterProblem, PatternColumn, ColumnGenerationEngine
from src.model.BPC.pricing import EarlyStopPricingEngine
from src.utility.config import config as Config

# IDE debug switches.
# 0 = False, 1 = True.
DBG_LOG_TO_CONSOLE = 0
DBG_USE_DOMINANCE = 1
DBG_USE_CUTS = 1
DBG_PRINT_BB_PROGRESS = 1
DBG_PRINT_SUBPROBLEM_PROGRESS = 0
DBG_PRINT_SUMMARY = 1

# Runtime settings (edit directly when debugging).
RUN_INSTANCE_DIR = "data/Instance/m7c7"
RUN_OUTPUT_DIR = "result"
RUN_MAX_NODES = 5000
RUN_MAX_CG_ITERS = 3000

@dataclass
class BBNode:
    node_id: int
    depth: int
    branch_a_bounds: Dict[int, Tuple[float | None, float | None]] = field(default_factory=dict)
    branch_q_bounds: Dict[int, Tuple[float | None, float | None]] = field(default_factory=dict)

@dataclass
class BPCResult:
    best_objective: float | None
    best_theta: Dict[str, float] | None
    explored_nodes: int
    generated_columns: int

def normalize_car_table(cars_path: Path) -> pd.DataFrame:
    car_info = pd.read_csv(cars_path)
    col_map = {
        "Brand": "program",
        "Model": "model",
        "Length": "length",
        "Height": "height",
        "Optional#": "optional",
        "Mandatory#": "mandatory",
    }
    car_info = car_info.rename(columns=col_map)
    required = ["program", "model", "length", "height", "optional", "mandatory"]
    missing = [c for c in required if c not in car_info.columns]
    if missing:
        raise ValueError(f"cars.csv missing required columns: {missing}")

    car_info["length"] = pd.to_numeric(car_info["length"])
    car_info["height"] = pd.to_numeric(car_info["height"])
    car_info["optional"] = pd.to_numeric(car_info["optional"]).astype(int)
    car_info["mandatory"] = pd.to_numeric(car_info["mandatory"]).astype(int)
    return car_info


class BBTree:
    """Branch-and-Bound tree maintenance for MLP-IC."""

    def __init__(
        self,
        instance_dir: Path,
        output_root: Path,
        max_nodes: int = 200,
        max_cg_iters: int = 100,
        log_to_console: bool = True,
        use_dominance: bool = True,
        use_cuts: bool = False,
        print_bb_progress: bool = True,
        print_subproblem_progress: bool = False,
    ) -> None:
        self.instance_dir = instance_dir
        self.output_dir = output_root / instance_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cars_path = instance_dir / "cars.csv"
        carriage_path = instance_dir / "carriage.csv"
        if not cars_path.exists() or not carriage_path.exists():
            raise FileNotFoundError("Instance folder must contain cars.csv and carriage.csv")

        car_info = normalize_car_table(cars_path)
        carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])

        self.master = MasterProblem(car_info=car_info, carriage_num=carriage_num)
        self.master.seed_initial_columns()

        self.pricing_engine = EarlyStopPricingEngine(
            use_dominance=use_dominance,
            use_cuts=use_cuts,
            verbose=print_subproblem_progress,
        )
        self.cg_engine = ColumnGenerationEngine(
            master=self.master,
            pricing_engine=self.pricing_engine,
            max_cg_iters=max_cg_iters,
            log_to_console=print_subproblem_progress
        )

        self.max_nodes = max_nodes
        self.log_to_console = log_to_console
        self.print_bb_progress = print_bb_progress

        self.best_obj: float | None = None
        self.best_theta: Dict[str, float] | None = None

    def solve(self) -> BPCResult:
        node_counter = 0
        queue: deque[BBNode] = deque([BBNode(node_id=node_counter, depth=0)])
        explored = 0

        if self.print_bb_progress:
            print(f"[BB] Start solve: max_nodes={self.max_nodes}")
            print(f"{'Node':>6}  {'Depth':>6}  {'Left':>6}  {'Current Bound':>14}  {'Best Incumbent':>14}  {'Time(s)':>8}")

        start_time = time.time()

        while queue and explored < self.max_nodes:
            node = queue.popleft()
            explored += 1

            lp_solution = self.cg_engine.solve(
                branch_a_bounds=node.branch_a_bounds, 
                branch_q_bounds=node.branch_q_bounds
            )
            
            elapsed = time.time() - start_time
            
            if lp_solution.objective is None:
                if self.print_bb_progress:
                    best_str = f"{self.best_obj:.2f}" if self.best_obj is not None else "-"
                    print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {'Infeasible':>14}  {best_str:>14}  {elapsed:8.2f}")
                continue

            current_bound = lp_solution.objective
            best_str = f"{self.best_obj:.2f}" if self.best_obj is not None else "-"
            
            if self.print_bb_progress:
                print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {current_bound:14.2f}  {best_str:>14}  {elapsed:8.2f}")

            # Bounding by incumbent
            if self.best_obj is not None and lp_solution.objective >= self.best_obj - 1e-6:
                continue

            if self.master.is_integral(lp_solution):
                if self.best_obj is None or lp_solution.objective < self.best_obj:
                    self.best_obj = lp_solution.objective
                    self.best_theta = {k: round(v) for k, v in lp_solution.theta_values.items()}
                continue

            # Branch on chosen a or q variable
            branch_var = self.master.choose_branch_var(lp_solution)
            if branch_var is None:
                continue
                
            b_type, car_type, value = branch_var
            floor_v = math.floor(value)
            ceil_v = math.ceil(value)

            left_a = dict(node.branch_a_bounds)
            left_q = dict(node.branch_q_bounds)
            right_a = dict(node.branch_a_bounds)
            right_q = dict(node.branch_q_bounds)

            if b_type == 'a':
                old_left = left_a.get(car_type, (None, None))
                left_a[car_type] = (old_left[0], floor_v)
                old_right = right_a.get(car_type, (None, None))
                right_a[car_type] = (ceil_v, old_right[1])
            else:
                old_left = left_q.get(car_type, (None, None))
                left_q[car_type] = (old_left[0], floor_v)
                old_right = right_q.get(car_type, (None, None))
                right_q[car_type] = (ceil_v, old_right[1])

            node_counter += 1
            queue.append(BBNode(node_id=node_counter, depth=node.depth + 1, branch_a_bounds=left_a, branch_q_bounds=left_q))
            node_counter += 1
            queue.append(BBNode(node_id=node_counter, depth=node.depth + 1, branch_a_bounds=right_a, branch_q_bounds=right_q))

        result = BPCResult(
            best_objective=self.best_obj,
            best_theta=self.best_theta,
            explored_nodes=explored,
            generated_columns=self.cg_engine.generated_columns,
        )

        with open(self.output_dir / "bb_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_objective": result.best_objective,
                    "best_theta": result.best_theta,
                    "explored_nodes": result.explored_nodes,
                    "generated_columns": result.generated_columns,
                    "total_columns_in_pool": len(self.master.columns),
                    "master_solve_time": self.cg_engine.stats.master_time,
                    "pricing_total_time": self.cg_engine.stats.pricing_time,
                    "labeling_time": self.cg_engine.stats.labeling_time,
                    "feasibility_check_time": self.cg_engine.stats.bs_time,
                    "merge_time": self.cg_engine.stats.merge_time,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return result

def main() -> None:
    solver = BBTree(
        instance_dir=PROJECT_ROOT / RUN_INSTANCE_DIR,
        output_root=PROJECT_ROOT / RUN_OUTPUT_DIR,
        max_nodes=RUN_MAX_NODES,
        max_cg_iters=RUN_MAX_CG_ITERS,
        log_to_console=bool(DBG_LOG_TO_CONSOLE),
        use_dominance=bool(DBG_USE_DOMINANCE),
        use_cuts=bool(DBG_USE_CUTS),
        print_bb_progress=bool(DBG_PRINT_BB_PROGRESS),
        print_subproblem_progress=bool(DBG_PRINT_SUBPROBLEM_PROGRESS),
    )
    
    t0 = time.time()
    result = solver.solve()
    total_time = time.time() - t0

    if bool(DBG_PRINT_SUMMARY):
        print("\n--- BBTree Finished ---")
        print(f"Explored Nodes     : {result.explored_nodes}")
        print(f"Generated Columns  : {result.generated_columns}")
        print(f"Best Objective     : {result.best_objective}")
        print("\n--- Time Profiling ---")
        print(f"Total BBTree Time  : {total_time:.2f} s")
        print(f"Master Solve Time  : {solver.cg_engine.stats.master_time:.2f} s")
        print(f"Pricing Total Time : {solver.cg_engine.stats.pricing_time:.2f} s")
        print(f"  ├─ Labeling Time : {solver.cg_engine.stats.labeling_time:.2f} s")
        print(f"  ├─ Feas Check BS : {solver.cg_engine.stats.bs_time:.2f} s")
        print(f"  └─ Merging Time  : {solver.cg_engine.stats.merge_time:.2f} s")

if __name__ == "__main__":
    main()