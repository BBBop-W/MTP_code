from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.BPC.CG import MasterLPSolution, MasterProblem, PatternColumn, ColumnGenerationEngine
from src.model.BPC.pricing import EarlyStopPricingEngine
from src.utility.config import config as Config

# IDE debug switches.
# 0 = False, 1 = True.
DBG_LOG_TO_CONSOLE = 0
DBG_USE_DOMINANCE = 1
DBG_USE_CUTS = 0
DBG_PRINT_BB_PROGRESS = 1
DBG_PRINT_SUBPROBLEM_PROGRESS = 0
DBG_PRINT_SUMMARY = 1

# Runtime settings (edit directly when debugging).
RUN_INSTANCE_DIR = "data/Instance/m10c10"
RUN_OUTPUT_DIR = "result"
RUN_MAX_NODES = 5000
RUN_MAX_CG_ITERS = 3000

@dataclass
class BBNode:
    node_id: int
    depth: int
    branch_bounds: Dict[str, Tuple[float | None, float | None]] = field(default_factory=dict)

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

    def _log_bb(self, msg: str) -> None:
        if self.print_bb_progress:
            print(f"[BB] {msg}")

    def solve(self) -> BPCResult:
        node_counter = 0
        queue: deque[BBNode] = deque([BBNode(node_id=node_counter, depth=0)])
        explored = 0

        self._log_bb(f"Start solve: max_nodes={self.max_nodes}")

        while queue and explored < self.max_nodes:
            node = queue.popleft()
            explored += 1
            self._log_bb(f"Explore node={node.node_id}, depth={node.depth}, queue_left={len(queue)}")

            lp_solution = self.cg_engine.solve(branch_bounds=node.branch_bounds)
            if lp_solution.objective is None:
                self._log_bb(f"Node={node.node_id} skipped due to invalid LP status={lp_solution.status}")
                continue

            # Bounding by incumbent
            if self.best_obj is not None and lp_solution.objective >= self.best_obj - 1e-6:
                self._log_bb(
                    f"Node={node.node_id} fathomed by bound: obj={lp_solution.objective:.6f} >= best={self.best_obj:.6f}"
                )
                continue

            if self.master.is_integral(lp_solution):
                if self.best_obj is None or lp_solution.objective < self.best_obj:
                    self.best_obj = lp_solution.objective
                    self.best_theta = {k: round(v) for k, v in lp_solution.theta_values.items()}
                    self._log_bb(f"Node={node.node_id} found new incumbent: obj={self.best_obj:.6f}")
                else:
                    self._log_bb(f"Node={node.node_id} integral but not improving")
                continue

            # Branch on first fractional theta_r
            branch_var = self.master.choose_branch_var(lp_solution)
            if branch_var is None:
                self._log_bb(f"Node={node.node_id} fractional expected but no branch var found")
                continue
            col_id, value = branch_var
            floor_v = math.floor(value)
            ceil_v = math.ceil(value)
            self._log_bb(f"Node={node.node_id} branch on {col_id}={value:.6f} -> <= {floor_v} and >= {ceil_v}")

            left_bounds = dict(node.branch_bounds)
            old_left = left_bounds.get(col_id, (None, None))
            left_bounds[col_id] = (old_left[0], floor_v)

            right_bounds = dict(node.branch_bounds)
            old_right = right_bounds.get(col_id, (None, None))
            right_bounds[col_id] = (ceil_v, old_right[1])

            node_counter += 1
            queue.append(BBNode(node_id=node_counter, depth=node.depth + 1, branch_bounds=left_bounds))
            node_counter += 1
            queue.append(BBNode(node_id=node_counter, depth=node.depth + 1, branch_bounds=right_bounds))
            self._log_bb(f"Create children nodes {node_counter-1} and {node_counter}")

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
    result = solver.solve()

    if bool(DBG_PRINT_SUMMARY):
        print("BBTree finished.")
        print(f"explored_nodes: {result.explored_nodes}")
        print(f"generated_columns: {result.generated_columns}")
        print(f"best_objective: {result.best_objective}")

if __name__ == "__main__":
    main()
