from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gurobipy as gp
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.master import MasterLPSolution, MasterProblem, PatternColumn
from src.model.pricing import EarlyStopPricingEngine
from src.utility.config import config as Config


PricingHook = Callable[[MasterLPSolution, MasterProblem], List[PatternColumn]]

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


class BPCSolver:
    """Branch-Price-and-Cut scaffold for MLP-IC.

    This file provides the algorithm entry and BB tree maintenance.
    Pricing subproblem is intentionally left as a pluggable hook.
    """

    def __init__(
        self,
        instance_dir: Path,
        output_root: Path,
        pricing_hook: PricingHook | None = None,
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

        self.pricing_hook = pricing_hook
        self.pricing_engine = EarlyStopPricingEngine(
            use_dominance=use_dominance,
            use_cuts=use_cuts,
            verbose=print_subproblem_progress,
            logger=self._log_subproblem,
        )
        self.max_nodes = max_nodes
        self.max_cg_iters = max_cg_iters
        self.log_to_console = log_to_console
        self.print_bb_progress = print_bb_progress
        self.print_subproblem_progress = print_subproblem_progress

        self.best_obj: float | None = None
        self.best_theta: Dict[str, float] | None = None
        self.generated_columns = 0

    def _log_bb(self, msg: str) -> None:
        if self.print_bb_progress:
            print(f"[BPC] {msg}")

    def _log_subproblem(self, msg: str) -> None:
        if self.print_subproblem_progress:
            print(f"[SP] {msg}")

    def _pricing(self, lp_solution: MasterLPSolution) -> List[PatternColumn]:
        # Integration point:
        # 1) Build layer specs for each compartment/deck configuration.
        # 2) Call labeling.generate_layer_patterns(...) for each layer.
        # 3) Merge upper/lower layer patterns into full wagon columns.
        # 4) Return only columns with negative reduced cost.
        if self.pricing_hook is not None:
            return self.pricing_hook(lp_solution, self.master)
        return self.pricing_engine.generate_columns(lp_solution, self.master)

    def _column_generation(self, node: BBNode) -> MasterLPSolution:
        self._log_bb(f"Start CG at node={node.node_id}, depth={node.depth}, columns={len(self.master.columns)}")
        last_solution = self.master.solve_lp(
            branch_bounds=node.branch_bounds,
            time_limit=Config.timelimit,
            log_to_console=self.log_to_console,
        )
        if last_solution.objective is None:
            self._log_bb(f"Node={node.node_id} RMP solve failed, status={last_solution.status}")
            return last_solution

        self._log_bb(f"Node={node.node_id} initial RMP objective={last_solution.objective:.6f}")

        for it in range(1, self.max_cg_iters + 1):
            new_columns = self._pricing(last_solution)
            if not new_columns:
                self._log_bb(f"Node={node.node_id} CG iter={it}: no new column, stop")
                break

            added = 0
            for col in new_columns:
                if col.column_id not in self.master.columns:
                    self.master.add_column(col)
                    self.generated_columns += 1
                    added += 1
                    self._log_bb(f"Node={node.node_id} add column={col.column_id}, cost={col.cost:.3f}, q_sum={sum(col.q.values())}")

            if added == 0:
                self._log_bb(f"Node={node.node_id} CG iter={it}: generated columns were duplicates, stop")
                break

            last_solution = self.master.solve_lp(
                branch_bounds=node.branch_bounds,
                time_limit=Config.timelimit,
                log_to_console=self.log_to_console,
            )
            if last_solution.objective is None:
                self._log_bb(f"Node={node.node_id} CG iter={it}: RMP resolve failed, status={last_solution.status}")
                break
            self._log_bb(f"Node={node.node_id} CG iter={it}: RMP objective={last_solution.objective:.6f}, added={added}")

        return last_solution

    def solve(self) -> BPCResult:
        node_counter = 0
        queue: deque[BBNode] = deque([BBNode(node_id=node_counter, depth=0)])
        explored = 0

        self._log_bb(
            f"Start solve: max_nodes={self.max_nodes}, max_cg_iters={self.max_cg_iters}, "
            f"dominance={self.pricing_engine.options.use_dominance}, cuts={self.pricing_engine.options.use_cuts}"
        )

        while queue and explored < self.max_nodes:
            node = queue.popleft()
            explored += 1
            self._log_bb(f"Explore node={node.node_id}, depth={node.depth}, queue_left={len(queue)}")

            lp_solution = self._column_generation(node)
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
            generated_columns=self.generated_columns,
        )

        with open(self.output_dir / "bpc_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_objective": result.best_objective,
                    "best_theta": result.best_theta,
                    "explored_nodes": result.explored_nodes,
                    "generated_columns": result.generated_columns,
                    "total_columns_in_pool": len(self.master.columns),
                    "use_dominance": self.pricing_engine.options.use_dominance,
                    "use_cuts": self.pricing_engine.options.use_cuts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return result


def main() -> None:
    solver = BPCSolver(
        instance_dir=Path(RUN_INSTANCE_DIR),
        output_root=Path(RUN_OUTPUT_DIR),
        pricing_hook=None,
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
        print("BPC scaffold finished.")
        print(f"explored_nodes: {result.explored_nodes}")
        print(f"generated_columns: {result.generated_columns}")
        print(f"best_objective: {result.best_objective}")


if __name__ == "__main__":
    main()
