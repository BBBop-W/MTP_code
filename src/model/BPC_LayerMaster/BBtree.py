import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import pandas as pd
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config

from src.model.BPC_LayerMaster.CG import ColumnGenerationEngine, MasterProblem
from src.model.BPC_LayerMaster.pricing import EarlyStopPricingEngine

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

class BBNode:
    def __init__(self, node_id: int, depth: int, branch_q_bounds: Dict[int, Tuple[float|None, float|None]] = None):
        self.node_id = node_id
        self.depth = depth
        self.branch_q_bounds = branch_q_bounds or {}
        self.lower_bound = -math.inf
        # In this LayerMaster decoupled version, we DO NOT branch on a_i because a_i requires
        # identifying columns as "a full wagon". A LayerColumn is only half a wagon. 

class BBTree:
    def __init__(
        self,
        instance_dir: Path,
        output_root: Path,
        max_nodes: int = 200,
        max_cg_iters: int = 100,
        log_to_console: bool = True,
        use_dominance: bool = True,
        use_cuts: bool = False,
        use_rc_bound: bool = True,
        use_residual_profile: bool = True,
        use_outer_inner_profile: bool = False,
        use_height_order: bool = True,
        use_local_residual_skyline: bool = True,
        residual_profile_mode: str = "full",
        compute_reachable_types: bool = False,
        print_bb_progress: bool = True,
        print_subproblem_progress: bool = False,
        mip_gap_tol: float = 5e-6,
    ) -> None:
        self.instance_dir = instance_dir
        self.output_dir = output_root / instance_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cars_path = instance_dir / "cars.csv"
        carriage_path = instance_dir / "carriage.csv"
        self.car_info = normalize_car_table(cars_path)
        self.carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])

        self.max_nodes = max_nodes
        self.log_to_console = log_to_console
        self.print_bb_progress = print_bb_progress
        self.mip_gap_tol = Config.gap

        self.master = MasterProblem(
            car_info=self.car_info,
            carriage_num=self.carriage_num,
        )
        self.master.seed_initial_columns()

        pricing_engine = EarlyStopPricingEngine(
            use_dominance=use_dominance,
            use_cuts=use_cuts,
            use_rc_bound=use_rc_bound,
            use_residual_profile=use_residual_profile,
            use_outer_inner_profile=use_outer_inner_profile,
            use_height_order=use_height_order,
            use_local_residual_skyline=use_local_residual_skyline,
            residual_profile_mode=residual_profile_mode,
            compute_reachable_types=compute_reachable_types,
            verbose=print_subproblem_progress,
        )
        
        self.cg_engine = ColumnGenerationEngine(
            master=self.master,
            pricing_engine=pricing_engine,
            max_cg_iters=max_cg_iters,
            log_to_console=log_to_console,
        )

        self.best_obj: float | None = None
        self.best_theta: Dict[str, float] | None = None
        self.global_lb: float = -math.inf

    def solve(self) -> BPCResult:
        node_counter = 0
        queue: deque[BBNode] = deque([BBNode(node_id=node_counter, depth=0)])
        explored = 0

        if self.print_bb_progress:
            print(f"[BB] Start solve: max_nodes={self.max_nodes}, MIPGap={self.mip_gap_tol}")
            print(f"{'Node':>6}  {'Depth':>6}  {'Left':>6}  {'Global LB':>14}  {'Current Node':>14}  {'Best Incumbent':>14}  {'Gap':>8}  {'Time(s)':>8}")

        start_time = time.time()

        while queue and explored < self.max_nodes:
            # Sort queue to process best bound first (Best-first search)
            queue = deque(sorted(list(queue), key=lambda x: x.lower_bound))
            
            node = queue.popleft()
            
            # Update global lower bound (LB can only increase)
            if node.lower_bound > self.global_lb:
                self.global_lb = node.lower_bound

            # Check gap before processing
            if self.best_obj is not None and self.global_lb != -math.inf:
                # GAP = |UB - LB| / |UB|
                gap = abs(self.best_obj - self.global_lb) / (abs(self.best_obj) + 1e-10)
                if gap <= self.mip_gap_tol:
                    if self.print_bb_progress:
                        print(f"[BB] Target MIP Gap reached: {gap:.4%} <= {self.mip_gap_tol:.4%}")
                    break
                    
            explored += 1

            lp_solution = self.cg_engine.solve(
                branch_a_bounds=None, 
                branch_q_bounds=node.branch_q_bounds
            )
            
            elapsed = time.time() - start_time
            
            if lp_solution.objective is None:
                if self.print_bb_progress:
                    best_str = f"{self.best_obj:.2f}" if self.best_obj is not None else "-"
                    lb_str = f"{self.global_lb:.2f}" if self.global_lb != -math.inf else "-"
                    print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {lb_str:>14}  {'Infeasible':>14}  {best_str:>14}  {'N/A':>8}  {elapsed:8.2f}")
                continue

            current_bound = lp_solution.objective
            
            def format_bound(val: float | None) -> str:
                return f"{val:.2f}" if val is not None and val != -math.inf else "-"
                
            def format_gap(best: float | None, lb: float) -> str:
                if best is None or lb == -math.inf:
                    return "-"
                g = abs(best - lb) / (abs(best) + 1e-10)
                return f"{g:.2%}"
            
            # Node bounding
            if self.best_obj is not None and current_bound >= self.best_obj - 1e-6:
                if self.print_bb_progress:
                    best_str = format_bound(self.best_obj)
                    lb_str = format_bound(self.global_lb)
                    gap_str = format_gap(self.best_obj, self.global_lb)
                    curr_str = f"{current_bound:.2f}"
                    print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {lb_str:>14}  {curr_str:>14}  {best_str:>14}  {gap_str:>8}  {elapsed:8.2f}")
                continue

            best_str = format_bound(self.best_obj)
            lb_str = format_bound(self.global_lb)
            gap_str = format_gap(self.best_obj, self.global_lb)
            curr_str = f"{current_bound:.2f}"
            
            if self.print_bb_progress:
                print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {lb_str:>14}  {curr_str:>14}  {best_str:>14}  {gap_str:>8}  {elapsed:8.2f}")

            if self.master.is_integral(lp_solution):
                if self.best_obj is None or lp_solution.objective < self.best_obj:
                    self.best_obj = lp_solution.objective
                    self.best_theta = {k: round(v) for k, v in lp_solution.theta_values.items()}
                    
                    # Re-check global gap after finding new best
                    if self.global_lb != -math.inf:
                        gap = abs(self.best_obj - self.global_lb) / (abs(self.best_obj) + 1e-10)
                        if gap <= self.mip_gap_tol:
                            if self.print_bb_progress:
                                print(f"[BB] Target MIP Gap reached upon finding new incumbent: {gap:.4%} <= {self.mip_gap_tol:.4%}")
                            break
                continue

            branch_var = self.master.choose_branch_var(lp_solution)
            if branch_var is None:
                continue
                
            b_type, car_type, value = branch_var
            floor_v = math.floor(value)
            ceil_v = math.ceil(value)

            left_q = dict(node.branch_q_bounds)
            right_q = dict(node.branch_q_bounds)

            old_left = left_q.get(car_type, (None, None))
            left_q[car_type] = (old_left[0], floor_v)
            old_right = right_q.get(car_type, (None, None))
            right_q[car_type] = (ceil_v, old_right[1])

            node_counter += 1
            left_node = BBNode(node_id=node_counter, depth=node.depth + 1, branch_q_bounds=left_q)
            left_node.lower_bound = current_bound
            queue.append(left_node)
            
            node_counter += 1
            right_node = BBNode(node_id=node_counter, depth=node.depth + 1, branch_q_bounds=right_q)
            right_node.lower_bound = current_bound
            queue.append(right_node)

        result = BPCResult(
            best_objective=self.best_obj,
            best_theta=self.best_theta,
            explored_nodes=explored,
            generated_columns=self.cg_engine.generated_columns,
        )
        return result
