import sys
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Set
from dataclasses import dataclass, field
from itertools import combinations

import gurobipy as gp
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config

@dataclass(frozen=True)
class PatternColumn:
    column_id: str
    q: Dict[int, int]
    cost: float
    compartment: str   # 'upper' or 'lower'
    deck_mode: str     # 'h-h', 'm-m', 'h-m', 'm-h'
    metadata: Dict[str, str] | None = None

@dataclass
class MasterLPSolution:
    status: int
    objective: float | None
    theta_values: Dict[str, float]
    dual_alpha: Dict[int, float]
    dual_beta: Dict[int, float]
    dual_gamma: Dict[str, float]  # indexed by deck_mode
    dual_kappa: float
    dual_branch_a: Dict[int, float] = field(default_factory=dict)
    dual_branch_q: Dict[int, float] = field(default_factory=dict)
    dual_eta: float = 0.0
    dual_sigma: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    
@dataclass
class CGStats:
    total_time: float = 0.0
    master_time: float = 0.0
    pricing_time: float = 0.0
    labeling_time: float = 0.0
    bs_time: float = 0.0
    merge_time: float = 0.0

class MasterProblem:
    def __init__(self, car_info: pd.DataFrame, carriage_num: int, penalty_unmet: float = 1e6) -> None:
        self.car_info = car_info.reset_index(drop=True)
        self.carriage_num = int(carriage_num)
        self.penalty_unmet = float(penalty_unmet)

        self.I: List[int] = list(range(1, len(self.car_info) + 1))
        self.length: Dict[int, float] = {i: float(self.car_info.iloc[i - 1]["length"]) for i in self.I}
        self.D: Dict[int, int] = {i: int(self.car_info.iloc[i - 1]["mandatory"]) for i in self.I}
        self.U: Dict[int, int] = {
            i: int(self.car_info.iloc[i - 1]["mandatory"] + self.car_info.iloc[i - 1]["optional"])
            for i in self.I
        }

        self.columns: Dict[str, PatternColumn] = {}

    def add_column(self, column: PatternColumn) -> None:
        if column.column_id in self.columns:
            return
        self.columns[column.column_id] = column

    def seed_initial_columns(self) -> None:
        for i in self.I:
            # Seed columns as 'upper' horizontal to just provide feasibility
            col = PatternColumn(
                column_id=f"seed_upper_i{i}",
                q={k: (1 if k == i else 0) for k in self.I},
                cost=-self.length[i],
                compartment="upper",
                deck_mode="h-h",
                metadata={"source": "seed"},
            )
            self.add_column(col)
            # Empty lower to match
            col_empty = PatternColumn(
                column_id=f"seed_lower_empty_{i}",
                q={},
                cost=0.0,
                compartment="lower",
                deck_mode="h-h",
                metadata={"source": "seed"},
            )
            self.add_column(col_empty)

    def solve_lp(
        self,
        branch_a_bounds: Dict[int, Tuple[float | None, float | None]] | None = None,
        branch_q_bounds: Dict[int, Tuple[float | None, float | None]] | None = None,
        active_sr_cuts: Set[Tuple[int, int, int]] | None = None,
        use_capacity_cut: bool = False,
        wagon_capacity: int = 10,
        time_limit: float | None = None,
        log_to_console: bool = False,
    ) -> MasterLPSolution:
        branch_a_bounds = branch_a_bounds or {}
        branch_q_bounds = branch_q_bounds or {}
        active_sr_cuts = active_sr_cuts or set()

        model = gp.Model("mlp_ic_master_layer")
        model.Params.OutputFlag = 1 if log_to_console else 0
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)

        theta: Dict[str, gp.Var] = {}
        for col_id, col in self.columns.items():
            theta[col_id] = model.addVar(lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, obj=col.cost, name=f"theta[{col_id}]")

        unmet = {
            i: model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, obj=self.penalty_unmet, name=f"unmet[{i}]")
            for i in self.I
        }

        mandatory_constr = {}
        optional_constr = {}
        for i in self.I:
            mandatory_constr[i] = model.addConstr(
                gp.quicksum(col.q.get(i, 0) * theta[cid] for cid, col in self.columns.items()) + unmet[i] >= self.D[i],
                name=f"mandatory[{i}]",
            )
            optional_constr[i] = model.addConstr(
                gp.quicksum(col.q.get(i, 0) * theta[cid] for cid, col in self.columns.items()) <= self.U[i],
                name=f"optional[{i}]",
            )

        # Wagon map constraints (sum upper_p - sum lower_p = 0)
        modes = ["h-h", "h-m", "m-h", "m-m"]
        wagon_map_constr = {}
        for p in modes:
            wagon_map_constr[p] = model.addConstr(
                gp.quicksum(theta[cid] for cid, col in self.columns.items() if col.compartment == "upper" and col.deck_mode == p) -
                gp.quicksum(theta[cid] for cid, col in self.columns.items() if col.compartment == "lower" and col.deck_mode == p) == 0,
                name=f"wagon_map[{p}]"
            )

        # Total wagons constraint (only count upper layers since map ensures they match)
        num_wagons_constr = model.addConstr(
            gp.quicksum(theta[cid] for cid, col in self.columns.items() if col.compartment == "upper") <= self.carriage_num,
            name="wagon_limit",
        )

        branch_q_constr = {}
        # Note: We skip branch_a_bounds in layer formulation since 'a_i' means car is in *a wagon*, which is harder to define linearly for separated layers without linking variables.
        # So we only enforce branch_q_bounds.
        for i, (lb, ub) in branch_q_bounds.items():
            expr = gp.quicksum(col.q.get(i, 0) * theta[cid] for cid, col in self.columns.items())
            if lb is not None:
                branch_q_constr[(i, 'lb')] = model.addConstr(expr >= lb, name=f"branch_q_lb_{i}")
            if ub is not None:
                branch_q_constr[(i, 'ub')] = model.addConstr(expr <= ub, name=f"branch_q_ub_{i}")

        model.ModelSense = gp.GRB.MINIMIZE
        model.optimize()

        if model.Status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT}:
            return MasterLPSolution(
                status=model.Status,
                objective=None,
                theta_values={},
                dual_alpha={},
                dual_beta={},
                dual_gamma={},
                dual_kappa=0.0,
            )

        theta_values = {cid: float(theta[cid].X) for cid in self.columns}
        dual_alpha = {i: float(mandatory_constr[i].Pi) for i in self.I}
        dual_beta = {i: float(optional_constr[i].Pi) for i in self.I}
        dual_gamma = {p: float(wagon_map_constr[p].Pi) for p in modes}
        dual_kappa = float(num_wagons_constr.Pi)

        dual_branch_q = {i: 0.0 for i in self.I}
        for (i, bound_type), constr in branch_q_constr.items():
            dual_branch_q[i] += float(constr.Pi)

        return MasterLPSolution(
            status=model.Status,
            objective=float(model.ObjVal),
            theta_values=theta_values,
            dual_alpha=dual_alpha,
            dual_beta=dual_beta,
            dual_gamma=dual_gamma,
            dual_kappa=dual_kappa,
            dual_branch_a={}, # Disabled for layer-based
            dual_branch_q=dual_branch_q,
            dual_eta=0.0,
            dual_sigma={}
        )

    def choose_branch_var(self, solution: MasterLPSolution, eps: float = 1e-5) -> tuple[str, int, float] | None:
        q_sums = {i: 0.0 for i in self.I}
        for col_id, theta_val in solution.theta_values.items():
            if theta_val <= eps:
                continue
            col = self.columns[col_id]
            for i in self.I:
                q_ir = col.q.get(i, 0)
                if q_ir > 0:
                    q_sums[i] += q_ir * theta_val

        for i in self.I:
            if abs(q_sums[i] - round(q_sums[i])) > eps:
                return 'q', i, q_sums[i]

        return None

    def is_integral(self, solution: MasterLPSolution, eps: float = 1e-5) -> bool:
        if not solution.theta_values:
            return True
        return self.choose_branch_var(solution, eps) is None


class ColumnGenerationEngine:
    def __init__(self, master: MasterProblem, pricing_engine, max_cg_iters: int = 100, log_to_console: bool = True):
        self.master = master
        self.pricing_engine = pricing_engine
        self.max_cg_iters = max_cg_iters
        self.log_to_console = log_to_console
        self.generated_columns = 0
        self.stats = CGStats()

    def _log_cg(self, msg: str):
        if self.log_to_console:
            print(f"[CG] {msg}")

    def solve(
        self, 
        branch_a_bounds: Dict[int, Tuple[float | None, float | None]] | None = None,
        branch_q_bounds: Dict[int, Tuple[float | None, float | None]] | None = None,
    ) -> MasterLPSolution:
        start_solve = time.time()
        
        t0 = time.time()
        last_solution = self.master.solve_lp(
            branch_a_bounds=branch_a_bounds,
            branch_q_bounds=branch_q_bounds,
            time_limit=Config.timelimit,
            log_to_console=False,
        )
        self.stats.master_time += (time.time() - t0)
        
        if last_solution.objective is None:
            return last_solution

        for it in range(1, self.max_cg_iters + 1):
            t0 = time.time()
            new_columns = self.pricing_engine.generate_columns(last_solution, self.master)
            p_time = time.time() - t0
            self.stats.pricing_time += p_time
            
            self.stats.labeling_time += self.pricing_engine.stats.labeling_time
            self.stats.bs_time += self.pricing_engine.stats.bs_time
            
            self.pricing_engine.stats.labeling_time = 0.0
            self.pricing_engine.stats.bs_time = 0.0

            if not new_columns:
                self._log_cg(f"Iter={it}: no new column found (reduced cost >= 0). Stop CG.")
                break

            added = 0
            for col in new_columns:
                if col.column_id not in self.master.columns:
                    self.master.add_column(col)
                    self.generated_columns += 1
                    added += 1

            if added == 0:
                break

            t0 = time.time()
            last_solution = self.master.solve_lp(
                branch_a_bounds=branch_a_bounds,
                branch_q_bounds=branch_q_bounds,
                time_limit=Config.timelimit,
                log_to_console=False,
            )
            self.stats.master_time += (time.time() - t0)
            
            if last_solution.objective is None:
                break

        self.stats.total_time += (time.time() - start_solve)
        return last_solution
