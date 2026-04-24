import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

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
    metadata: Dict[str, str] | None = None

@dataclass
class MasterLPSolution:
    status: int
    objective: float | None
    theta_values: Dict[str, float]
    dual_alpha: Dict[int, float]
    dual_beta: Dict[int, float]
    dual_gamma: float | None

class MasterProblem:
    """Restricted master problem (set covering form) for MLP-IC.

    MP (LP relaxation):
      min sum_r c_r * theta_r
      s.t. sum_r q_i^r * theta_r >= D_i
           sum_r q_i^r * theta_r <= C_i + D_i
           sum_r theta_r <= |J|
           theta_r >= 0
    """

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
        # Very light seed: one single-unit column per automobile type.
        for i in self.I:
            col = PatternColumn(
                column_id=f"seed_i{i}",
                q={k: (1 if k == i else 0) for k in self.I},
                cost=-self.length[i],
                metadata={"source": "seed"},
            )
            self.add_column(col)

    def solve_lp(
        self,
        branch_bounds: Dict[str, Tuple[float | None, float | None]] | None = None,
        time_limit: float | None = None,
        log_to_console: bool = False,
    ) -> MasterLPSolution:
        branch_bounds = branch_bounds or {}

        model = gp.Model("mlp_ic_master")
        model.Params.OutputFlag = 1 if log_to_console else 0
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)

        theta: Dict[str, gp.Var] = {}
        for col_id, col in self.columns.items():
            lb = 0.0
            ub = gp.GRB.INFINITY
            if col_id in branch_bounds:
                b_lb, b_ub = branch_bounds[col_id]
                if b_lb is not None:
                    lb = max(lb, float(b_lb))
                if b_ub is not None:
                    ub = min(ub, float(b_ub))
            theta[col_id] = model.addVar(lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, obj=col.cost, name=f"theta[{col_id}]")

        # Artificial unmet-demand variables keep RMP feasible at every node.
        unmet = {
            i: model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, obj=self.penalty_unmet, name=f"unmet[{i}]")
            for i in self.I
        }

        mandatory_constr = {}
        optional_constr = {}
        for i in self.I:
            mandatory_constr[i] = model.addConstr(
                gp.quicksum(self.columns[cid].q.get(i, 0) * theta[cid] for cid in self.columns) + unmet[i] >= self.D[i],
                name=f"mandatory[{i}]",
            )
            optional_constr[i] = model.addConstr(
                gp.quicksum(self.columns[cid].q.get(i, 0) * theta[cid] for cid in self.columns) <= self.U[i],
                name=f"optional[{i}]",
            )

        wagon_constr = model.addConstr(
            gp.quicksum(theta[cid] for cid in self.columns) <= self.carriage_num,
            name="wagon_limit",
        )

        model.ModelSense = gp.GRB.MINIMIZE
        model.optimize()

        if model.Status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT}:
            return MasterLPSolution(
                status=model.Status,
                objective=None,
                theta_values={},
                dual_alpha={},
                dual_beta={},
                dual_gamma=None,
            )

        theta_values = {cid: float(theta[cid].X) for cid in self.columns}
        dual_alpha = {i: float(mandatory_constr[i].Pi) for i in self.I}
        dual_beta = {i: float(optional_constr[i].Pi) for i in self.I}
        dual_gamma = float(wagon_constr.Pi)

        return MasterLPSolution(
            status=model.Status,
            objective=float(model.ObjVal),
            theta_values=theta_values,
            dual_alpha=dual_alpha,
            dual_beta=dual_beta,
            dual_gamma=dual_gamma,
        )

    @staticmethod
    def is_integral(solution: MasterLPSolution, eps: float = 1e-6) -> bool:
        if not solution.theta_values:
            return True
        return all(abs(v - round(v)) <= eps for v in solution.theta_values.values())

    @staticmethod
    def choose_branch_var(solution: MasterLPSolution, eps: float = 1e-6) -> tuple[str, float] | None:
        for col_id, value in solution.theta_values.items():
            if abs(value - round(value)) > eps:
                return col_id, value
        return None

class ColumnGenerationEngine:
    def __init__(self, master: MasterProblem, pricing_engine, max_cg_iters: int = 100, log_to_console: bool = True):
        self.master = master
        self.pricing_engine = pricing_engine
        self.max_cg_iters = max_cg_iters
        self.log_to_console = log_to_console
        self.generated_columns = 0

    def _log_cg(self, msg: str):
        if self.log_to_console:
            print(f"[CG] {msg}")

    def solve(self, branch_bounds: Dict[str, Tuple[float | None, float | None]] | None = None) -> MasterLPSolution:
        self._log_cg(f"Start CG, initial columns={len(self.master.columns)}")
        last_solution = self.master.solve_lp(
            branch_bounds=branch_bounds,
            time_limit=Config.timelimit,
            log_to_console=False,
        )
        if last_solution.objective is None:
            self._log_cg(f"RMP solve failed, status={last_solution.status}")
            return last_solution

        self._log_cg(f"Initial RMP objective={last_solution.objective:.6f}")

        for it in range(1, self.max_cg_iters + 1):
            new_columns = self.pricing_engine.generate_columns(last_solution, self.master)
            if not new_columns:
                self._log_cg(f"Iter={it}: no new column found (reduced cost >= 0). Stop CG.")
                break

            added = 0
            for col in new_columns:
                if col.column_id not in self.master.columns:
                    self.master.add_column(col)
                    self.generated_columns += 1
                    added += 1
                    rc = float(col.metadata.get("reduced_cost", 0.0)) if col.metadata else 0.0
                    self._log_cg(f"  + Add column={col.column_id}, length={-col.cost:.1f}, reduced_cost={rc:.6f}, q_sum={sum(col.q.values())}")

            if added == 0:
                self._log_cg(f"Iter={it}: generated columns were duplicates. Stop CG.")
                break

            last_solution = self.master.solve_lp(
                branch_bounds=branch_bounds,
                time_limit=Config.timelimit,
                log_to_console=False,
            )
            if last_solution.objective is None:
                self._log_cg(f"Iter={it}: RMP resolve failed, status={last_solution.status}")
                break
            
            self._log_cg(f"Iter={it} complete: RMP objective={last_solution.objective:.6f}, total_columns={len(self.master.columns)}")

        return last_solution

if __name__ == "__main__":
    from src.model.BPC.pricing import EarlyStopPricingEngine

    print("--- Testing Column Generation (Root Node) ---")
    instance_dir = PROJECT_ROOT / "data/Instance/m10c10"
    cars_path = instance_dir / "cars.csv"
    carriage_path = instance_dir / "carriage.csv"
    
    if not cars_path.exists():
        print(f"Cannot find instance data at {cars_path}")
        sys.exit(1)
        
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
    car_info["length"] = pd.to_numeric(car_info["length"])
    car_info["height"] = pd.to_numeric(car_info["height"])
    car_info["optional"] = pd.to_numeric(car_info["optional"]).astype(int)
    car_info["mandatory"] = pd.to_numeric(car_info["mandatory"]).astype(int)
    
    carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])

    master = MasterProblem(car_info=car_info, carriage_num=carriage_num)
    master.seed_initial_columns()

    pricing_engine = EarlyStopPricingEngine(
        use_dominance=True,
        use_cuts=False,
        verbose=False,
    )
    
    cg_engine = ColumnGenerationEngine(
        master=master,
        pricing_engine=pricing_engine,
        max_cg_iters=100,
        log_to_console=True
    )
    
    import time
    t0 = time.time()
    final_sol = cg_engine.solve()
    t1 = time.time()
    
    print(f"\n--- CG Finished in {t1-t0:.2f}s ---")
    print(f"Final Objective: {final_sol.objective:.6f}")
    print(f"Generated Columns: {cg_engine.generated_columns}")
