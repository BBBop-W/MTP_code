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
    metadata: Dict[str, str] | None = None

@dataclass
class MasterLPSolution:
    status: int
    objective: float | None
    theta_values: Dict[str, float]
    dual_alpha: Dict[int, float]
    dual_beta: Dict[int, float]
    dual_gamma: float | None
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
    """Restricted master problem (set covering form) for MLP-IC."""

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
            col = PatternColumn(
                column_id=f"seed_i{i}",
                q={k: (1 if k == i else 0) for k in self.I},
                cost=-self.length[i],
                metadata={"source": "seed"},
            )
            self.add_column(col)

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

        model = gp.Model("mlp_ic_master")
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

        branch_a_constr = {}
        branch_q_constr = {}

        for i, (lb, ub) in branch_a_bounds.items():
            expr = gp.quicksum((1 if self.columns[cid].q.get(i, 0) > 0 else 0) * theta[cid] for cid in self.columns)
            if lb is not None:
                branch_a_constr[(i, 'lb')] = model.addConstr(expr >= lb, name=f"branch_a_lb_{i}")
            if ub is not None:
                branch_a_constr[(i, 'ub')] = model.addConstr(expr <= ub, name=f"branch_a_ub_{i}")

        for i, (lb, ub) in branch_q_bounds.items():
            expr = gp.quicksum(self.columns[cid].q.get(i, 0) * theta[cid] for cid in self.columns)
            if lb is not None:
                branch_q_constr[(i, 'lb')] = model.addConstr(expr >= lb, name=f"branch_q_lb_{i}")
            if ub is not None:
                branch_q_constr[(i, 'ub')] = model.addConstr(expr <= ub, name=f"branch_q_ub_{i}")

        # Cuts
        capacity_constr = None
        if use_capacity_cut:
            min_wagons = math.ceil(sum(self.D[i] for i in self.I) / wagon_capacity)
            capacity_constr = model.addConstr(
                gp.quicksum(theta[cid] for cid in self.columns) >= min_wagons,
                name="capacity_cut"
            )

        sr_constrs = {}
        for subset in active_sr_cuts:
            expr = gp.LinExpr()
            for cid, col in self.columns.items():
                val = sum(1 for i in subset if col.q.get(i, 0) > self.U[i] / 2.0)
                coeff = math.floor(0.5 * val)
                if coeff > 0:
                    expr.addTerms(coeff, theta[cid])
            sr_constrs[subset] = model.addConstr(expr <= 1, name=f"sr_cut_{subset}")

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

        dual_branch_a = {i: 0.0 for i in self.I}
        for (i, bound_type), constr in branch_a_constr.items():
            dual_branch_a[i] += float(constr.Pi)

        dual_branch_q = {i: 0.0 for i in self.I}
        for (i, bound_type), constr in branch_q_constr.items():
            dual_branch_q[i] += float(constr.Pi)

        dual_eta = float(capacity_constr.Pi) if capacity_constr is not None else 0.0
        dual_sigma = {subset: float(constr.Pi) for subset, constr in sr_constrs.items()}

        return MasterLPSolution(
            status=model.Status,
            objective=float(model.ObjVal),
            theta_values=theta_values,
            dual_alpha=dual_alpha,
            dual_beta=dual_beta,
            dual_gamma=dual_gamma,
            dual_branch_a=dual_branch_a,
            dual_branch_q=dual_branch_q,
            dual_eta=dual_eta,
            dual_sigma=dual_sigma
        )

    def separate_3sr_cuts(self, solution: MasterLPSolution, eps: float = 1e-4) -> List[Tuple[int, int, int]]:
        violated = []
        for subset in combinations(self.I, 3):
            lhs = 0.0
            for col_id, theta_val in solution.theta_values.items():
                if theta_val > eps:
                    col = self.columns[col_id]
                    val = sum(1 for i in subset if col.q.get(i, 0) > self.U[i] / 2.0)
                    coeff = math.floor(0.5 * val)
                    if coeff > 0:
                        lhs += coeff * theta_val
            if lhs > 1 + eps:
                violated.append(subset)
        return violated

    def choose_branch_var(self, solution: MasterLPSolution, eps: float = 1e-5) -> tuple[str, int, float] | None:
        a_sums = {i: 0.0 for i in self.I}
        q_sums = {i: 0.0 for i in self.I}

        for col_id, theta_val in solution.theta_values.items():
            if theta_val <= eps:
                continue
            col = self.columns[col_id]
            for i in self.I:
                q_ir = col.q.get(i, 0)
                if q_ir > 0:
                    a_sums[i] += theta_val
                    q_sums[i] += q_ir * theta_val

        for i in self.I:
            if abs(q_sums[i] - round(q_sums[i])) > eps:
                return 'q', i, q_sums[i]

        for i in self.I:
            if abs(a_sums[i] - round(a_sums[i])) > eps:
                return 'a', i, a_sums[i]

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
        self.active_sr_cuts: Set[Tuple[int, int, int]] = set()

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
            active_sr_cuts=self.active_sr_cuts,
            use_capacity_cut=self.pricing_engine.options.use_cuts,
            wagon_capacity=self.pricing_engine.wagon_capacity_cut or 10,
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
            
            # Extract sub-stats from pricing engine
            self.stats.labeling_time += self.pricing_engine.stats.labeling_time
            self.stats.bs_time += self.pricing_engine.stats.bs_time
            self.stats.merge_time += self.pricing_engine.stats.merge_time
            
            # Reset pricing engine stats for next iteration to avoid double counting
            self.pricing_engine.stats.labeling_time = 0.0
            self.pricing_engine.stats.bs_time = 0.0
            self.pricing_engine.stats.merge_time = 0.0

            if not new_columns:
                if self.pricing_engine.options.use_cuts:
                    violated_cuts = self.master.separate_3sr_cuts(last_solution)
                    if violated_cuts:
                        new_cuts_added = False
                        for cut in violated_cuts:
                            if cut not in self.active_sr_cuts:
                                self.active_sr_cuts.add(cut)
                                new_cuts_added = True
                        if new_cuts_added:
                            self._log_cg(f"Iter={it}: Separated {len(violated_cuts)} new 3-SR cuts. Resuming CG.")
                            t0 = time.time()
                            last_solution = self.master.solve_lp(
                                branch_a_bounds=branch_a_bounds,
                                branch_q_bounds=branch_q_bounds,
                                active_sr_cuts=self.active_sr_cuts,
                                use_capacity_cut=self.pricing_engine.options.use_cuts,
                                wagon_capacity=self.pricing_engine.wagon_capacity_cut or 10,
                                time_limit=Config.timelimit,
                                log_to_console=False,
                            )
                            self.stats.master_time += (time.time() - t0)
                            continue
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
                active_sr_cuts=self.active_sr_cuts,
                use_capacity_cut=self.pricing_engine.options.use_cuts,
                wagon_capacity=self.pricing_engine.wagon_capacity_cut or 10,
                time_limit=Config.timelimit,
                log_to_console=False,
            )
            self.stats.master_time += (time.time() - t0)
            
            if last_solution.objective is None:
                break

        self.stats.total_time += (time.time() - start_solve)
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
