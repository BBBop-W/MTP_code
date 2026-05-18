from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import gurobipy as gp
import pandas as pd

from src.model.BPC_wagon.CG import MasterLPSolution, MasterProblem, PatternColumn
from src.utility.config import config as Config
from src.utility.dynamic_segmentation import get_model_segments


@dataclass
class SolverPricingStats:
    labeling_time: float = 0.0
    feasibility_time: float = 0.0
    merge_time: float = 0.0
    solver_time: float = 0.0
    solver_nodes: float = 0.0


@dataclass(frozen=True)
class SolverPricingOptions:
    use_cuts: bool = False


class SolverPricingEngine:
    """Solve the full-wagon DW pricing subproblem with Gurobi."""

    def __init__(
        self,
        num_splits: int = 1,
        independent_mode_split: bool = True,
        max_units_per_compartment: int = 10,
        max_columns_per_pricing: int = 1,
        use_cuts: bool = False,
        verbose: bool = False,
    ) -> None:
        self.num_splits = int(num_splits)
        self.independent_mode_split = bool(independent_mode_split)
        self.max_units_per_compartment = int(max_units_per_compartment)
        self.max_columns_per_pricing = 1
        self.verbose = verbose
        self.options = SolverPricingOptions(use_cuts=use_cuts)
        self.wagon_capacity_cut = self.max_units_per_compartment
        self.stats = SolverPricingStats()
        self._column_seq = 0
        self._seen_signatures: Set[Tuple[int, ...]] = set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[WagonSolverPricing] {msg}")

    def generate_columns(self, lp_solution: MasterLPSolution, master: MasterProblem) -> List[PatternColumn]:
        if lp_solution.dual_gamma is None:
            return []

        columns: List[PatternColumn] = []
        forbidden = list(self._seen_signatures)
        for _ in range(self.max_columns_per_pricing):
            t0 = time.time()
            column = self._solve_one_wagon(lp_solution, master, forbidden)
            self.stats.solver_time += time.time() - t0
            if column is None:
                break
            signature = tuple(int(column.q.get(i, 0)) for i in master.I)
            forbidden.append(signature)
            columns.append(column)
        return columns

    def _solve_one_wagon(
        self,
        lp_solution: MasterLPSolution,
        master: MasterProblem,
        forbidden_signatures: List[Tuple[int, ...]],
    ) -> PatternColumn | None:
        car_info = master.car_info
        I = list(master.I)
        P = ["h-h", "h-m", "m-h", "m-m"]
        deck_side_mode = {
            "h-h": {"left": "h", "right": "h"},
            "h-m": {"left": "h", "right": "m"},
            "m-h": {"left": "m", "right": "h"},
            "m-m": {"left": "m", "right": "m"},
        }
        heights = {i: float(car_info.iloc[i - 1]["height"]) for i in I}
        segments_data = get_model_segments(
            pd.DataFrame({"height": sorted(set(heights.values()))}),
            self.num_splits,
            self.independent_mode_split,
        )

        H: List[str] = []
        H_lower: List[str] = []
        H_upper: List[str] = []
        component_side: Dict[str, str] = {}
        L: Dict[str, float] = {}
        h_h_limits: Dict[str, float] = {}
        h_m_limits: Dict[str, float] = {}

        def add_layer(compartment_prefix: str, target: List[str]) -> int:
            central = f"{compartment_prefix}_central"
            H.append(central)
            target.append(central)
            component_side[central] = "central"
            L[central] = float(segments_data[compartment_prefix]["central"]["len"])
            h_h_limits[central] = float(segments_data[compartment_prefix]["central"]["h_h"])
            h_m_limits[central] = float(segments_data[compartment_prefix]["central"]["h_m"])
            blocks = segments_data[compartment_prefix]["blocks"]
            for block in blocks:
                for side in ["left", "right"]:
                    name = f"{compartment_prefix}_{block['name']}_{side}"
                    H.append(name)
                    target.append(name)
                    component_side[name] = side
                    L[name] = float(block["len"])
                    h_h_limits[name] = float(block["h_h"])
                    h_m_limits[name] = float(block["h_m"])
            return len(blocks)

        num_lower_blocks = add_layer("lower", H_lower)
        num_upper_blocks = add_layer("upper", H_upper)

        def height_limit(h: str, side_mode: str) -> float:
            return h_m_limits[h] if side_mode == "m" else h_h_limits[h]

        def eta(i: int, p: str, h: str) -> int:
            side = component_side[h]
            if side == "central":
                limit = min(height_limit(h, mode) for mode in deck_side_mode[p].values())
            else:
                limit = height_limit(h, deck_side_mode[p][side])
            return int(heights[i] <= limit)

        def get_intervals(compartment_prefix: str, num_blocks: int):
            intervals = []
            for left_count in range(num_blocks + 1):
                for right_count in range(num_blocks + 1):
                    blocks = [f"{compartment_prefix}_central"]
                    blocks.extend(f"{compartment_prefix}_block_{idx}_left" for idx in range(1, left_count + 1))
                    blocks.extend(f"{compartment_prefix}_block_{idx}_right" for idx in range(1, right_count + 1))
                    if left_count == num_blocks and right_count == num_blocks:
                        mod = -400.0
                    elif left_count == num_blocks or right_count == num_blocks:
                        mod = 0.0
                    else:
                        mod = 400.0
                    intervals.append((blocks, mod))
            return intervals

        model = gp.Model("wagon_solver_pricing")
        model.Params.OutputFlag = 0
        Config.apply_gurobi_params(model)

        N = self.max_units_per_compartment
        Delta = 400.0
        z = model.addVars(P, vtype=gp.GRB.BINARY, name="z")
        x = model.addVars(I, H, vtype=gp.GRB.INTEGER, lb=0, ub=N, name="x")
        a = model.addVars(I, vtype=gp.GRB.BINARY, name="a")
        q = {i: gp.quicksum(x[i, h] for h in H) for i in I}

        for forbid_idx, signature in enumerate(forbidden_signatures):
            y_pos = model.addVars(I, vtype=gp.GRB.BINARY, name=f"forbid_pos[{forbid_idx}]")
            y_neg = model.addVars(I, vtype=gp.GRB.BINARY, name=f"forbid_neg[{forbid_idx}]")
            big_m = max(max(int(master.U[i]) for i in I), 2 * N) + 1
            for i, value in zip(I, signature):
                model.addConstr(
                    q[i] - int(value) >= 1 - big_m * (1 - y_pos[i]),
                    name=f"forbid_gt[{forbid_idx},{i}]",
                )
                model.addConstr(
                    int(value) - q[i] >= 1 - big_m * (1 - y_neg[i]),
                    name=f"forbid_lt[{forbid_idx},{i}]",
                )
            model.addConstr(
                gp.quicksum(y_pos[i] + y_neg[i] for i in I) >= 1,
                name=f"forbid_signature[{forbid_idx}]",
            )

        model.addConstr(gp.quicksum(z[p] for p in P) == 1, name="deck_position")
        for i in I:
            model.addConstr(q[i] <= int(master.U[i]), name=f"type_upper[{i}]")
            model.addConstr(q[i] <= 2 * N * a[i], name=f"activate_upper[{i}]")
            model.addConstr(q[i] >= a[i], name=f"activate_lower[{i}]")

        for blocks, mod in get_intervals("lower", num_lower_blocks) + get_intervals("upper", num_upper_blocks):
            cap = sum(L[h] for h in blocks) + mod
            model.addConstr(
                gp.quicksum(x[i, h] * (master.length[i] + Delta) for i in I for h in blocks) <= cap
            )

        for i in I:
            for h in H:
                model.addConstr(x[i, h] <= N * gp.quicksum(eta(i, p, h) * z[p] for p in P))

        model.addConstr(gp.quicksum(x[i, h] for i in I for h in H_lower) <= N, name="lower_quantity")
        model.addConstr(gp.quicksum(x[i, h] for i in I for h in H_upper) <= N, name="upper_quantity")

        sr_terms: List[Tuple[float, gp.Var]] = []
        if self.options.use_cuts:
            for cut_idx, (subset, sigma) in enumerate(lp_solution.dual_sigma.items()):
                if abs(float(sigma)) <= 1e-12:
                    continue
                high_vars = []
                for i in subset:
                    high = model.addVar(vtype=gp.GRB.BINARY, name=f"sr_high[{cut_idx},{i}]")
                    threshold = int(master.U[i] // 2 + 1)
                    max_q = int(master.U[i])
                    if threshold <= 0 or max_q < threshold:
                        model.addConstr(high == 0, name=f"sr_high_off[{cut_idx},{i}]")
                    else:
                        model.addConstr(q[i] >= threshold * high, name=f"sr_high_lb[{cut_idx},{i}]")
                        model.addConstr(q[i] <= threshold - 1 + max_q * high, name=f"sr_high_ub[{cut_idx},{i}]")
                    high_vars.append(high)
                coeff = model.addVar(vtype=gp.GRB.BINARY, name=f"sr_coeff[{cut_idx}]")
                count = gp.quicksum(high_vars)
                model.addConstr(count >= 2 * coeff, name=f"sr_coeff_lb[{cut_idx}]")
                model.addConstr(count <= 1 + 2 * coeff, name=f"sr_coeff_ub[{cut_idx}]")
                sr_terms.append((float(sigma), coeff))

        rc = gp.LinExpr(-float(lp_solution.dual_gamma or 0.0))
        if self.options.use_cuts:
            rc -= float(lp_solution.dual_eta)
        for i in I:
            coef = (
                master.length[i]
                + lp_solution.dual_alpha.get(i, 0.0)
                + lp_solution.dual_beta.get(i, 0.0)
                + lp_solution.dual_branch_q.get(i, 0.0)
            )
            rc -= coef * q[i]
            rc -= lp_solution.dual_branch_a.get(i, 0.0) * a[i]
        for sigma, coeff in sr_terms:
            rc -= sigma * coeff
        model.setObjective(rc, gp.GRB.MINIMIZE)
        model.optimize()
        self.stats.solver_nodes += float(model.NodeCount)

        if model.Status != gp.GRB.OPTIMAL or model.ObjVal >= -1e-5:
            return None

        quantities = {i: int(round(q[i].getValue())) for i in I}
        signature = tuple(quantities[i] for i in I)
        if signature in self._seen_signatures:
            return None
        self._seen_signatures.add(signature)

        self._column_seq += 1
        selected_deck = max(P, key=lambda p: z[p].X)
        cost = -sum(master.length[i] * quantities[i] for i in I)
        self._log(f"generated column rc={model.ObjVal:.6f}, deck={selected_deck}, q={quantities}")
        return PatternColumn(
            column_id=f"solver_{selected_deck}_{self._column_seq}",
            q=quantities,
            cost=float(cost),
            metadata={
                "source": "solver_pricing",
                "deck": selected_deck,
                "reduced_cost": f"{model.ObjVal:.6f}",
            },
        )
