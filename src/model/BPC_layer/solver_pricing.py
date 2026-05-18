from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import gurobipy as gp

from src.model.BPC_layer import feasibility_check
from src.model.BPC_layer.CG import MasterLPSolution, MasterProblem, PatternColumn
from src.model.BPC_layer.labeling import DualValues, LayerSpec
from src.utility.config import config as Config


@dataclass
class SolverPricingStats:
    labeling_time: float = 0.0
    bs_time: float = 0.0
    labels_feasible: int = 0
    labels_pruned_by_bound: int = 0
    labels_pruned_by_dominance: int = 0
    labels_pruned_by_local_skyline: int = 0
    labels_after_dominance: int = 0
    reachability_probes: int = 0
    solver_time: float = 0.0
    solver_nodes: float = 0.0


@dataclass(frozen=True)
class SolverPricingOptions:
    use_cuts: bool = False


class SolverPricingEngine:
    """Solve every fixed deck/compartment pricing subproblem with Gurobi."""

    def __init__(
        self,
        max_units_per_type: int = Config.max_units_per_compartment,
        max_columns_per_subproblem: int = 1,
        use_cuts: bool = False,
        verbose: bool = False,
    ) -> None:
        self.max_units_per_type = int(max_units_per_type)
        self.max_columns_per_subproblem = 1
        self.options = SolverPricingOptions(use_cuts=use_cuts)
        self.wagon_capacity_cut = Config.max_units_per_compartment
        self.verbose = verbose
        self.stats = SolverPricingStats()
        self._col_counter = 0
        self._seen_signatures: Set[Tuple[str, str, Tuple[int, ...]]] = set()

    def generate_columns(self, solution: MasterLPSolution, master: MasterProblem) -> List[PatternColumn]:
        self.stats = SolverPricingStats()
        modes = ["h-h", "h-m", "m-h", "m-m"]
        car_heights = {i: float(master.car_info.iloc[i - 1]["height"]) for i in master.I}
        new_columns: List[PatternColumn] = []

        for p in modes:
            gamma_p = solution.dual_gamma.get(p, 0.0)
            eta = solution.dual_eta if self.options.use_cuts else 0.0
            upper_duals = DualValues(
                alpha=solution.dual_alpha,
                beta=solution.dual_beta,
                gamma=2.0 * (gamma_p + solution.dual_kappa + eta),
                branch_a=solution.dual_branch_a,
                branch_q=solution.dual_branch_q,
            )
            lower_duals = DualValues(
                alpha=solution.dual_alpha,
                beta=solution.dual_beta,
                gamma=-2.0 * gamma_p,
                branch_a=solution.dual_branch_a,
                branch_q=solution.dual_branch_q,
            )
            for compartment, duals, length_limit in [
                ("upper", upper_duals, Config.top_len),
                ("lower", lower_duals, Config.bottom_len),
            ]:
                spec = LayerSpec(
                    layer_id=f"{compartment}_{p}",
                    car_types=master.I,
                    car_lengths=master.length,
                    car_heights=car_heights,
                    layer_length_limit=length_limit,
                    shape_params={"compartment": compartment, "deck": p},
                    max_quantity_by_type={
                        i: min(self.max_units_per_type, int(master.U[i]))
                        for i in master.I
                    },
                )
                forbidden = [
                    signature
                    for comp, deck, signature in self._seen_signatures
                    if comp == compartment and deck == p
                ]
                for _ in range(self.max_columns_per_subproblem):
                    t0 = time.time()
                    result = solve_layer_pricing_mip(
                        spec,
                        duals,
                        forbidden_signatures=forbidden,
                        sr_cut_duals=solution.dual_sigma if self.options.use_cuts else None,
                        max_total_by_type=master.U,
                    )
                    self.stats.solver_time += time.time() - t0
                    if result is None:
                        break
                    quantities, reduced_cost, node_count = result
                    self.stats.solver_nodes += node_count
                    signature = tuple(int(quantities.get(i, 0)) for i in master.I)
                    forbidden.append(signature)
                    if reduced_cost >= -1e-5:
                        break
                    full_signature = (compartment, p, signature)
                    if full_signature in self._seen_signatures:
                        continue
                    self._seen_signatures.add(full_signature)
                    self._col_counter += 1
                    new_columns.append(
                        PatternColumn(
                            column_id=f"solver_{self._col_counter}",
                            q=quantities,
                            cost=-sum(master.length[i] * quantities.get(i, 0) for i in master.I),
                            compartment=compartment,
                            deck_mode=p,
                            metadata={"source": "solver_pricing", "rc": reduced_cost},
                        )
                    )

        new_columns.sort(key=lambda column: float(column.metadata["rc"]))
        return new_columns


def solve_layer_pricing_mip(
    layer: LayerSpec,
    duals: DualValues,
    forbidden_signatures: List[Tuple[int, ...]] | None = None,
    sr_cut_duals: Dict[Tuple[int, int, int], float] | None = None,
    max_total_by_type: Dict[int, int] | None = None,
) -> Tuple[Dict[int, int], float, float] | None:
    forbidden_signatures = forbidden_signatures or []
    sr_cut_duals = sr_cut_duals or {}
    max_total_by_type = max_total_by_type or {}
    compartment = str(layer.shape_params.get("compartment", "lower"))
    deck = str(layer.shape_params.get("deck", "h-h"))
    mode_left, mode_right = deck.split("-")

    segments = feasibility_check._get_segments_for_layer(layer.car_heights)
    central = segments[compartment]["central"]
    blocks = segments[compartment]["blocks"]
    n_blocks = len(blocks)

    limit_left = [central["h_m"] if mode_left == "m" else central["h_h"]]
    limit_right = [central["h_m"] if mode_right == "m" else central["h_h"]]
    for block in blocks:
        limit_left.append(block["h_m"] if mode_left == "m" else block["h_h"])
        limit_right.append(block["h_m"] if mode_right == "m" else block["h_h"])
    lengths = [float(central["len"])] + [float(block["len"]) for block in blocks]

    region_names = ["central"]
    region_names.extend(f"left_{idx}" for idx in range(1, n_blocks + 1))
    region_names.extend(f"right_{idx}" for idx in range(1, n_blocks + 1))

    model = gp.Model("layer_solver_pricing")
    model.Params.OutputFlag = 0
    Config.apply_gurobi_params(model)

    N = Config.max_units_per_compartment
    Delta = 400.0
    I = list(layer.car_types)
    x = model.addVars(I, region_names, vtype=gp.GRB.INTEGER, lb=0, ub=N, name="x")
    a = model.addVars(I, vtype=gp.GRB.BINARY, name="a")
    q = {i: gp.quicksum(x[i, h] for h in region_names) for i in I}

    for forbid_idx, signature in enumerate(forbidden_signatures):
        y_pos = model.addVars(I, vtype=gp.GRB.BINARY, name=f"forbid_pos[{forbid_idx}]")
        y_neg = model.addVars(I, vtype=gp.GRB.BINARY, name=f"forbid_neg[{forbid_idx}]")
        big_m = N + 1
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

    for i in I:
        max_q = min(N, int(layer.max_quantity_by_type.get(i, N)))
        model.addConstr(q[i] <= max_q, name=f"type_upper[{i}]")
        model.addConstr(q[i] <= N * a[i], name=f"activate_upper[{i}]")
        model.addConstr(q[i] >= a[i], name=f"activate_lower[{i}]")

    model.addConstr(gp.quicksum(x[i, h] for i in I for h in region_names) <= N, name="layer_quantity")

    actual_limit_central = min(limit_left[0], limit_right[0])
    for i in I:
        if layer.car_heights[i] > actual_limit_central:
            model.addConstr(x[i, "central"] == 0, name=f"height[{i},central]")
        for idx in range(1, n_blocks + 1):
            if layer.car_heights[i] > limit_left[idx]:
                model.addConstr(x[i, f"left_{idx}"] == 0, name=f"height[{i},left_{idx}]")
            if layer.car_heights[i] > limit_right[idx]:
                model.addConstr(x[i, f"right_{idx}"] == 0, name=f"height[{i},right_{idx}]")

    for left_count in range(n_blocks + 1):
        for right_count in range(n_blocks + 1):
            if left_count == n_blocks and right_count == n_blocks:
                mod = -Delta
            elif left_count == n_blocks or right_count == n_blocks:
                mod = 0.0
            else:
                mod = Delta
            cap = lengths[0] + sum(lengths[1:left_count + 1]) + sum(lengths[1:right_count + 1]) + mod
            interval_regions = ["central"]
            interval_regions.extend(f"left_{idx}" for idx in range(1, left_count + 1))
            interval_regions.extend(f"right_{idx}" for idx in range(1, right_count + 1))
            model.addConstr(
                gp.quicksum(x[i, h] * (layer.car_lengths[i] + Delta) for i in I for h in interval_regions) <= cap
            )

    sr_terms: List[Tuple[float, gp.Var]] = []
    for cut_idx, (subset, sigma) in enumerate(sr_cut_duals.items()):
        if abs(float(sigma)) <= 1e-12:
            continue
        high_vars = []
        for i in subset:
            high = model.addVar(vtype=gp.GRB.BINARY, name=f"sr_high[{cut_idx},{i}]")
            threshold = int(max_total_by_type.get(i, 0) // 2 + 1)
            max_q = min(N, int(layer.max_quantity_by_type.get(i, N)))
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

    rc = gp.LinExpr(-duals.gamma / 2.0)
    for i in I:
        coef = (
            layer.car_lengths[i]
            + duals.alpha.get(i, 0.0)
            + duals.beta.get(i, 0.0)
            + duals.branch_q.get(i, 0.0)
        )
        rc -= coef * q[i]
        rc -= duals.branch_a.get(i, 0.0) * a[i]
    for sigma, coeff in sr_terms:
        rc -= sigma * coeff
    model.setObjective(rc, gp.GRB.MINIMIZE)
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        return None
    quantities = {i: int(round(q[i].getValue())) for i in I}
    return quantities, float(model.ObjVal), float(model.NodeCount)
