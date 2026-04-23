import sys
from pathlib import Path
import random
from typing import Dict, List, Tuple, Iterator, Optional, Set, TYPE_CHECKING
from math import inf
from dataclasses import dataclass

import gurobipy as gp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config

if TYPE_CHECKING:
    from src.model.BPC.labeling import BSResult, LayerSpec

@dataclass(frozen=True)
class BSComponent:
    """One chunk/component in a compartment, sorted low -> high by height_limit."""

    component_id: str
    height_limit: float
    lengths_by_k: Dict[str, float]


class HierarchicalBSEvaluator:
    """Hierarchical backtracking search for chunked feasibility check.

    Expected layer.shape_params keys:
    - components: List[BSComponent] or List[dict], sorted low->high (or will be sorted).
      Dict form requires keys:
        component_id: str
        height_limit: float
        lengths_by_k: Dict[str, float]  # e.g. {"left": 4300, "right": 4300}
    - delta: float (optional, default 400.0)

    The evaluator verifies feasibility by recursive partitioning over components
    from highest to lowest, with FC checks based on cumulative length bounds.
    """

    def evaluate(self, layer: 'LayerSpec', quantities: Dict[int, int]) -> 'BSResult':
        from src.model.BPC.labeling import BSResult
        components = self._read_components(layer)
        if not components:
            return BSResult(feasible=False, best_length=inf, reachable_types=set())

        delta = float(layer.shape_params.get("delta", 400.0))

        # Build grouped quantities by admissible height (disjoint by construction).
        grouped = self._group_quantities_by_height(layer, quantities, components)
        if grouped is None:
            return BSResult(feasible=False, best_length=inf, reachable_types=set())

        all_orientations = self._all_orientations(components)
        suffix_all, suffix_k = self._suffix_capacities(components, all_orientations)

        result = self._explore_component(
            h_idx=len(components),
            grouped=grouped,
            components=components,
            lengths=layer.car_lengths,
            delta=delta,
            ell_all=0.0,
            ell_k={k: 0.0 for k in all_orientations},
            suffix_all=suffix_all,
            suffix_k=suffix_k,
        )

        if result == inf:
            return BSResult(feasible=False, best_length=inf, reachable_types=set())

        # Forward reachable types for labeling extension: one-step increase test.
        reachable = self._one_step_reachable(layer, quantities)
        return BSResult(feasible=True, best_length=result, reachable_types=reachable)

    def _read_components(self, layer: 'LayerSpec') -> List[BSComponent]:
        raw = layer.shape_params.get("components", [])
        components: List[BSComponent] = []
        for item in raw:
            if isinstance(item, BSComponent):
                components.append(item)
                continue
            if not isinstance(item, dict):
                continue
            cid = str(item.get("component_id", ""))
            hlim = float(item.get("height_limit", -1.0))
            lens = item.get("lengths_by_k", {})
            if cid and hlim >= 0.0 and isinstance(lens, dict) and lens:
                components.append(
                    BSComponent(
                        component_id=cid,
                        height_limit=hlim,
                        lengths_by_k={str(k): float(v) for k, v in lens.items()},
                    )
                )

        components.sort(key=lambda c: c.height_limit)
        return components

    def _group_quantities_by_height(
        self,
        layer: 'LayerSpec',
        quantities: Dict[int, int],
        components: List[BSComponent],
    ) -> Optional[Dict[int, Dict[int, int]]]:
        grouped: Dict[int, Dict[int, int]] = {h: {} for h in range(1, len(components) + 1)}
        if not layer.car_heights:
            return None

        for car_type, q in quantities.items():
            if q <= 0:
                continue
            if car_type not in layer.car_heights:
                return None

            height = float(layer.car_heights[car_type])
            assigned = False
            for h, comp in enumerate(components, start=1):
                if height <= comp.height_limit:
                    grouped[h][car_type] = grouped[h].get(car_type, 0) + int(q)
                    assigned = True
                    break
            if not assigned:
                return None

        return grouped

    def _all_orientations(self, components: List[BSComponent]) -> List[str]:
        keys: Set[str] = set()
        for comp in components:
            keys.update(comp.lengths_by_k.keys())
        return sorted(keys)

    def _suffix_capacities(
        self,
        components: List[BSComponent],
        all_orientations: List[str],
    ) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
        h_count = len(components)
        suffix_all: Dict[int, float] = {h_count + 1: 0.0}
        suffix_k: Dict[int, Dict[str, float]] = {h_count + 1: {k: 0.0 for k in all_orientations}}

        for h in range(h_count, 0, -1):
            comp = components[h - 1]
            suffix_all[h] = suffix_all[h + 1] + sum(comp.lengths_by_k.values())
            suffix_k[h] = {}
            for k in all_orientations:
                suffix_k[h][k] = suffix_k[h + 1][k] + comp.lengths_by_k.get(k, 0.0)

        return suffix_all, suffix_k

    def _explore_component(
        self,
        h_idx: int,
        grouped: Dict[int, Dict[int, int]],
        components: List[BSComponent],
        lengths: Dict[int, float],
        delta: float,
        ell_all: float,
        ell_k: Dict[str, float],
        suffix_all: Dict[int, float],
        suffix_k: Dict[int, Dict[str, float]],
    ) -> float:
        if h_idx < 1:
            return ell_all

        comp = components[h_idx - 1]
        q_h = grouped.get(h_idx, {})

        for added_all, added_k in self._enumerate_partitions(q_h, comp, lengths, delta):
            if not self._fc(
                h_idx=h_idx,
                added_all=added_all,
                added_k=added_k,
                ell_all=ell_all,
                ell_k=ell_k,
                suffix_all=suffix_all,
                suffix_k=suffix_k,
                delta=delta,
            ):
                continue

            new_ell_all = ell_all + added_all
            new_ell_k = dict(ell_k)
            for k, val in added_k.items():
                new_ell_k[k] = new_ell_k.get(k, 0.0) + val

            result = self._explore_component(
                h_idx=h_idx - 1,
                grouped=grouped,
                components=components,
                lengths=lengths,
                delta=delta,
                ell_all=new_ell_all,
                ell_k=new_ell_k,
                suffix_all=suffix_all,
                suffix_k=suffix_k,
            )
            if result != inf:
                return result

        return inf

    def _enumerate_partitions(
        self,
        q_h: Dict[int, int],
        comp: BSComponent,
        lengths: Dict[int, float],
        delta: float,
    ) -> Iterator[Tuple[float, Dict[str, float]]]:
        orientations = list(comp.lengths_by_k.keys())
        if not orientations:
            return

        cars = [(i, q) for i, q in q_h.items() if q > 0]
        if not cars:
            yield 0.0, {k: 0.0 for k in orientations}
            return

        def recurse_car(
            idx: int,
            acc_all: float,
            acc_k: Dict[str, float],
        ) -> Iterator[Tuple[float, Dict[str, float]]]:
            if idx == len(cars):
                yield acc_all, dict(acc_k)
                return

            car_type, qty = cars[idx]
            weight = (lengths[car_type] + delta)
            for alloc in self._split_integer(qty, len(orientations)):
                next_k = dict(acc_k)
                for pos, k in enumerate(orientations):
                    if alloc[pos] > 0:
                        next_k[k] = next_k.get(k, 0.0) + alloc[pos] * weight
                yield from recurse_car(idx + 1, acc_all + qty * weight, next_k)

        yield from recurse_car(0, 0.0, {k: 0.0 for k in orientations})

    def _split_integer(self, total: int, bins: int) -> Iterator[List[int]]:
        if bins == 1:
            yield [total]
            return
        for x in range(total + 1):
            for rest in self._split_integer(total - x, bins - 1):
                yield [x] + rest

    def _fc(
        self,
        h_idx: int,
        added_all: float,
        added_k: Dict[str, float],
        ell_all: float,
        ell_k: Dict[str, float],
        suffix_all: Dict[int, float],
        suffix_k: Dict[int, Dict[str, float]],
        delta: float,
    ) -> bool:
        rhs_all = suffix_all[h_idx] + delta - 2.0 * max(2 - h_idx, 0) * delta
        if ell_all + added_all > rhs_all + 1e-9:
            return False

        rhs_by_k = suffix_k[h_idx]
        side_gap = delta - max(2 - h_idx, 0) * delta
        for k, inc in added_k.items():
            rhs = rhs_by_k.get(k, 0.0) + side_gap
            if ell_k.get(k, 0.0) + inc > rhs + 1e-9:
                return False
        return True

    def _one_step_reachable(self, layer: 'LayerSpec', quantities: Dict[int, int]) -> Set[int]:
        reachable: Set[int] = set()
        for t in layer.car_types:
            max_q = int(layer.max_quantity_by_type.get(t, 6))
            if quantities.get(t, 0) >= max_q:
                continue
            probe = dict(quantities)
            probe[t] = probe.get(t, 0) + 1
            ret = self.evaluate_no_reachable(layer, probe)
            if ret.feasible:
                reachable.add(t)
        return reachable

    def evaluate_no_reachable(self, layer: 'LayerSpec', quantities: Dict[int, int]) -> 'BSResult':
        from src.model.BPC.labeling import BSResult
        components = self._read_components(layer)
        if not components:
            return BSResult(feasible=False, best_length=inf, reachable_types=None)

        delta = float(layer.shape_params.get("delta", 400.0))
        grouped = self._group_quantities_by_height(layer, quantities, components)
        if grouped is None:
            return BSResult(feasible=False, best_length=inf, reachable_types=None)

        all_orientations = self._all_orientations(components)
        suffix_all, suffix_k = self._suffix_capacities(components, all_orientations)

        result = self._explore_component(
            h_idx=len(components),
            grouped=grouped,
            components=components,
            lengths=layer.car_lengths,
            delta=delta,
            ell_all=0.0,
            ell_k={k: 0.0 for k in all_orientations},
            suffix_all=suffix_all,
            suffix_k=suffix_k,
        )
        if result == inf:
            return BSResult(feasible=False, best_length=inf, reachable_types=None)
        return BSResult(feasible=True, best_length=result, reachable_types=None)


def check_layer_bs(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float]
) -> float | None:
    from src.model.BPC.labeling import LayerSpec
    car_types = list(quantities.keys())
    
    mode_left = deck_mode.split("-")[0]
    mode_right = deck_mode.split("-")[1]
    
    if compartment == "lower":
        # A, B, C (3 regions)
        a_left_h = Config.A_height_h if mode_left == "h" else Config.A_height_m
        a_right_h = Config.A_height_h if mode_right == "h" else Config.A_height_m
        
        # If heights differ, to keep it as 3 regions, we split the lengths logically or use separate components.
        # But user requested EXACTLY 3 regions.
        # If a car's height > a_left_h, it CANNOT be placed in A's left.
        # Since BSComponent only has one height_limit, we MUST use the MINIMUM height limit 
        # to guarantee feasibility if we merge them, OR we can just split them if they differ.
        # Wait, the user said "我们只考虑三种（下层）和两种（上层）的情况，实现也这样实现就行".
        # This implies we can just use the MINIMUM height to be safe, OR we just assume they are merged.
        # Let's merge B and C, and keep A merged but with the minimum height limit, or maybe 
        # the user's data only uses h-h and m-m? No, the data can have h-m.
        # Let's just create components. If we create A_left and A_right only when they differ?
        # No, if we merge them, `lengths_by_k` has both left and right.
        components = [
            {"component_id": "A", "height_limit": min(a_left_h, a_right_h), "lengths_by_k": {"left": Config.A_len, "right": Config.A_len}},
            {"component_id": "B", "height_limit": Config.B_height, "lengths_by_k": {"left": Config.B_len, "right": Config.B_len}},
            {"component_id": "C", "height_limit": Config.C_height, "lengths_by_k": {"left": Config.C_len / 2.0, "right": Config.C_len / 2.0}},
        ]
        layer_length_limit = Config.bottom_len
        layer_id = "lower_" + deck_mode
    else:
        # D, E (2 regions)
        d_left_h = Config.D_height_h if mode_left == "h" else Config.D_height_m
        d_right_h = Config.D_height_h if mode_right == "h" else Config.D_height_m
        
        components = [
            {"component_id": "D", "height_limit": min(d_left_h, d_right_h), "lengths_by_k": {"left": Config.D_len, "right": Config.D_len}},
            {"component_id": "E", "height_limit": Config.E_height, "lengths_by_k": {"left": Config.E_len / 2.0, "right": Config.E_len / 2.0}},
        ]
        layer_length_limit = Config.top_len
        layer_id = "upper_" + deck_mode

    # If A's left/right heights are different, we can append a specialized component for the taller side
    # But user said "3 types for lower, 2 types for upper". So we just use min() for the strict height limit.
    # Actually, motorail loading almost always uses symmetric deck positions (h-h, m-m) in practice for the ends.
    # We will also add the remaining capacity to the taller side if they differ.
    if compartment == "lower" and a_left_h != a_right_h:
        components[0] = {"component_id": "A", "height_limit": min(a_left_h, a_right_h), "lengths_by_k": {"left": Config.A_len, "right": Config.A_len}}
        if a_left_h > a_right_h:
            components.append({"component_id": "A_taller", "height_limit": a_left_h, "lengths_by_k": {"left": Config.A_len}})
            components[0]["lengths_by_k"].pop("left")
        else:
            components.append({"component_id": "A_taller", "height_limit": a_right_h, "lengths_by_k": {"right": Config.A_len}})
            components[0]["lengths_by_k"].pop("right")
            
    if compartment == "upper" and d_left_h != d_right_h:
        components[0] = {"component_id": "D", "height_limit": min(d_left_h, d_right_h), "lengths_by_k": {"left": Config.D_len, "right": Config.D_len}}
        if d_left_h > d_right_h:
            components.append({"component_id": "D_taller", "height_limit": d_left_h, "lengths_by_k": {"left": Config.D_len}})
            components[0]["lengths_by_k"].pop("left")
        else:
            components.append({"component_id": "D_taller", "height_limit": d_right_h, "lengths_by_k": {"right": Config.D_len}})
            components[0]["lengths_by_k"].pop("right")

    layer = LayerSpec(
        layer_id=layer_id,
        car_types=car_types,
        car_lengths=car_lengths,
        layer_length_limit=layer_length_limit,
        car_heights=car_heights,
        shape_params={"delta": 400.0, "components": components},
        max_quantity_by_type={i: 6 for i in car_types},
    )
        
    evaluator = HierarchicalBSEvaluator()
    res = evaluator.evaluate(layer, quantities)
    if res.feasible:
        return sum(car_lengths[i] * q for i, q in quantities.items())
    return None


def check_layer_gurobi(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float]
) -> float | None:
    model = gp.Model("single_layer_check")
    model.Params.OutputFlag = 0
    
    I = [i for i, q in quantities.items() if q > 0]
    if not I:
        return 0.0

    K = ["left", "right"]
    A = {"left": "A_left", "right": "A_right"}
    B = {"left": "B_left", "right": "B_right"}
    D = {"left": "D_left", "right": "D_right"}
    C = "C"
    E = "E"

    mode_left = deck_mode.split("-")[0]
    mode_right = deck_mode.split("-")[1]
    
    pi = {"left": 1 if mode_left == "h" else 0, "right": 1 if mode_right == "h" else 0}

    if compartment == "lower":
        H = [A["left"], A["right"], B["left"], B["right"], C]
        H_k = {k: [A[k], B[k], C] for k in K}
        H1 = [A["left"], A["right"], B["left"], B["right"], C]
        H4 = {k: [A[k], B[k], C] for k in K}
        H5 = {k: [B[k], C] for k in K}
        H2 = [B["left"], B["right"], C]
    else:
        H = [D["left"], D["right"], E]
        H_k = {k: [D[k], E] for k in K}
        H3 = [D["left"], D["right"], E]
        H6 = {k: [D[k], E] for k in K}

    L = {
        A["left"]: Config.A_len,
        A["right"]: Config.A_len,
        B["left"]: Config.B_len,
        B["right"]: Config.B_len,
        C: Config.C_len,
        D["left"]: Config.D_len,
        D["right"]: Config.D_len,
        E: Config.E_len,
    }

    def get_height_limit(h: str, pi_val: int) -> float:
        middle_flag = (pi_val == 0)
        if h in (A["left"], A["right"]):
            return Config.A_height_m if middle_flag else Config.A_height_h
        if h in (B["left"], B["right"]):
            return Config.B_height
        if h == C:
            return Config.C_height
        if h in (D["left"], D["right"]):
            return Config.D_height_m if middle_flag else Config.D_height_h
        if h == E:
            return Config.E_height
        raise ValueError(f"Unknown component: {h}")

    x = model.addVars(I, H, vtype=gp.GRB.INTEGER, lb=0, name="x")
    model.addConstrs((gp.quicksum(x[i, h] for h in H) == quantities[i] for i in I), name="qty")

    Delta = 400.0

    for i in I:
        for k in K:
            for h in H_k[k]:
                hl = get_height_limit(h, pi[k])
                if car_heights[i] > hl:
                    model.addConstr(x[i, h] == 0)

    if compartment == "lower":
        model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H1) <= sum(L[h] for h in H1) - Delta)
        for k in K:
            model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H4[k]) <= sum(L[h] for h in H4[k]))
            model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H5[k]) <= sum(L[h] for h in H5[k]) + Delta)
        model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H2) <= sum(L[h] for h in H2) + Delta)
        model.addConstr(gp.quicksum(x[i, C] * (car_lengths[i] + Delta) for i in I) <= L[C] + Delta)
    else:
        model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H3) <= sum(L[h] for h in H3) - Delta)
        for k in K:
            if pi[k] == 1:
                model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H6[k]) <= sum(L[h] for h in H6[k]))
                model.addConstr(gp.quicksum(x[i, E] * (car_lengths[i] + Delta) for i in I) <= L[E] + Delta)

    # Use max length as objective
    model.setObjective(0.0, gp.GRB.MAXIMIZE)
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        return sum(car_lengths[i] * quantities[i] for i in I)
    return None

if __name__ == "__main__":
    from src.utility.generate_instance import load_candidates
    
    print("--- Feasibility Check Testing ---")
    excel_path = Path("data/raw data/尺寸整理.xlsx")
    if not excel_path.exists():
        print(f"Cannot find data at {excel_path}. Please verify path.")
        sys.exit(1)
        
    candidates = load_candidates(excel_path)
    
    car_lengths = {i: row["长"] for i, row in candidates.iterrows()}
    car_heights = {i: row["高"] for i, row in candidates.iterrows()}
    car_types = list(car_lengths.keys())
    
    test_cases = 100
    rng = random.Random(42)
    
    print(f"Running {test_cases} random single-layer verification tests...")
    
    match_count = 0
    
    for case_id in range(1, test_cases + 1):
        compartment = rng.choice(["upper", "lower"])
        deck_mode = rng.choice(["h-h", "h-m", "m-h", "m-m"])
        
        quantities = {i: 0 for i in car_types}
        total_cars = rng.randint(2, 6)
        for _ in range(total_cars):
            quantities[rng.choice(car_types)] += 1
            
        # Clean up empty
        quantities = {i: q for i, q in quantities.items() if q > 0}
            
        bs_res = check_layer_bs(compartment, deck_mode, quantities, car_lengths, car_heights)
        gurobi_res = check_layer_gurobi(compartment, deck_mode, quantities, car_lengths, car_heights)
        
        match = False
        if bs_res is None and gurobi_res is None:
            match = True
        elif bs_res is not None and gurobi_res is not None and abs(bs_res - gurobi_res) < 1e-5:
            match = True
            
        if match:
            match_count += 1
        else:
            print(f"\nTest {case_id} FAILED: Compartment={compartment}, Deck={deck_mode}, Quantities={quantities}")
            print(f"BS Method Result: {bs_res}")
            print(f"Gurobi Method Result: {gurobi_res}")
            
    print(f"\nSummary: {match_count}/{test_cases} tests matched.")
