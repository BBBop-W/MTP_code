import sys
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple, Iterator, Optional, Set
import itertools
from math import inf
from dataclasses import dataclass
import pandas as pd
import gurobipy as gp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config

_SEGMENTS_CACHE = {}
GLOBAL_NUM_SPLITS = 1
GLOBAL_INDEP_MODE = True

def _get_segments_for_compartment(car_heights: Dict[int, float]):
    unique_heights = tuple(sorted(set(car_heights.values())))
    cache_key = (unique_heights, GLOBAL_NUM_SPLITS, GLOBAL_INDEP_MODE)
    if cache_key in _SEGMENTS_CACHE:
        return _SEGMENTS_CACHE[cache_key]
        
    df = pd.DataFrame({"height": list(unique_heights)})
    from src.utility.dynamic_segmentation import get_model_segments
    seg = get_model_segments(df, num_splits=GLOBAL_NUM_SPLITS, independent_mode_split=GLOBAL_INDEP_MODE)
    _SEGMENTS_CACHE[cache_key] = seg
    return seg

@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    best_length: float
    reachable_types: Optional[Set[int]] = None


def check_compartment_exact(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float],
    segments: dict = None
) -> float | None:
    """
    Exact interval-resource DFS for a single compartment.

    The implementation enumerates the admissible side/block placements induced
    by the dynamic segmentation and checks all nested interval capacities.
    """
    cars = []
    for cid, q in quantities.items():
        for _ in range(q):
            cars.append(cid)
            
    if not cars:
        return 0.0
    if len(cars) > Config.max_units_per_compartment:
        return None

    # SORTING: Tallest cars first. This naturally forces the algorithm to build from the center outwards!
    cars.sort(key=lambda c: car_heights[c], reverse=True)

    if "-" in deck_mode:
        mode_left = deck_mode.split("-")[0]
        mode_right = deck_mode.split("-")[1]
    else:
        m = "h" if deck_mode == "horizontal" else "m"
        mode_left = m
        mode_right = m

    pi_left = 1 if mode_left == "m" else 0
    pi_right = 1 if mode_right == "m" else 0

    delta = 400.0
    total_length = sum(car_lengths[c] for c in cars)
    
    if segments is None:
        segments = _get_segments_for_compartment(car_heights)
        
    central = segments[compartment]["central"]
    blocks = segments[compartment]["blocks"]
    N_blocks = len(blocks)
    
    limit_left = [central["h_m"] if pi_left else central["h_h"]]
    limit_right = [central["h_m"] if pi_right else central["h_h"]]
    
    for b in blocks:
        limit_left.append(b["h_m"] if pi_left else b["h_h"])
        limit_right.append(b["h_m"] if pi_right else b["h_h"])
        
    actual_limit_central = min(limit_left[0], limit_right[0])
    L_array = [central["len"]] + [b["len"] for b in blocks]
    
    car_choices = []
    for c in cars:
        h = car_heights[c]
        
        max_l = -1
        for idx in range(N_blocks, 0, -1):
            if h <= limit_left[idx]:
                max_l = idx
                break
        if max_l == -1 and h <= actual_limit_central:
            max_l = 0
            
        max_r = -1
        for idx in range(N_blocks, 0, -1):
            if h <= limit_right[idx]:
                max_r = idx
                break
        if max_r == -1 and h <= actual_limit_central:
            max_r = 0
                
        if max_l == -1 and max_r == -1:
            return None
            
        choices = []
        if max_l == 0 and max_r == 0:
            choices.append(("central", 0))
        else:
            # We only append the outermost valid block. Mathematical interval capacity makes this sufficient.
            if max_l >= 0:
                choices.append(("left", max_l))
            if max_r >= 0:
                choices.append(("right", max_r))
                
        choices = list(set(choices))
        car_choices.append(choices)
        
    intervals = []
    for l in range(N_blocks + 1):
        for r in range(N_blocks + 1):
            if l == N_blocks and r == N_blocks:
                mod = -delta
            elif l == N_blocks or r == N_blocks:
                mod = 0
            else:
                mod = delta
                
            cap = L_array[0] + sum(L_array[1:l+1]) + sum(L_array[1:r+1]) + mod
            intervals.append((l, r, cap))
            
    # Convert interval checks into a fast matrix mapping for DFS
    car_choice_hits = []
    for i, c_len in enumerate([car_lengths[c] + delta for c in cars]):
        choice_hits = []
        for side, b_idx in car_choices[i]:
            hits = []
            for j, (l_int, r_int, cap) in enumerate(intervals):
                inside = False
                if side == "central":
                    inside = True
                elif side == "left" and b_idx <= l_int:
                    inside = True
                elif side == "right" and b_idx <= r_int:
                    inside = True
                if inside:
                    hits.append(j)
            choice_hits.append((c_len, hits))
        car_choice_hits.append(choice_hits)
        
    caps = [cap for l, r, cap in intervals]
    
    # True DFS Backtracking
    def dfs(car_idx, current_usage):
        if car_idx == len(cars):
            return True
            
        for c_len, hits in car_choice_hits[car_idx]:
            valid = True
            for j in hits:
                if current_usage[j] + c_len > caps[j] + 1e-5:
                    valid = False
                    break
                    
            if valid:
                for j in hits:
                    current_usage[j] += c_len
                if dfs(car_idx + 1, current_usage):
                    return True
                for j in hits:
                    current_usage[j] -= c_len
                    
        return False

    if dfs(0, [0.0] * len(intervals)):
        return total_length
    return None

@dataclass(frozen=True)
class PlacementChoice:
    side: str
    block_idx: int
    hits: Tuple[int, ...]


@dataclass(frozen=True)
class CompartmentResourceModel:
    capacities: Tuple[float, ...]
    intervals: Tuple[Tuple[int, int, float], ...]
    choices_by_type: Dict[int, Tuple[PlacementChoice, ...]]
    delta: float = 400.0


def _split_deck_mode(deck_mode: str) -> Tuple[str, str]:
    if "-" in deck_mode:
        mode_left, mode_right = deck_mode.split("-")
        return mode_left, mode_right
    mode = "h" if deck_mode == "horizontal" else "m"
    return mode, mode


def _keep_interval_for_profile(l_idx: int, r_idx: int, n_blocks: int, interval_profile: str) -> bool:
    if interval_profile == "full":
        return True
    if interval_profile == "fans_diag":
        return l_idx == n_blocks or r_idx == n_blocks or l_idx == r_idx
    raise ValueError(f"Unknown interval_profile: {interval_profile}")


def build_compartment_resource_model(compartment_spec: 'CompartmentSpec', interval_profile: str = "full") -> CompartmentResourceModel:
    """Build the interval-resource model used by residual-profile labels.

    This mirrors the interval constraints used in ``check_compartment_gurobi`` and
    keeps every height-feasible central/left/right component choice. Safe
    dominance-based reductions are applied later by the label generator, so the
    exact generator can use this model as the complete baseline.
    """

    mode = compartment_spec.shape_params.get("deck", "h-h")
    compartment = compartment_spec.shape_params.get("compartment", "lower")
    mode_left, mode_right = _split_deck_mode(mode)
    pi_left = 1 if mode_left == "m" else 0
    pi_right = 1 if mode_right == "m" else 0

    segments = _get_segments_for_compartment(compartment_spec.car_heights)
    central = segments[compartment]["central"]
    blocks = segments[compartment]["blocks"]
    n_blocks = len(blocks)

    limit_left = [central["h_m"] if pi_left else central["h_h"]]
    limit_right = [central["h_m"] if pi_right else central["h_h"]]
    for block in blocks:
        limit_left.append(block["h_m"] if pi_left else block["h_h"])
        limit_right.append(block["h_m"] if pi_right else block["h_h"])

    actual_limit_central = min(limit_left[0], limit_right[0])
    lengths = [central["len"]] + [block["len"] for block in blocks]
    delta = 400.0

    intervals: List[Tuple[int, int, float]] = []
    for l_idx in range(n_blocks + 1):
        for r_idx in range(n_blocks + 1):
            if l_idx == n_blocks and r_idx == n_blocks:
                mod = -delta
            elif l_idx == n_blocks or r_idx == n_blocks:
                mod = 0.0
            else:
                mod = delta

            cap = lengths[0] + sum(lengths[1:l_idx + 1]) + sum(lengths[1:r_idx + 1]) + mod
            if _keep_interval_for_profile(l_idx, r_idx, n_blocks, interval_profile):
                intervals.append((l_idx, r_idx, cap))

    def hits_for(side: str, block_idx: int) -> Tuple[int, ...]:
        hits: List[int] = []
        for idx, (l_int, r_int, _cap) in enumerate(intervals):
            inside = False
            if side == "central":
                inside = True
            elif side == "left" and block_idx <= l_int:
                inside = True
            elif side == "right" and block_idx <= r_int:
                inside = True
            if inside:
                hits.append(idx)
        return tuple(hits)

    choices_by_type: Dict[int, Tuple[PlacementChoice, ...]] = {}
    for car_type in compartment_spec.car_types:
        height = compartment_spec.car_heights[car_type]

        choices: List[PlacementChoice] = []
        if height <= actual_limit_central:
            choices.append(PlacementChoice("central", 0, hits_for("central", 0)))
        for idx in range(1, n_blocks + 1):
            if height <= limit_left[idx]:
                choices.append(PlacementChoice("left", idx, hits_for("left", idx)))
            if height <= limit_right[idx]:
                choices.append(PlacementChoice("right", idx, hits_for("right", idx)))

        dedup: Dict[Tuple[int, ...], PlacementChoice] = {}
        for choice in choices:
            dedup.setdefault(choice.hits, choice)
        choices_by_type[car_type] = tuple(dedup.values())

    return CompartmentResourceModel(
        capacities=tuple(cap for _l, _r, cap in intervals),
        intervals=tuple(intervals),
        choices_by_type=choices_by_type,
        delta=delta,
    )


class ExactFeasibilityEvaluator:
    def __init__(self, compute_reachable_types: bool = True):
        self.compute_reachable_types = compute_reachable_types
        self.accumulated_time = 0.0
        self.reachability_probes = 0
        self._cache = {}

    def evaluate(self, compartment_spec, quantities: Dict[int, int]) -> FeasibilityResult:
        import time
        t0 = time.time()
        
        mode = compartment_spec.shape_params.get("deck", "h-h")
        compartment = compartment_spec.shape_params.get("compartment", "lower")
        
        clean_q = {i: q for i, q in quantities.items() if q > 0}
        
        cache_key = (compartment, mode, tuple(sorted(clean_q.items())), self.compute_reachable_types)
        if cache_key in self._cache:
            self.accumulated_time += (time.time() - t0)
            return self._cache[cache_key]
        
        res = check_compartment_exact(
            compartment=compartment,
            deck_mode=mode,
            quantities=clean_q,
            car_lengths=compartment_spec.car_lengths,
            car_heights=compartment_spec.car_heights
        )
        
        if res is not None:
            reachable = set()
            if self.compute_reachable_types:
                for t in compartment_spec.car_types:
                    max_q = int(compartment_spec.max_quantity_by_type.get(t, Config.max_units_per_compartment))
                    if clean_q.get(t, 0) >= max_q:
                        continue
                    probe = dict(clean_q)
                    probe[t] = probe.get(t, 0) + 1

                    probe_key = (compartment, mode, tuple(sorted(probe.items())), self.compute_reachable_types)
                    if probe_key in self._cache:
                        if self._cache[probe_key].feasible:
                            reachable.add(t)
                        continue

                    self.reachability_probes += 1
                    probe_res = check_compartment_exact(
                        compartment=compartment,
                        deck_mode=mode,
                        quantities=probe,
                        car_lengths=compartment_spec.car_lengths,
                        car_heights=compartment_spec.car_heights
                    )
                    if probe_res is not None:
                        reachable.add(t)
                reachable_types = reachable
            else:
                reachable_types = None
                    
            result = FeasibilityResult(feasible=True, best_length=res, reachable_types=reachable_types)
            self._cache[cache_key] = result
            self.accumulated_time += (time.time() - t0)
            return result
            
        result = FeasibilityResult(feasible=False, best_length=inf, reachable_types=set())
        self._cache[cache_key] = result
        self.accumulated_time += (time.time() - t0)
        return result

def check_compartment_gurobi(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float],
    segments: dict = None
) -> float | None:
    model = gp.Model("single_compartment_check")
    model.Params.OutputFlag = 0
    Config.apply_gurobi_params(model)
    
    I_list = [i for i, q in quantities.items() if q > 0]
    if not I_list:
        return 0.0

    if segments is None:
        segments = _get_segments_for_compartment(car_heights)
        
    central = segments[compartment]["central"]
    blocks = segments[compartment]["blocks"]
    N_blocks = len(blocks)
    
    if "-" in deck_mode:
        mode_left = deck_mode.split("-")[0]
        mode_right = deck_mode.split("-")[1]
    else:
        m = "h" if deck_mode == "horizontal" else "m"
        mode_left = m
        mode_right = m
        
    pi_left = 1 if mode_left == "m" else 0
    pi_right = 1 if mode_right == "m" else 0
    
    limit_left = [central["h_m"] if pi_left else central["h_h"]]
    limit_right = [central["h_m"] if pi_right else central["h_h"]]
    for b in blocks:
        limit_left.append(b["h_m"] if pi_left else b["h_h"])
        limit_right.append(b["h_m"] if pi_right else b["h_h"])
        
    L_array = [central["len"]] + [b["len"] for b in blocks]
    
    H_names = ["central"] + [f"left_{i}" for i in range(1, N_blocks+1)] + [f"right_{i}" for i in range(1, N_blocks+1)]
    
    x = model.addVars(I_list, H_names, vtype=gp.GRB.INTEGER, lb=0, name="x")
    model.addConstrs((gp.quicksum(x[i, h] for h in H_names) == quantities[i] for i in I_list), name="qty")
    
    delta = 400.0
    
    for i in I_list:
        h_car = car_heights[i]
        
        actual_limit_central = min(limit_left[0], limit_right[0])
        if h_car > actual_limit_central:
            model.addConstr(x[i, "central"] == 0)
            
        for b in range(1, N_blocks + 1):
            if h_car > limit_left[b]:
                model.addConstr(x[i, f"left_{b}"] == 0)
            if h_car > limit_right[b]:
                model.addConstr(x[i, f"right_{b}"] == 0)
                
    for l in range(N_blocks + 1):
        for r in range(N_blocks + 1):
            if l == N_blocks and r == N_blocks:
                mod = -delta
            elif l == N_blocks or r == N_blocks:
                mod = 0
            else:
                mod = delta
                
            cap = L_array[0] + sum(L_array[1:l+1]) + sum(L_array[1:r+1]) + mod
            
            blocks_in_interval = ["central"] + [f"left_{i}" for i in range(1, l+1)] + [f"right_{i}" for i in range(1, r+1)]
            
            model.addConstr(
                gp.quicksum(x[i, h] * (car_lengths[i] + delta) for i in I_list for h in blocks_in_interval) <= cap
            )
            
    model.setObjective(0.0, gp.GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        return sum(car_lengths[i] * quantities[i] for i in I_list)
    return None

if __name__ == "__main__":
    from src.utility.generate_instance import load_candidates
    from src.utility.dynamic_segmentation import get_model_segments
    
    print("--- Feasibility Check Testing (Exact DFS vs Gurobi) ---")
    excel_path = Path("data/raw data/尺寸整理.xlsx")
    if not excel_path.exists():
        print(f"Cannot find data at {excel_path}. Please verify path.")
        sys.exit(1)
        
    candidates = load_candidates(excel_path)
    
    car_lengths = {i: row["长"] for i, row in candidates.iterrows()}
    car_heights = {i: row["高"] for i, row in candidates.iterrows()}
    car_types = list(car_lengths.keys())
    all_heights = list(car_heights.values())
    
    tests = 1000
    rng = random.Random(42)
    
    print(f"Running {tests} random verification tests with DYNAMIC segmentations...")
    
    match_count = 0
    grb_time = 0.0
    exact_time = 0.0

    for case_id in range(tests):
        compartment = rng.choice(["upper", "lower"])
        deck_mode = rng.choice(["h-h", "h-m", "m-h", "m-m"])
        
        # Randomly generate different segmentation scenarios
        num_splits = rng.randint(1, 4)
        indep_mode = rng.choice([True, False])
        
        # Use a random subset of heights to create unpredictable segmentations
        sampled_heights = rng.sample(all_heights, k=rng.randint(2, 5))
        df_dummy = pd.DataFrame({"height": sampled_heights})
        segments_data = get_model_segments(df_dummy, num_splits=num_splits, independent_mode_split=indep_mode)
        
        quantities = {i: 0 for i in car_types}
        total_cars = rng.randint(6, 8)
        for _ in range(total_cars):
            quantities[rng.choice(car_types)] += 1
        quantities = {i: q for i, q in quantities.items() if q > 0}
            
        t0 = time.time()
        gurobi_res = check_compartment_gurobi(compartment, deck_mode, quantities, car_lengths, car_heights, segments_data)
        t1 = time.time()
        exact_res = check_compartment_exact(compartment, deck_mode, quantities, car_lengths, car_heights, segments_data)
        t2 = time.time()
        
        grb_time += (t1 - t0)
        exact_time += (t2 - t1)
        
        match = False
        if exact_res is None and gurobi_res is None:
            match = True
        elif exact_res is not None and gurobi_res is not None and abs(exact_res - gurobi_res) < 1e-5:
            match = True
            
        if match:
            match_count += 1
        else:
            print(f"\nTest {case_id} FAILED: {compartment} {deck_mode}, splits={num_splits}, indep={indep_mode}")
            print("Cars:")
            for cid, q in quantities.items():
                print(f"  ID {cid}: Qty={q}, Len={car_lengths[cid]}, Height={car_heights[cid]}")
            print(f"Gurobi: {gurobi_res}, exact DFS: {exact_res}")
            break

    print(f"\nMatch rate: {match_count}/{tests}")
    print(f"Gurobi total time: {grb_time:.4f} s")
    print(f"Exact DFS total time: {exact_time:.4f} s")
