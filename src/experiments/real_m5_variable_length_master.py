from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

import gurobipy as gp
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.experiments.variable_length_gr_hybrid import (
    LabelingLimitExceeded,
    MethodConfig,
    PlacementChoice,
    QuantityKey,
    VariableLengthInstance,
    compare_profile_maps,
    solve_instance,
)
from src.model.BPC_compartment.BBtree import normalize_car_table
from src.utility.dynamic_segmentation import get_model_segments


P_MODES = ("h-h", "h-m", "m-h", "m-m")
DELTA = 400.0
BIG_M = 1e7


@dataclass(frozen=True)
class RealVariant:
    name: str
    adjusted_loads: Dict[Tuple[int, str, int], float]
    good_types: Tuple[int, ...]


@dataclass(frozen=True)
class CompartmentPattern:
    compartment: str
    deck: str
    quantities: QuantityKey
    value: float


def _split_deck(deck: str) -> Tuple[str, str]:
    left, right = deck.split("-")
    return left, right


def _component_id(compartment: str, side: str, block: int) -> str:
    if side == "central":
        return f"{compartment}_central"
    return f"{compartment}_{side}_{block}"


def _hits_for(intervals: Sequence[Tuple[int, int]], side: str, block: int) -> Tuple[int, ...]:
    hits: List[int] = []
    for idx, (left_idx, right_idx) in enumerate(intervals):
        if side == "central":
            hits.append(idx)
        elif side == "left" and block <= left_idx:
            hits.append(idx)
        elif side == "right" and block <= right_idx:
            hits.append(idx)
    return tuple(hits)


def _layer_intervals(compartment: str, deck: str, n_blocks: int) -> List[Tuple[int, int]]:
    left_mode, right_mode = _split_deck(deck)
    pi_left = 1 if left_mode == "m" else 0
    pi_right = 1 if right_mode == "m" else 0
    intervals: List[Tuple[int, int]] = []
    for left_idx in range(n_blocks + 1):
        for right_idx in range(n_blocks + 1):
            if compartment == "upper":
                enforce_left = (left_idx == n_blocks) or (pi_left == 1)
                enforce_right = (right_idx == n_blocks) or (pi_right == 1)
                if not (enforce_left and enforce_right):
                    continue
            intervals.append((left_idx, right_idx))
    return intervals


def _capacity(lengths: Sequence[float], left_idx: int, right_idx: int) -> float:
    n_blocks = len(lengths) - 1
    if left_idx == n_blocks and right_idx == n_blocks:
        mod = -DELTA
    elif left_idx == n_blocks or right_idx == n_blocks:
        mod = 0.0
    else:
        mod = DELTA
    return lengths[0] + sum(lengths[1:left_idx + 1]) + sum(lengths[1:right_idx + 1]) + mod


def _height_limit_lists(segments: dict, compartment: str, deck: str) -> Tuple[List[float], List[float]]:
    central = segments[compartment]["central"]
    blocks = segments[compartment]["blocks"]
    left_mode, right_mode = _split_deck(deck)
    left_limits = [central["h_m"] if left_mode == "m" else central["h_h"]]
    right_limits = [central["h_m"] if right_mode == "m" else central["h_h"]]
    for block in blocks:
        left_limits.append(block["h_m"] if left_mode == "m" else block["h_h"])
        right_limits.append(block["h_m"] if right_mode == "m" else block["h_h"])
    return left_limits, right_limits


def _choice_dominates(a: PlacementChoice, b: PlacementChoice, eps: float = 1e-9) -> bool:
    return set(a.hits).issubset(set(b.hits)) and a.load <= b.load + eps


def _gr_choices(choices: Sequence[PlacementChoice]) -> Tuple[PlacementChoice, ...]:
    kept: List[PlacementChoice] = []
    for side in ("left", "right"):
        side_choices = [choice for choice in choices if choice.side == side]
        if side_choices:
            kept.append(max(side_choices, key=lambda choice: choice.block))
    if not kept:
        kept.extend(choice for choice in choices if choice.side == "central")
    return tuple(kept)


def _is_type_gr_safe(choices: Sequence[PlacementChoice]) -> bool:
    gr_choices = _gr_choices(choices)
    return all(any(_choice_dominates(gr, choice) for gr in gr_choices) for choice in choices)


def make_adjusted_loads(car_info: pd.DataFrame, segments: dict, variant: str, seed: int) -> RealVariant:
    rng = random.Random(seed)
    adjusted: Dict[Tuple[int, str, int], float] = {}
    good_types: List[int] = []
    n_blocks = len(segments["lower"]["blocks"])
    car_types = list(range(1, len(car_info) + 1))
    for car_type in car_types:
        nominal = float(car_info.iloc[car_type - 1]["length"])
        if variant == "monotone":
            is_good = True
        elif variant == "hybrid":
            is_good = car_type <= max(1, len(car_types) // 2)
        elif variant == "ex":
            is_good = False
        else:
            raise ValueError(f"unknown variant: {variant}")
        if is_good:
            good_types.append(car_type)
        for compartment in ("lower", "upper"):
            adjusted[car_type, _component_id(compartment, "central", 0), 0] = nominal + DELTA + (520.0 if is_good else rng.uniform(-250.0, 180.0))
            for side in ("left", "right"):
                side_noise = rng.uniform(-35.0, 35.0)
                for block in range(1, n_blocks + 1):
                    component = _component_id(compartment, side, block)
                    if is_good:
                        outward_bonus = 180.0 * (n_blocks - block)
                        load = nominal + DELTA + outward_bonus + side_noise
                    else:
                        load = nominal + DELTA + rng.uniform(-320.0, 360.0)
                        if block == 1:
                            load -= rng.uniform(160.0, 420.0)
                        if block == n_blocks:
                            load += rng.uniform(120.0, 360.0)
                    adjusted[car_type, component, block] = max(nominal + 80.0, load)
    return RealVariant(name=variant, adjusted_loads=adjusted, good_types=tuple(good_types))


def build_layer_instance(
    car_info: pd.DataFrame,
    segments: dict,
    variant: RealVariant,
    compartment: str,
    deck: str,
    max_quantity_by_type: Dict[int, int],
) -> VariableLengthInstance:
    central = segments[compartment]["central"]
    blocks = segments[compartment]["blocks"]
    n_blocks = len(blocks)
    intervals = _layer_intervals(compartment, deck, n_blocks)
    lengths = [float(central["len"])] + [float(block["len"]) for block in blocks]
    capacities = tuple(_capacity(lengths, left_idx, right_idx) for left_idx, right_idx in intervals)
    left_limits, right_limits = _height_limit_lists(segments, compartment, deck)
    central_limit = min(left_limits[0], right_limits[0])
    choices_by_type: Dict[int, Tuple[PlacementChoice, ...]] = {}
    layer_max_quantity_by_type: Dict[int, int] = {}
    actual_good: List[int] = []
    full_interval = (n_blocks, n_blocks)
    full_capacity = capacities[intervals.index(full_interval)] if full_interval in intervals else min(capacities)
    for car_type in range(1, len(car_info) + 1):
        height = float(car_info.iloc[car_type - 1]["height"])
        choices: List[PlacementChoice] = []
        if height <= central_limit:
            component = _component_id(compartment, "central", 0)
            choices.append(PlacementChoice("central", 0, _hits_for(intervals, "central", 0), variant.adjusted_loads[car_type, component, 0]))
        for block in range(1, n_blocks + 1):
            if height <= left_limits[block]:
                component = _component_id(compartment, "left", block)
                choices.append(PlacementChoice("left", block, _hits_for(intervals, "left", block), variant.adjusted_loads[car_type, component, block]))
            if height <= right_limits[block]:
                component = _component_id(compartment, "right", block)
                choices.append(PlacementChoice("right", block, _hits_for(intervals, "right", block), variant.adjusted_loads[car_type, component, block]))
        choices_by_type[car_type] = tuple(choices)
        if choices:
            min_load = min(choice.load for choice in choices)
            layer_max_quantity_by_type[car_type] = min(int(max_quantity_by_type[car_type]), int((full_capacity + 1e-9) // min_load))
        else:
            layer_max_quantity_by_type[car_type] = 0
        if choices and _is_type_gr_safe(choices):
            actual_good.append(car_type)
    return VariableLengthInstance(
        name=f"m5c5_{variant.name}_{compartment}_{deck}",
        capacities=capacities,
        intervals=tuple(intervals),
        choices_by_type=choices_by_type,
        max_quantity_by_type=layer_max_quantity_by_type,
        good_types=tuple(actual_good),
    )


def _pattern_value(key: QuantityKey, nominal: Dict[int, float]) -> float:
    return sum(nominal[car_type] * qty for car_type, qty in key)


def generate_layer_patterns(
    car_info: pd.DataFrame,
    segments: dict,
    variant: RealVariant,
    method: MethodConfig,
    max_quantity_by_type: Dict[int, int],
    reference_maps: Dict[Tuple[str, str], Dict[QuantityKey, List[Tuple[float, ...]]]] | None = None,
    generation_time_limit: float | None = None,
) -> Tuple[List[CompartmentPattern], Dict[str, object], Dict[Tuple[str, str], Dict[QuantityKey, List[Tuple[float, ...]]]]]:
    nominal = {i: float(car_info.iloc[i - 1]["length"]) for i in range(1, len(car_info) + 1)}
    patterns: List[CompartmentPattern] = []
    profile_maps: Dict[Tuple[str, str], Dict[QuantityKey, List[Tuple[float, ...]]]] = {}
    start = time.perf_counter()
    totals = {
        "generation_status": "ok",
        "layer_count": 0,
        "quantity_keys_total": 0,
        "profiles_total": 0,
        "placement_attempts": 0,
        "raw_feasible_children": 0,
        "labels_created": 0,
        "local_pruned": 0,
        "quantity_skyline_pruned": 0,
        "dominance_pruned": 0,
        "generation_time": 0.0,
        "profile_cover_failures": "",
        "profile_uncovered_total": "",
        "profile_missing_keys_total": "",
    }
    cover_failures = 0
    uncovered_total = 0
    missing_keys_total = 0
    checked_layers = 0
    for compartment in ("upper", "lower"):
        for deck in P_MODES:
            if generation_time_limit is not None:
                remaining = generation_time_limit - (time.perf_counter() - start)
                if remaining <= 0:
                    totals["generation_status"] = "time_limit"
                    totals["generation_time"] = time.perf_counter() - start
                    return list({(p.compartment, p.deck, p.quantities): p for p in patterns}.values()), totals, profile_maps
                layer_method = MethodConfig(
                    method.name,
                    method.mode,
                    method.use_dominance,
                    method.use_local_skyline,
                    time_limit_sec=remaining,
                    label_limit=method.label_limit,
                )
            else:
                layer_method = method
            layer = build_layer_instance(car_info, segments, variant, compartment, deck, max_quantity_by_type)
            try:
                profile_map, stats, _ = solve_instance(layer, layer_method)
            except LabelingLimitExceeded as exc:
                totals["generation_status"] = exc.reason
                totals["generation_time"] = time.perf_counter() - start
                totals["placement_attempts"] += exc.stats.placement_assignments_attempted
                totals["raw_feasible_children"] += exc.stats.placement_assignments_feasible_raw
                totals["labels_created"] += exc.stats.labels_created
                totals["local_pruned"] += exc.stats.labels_pruned_by_local_skyline
                totals["quantity_skyline_pruned"] += exc.stats.labels_pruned_by_quantity_skyline
                totals["dominance_pruned"] += exc.stats.labels_pruned_by_dominance
                return list({(p.compartment, p.deck, p.quantities): p for p in patterns}.values()), totals, profile_maps
            totals["generation_time"] = time.perf_counter() - start
            profile_maps[(compartment, deck)] = profile_map
            if reference_maps is not None and (compartment, deck) in reference_maps:
                cmp = compare_profile_maps(reference_maps[(compartment, deck)], profile_map)
                checked_layers += 1
                if not cmp["profile_coverage"]:
                    cover_failures += 1
                uncovered_total += int(cmp["uncovered_profiles"])
                missing_keys_total += int(cmp["missing_keys"])
            for key in profile_map:
                if key:
                    patterns.append(CompartmentPattern(compartment, deck, key, _pattern_value(key, nominal)))
            totals["layer_count"] += 1
            totals["quantity_keys_total"] += len(profile_map)
            totals["profiles_total"] += sum(len(v) for v in profile_map.values())
            totals["placement_attempts"] += stats.placement_assignments_attempted
            totals["raw_feasible_children"] += stats.placement_assignments_feasible_raw
            totals["labels_created"] += stats.labels_created
            totals["local_pruned"] += stats.labels_pruned_by_local_skyline
            totals["quantity_skyline_pruned"] += stats.labels_pruned_by_quantity_skyline
            totals["dominance_pruned"] += stats.labels_pruned_by_dominance
    if checked_layers:
        totals["profile_cover_failures"] = cover_failures
        totals["profile_uncovered_total"] = uncovered_total
        totals["profile_missing_keys_total"] = missing_keys_total
    unique = {}
    for pattern in patterns:
        unique[(pattern.compartment, pattern.deck, pattern.quantities)] = pattern
    return list(unique.values()), totals, profile_maps


def solve_pattern_master(
    car_info: pd.DataFrame,
    carriage_num: int,
    patterns: List[CompartmentPattern],
    mandatory: Dict[int, int],
    optional: Dict[int, int],
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int | None = None,
) -> Dict[str, object]:
    model = gp.Model("real_m5_pattern_master")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    y = model.addVars(range(len(patterns)), vtype=gp.GRB.INTEGER, lb=0, ub=carriage_num, name="theta")
    for car_type in range(1, len(car_info) + 1):
        total = gp.quicksum(dict(patterns[p].quantities).get(car_type, 0) * y[p] for p in range(len(patterns)))
        model.addConstr(total >= mandatory[car_type], name=f"mandatory[{car_type}]")
        model.addConstr(total <= mandatory[car_type] + optional[car_type], name=f"optional[{car_type}]")
    for deck in P_MODES:
        model.addConstr(
            gp.quicksum(y[p] for p, pattern in enumerate(patterns) if pattern.compartment == "upper" and pattern.deck == deck) ==
            gp.quicksum(y[p] for p, pattern in enumerate(patterns) if pattern.compartment == "lower" and pattern.deck == deck),
            name=f"map[{deck}]",
        )
    model.addConstr(gp.quicksum(y[p] for p, pattern in enumerate(patterns) if pattern.compartment == "upper") <= carriage_num, name="wagons")
    model.setObjective(gp.quicksum(patterns[p].value * y[p] for p in range(len(patterns))), gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    model.optimize()
    return {
        "status": int(model.Status),
        "objective": float(model.ObjVal) if model.SolCount else None,
        "bound": float(model.ObjBound) if model.SolCount or model.Status in {gp.GRB.TIME_LIMIT, gp.GRB.OPTIMAL} else None,
        "mip_gap": float(model.MIPGap) if model.SolCount and model.IsMIP else None,
        "solve_time": time.perf_counter() - t0,
        "runtime": float(model.Runtime),
        "node_count": float(model.NodeCount),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
    }


def _status_name(status: int | str) -> str:
    if isinstance(status, str):
        return status
    names = {
        gp.GRB.LOADED: "LOADED",
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.CUTOFF: "CUTOFF",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(int(status), str(status))


def _component_data(segments: dict) -> Tuple[List[str], Dict[str, str], Dict[str, int], Dict[str, str], Dict[str, float]]:
    components: List[str] = []
    comp_layer: Dict[str, str] = {}
    comp_block: Dict[str, int] = {}
    comp_side: Dict[str, str] = {}
    comp_len: Dict[str, float] = {}
    for compartment in ("lower", "upper"):
        central = _component_id(compartment, "central", 0)
        components.append(central)
        comp_layer[central] = compartment
        comp_block[central] = 0
        comp_side[central] = "central"
        comp_len[central] = float(segments[compartment]["central"]["len"])
        for block_idx, block in enumerate(segments[compartment]["blocks"], start=1):
            for side in ("left", "right"):
                component = _component_id(compartment, side, block_idx)
                components.append(component)
                comp_layer[component] = compartment
                comp_block[component] = block_idx
                comp_side[component] = side
                comp_len[component] = float(block["len"])
    return components, comp_layer, comp_block, comp_side, comp_len


def _height_feasible(car_height: float, segments: dict, component: str, deck: str) -> bool:
    parts = component.split("_")
    compartment = parts[0]
    side = parts[1]
    block = 0 if side == "central" else int(parts[2])
    left_limits, right_limits = _height_limit_lists(segments, compartment, deck)
    if side == "central":
        return car_height <= min(left_limits[0], right_limits[0])
    if side == "left":
        return car_height <= left_limits[block]
    return car_height <= right_limits[block]


def solve_compact(
    car_info: pd.DataFrame,
    segments: dict,
    variant: RealVariant,
    carriage_num: int,
    mandatory: Dict[int, int],
    optional: Dict[int, int],
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int | None = None,
) -> Dict[str, object]:
    model = gp.Model("real_m5_variable_length_compact")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    car_types = list(range(1, len(car_info) + 1))
    wagons = list(range(carriage_num))
    components, comp_layer, comp_block, comp_side, comp_len = _component_data(segments)
    nominal = {i: float(car_info.iloc[i - 1]["length"]) for i in car_types}
    heights = {i: float(car_info.iloc[i - 1]["height"]) for i in car_types}
    z = model.addVars(wagons, P_MODES, vtype=gp.GRB.BINARY, name="z")
    x = model.addVars(car_types, wagons, components, vtype=gp.GRB.INTEGER, lb=0, ub=10, name="x")
    for w in wagons:
        model.addConstr(gp.quicksum(z[w, deck] for deck in P_MODES) == 1, name=f"deck[{w}]")
    for car_type in car_types:
        total = gp.quicksum(x[car_type, w, component] for w in wagons for component in components)
        model.addConstr(total >= mandatory[car_type], name=f"mandatory[{car_type}]")
        model.addConstr(total <= mandatory[car_type] + optional[car_type], name=f"optional[{car_type}]")
    for car_type in car_types:
        for w in wagons:
            for component in components:
                model.addConstr(
                    x[car_type, w, component] <= 10 * gp.quicksum(
                        int(_height_feasible(heights[car_type], segments, component, deck)) * z[w, deck]
                        for deck in P_MODES
                    ),
                    name=f"eta[{car_type},{w},{component}]",
                )
    for w in wagons:
        for compartment in ("lower", "upper"):
            layer_components = [component for component in components if comp_layer[component] == compartment]
            model.addConstr(gp.quicksum(x[car_type, w, component] for car_type in car_types for component in layer_components) <= 10, name=f"count[{w},{compartment}]")
            n_blocks = len(segments[compartment]["blocks"])
            lengths = [float(segments[compartment]["central"]["len"])] + [float(block["len"]) for block in segments[compartment]["blocks"]]
            for deck in P_MODES:
                for left_idx, right_idx in _layer_intervals(compartment, deck, n_blocks):
                    cap = _capacity(lengths, left_idx, right_idx)
                    interval_components = [_component_id(compartment, "central", 0)]
                    interval_components += [_component_id(compartment, "left", idx) for idx in range(1, left_idx + 1)]
                    interval_components += [_component_id(compartment, "right", idx) for idx in range(1, right_idx + 1)]
                    expr = gp.quicksum(
                        x[car_type, w, component] * variant.adjusted_loads[car_type, component, comp_block[component]]
                        for car_type in car_types
                        for component in interval_components
                    )
                    model.addConstr(expr <= cap + BIG_M * (1 - z[w, deck]), name=f"len[{w},{compartment},{deck},{left_idx},{right_idx}]")
    model.setObjective(gp.quicksum(nominal[car_type] * x[car_type, w, component] for car_type in car_types for w in wagons for component in components), gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    model.optimize()
    return {
        "status": int(model.Status),
        "objective": float(model.ObjVal) if model.SolCount else None,
        "bound": float(model.ObjBound) if model.SolCount or model.Status in {gp.GRB.TIME_LIMIT, gp.GRB.OPTIMAL} else None,
        "mip_gap": float(model.MIPGap) if model.SolCount and model.IsMIP else None,
        "solve_time": time.perf_counter() - t0,
        "runtime": float(model.Runtime),
        "node_count": float(model.NodeCount),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
    }


def run_variant(
    car_info: pd.DataFrame,
    segments: dict,
    variant: RealVariant,
    carriage_num: int,
    mandatory: Dict[int, int],
    optional: Dict[int, int],
    max_quantity_by_type: Dict[int, int],
    methods: Sequence[MethodConfig],
    generation_time_limit: float | None,
    solver_time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    compact = solve_compact(car_info, segments, variant, carriage_num, mandatory, optional, solver_time_limit, mip_gap, threads)
    compact_obj = compact["objective"]
    rows.append({
        "variant": variant.name,
        "method": "compact",
        "status": compact["status"],
        "status_name": _status_name(compact["status"]),
        "generation_status": "",
        "objective": compact_obj if compact_obj is not None else "",
        "objective_bound": compact["bound"] if compact["bound"] is not None else "",
        "mip_gap": compact["mip_gap"] if compact["mip_gap"] is not None else "",
        "compact_objective": compact_obj if compact_obj is not None else "",
        "objective_gap_to_compact": 0.0 if compact_obj is not None else "",
        "relative_gap_to_compact": 0.0 if compact_obj else "",
        "aligned_with_compact": True,
        "patterns": "",
        "quantity_keys_total": "",
        "profiles_total": "",
        "profile_cover_failures": "",
        "profile_uncovered_total": "",
        "profile_missing_keys_total": "",
        "generation_time": 0.0,
        "master_solve_time": compact["solve_time"],
        "total_time": compact["solve_time"],
        "placement_attempts": "",
        "raw_feasible_children": "",
        "labels_created": "",
        "local_pruned": "",
        "quantity_skyline_pruned": "",
        "dominance_pruned": "",
        "declared_gr_safe_types": ",".join(str(v) for v in variant.good_types) if variant.good_types else "none",
        "speedup_vs_ex": "",
        "attempt_reduction_vs_ex": "",
        "label_reduction_vs_ex": "",
        "num_vars": compact["num_vars"],
        "num_constrs": compact["num_constrs"],
        "node_count": compact["node_count"],
    })
    reference_maps: Dict[Tuple[str, str], Dict[QuantityKey, List[Tuple[float, ...]]]] | None = None
    ex_baseline: Dict[str, object] | None = None
    for method in methods:
        patterns, stats, profile_maps = generate_layer_patterns(
            car_info,
            segments,
            variant,
            method,
            max_quantity_by_type,
            reference_maps=reference_maps if method.mode != "ex" else None,
            generation_time_limit=generation_time_limit,
        )
        generation_ok = stats["generation_status"] == "ok"
        if method.mode == "ex" and generation_ok:
            reference_maps = profile_maps
            stats["profile_cover_failures"] = 0
            stats["profile_uncovered_total"] = 0
            stats["profile_missing_keys_total"] = 0
        if generation_ok:
            master = solve_pattern_master(car_info, carriage_num, patterns, mandatory, optional, solver_time_limit, mip_gap, threads)
        else:
            master = {
                "status": stats["generation_status"],
                "objective": None,
                "bound": None,
                "mip_gap": None,
                "solve_time": 0.0,
                "runtime": 0.0,
                "node_count": 0.0,
                "num_vars": 0,
                "num_constrs": 0,
            }
        objective = master["objective"]
        aligned = objective is not None and compact_obj is not None and abs(objective - compact_obj) <= 1e-6
        abs_gap = abs(objective - compact_obj) if objective is not None and compact_obj is not None else ""
        rel_gap = abs_gap / abs(compact_obj) if isinstance(abs_gap, float) and compact_obj else ""
        total_time = float(stats["generation_time"]) + float(master["solve_time"])
        row = {
            "variant": variant.name,
            "method": method.name,
            "status": master["status"],
            "status_name": _status_name(master["status"]),
            "generation_status": stats["generation_status"],
            "objective": objective if objective is not None else "",
            "objective_bound": master["bound"] if master["bound"] is not None else "",
            "mip_gap": master["mip_gap"] if master["mip_gap"] is not None else "",
            "compact_objective": compact_obj if compact_obj is not None else "",
            "objective_gap_to_compact": abs_gap,
            "relative_gap_to_compact": rel_gap,
            "aligned_with_compact": aligned,
            "patterns": len(patterns),
            "quantity_keys_total": stats["quantity_keys_total"],
            "profiles_total": stats["profiles_total"],
            "profile_cover_failures": stats["profile_cover_failures"],
            "profile_uncovered_total": stats["profile_uncovered_total"],
            "profile_missing_keys_total": stats["profile_missing_keys_total"],
            "generation_time": stats["generation_time"],
            "master_solve_time": master["solve_time"],
            "total_time": total_time,
            "placement_attempts": stats["placement_attempts"],
            "raw_feasible_children": stats["raw_feasible_children"],
            "labels_created": stats["labels_created"],
            "local_pruned": stats["local_pruned"],
            "quantity_skyline_pruned": stats["quantity_skyline_pruned"],
            "dominance_pruned": stats["dominance_pruned"],
            "declared_gr_safe_types": ",".join(str(v) for v in variant.good_types) if variant.good_types else "none",
            "speedup_vs_ex": "",
            "attempt_reduction_vs_ex": "",
            "label_reduction_vs_ex": "",
            "num_vars": master["num_vars"],
            "num_constrs": master["num_constrs"],
            "node_count": master["node_count"],
        }
        if method.name == "EX" and generation_ok:
            ex_baseline = row
        elif ex_baseline is not None and generation_ok:
            ex_time = float(ex_baseline["total_time"])
            ex_attempts = float(ex_baseline["placement_attempts"])
            ex_labels = float(ex_baseline["labels_created"])
            row["speedup_vs_ex"] = ex_time / total_time if total_time > 0 else ""
            row["attempt_reduction_vs_ex"] = 1.0 - float(row["placement_attempts"]) / ex_attempts if ex_attempts > 0 else ""
            row["label_reduction_vs_ex"] = 1.0 - float(row["labels_created"]) / ex_labels if ex_labels > 0 else ""
        rows.append({
            **row,
        })
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=PROJECT_ROOT / "data" / "Instance" / "m5c5")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--carriages", type=int, default=None)
    parser.add_argument("--max-q", type=int, default=None)
    parser.add_argument("--max-units-per-layer", type=int, default=10)
    parser.add_argument("--generation-time-limit", type=float, default=120.0)
    parser.add_argument("--solver-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--methods", nargs="*", default=["EX", "GR", "HYB"])
    parser.add_argument("--variants", nargs="*", default=["monotone", "hybrid", "ex"])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "result" / "real_m5_variable_length" / "summary_original.csv")
    args = parser.parse_args()

    car_info = normalize_car_table(args.instance / "cars.csv")
    carriage_num = int(pd.read_csv(args.instance / "carriage.csv")["carriage_num"].iloc[0])
    if args.carriages is not None:
        carriage_num = int(args.carriages)
    segments = get_model_segments(car_info, num_splits=args.splits, independent_mode_split=False)
    mandatory = {i: int(car_info.iloc[i - 1]["mandatory"]) for i in range(1, len(car_info) + 1)}
    optional = {i: int(car_info.iloc[i - 1]["optional"]) for i in range(1, len(car_info) + 1)}
    max_quantity_by_type = {
        i: min(
            int(args.max_units_per_layer),
            int(mandatory[i] + optional[i]),
            int(args.max_q) if args.max_q is not None else int(args.max_units_per_layer),
        )
        for i in range(1, len(car_info) + 1)
    }
    method_map = {
        "EX": MethodConfig("EX", "ex", use_dominance=False),
        "EX_dom": MethodConfig("EX_dom", "ex", use_dominance=True),
        "GR": MethodConfig("GR", "gr", use_dominance=False),
        "GR_dom": MethodConfig("GR_dom", "gr", use_dominance=True),
        "HYB": MethodConfig("HYB", "hyb", use_dominance=False),
        "HYB_dom": MethodConfig("HYB_dom", "hyb", use_dominance=True),
    }
    methods = [method_map[name] for name in args.methods]
    rows: List[Dict[str, object]] = []
    print(
        "original demand: "
        f"carriages={carriage_num}, mandatory={sum(mandatory.values())}, "
        f"upper={sum(mandatory[i] + optional[i] for i in mandatory)}, "
        f"max_q_by_type={max_quantity_by_type}",
        flush=True,
    )
    for offset, variant_name in enumerate(args.variants):
        variant = make_adjusted_loads(car_info, segments, variant_name, args.seed + offset)
        variant_rows = run_variant(
            car_info,
            segments,
            variant,
            carriage_num,
            mandatory,
            optional,
            max_quantity_by_type,
            methods,
            args.generation_time_limit,
            args.solver_time_limit,
            args.mip_gap,
            args.threads,
        )
        rows.extend(variant_rows)
        for row in variant_rows:
            print(
                f"{row['variant']:8s} {row['method']:7s} status={row['status_name']} "
                f"obj={row['objective']} gap={row['mip_gap']} align={row['aligned_with_compact']} "
                f"patterns={row['patterns']} total={row['total_time']} "
                f"labels={row['labels_created']} attempts={row['placement_attempts']}",
                flush=True,
            )
    write_csv(rows, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
