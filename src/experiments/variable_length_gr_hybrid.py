from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import sys

import gurobipy as gp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.config import config as Config


Residual = Tuple[float, ...]
QuantityKey = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class PlacementChoice:
    side: str
    block: int
    hits: Tuple[int, ...]
    load: float

    @property
    def name(self) -> str:
        if self.side == "central":
            return "C"
        return f"{self.side[0].upper()}{self.block}"


@dataclass(frozen=True)
class VariableLengthInstance:
    name: str
    capacities: Residual
    intervals: Tuple[Tuple[int, int], ...]
    choices_by_type: Dict[int, Tuple[PlacementChoice, ...]]
    max_quantity_by_type: Dict[int, int]
    good_types: Tuple[int, ...]


@dataclass(frozen=True)
class WholeModelCase:
    name: str
    instance: VariableLengthInstance
    num_compartments: int
    mandatory: Dict[int, int]
    optional: Dict[int, int]
    nominal_value: Dict[int, float]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    mode: str
    use_dominance: bool
    use_local_skyline: bool = True
    time_limit_sec: float | None = None
    label_limit: int | None = None


class LabelingLimitExceeded(TimeoutError):
    def __init__(self, reason: str, elapsed: float, stats: "LabelStats") -> None:
        super().__init__(reason)
        self.reason = reason
        self.elapsed = elapsed
        self.stats = stats


@dataclass(frozen=True)
class ResidualLabel:
    stage: int
    reduced_cost: float
    quantities: Tuple[int, ...]
    residual: Residual


@dataclass
class LabelStats:
    placement_assignments_attempted: int = 0
    placement_assignments_feasible_raw: int = 0
    labels_created: int = 0
    labels_pruned_by_local_skyline: int = 0
    labels_pruned_by_quantity_skyline: int = 0
    labels_pruned_by_dominance: int = 0
    labels_after_dominance: int = 0
    stage_attempted: List[int] | None = None
    stage_raw_feasible: List[int] | None = None
    stage_created: List[int] | None = None
    stage_kept: List[int] | None = None

    def __post_init__(self) -> None:
        self.stage_attempted = []
        self.stage_raw_feasible = []
        self.stage_created = []
        self.stage_kept = []


def _choice_count_vectors(total: int, num_choices: int) -> Iterator[Tuple[int, ...]]:
    if num_choices == 0:
        if total == 0:
            yield ()
        return
    if num_choices == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _choice_count_vectors(total - first, num_choices - 1):
            yield (first,) + rest


def _build_topology(n_blocks: int = 3) -> Tuple[Tuple[Tuple[int, int], ...], Residual]:
    central_len = 6200.0
    side_len = [2200.0, 1800.0, 1500.0]
    delta = 400.0

    intervals: List[Tuple[int, int]] = []
    capacities: List[float] = []
    for l_idx in range(n_blocks + 1):
        for r_idx in range(n_blocks + 1):
            intervals.append((l_idx, r_idx))
            if l_idx == n_blocks and r_idx == n_blocks:
                mod = -delta
            elif l_idx == n_blocks or r_idx == n_blocks:
                mod = 0.0
            else:
                mod = delta
            capacities.append(central_len + sum(side_len[:l_idx]) + sum(side_len[:r_idx]) + mod)
    return tuple(intervals), tuple(capacities)


def _hits_for(intervals: Sequence[Tuple[int, int]], side: str, block: int) -> Tuple[int, ...]:
    hits: List[int] = []
    for idx, (l_idx, r_idx) in enumerate(intervals):
        if side == "central":
            hits.append(idx)
        elif side == "left" and block <= l_idx:
            hits.append(idx)
        elif side == "right" and block <= r_idx:
            hits.append(idx)
    return tuple(hits)


def _all_choices_from_loads(
    intervals: Sequence[Tuple[int, int]],
    central: float,
    left: Sequence[float],
    right: Sequence[float],
) -> Tuple[PlacementChoice, ...]:
    choices = [PlacementChoice("central", 0, _hits_for(intervals, "central", 0), central)]
    for block, load in enumerate(left, start=1):
        choices.append(PlacementChoice("left", block, _hits_for(intervals, "left", block), load))
    for block, load in enumerate(right, start=1):
        choices.append(PlacementChoice("right", block, _hits_for(intervals, "right", block), load))
    return tuple(choices)


def _monotone_good_loads(rng: random.Random, type_idx: int, n_blocks: int) -> Tuple[float, List[float], List[float]]:
    base = 1700.0 + 180.0 * type_idx + rng.uniform(-60.0, 60.0)
    central = base + 680.0
    left_offset = rng.uniform(-40.0, 40.0)
    right_offset = rng.uniform(-40.0, 40.0)
    left = [base + 210.0 * (n_blocks - block) + left_offset for block in range(1, n_blocks + 1)]
    right = [base + 210.0 * (n_blocks - block) + right_offset for block in range(1, n_blocks + 1)]
    return central, left, right


def _bad_nonmonotone_loads(rng: random.Random, type_idx: int, n_blocks: int) -> Tuple[float, List[float], List[float]]:
    base = 1650.0 + 170.0 * type_idx + rng.uniform(-80.0, 80.0)
    central = base - rng.uniform(120.0, 260.0)
    left = [base + rng.uniform(-250.0, 250.0) for _ in range(n_blocks)]
    right = [base + rng.uniform(-250.0, 250.0) for _ in range(n_blocks)]
    left[0] = min(left[0], left[-1] - rng.uniform(250.0, 650.0))
    right[0] = min(right[0], right[-1] - rng.uniform(250.0, 650.0))
    return central, left, right


def make_instance(kind: str, seed: int, n_types: int = 4, n_blocks: int = 3, max_q: int = 2) -> VariableLengthInstance:
    rng = random.Random(seed)
    intervals, capacities = _build_topology(n_blocks)
    choices_by_type: Dict[int, Tuple[PlacementChoice, ...]] = {}
    declared_good: List[int] = []
    for car_type in range(1, n_types + 1):
        if kind == "monotone":
            central, left, right = _monotone_good_loads(rng, car_type, n_blocks)
        elif kind == "mixed":
            if car_type <= max(1, n_types // 2):
                central, left, right = _monotone_good_loads(rng, car_type, n_blocks)
            else:
                central, left, right = _bad_nonmonotone_loads(rng, car_type, n_blocks)
        elif kind == "nonuniform":
            central, left, right = _bad_nonmonotone_loads(rng, car_type, n_blocks)
        else:
            raise ValueError(f"unknown kind: {kind}")
        choices_by_type[car_type] = _all_choices_from_loads(intervals, central, left, right)

    for car_type, choices in choices_by_type.items():
        if _is_gr_safe(choices, _gr_choices(choices)):
            declared_good.append(car_type)

    return VariableLengthInstance(
        name=f"{kind}_seed{seed}",
        capacities=capacities,
        intervals=intervals,
        choices_by_type=choices_by_type,
        max_quantity_by_type={car_type: max_q for car_type in choices_by_type},
        good_types=tuple(declared_good),
    )


def _dominates_choice(a: PlacementChoice, b: PlacementChoice, eps: float = 1e-9) -> bool:
    return set(a.hits).issubset(set(b.hits)) and a.load <= b.load + eps


def _gr_choices(choices: Sequence[PlacementChoice]) -> Tuple[PlacementChoice, ...]:
    kept: List[PlacementChoice] = []
    for side in ("left", "right"):
        side_choices = [choice for choice in choices if choice.side == side]
        if side_choices:
            kept.append(max(side_choices, key=lambda choice: choice.block))
    if not kept:
        central = [choice for choice in choices if choice.side == "central"]
        kept.extend(central)
    return tuple(kept)


def _is_gr_safe(all_choices: Sequence[PlacementChoice], gr_choices: Sequence[PlacementChoice]) -> bool:
    return all(any(_dominates_choice(gr, choice) for gr in gr_choices) for choice in all_choices)


def _choices_for_method(instance: VariableLengthInstance, method: MethodConfig) -> Dict[int, Tuple[PlacementChoice, ...]]:
    if method.mode == "ex":
        return instance.choices_by_type
    if method.mode == "gr":
        return {car_type: _gr_choices(choices) for car_type, choices in instance.choices_by_type.items()}
    if method.mode == "hyb":
        good = set(instance.good_types)
        return {
            car_type: _gr_choices(choices) if car_type in good else choices
            for car_type, choices in instance.choices_by_type.items()
        }
    raise ValueError(f"unknown method mode: {method.mode}")


def _apply_choice(residual: Residual, choice: PlacementChoice, count: int, eps: float = 1e-9) -> Residual | None:
    updated = list(residual)
    amount = count * choice.load
    for idx in choice.hits:
        updated[idx] -= amount
        if updated[idx] < -eps:
            return None
    return tuple(updated)


def _residual_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b)) and any(x > y + eps for x, y in zip(a, b))


def _residual_weakly_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b))


def _local_skyline(residuals: Iterable[Residual], stats: LabelStats, eps: float = 1e-9) -> List[Residual]:
    kept: List[Residual] = []
    seen = set()
    for residual in residuals:
        key = tuple(round(v, 8) for v in residual)
        if key in seen:
            stats.labels_pruned_by_local_skyline += 1
            continue
        seen.add(key)
        dominated = False
        remove_idx: List[int] = []
        for idx, old in enumerate(kept):
            if _residual_weakly_dominates(old, residual, eps):
                dominated = True
                break
            if _residual_dominates(residual, old, eps):
                remove_idx.append(idx)
        if dominated:
            stats.labels_pruned_by_local_skyline += 1
            continue
        if remove_idx:
            stats.labels_pruned_by_local_skyline += len(remove_idx)
            remove_set = set(remove_idx)
            kept = [value for idx, value in enumerate(kept) if idx not in remove_set]
        kept.append(residual)
    return kept


def _placement_residuals(
    residual: Residual,
    choices: Sequence[PlacementChoice],
    quantity: int,
    stats: LabelStats,
    deadline: float | None = None,
    start_time: float | None = None,
) -> Iterator[Residual]:
    if quantity == 0:
        stats.placement_assignments_attempted += 1
        stats.placement_assignments_feasible_raw += 1
        yield residual
        return
    for counts in _choice_count_vectors(quantity, len(choices)):
        if deadline is not None and time.perf_counter() > deadline:
            elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
            raise LabelingLimitExceeded("labeling_time_limit", elapsed, stats)
        stats.placement_assignments_attempted += 1
        updated = residual
        feasible = True
        for count, choice in zip(counts, choices):
            if count == 0:
                continue
            updated = _apply_choice(updated, choice, count)
            if updated is None:
                feasible = False
                break
        if feasible:
            stats.placement_assignments_feasible_raw += 1
            yield updated


def _label_dominates(a: ResidualLabel, b: ResidualLabel, eps: float = 1e-9) -> bool:
    if a.reduced_cost > b.reduced_cost + eps:
        return False
    if not _residual_weakly_dominates(a.residual, b.residual, eps):
        return False
    return a.reduced_cost < b.reduced_cost - eps or _residual_dominates(a.residual, b.residual, eps)


def _apply_label_dominance(labels: List[ResidualLabel], stats: LabelStats) -> List[ResidualLabel]:
    kept: List[ResidualLabel] = []
    for cand in sorted(labels, key=lambda label: label.reduced_cost):
        if any(_label_dominates(old, cand) for old in kept):
            stats.labels_pruned_by_dominance += 1
            continue
        kept.append(cand)
    stats.labels_after_dominance += len(kept)
    return kept


def _apply_quantity_skyline(labels: List[ResidualLabel], stats: LabelStats) -> List[ResidualLabel]:
    grouped: Dict[Tuple[int, ...], List[ResidualLabel]] = {}
    for label in labels:
        grouped.setdefault(label.quantities, []).append(label)
    kept: List[ResidualLabel] = []
    for group in grouped.values():
        local_kept: List[ResidualLabel] = []
        for cand in group:
            dominated = False
            remove_idx: List[int] = []
            for idx, old in enumerate(local_kept):
                if _residual_weakly_dominates(old.residual, cand.residual):
                    dominated = True
                    break
                if _residual_dominates(cand.residual, old.residual):
                    remove_idx.append(idx)
            if dominated:
                stats.labels_pruned_by_quantity_skyline += 1
                continue
            if remove_idx:
                stats.labels_pruned_by_quantity_skyline += len(remove_idx)
                remove_set = set(remove_idx)
                local_kept = [label for idx, label in enumerate(local_kept) if idx not in remove_set]
            local_kept.append(cand)
        kept.extend(local_kept)
    return kept


def _quantity_key(car_types: Sequence[int], quantities: Sequence[int]) -> QuantityKey:
    return tuple((car_type, qty) for car_type, qty in zip(car_types, quantities) if qty > 0)


def _skyline_profiles(profiles: Iterable[Residual]) -> List[Residual]:
    stats = LabelStats()
    return _local_skyline(profiles, stats)


def solve_instance(instance: VariableLengthInstance, method: MethodConfig) -> Tuple[Dict[QuantityKey, List[Residual]], LabelStats, float]:
    choices_by_type = _choices_for_method(instance, method)
    car_types = sorted(instance.choices_by_type)
    root = ResidualLabel(
        stage=0,
        reduced_cost=0.0,
        quantities=tuple(0 for _ in car_types),
        residual=instance.capacities,
    )
    labels = [root]
    stats = LabelStats()
    t0 = time.perf_counter()
    deadline = t0 + method.time_limit_sec if method.time_limit_sec is not None else None
    for stage, car_type in enumerate(car_types, start=1):
        next_labels: List[ResidualLabel] = []
        attempted_before = stats.placement_assignments_attempted
        raw_before = stats.placement_assignments_feasible_raw
        max_q = instance.max_quantity_by_type[car_type]
        choices = choices_by_type[car_type]
        for label in labels:
            if deadline is not None and time.perf_counter() > deadline:
                raise LabelingLimitExceeded("labeling_time_limit", time.perf_counter() - t0, stats)
            for qty in range(max_q + 1):
                quantities = list(label.quantities)
                quantities[stage - 1] = qty
                residuals = list(_placement_residuals(label.residual, choices, qty, stats, deadline, t0))
                if method.use_local_skyline:
                    residuals = _local_skyline(residuals, stats)
                for residual in residuals:
                    next_labels.append(
                        ResidualLabel(
                            stage=stage,
                            reduced_cost=label.reduced_cost - 1000.0 * qty,
                            quantities=tuple(quantities),
                            residual=residual,
                        )
                    )
                if method.label_limit is not None and len(next_labels) > method.label_limit:
                    raise LabelingLimitExceeded("label_limit", time.perf_counter() - t0, stats)
        stats.labels_created += len(next_labels)
        stats.stage_attempted.append(stats.placement_assignments_attempted - attempted_before)
        stats.stage_raw_feasible.append(stats.placement_assignments_feasible_raw - raw_before)
        stats.stage_created.append(len(next_labels))
        next_labels = _apply_quantity_skyline(next_labels, stats)
        if method.use_dominance:
            next_labels = _apply_label_dominance(next_labels, stats)
        stats.stage_kept.append(len(next_labels))
        labels = next_labels
        if not labels:
            break
    elapsed = time.perf_counter() - t0
    profile_map: Dict[QuantityKey, List[Residual]] = {}
    for label in labels:
        if label.stage != len(car_types):
            continue
        key = _quantity_key(car_types, label.quantities)
        if not key:
            continue
        profile_map.setdefault(key, []).append(label.residual)
    profile_map = {key: _skyline_profiles(values) for key, values in profile_map.items()}
    return profile_map, stats, elapsed


def compare_profile_maps(reference: Dict[QuantityKey, List[Residual]], candidate: Dict[QuantityKey, List[Residual]]) -> Dict[str, object]:
    missing_keys = set(reference) - set(candidate)
    extra_keys = set(candidate) - set(reference)
    uncovered = 0
    for key, ref_profiles in reference.items():
        cand_profiles = candidate.get(key, [])
        for profile in ref_profiles:
            if not any(_residual_weakly_dominates(cand, profile) for cand in cand_profiles):
                uncovered += 1
    return {
        "keys_equal": not missing_keys and not extra_keys,
        "profile_coverage": uncovered == 0 and not missing_keys,
        "missing_keys": len(missing_keys),
        "extra_keys": len(extra_keys),
        "uncovered_profiles": uncovered,
    }


def summarize_method(instance: VariableLengthInstance, method: MethodConfig, reference: Dict[QuantityKey, List[Residual]] | None = None) -> Dict[str, object]:
    profile_map, stats, elapsed = solve_instance(instance, method)
    cmp = compare_profile_maps(reference, profile_map) if reference is not None else {
        "keys_equal": True,
        "profile_coverage": True,
        "missing_keys": 0,
        "extra_keys": 0,
        "uncovered_profiles": 0,
    }
    choices = _choices_for_method(instance, method)
    return {
        "instance": instance.name,
        "method": method.name,
        "mode": method.mode,
        "good_types": "|".join(str(v) for v in instance.good_types),
        "choices_total": sum(len(v) for v in choices.values()),
        "quantity_keys": len(profile_map),
        "profiles": sum(len(v) for v in profile_map.values()),
        "placement_attempts": stats.placement_assignments_attempted,
        "raw_feasible_children": stats.placement_assignments_feasible_raw,
        "labels_created": stats.labels_created,
        "local_pruned": stats.labels_pruned_by_local_skyline,
        "quantity_skyline_pruned": stats.labels_pruned_by_quantity_skyline,
        "dominance_pruned": stats.labels_pruned_by_dominance,
        "labels_after_dominance": stats.labels_after_dominance,
        "stage_attempted": "|".join(str(v) for v in stats.stage_attempted),
        "stage_raw_feasible": "|".join(str(v) for v in stats.stage_raw_feasible),
        "stage_created": "|".join(str(v) for v in stats.stage_created),
        "stage_kept": "|".join(str(v) for v in stats.stage_kept),
        "time_sec": elapsed,
        **cmp,
    }


def run_named_scenarios(seed: int, n_types: int, max_q: int, include_dominance: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    methods = [
        MethodConfig("EX", "ex", use_dominance=False),
        MethodConfig("GR_exact", "gr", use_dominance=False),
        MethodConfig("HYB", "hyb", use_dominance=False),
    ]
    if include_dominance:
        methods.extend([
            MethodConfig("EX_dom", "ex", use_dominance=True),
            MethodConfig("GR_dom", "gr", use_dominance=True),
            MethodConfig("HYB_dom", "hyb", use_dominance=True),
        ])
    for offset, kind in enumerate(("monotone", "mixed", "nonuniform")):
        instance = make_instance(kind, seed + offset, n_types=n_types, max_q=max_q)
        reference, _, _ = solve_instance(instance, methods[0])
        for method in methods:
            rows.append(summarize_method(instance, method, reference))
    return rows


def run_nonuniform_trials(seed: int, trials: int, n_types: int, max_q: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ex = MethodConfig("EX", "ex", use_dominance=False)
    gr = MethodConfig("GR_exact_assumption", "gr", use_dominance=False)
    hyb = MethodConfig("HYB", "hyb", use_dominance=False)
    for idx in range(trials):
        instance = make_instance("nonuniform", seed + 1000 + idx, n_types=n_types, max_q=max_q)
        ref, ref_stats, ref_time = solve_instance(instance, ex)
        gr_map, gr_stats, gr_time = solve_instance(instance, gr)
        hyb_map, hyb_stats, hyb_time = solve_instance(instance, hyb)
        gr_cmp = compare_profile_maps(ref, gr_map)
        hyb_cmp = compare_profile_maps(ref, hyb_map)
        rows.append({
            "trial": idx,
            "instance": instance.name,
            "good_types": "|".join(str(v) for v in instance.good_types),
            "num_good_types": len(instance.good_types),
            "ex_keys": len(ref),
            "gr_keys": len(gr_map),
            "hyb_keys": len(hyb_map),
            "ex_attempts": ref_stats.placement_assignments_attempted,
            "gr_attempts": gr_stats.placement_assignments_attempted,
            "hyb_attempts": hyb_stats.placement_assignments_attempted,
            "ex_time": ref_time,
            "gr_time": gr_time,
            "hyb_time": hyb_time,
            "gr_keys_equal": gr_cmp["keys_equal"],
            "gr_profile_coverage": gr_cmp["profile_coverage"],
            "gr_missing_keys": gr_cmp["missing_keys"],
            "gr_uncovered_profiles": gr_cmp["uncovered_profiles"],
            "hyb_keys_equal": hyb_cmp["keys_equal"],
            "hyb_profile_coverage": hyb_cmp["profile_coverage"],
            "hyb_missing_keys": hyb_cmp["missing_keys"],
            "hyb_uncovered_profiles": hyb_cmp["uncovered_profiles"],
        })
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_named_summary(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        if row["method"] not in {"EX", "GR_exact", "HYB"}:
            continue
        print(
            f"{row['instance']:18s} {row['method']:8s} "
            f"good={row['good_types'] or '-':7s} keys={row['quantity_keys']:4} "
            f"cover={str(row['profile_coverage']):5s} miss={row['missing_keys']:3} "
            f"attempts={row['placement_attempts']:9} labels={row['labels_created']:7} "
            f"local={row['local_pruned']:6} time={float(row['time_sec']):7.3f}s"
        )


def print_trial_summary(rows: List[Dict[str, object]]) -> None:
    total = len(rows)
    gr_exact = sum(1 for row in rows if row["gr_profile_coverage"])
    hyb_exact = sum(1 for row in rows if row["hyb_profile_coverage"])
    all_bad = sum(1 for row in rows if row["num_good_types"] == 0)
    print(
        f"nonuniform trials={total}, all_bad={all_bad}, "
        f"GR exact in {gr_exact}/{total}, HYB exact in {hyb_exact}/{total}"
    )
    bad_examples = [row for row in rows if not row["gr_profile_coverage"]][:5]
    for row in bad_examples:
        print(
            f"  trial={row['trial']:3} good={row['good_types'] or '-':5s} "
            f"EX_keys={row['ex_keys']:4} GR_keys={row['gr_keys']:4} "
            f"GR_missing={row['gr_missing_keys']:3} uncovered={row['gr_uncovered_profiles']:4} "
            f"HYB_cover={row['hyb_profile_coverage']}"
        )


def _pattern_value(key: QuantityKey, nominal_value: Dict[int, float]) -> float:
    return sum(nominal_value[car_type] * qty for car_type, qty in key)


def solve_pattern_master(case: WholeModelCase, method: MethodConfig) -> Dict[str, object]:
    profile_map, stats, gen_time = solve_instance(case.instance, method)
    patterns = sorted(profile_map)
    model = gp.Model(f"pattern_master_{case.name}_{method.name}")
    model.Params.OutputFlag = 0
    Config.apply_gurobi_params(model)
    y = model.addVars(range(len(patterns)), vtype=gp.GRB.INTEGER, lb=0, ub=case.num_compartments, name="y")
    model.addConstr(gp.quicksum(y[p] for p in range(len(patterns))) <= case.num_compartments, name="compartments")
    for car_type in sorted(case.instance.choices_by_type):
        model.addConstr(
            gp.quicksum(dict(patterns[p]).get(car_type, 0) * y[p] for p in range(len(patterns))) >= case.mandatory[car_type],
            name=f"mandatory[{car_type}]",
        )
        model.addConstr(
            gp.quicksum(dict(patterns[p]).get(car_type, 0) * y[p] for p in range(len(patterns))) <= case.mandatory[car_type] + case.optional[car_type],
            name=f"optional[{car_type}]",
        )
    model.setObjective(gp.quicksum(_pattern_value(patterns[p], case.nominal_value) * y[p] for p in range(len(patterns))), gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0
    return {
        "method": method.name,
        "status": int(model.Status),
        "objective": float(model.ObjVal) if model.SolCount else None,
        "patterns": len(patterns),
        "generation_time": gen_time,
        "solve_time": solve_time,
        "placement_attempts": stats.placement_assignments_attempted,
        "labels_created": stats.labels_created,
        "local_pruned": stats.labels_pruned_by_local_skyline,
        "quantity_skyline_pruned": stats.labels_pruned_by_quantity_skyline,
        "dominance_pruned": stats.labels_pruned_by_dominance,
    }


def solve_compact_whole(case: WholeModelCase) -> Dict[str, object]:
    model = gp.Model(f"compact_{case.name}")
    model.Params.OutputFlag = 0
    Config.apply_gurobi_params(model)
    car_types = sorted(case.instance.choices_by_type)
    compartments = range(case.num_compartments)
    choice_index = {
        car_type: list(enumerate(case.instance.choices_by_type[car_type]))
        for car_type in car_types
    }
    x = {}
    for car_type in car_types:
        for w in compartments:
            for choice_idx, _choice in choice_index[car_type]:
                x[car_type, w, choice_idx] = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=case.instance.max_quantity_by_type[car_type], name=f"x[{car_type},{w},{choice_idx}]")
    for car_type in car_types:
        total = gp.quicksum(x[car_type, w, choice_idx] for w in compartments for choice_idx, _choice in choice_index[car_type])
        model.addConstr(total >= case.mandatory[car_type], name=f"mandatory[{car_type}]")
        model.addConstr(total <= case.mandatory[car_type] + case.optional[car_type], name=f"optional[{car_type}]")
    for w in compartments:
        for interval_idx, cap in enumerate(case.instance.capacities):
            expr = gp.LinExpr()
            for car_type in car_types:
                for choice_idx, choice in choice_index[car_type]:
                    if interval_idx in choice.hits:
                        expr += choice.load * x[car_type, w, choice_idx]
            model.addConstr(expr <= cap, name=f"interval[{w},{interval_idx}]")
    model.setObjective(
        gp.quicksum(case.nominal_value[car_type] * x[car_type, w, choice_idx] for car_type in car_types for w in compartments for choice_idx, _choice in choice_index[car_type]),
        gp.GRB.MAXIMIZE,
    )
    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0
    return {
        "method": "compact",
        "status": int(model.Status),
        "objective": float(model.ObjVal) if model.SolCount else None,
        "patterns": "",
        "generation_time": 0.0,
        "solve_time": solve_time,
        "placement_attempts": "",
        "labels_created": "",
        "local_pruned": "",
        "quantity_skyline_pruned": "",
        "dominance_pruned": "",
    }


def make_whole_case(kind: str, seed: int, n_types: int = 3, max_q: int = 2) -> WholeModelCase:
    instance = make_instance(kind, seed, n_types=n_types, max_q=max_q)
    rng = random.Random(seed + 17)
    mandatory = {car_type: 1 for car_type in instance.choices_by_type}
    optional = {car_type: rng.randint(1, 2) for car_type in instance.choices_by_type}
    nominal_value = {car_type: 1000.0 + 150.0 * car_type for car_type in instance.choices_by_type}
    return WholeModelCase(
        name=f"whole_{instance.name}",
        instance=instance,
        num_compartments=2,
        mandatory=mandatory,
        optional=optional,
        nominal_value=nominal_value,
    )


def run_whole_model_tests(seed: int, trials: int, n_types: int, max_q: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    methods = [
        MethodConfig("EX", "ex", use_dominance=False),
        MethodConfig("GR_exact_assumption", "gr", use_dominance=False),
        MethodConfig("HYB", "hyb", use_dominance=False),
    ]
    kinds = ["monotone", "mixed", "nonuniform"]
    for idx in range(trials):
        kind = kinds[idx % len(kinds)]
        case = make_whole_case(kind, seed + 2000 + idx, n_types=n_types, max_q=max_q)
        compact = solve_compact_whole(case)
        for result in [compact] + [solve_pattern_master(case, method) for method in methods]:
            objective = result["objective"]
            compact_obj = compact["objective"]
            aligned = objective is not None and compact_obj is not None and abs(objective - compact_obj) <= 1e-6
            rows.append({
                "case": case.name,
                "kind": kind,
                "good_types": "|".join(str(v) for v in case.instance.good_types),
                "method": result["method"],
                "status": result["status"],
                "objective": objective if objective is not None else "",
                "compact_objective": compact_obj if compact_obj is not None else "",
                "aligned_with_compact": aligned,
                "patterns": result["patterns"],
                "generation_time": result["generation_time"],
                "solve_time": result["solve_time"],
                "placement_attempts": result["placement_attempts"],
                "labels_created": result["labels_created"],
                "local_pruned": result["local_pruned"],
                "quantity_skyline_pruned": result["quantity_skyline_pruned"],
                "dominance_pruned": result["dominance_pruned"],
            })
    return rows


def print_whole_summary(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        if row["method"] == "compact":
            print(f"{row['case']:28s} compact obj={row['objective']}")
            continue
        print(
            f"{row['case']:28s} {row['method']:19s} "
            f"align={str(row['aligned_with_compact']):5s} obj={row['objective']} "
            f"patterns={row['patterns']} attempts={row['placement_attempts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["scenario", "trials", "whole", "all"], default="whole")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--n-types", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=2)
    parser.add_argument("--include-dominance", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "result" / "variable_length_gr_hybrid")
    args = parser.parse_args()

    if args.scope in {"scenario", "all"}:
        scenario_rows = run_named_scenarios(args.seed, args.n_types, args.max_q, args.include_dominance)
        write_csv(scenario_rows, args.out_dir / "scenario_summary.csv")
        print_named_summary(scenario_rows)
        print(f"\nwrote {args.out_dir / 'scenario_summary.csv'}")
    if args.scope in {"trials", "all"}:
        trials = run_nonuniform_trials(args.seed, args.trials, args.n_types, args.max_q)
        write_csv(trials, args.out_dir / "nonuniform_trials.csv")
        print_trial_summary(trials)
        print(f"wrote {args.out_dir / 'nonuniform_trials.csv'}")
    if args.scope in {"whole", "all"}:
        whole_rows = run_whole_model_tests(args.seed, args.trials, args.n_types, args.max_q)
        write_csv(whole_rows, args.out_dir / "whole_model_summary.csv")
        print_whole_summary(whole_rows)
        print(f"wrote {args.out_dir / 'whole_model_summary.csv'}")


if __name__ == "__main__":
    main()
