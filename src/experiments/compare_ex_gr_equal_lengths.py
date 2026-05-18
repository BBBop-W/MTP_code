from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.model.BPC_compartment.feasibility_check as feasibility_check
from src.model.BPC_compartment.labeling import CompartmentSpec


Residual = Tuple[float, ...]
Quantity = Dict[int, int]


@dataclass(frozen=True)
class ToyCase:
    name: str
    compartment: str
    deck: str
    num_splits: int
    heights: Dict[int, float]
    max_per_type: int = 2
    max_total: int = 6
    equal_length: float = 1600.0


@dataclass(frozen=True)
class ExperimentResourceModel:
    capacities: Tuple[float, ...]
    intervals: Tuple[Tuple[int, int, float], ...]
    choices_by_type: Dict[int, Tuple[feasibility_check.PlacementChoice, ...]]
    delta: float = 400.0


def _build_full_resource_model(layer: CompartmentSpec, interval_profile: str = "full") -> ExperimentResourceModel:
    """Build all feasible central/left/right choices for the toy experiment."""

    mode = layer.shape_params.get("deck", "h-h")
    compartment = layer.shape_params.get("compartment", "lower")
    mode_left, mode_right = feasibility_check._split_deck_mode(mode)
    pi_left = 1 if mode_left == "m" else 0
    pi_right = 1 if mode_right == "m" else 0

    segments = feasibility_check._get_segments_for_layer(layer.car_heights)
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
            if compartment == "upper":
                enforce_left = (l_idx == n_blocks) or (pi_left == 1)
                enforce_right = (r_idx == n_blocks) or (pi_right == 1)
                if not (enforce_left and enforce_right):
                    continue

            if l_idx == n_blocks and r_idx == n_blocks:
                mod = -delta
            elif l_idx == n_blocks or r_idx == n_blocks:
                mod = 0.0
            else:
                mod = delta

            cap = lengths[0] + sum(lengths[1:l_idx + 1]) + sum(lengths[1:r_idx + 1]) + mod
            if feasibility_check._keep_interval_for_profile(l_idx, r_idx, n_blocks, interval_profile):
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

    choices_by_type: Dict[int, Tuple[feasibility_check.PlacementChoice, ...]] = {}
    for car_type in layer.car_types:
        height = layer.car_heights[car_type]
        choices: List[feasibility_check.PlacementChoice] = []

        if height <= actual_limit_central:
            choices.append(feasibility_check.PlacementChoice("central", 0, hits_for("central", 0)))
        for idx in range(1, n_blocks + 1):
            if height <= limit_left[idx]:
                choices.append(feasibility_check.PlacementChoice("left", idx, hits_for("left", idx)))
            if height <= limit_right[idx]:
                choices.append(feasibility_check.PlacementChoice("right", idx, hits_for("right", idx)))

        dedup: Dict[Tuple[int, ...], feasibility_check.PlacementChoice] = {}
        for choice in choices:
            dedup.setdefault(choice.hits, choice)
        choices_by_type[car_type] = tuple(dedup.values())

    return ExperimentResourceModel(
        capacities=tuple(cap for _l, _r, cap in intervals),
        intervals=tuple(intervals),
        choices_by_type=choices_by_type,
        delta=delta,
    )


def _choice_weakly_dominates(a, b) -> bool:
    return set(a.hits).issubset(set(b.hits))


def _prune_outer_dominated_choices(resource_model: ExperimentResourceModel) -> ExperimentResourceModel:
    """Keep only placement choices not dominated by an outer placement of the same type."""

    pruned: Dict[int, Tuple[feasibility_check.PlacementChoice, ...]] = {}
    for car_type, choices in resource_model.choices_by_type.items():
        kept: List[feasibility_check.PlacementChoice] = []
        for choice in choices:
            dominated = False
            for other in choices:
                if other == choice:
                    continue
                if _choice_weakly_dominates(other, choice):
                    dominated = True
                    break
            if not dominated:
                kept.append(choice)
        pruned[car_type] = tuple(kept)
    return ExperimentResourceModel(
        capacities=resource_model.capacities,
        intervals=resource_model.intervals,
        choices_by_type=pruned,
        delta=resource_model.delta,
    )


def _choice_count_vectors(total: int, num_choices: int):
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


def _dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b)) and any(x > y + eps for x, y in zip(a, b))


def _weakly_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b))


def _skyline(profiles: Iterable[Residual], eps: float = 1e-9) -> List[Residual]:
    kept: List[Residual] = []
    seen = set()
    for profile in profiles:
        key = tuple(round(v, 8) for v in profile)
        if key in seen:
            continue
        seen.add(key)
        dominated = False
        remove: List[int] = []
        for idx, old in enumerate(kept):
            if _weakly_dominates(old, profile, eps):
                dominated = True
                break
            if _dominates(profile, old, eps):
                remove.append(idx)
        if dominated:
            continue
        if remove:
            remove_set = set(remove)
            kept = [value for idx, value in enumerate(kept) if idx not in remove_set]
        kept.append(profile)
    return kept


def _apply_choice(residual: Residual, hits: Sequence[int], amount: float, eps: float = 1e-9) -> Residual | None:
    updated = list(residual)
    for idx in hits:
        updated[idx] -= amount
        if updated[idx] < -eps:
            return None
    return tuple(updated)


def _exhaustive_profiles(resource_model, quantities: Quantity, unit_resource: float) -> List[Residual]:
    profiles: List[Residual] = [resource_model.capacities]
    for car_type in sorted(quantities):
        quantity = quantities[car_type]
        choices = resource_model.choices_by_type.get(car_type, ())
        next_profiles: List[Residual] = []
        for residual in profiles:
            for counts in _choice_count_vectors(quantity, len(choices)):
                updated = residual
                feasible = True
                for count, choice in zip(counts, choices):
                    if count == 0:
                        continue
                    updated = _apply_choice(updated, choice.hits, count * unit_resource)
                    if updated is None:
                        feasible = False
                        break
                if feasible:
                    next_profiles.append(updated)
        profiles = _skyline(next_profiles)
        if not profiles:
            break
    return profiles


def _type_order(resource_model, heights: Dict[int, float]) -> List[int]:
    def rank(car_type: int):
        choices = resource_model.choices_by_type.get(car_type, ())
        min_hits = min((len(choice.hits) for choice in choices), default=10**9)
        return (min_hits, heights[car_type], car_type)

    return sorted(heights, key=rank)


def _quantity_vectors(car_types: Sequence[int], max_per_type: int, max_total: int):
    def rec(pos: int, remaining: int, current: Dict[int, int]):
        if pos == len(car_types):
            if any(current.values()):
                yield dict(current)
            return
        car_type = car_types[pos]
        for q in range(min(max_per_type, remaining) + 1):
            current[car_type] = q
            yield from rec(pos + 1, remaining - q, current)
        current.pop(car_type, None)

    yield from rec(0, max_total, {})


def _covers_all(candidates: List[Residual], exhaustive: List[Residual]) -> bool:
    return all(any(_weakly_dominates(candidate, profile) for candidate in candidates) for profile in exhaustive)


def _same_profile_set(a: List[Residual], b: List[Residual]) -> bool:
    rounded_a = {tuple(round(v, 8) for v in profile) for profile in a}
    rounded_b = {tuple(round(v, 8) for v in profile) for profile in b}
    return rounded_a == rounded_b


def run_case(case: ToyCase) -> None:
    feasibility_check.GLOBAL_NUM_SPLITS = case.num_splits
    feasibility_check._SEGMENTS_CACHE.clear()

    car_types = sorted(case.heights)
    layer = CompartmentSpec(
        compartment_id=case.name,
        car_types=car_types,
        car_lengths={car_type: case.equal_length for car_type in car_types},
        compartment_length_limit=10**9,
        car_heights=case.heights,
        shape_params={"deck": case.deck, "compartment": case.compartment},
        max_quantity_by_type={car_type: case.max_per_type for car_type in car_types},
    )
    ex_model = _build_full_resource_model(layer, interval_profile="full")
    gr_model = _prune_outer_dominated_choices(ex_model)
    unit_resource = case.equal_length + ex_model.delta

    print(f"\n=== {case.name} ===")
    print(f"compartment={case.compartment}, deck={case.deck}, splits={case.num_splits}, unit_resource={unit_resource:.0f}")
    print(f"intervals={len(ex_model.capacities)}, type_order={_type_order(gr_model, case.heights)}")
    for car_type in car_types:
        ex_choices = ex_model.choices_by_type.get(car_type, ())
        gr_choices = gr_model.choices_by_type.get(car_type, ())
        ex_desc = ", ".join(f"{choice.side}{choice.block_idx}" for choice in ex_choices) or "none"
        gr_desc = ", ".join(f"{choice.side}{choice.block_idx}" for choice in gr_choices) or "none"
        print(f"type {car_type}: height={case.heights[car_type]:.0f}, EX choices={ex_desc}; GR choices={gr_desc}")

    stats = {
        "vectors": 0,
        "ex_feasible": 0,
        "gr_feasible": 0,
        "gr_covers": 0,
        "gr_equals_ex": 0,
        "ex_profiles_total": 0,
        "ex_profiles_max": 0,
        "gr_profiles_total": 0,
        "gr_profiles_max": 0,
    }
    examples = []

    t0 = time.perf_counter()
    for quantities in _quantity_vectors(car_types, case.max_per_type, case.max_total):
        stats["vectors"] += 1
        ex_profiles = _exhaustive_profiles(ex_model, quantities, unit_resource)
        gr_profiles = _exhaustive_profiles(gr_model, quantities, unit_resource)

        if ex_profiles:
            stats["ex_feasible"] += 1
            stats["ex_profiles_total"] += len(ex_profiles)
            stats["ex_profiles_max"] = max(stats["ex_profiles_max"], len(ex_profiles))
        if gr_profiles:
            stats["gr_feasible"] += 1
            stats["gr_profiles_total"] += len(gr_profiles)
            stats["gr_profiles_max"] = max(stats["gr_profiles_max"], len(gr_profiles))
        if ex_profiles and _covers_all(gr_profiles, ex_profiles):
            stats["gr_covers"] += 1
        if ex_profiles and _same_profile_set(gr_profiles, ex_profiles):
            stats["gr_equals_ex"] += 1
        if ex_profiles and not _covers_all(gr_profiles, ex_profiles):
            if len(examples) < 5:
                examples.append((
                    dict(quantities),
                    len(ex_profiles),
                    len(gr_profiles),
                    _same_profile_set(gr_profiles, ex_profiles),
                ))

    elapsed = time.perf_counter() - t0
    avg_ex = stats["ex_profiles_total"] / max(stats["ex_feasible"], 1)
    avg_gr = stats["gr_profiles_total"] / max(stats["gr_feasible"], 1)
    print(
        "summary: "
        f"vectors={stats['vectors']}, ex_feasible={stats['ex_feasible']}, "
        f"gr_feasible={stats['gr_feasible']}, "
        f"gr_covers={stats['gr_covers']}/{stats['ex_feasible']}, "
        f"gr_equals_ex={stats['gr_equals_ex']}/{stats['ex_feasible']}"
    )
    print(
        "profiles: "
        f"avg_ex={avg_ex:.2f}, max_ex={stats['ex_profiles_max']}, "
        f"avg_gr={avg_gr:.2f}, max_gr={stats['gr_profiles_max']}, "
        f"elapsed={elapsed:.4f}s"
    )
    if examples:
        print("first non-cover examples: quantities, ex_count, gr_count, same_profile_set")
        for example in examples:
            print(f"  {example}")


def main() -> None:
    cases = [
        ToyCase(
            name="lower_hh_splits1_3types_equal_length",
            compartment="lower",
            deck="h-h",
            num_splits=1,
            heights={1: 1500.0, 2: 1700.0, 3: 1900.0},
            max_per_type=2,
            max_total=5,
        ),
        ToyCase(
            name="lower_hh_splits3_4types_equal_length",
            compartment="lower",
            deck="h-h",
            num_splits=3,
            heights={1: 1500.0, 2: 1650.0, 3: 1800.0, 4: 2000.0},
            max_per_type=2,
            max_total=6,
        ),
        ToyCase(
            name="upper_hm_splits3_4types_equal_length",
            compartment="upper",
            deck="h-m",
            num_splits=3,
            heights={1: 1500.0, 2: 1650.0, 3: 1780.0, 4: 1950.0},
            max_per_type=2,
            max_total=6,
        ),
        ToyCase(
            name="lower_hh_splits4_5types_equal_length",
            compartment="lower",
            deck="h-h",
            num_splits=4,
            heights={1: 1450.0, 2: 1550.0, 3: 1650.0, 4: 1800.0, 5: 2000.0},
            max_per_type=2,
            max_total=7,
        ),
        ToyCase(
            name="lower_hh_splits5_6types_equal_length",
            compartment="lower",
            deck="h-h",
            num_splits=5,
            heights={1: 1450.0, 2: 1525.0, 3: 1600.0, 4: 1700.0, 5: 1850.0, 6: 2000.0},
            max_per_type=2,
            max_total=8,
        ),
        ToyCase(
            name="upper_hm_splits5_6types_equal_length",
            compartment="upper",
            deck="h-m",
            num_splits=5,
            heights={1: 1450.0, 2: 1525.0, 3: 1600.0, 4: 1700.0, 5: 1800.0, 6: 1950.0},
            max_per_type=2,
            max_total=8,
        ),
        ToyCase(
            name="lower_hh_splits6_7types_equal_length",
            compartment="lower",
            deck="h-h",
            num_splits=6,
            heights={1: 1425.0, 2: 1500.0, 3: 1575.0, 4: 1650.0, 5: 1750.0, 6: 1875.0, 7: 2000.0},
            max_per_type=2,
            max_total=9,
        ),
    ]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
