from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


Residual = Tuple[float, ...]
QuantityKey = Tuple[Tuple[int, int], ...]
ProfileMap = Dict[QuantityKey, List[Residual]]


@dataclass(frozen=True)
class PlacementChoice:
    """A feasible placement choice for one automobile in one component."""

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
class ResidualProfileInstance:
    """A single layer/sub-pattern instance for residual-profile labeling."""

    name: str
    capacities: Residual
    choices_by_type: Dict[int, Tuple[PlacementChoice, ...]]
    max_quantity_by_type: Dict[int, int]
    gr_safe_types: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ProfileLabelingOptions:
    """Options for EX, GR, and HYB residual-profile label expansion.

    The dominance switch is intended for pricing with a fixed reduced-cost
    continuation structure. For full pattern enumeration, keep it disabled and
    rely on quantity skyline pruning.
    """

    name: str
    mode: str
    use_dominance: bool = False
    use_local_skyline: bool = True
    unit_reward: float = 1000.0
    time_limit_sec: float | None = None
    label_limit: int | None = None


@dataclass(frozen=True)
class ResidualLabel:
    stage: int
    reduced_cost: float
    quantities: Tuple[int, ...]
    residual: Residual


@dataclass
class ProfileLabelingStats:
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


class LabelingLimitExceeded(TimeoutError):
    def __init__(self, reason: str, elapsed: float, stats: ProfileLabelingStats) -> None:
        super().__init__(reason)
        self.reason = reason
        self.elapsed = elapsed
        self.stats = stats


def choice_dominates(a: PlacementChoice, b: PlacementChoice, eps: float = 1e-9) -> bool:
    """Return true if choice a consumes no more nested resources than choice b."""

    return set(a.hits).issubset(set(b.hits)) and a.load <= b.load + eps


def gr_choices(choices: Sequence[PlacementChoice]) -> Tuple[PlacementChoice, ...]:
    """Keep the outermost feasible placement on each side.

    Left and right are both retained because asymmetric profiles are generally
    incomparable. The central choice is retained only when no side choice exists.
    """

    kept: List[PlacementChoice] = []
    for side in ("left", "right"):
        side_choices = [choice for choice in choices if choice.side == side]
        if side_choices:
            kept.append(max(side_choices, key=lambda choice: choice.block))
    if not kept:
        kept.extend(choice for choice in choices if choice.side == "central")
    return tuple(kept)


def is_gr_safe(choices: Sequence[PlacementChoice]) -> bool:
    retained = gr_choices(choices)
    return all(any(choice_dominates(gr, choice) for gr in retained) for choice in choices)


def choices_for_options(
    instance: ResidualProfileInstance,
    options: ProfileLabelingOptions,
) -> Dict[int, Tuple[PlacementChoice, ...]]:
    if options.mode == "ex":
        return instance.choices_by_type
    if options.mode == "gr":
        return {car_type: gr_choices(choices) for car_type, choices in instance.choices_by_type.items()}
    if options.mode == "hyb":
        safe = set(instance.gr_safe_types)
        return {
            car_type: gr_choices(choices) if car_type in safe else choices
            for car_type, choices in instance.choices_by_type.items()
        }
    raise ValueError(f"unknown residual-profile labeling mode: {options.mode}")


def compare_profile_maps(reference: ProfileMap, candidate: ProfileMap) -> Dict[str, object]:
    missing_keys = set(reference) - set(candidate)
    extra_keys = set(candidate) - set(reference)
    uncovered = 0
    for key, ref_profiles in reference.items():
        cand_profiles = candidate.get(key, [])
        for profile in ref_profiles:
            if not any(residual_weakly_dominates(cand, profile) for cand in cand_profiles):
                uncovered += 1
    return {
        "keys_equal": not missing_keys and not extra_keys,
        "profile_coverage": uncovered == 0 and not missing_keys,
        "missing_keys": len(missing_keys),
        "extra_keys": len(extra_keys),
        "uncovered_profiles": uncovered,
    }


def solve_residual_profile_labeling(
    instance: ResidualProfileInstance,
    options: ProfileLabelingOptions,
) -> Tuple[ProfileMap, ProfileLabelingStats, float]:
    choices_by_type = choices_for_options(instance, options)
    car_types = sorted(instance.choices_by_type)
    root = ResidualLabel(
        stage=0,
        reduced_cost=0.0,
        quantities=tuple(0 for _ in car_types),
        residual=instance.capacities,
    )
    labels = [root]
    stats = ProfileLabelingStats()
    start = time.perf_counter()
    deadline = start + options.time_limit_sec if options.time_limit_sec is not None else None

    for stage, car_type in enumerate(car_types, start=1):
        next_labels: List[ResidualLabel] = []
        attempted_before = stats.placement_assignments_attempted
        raw_before = stats.placement_assignments_feasible_raw
        max_q = int(instance.max_quantity_by_type.get(car_type, 0))
        choices = choices_by_type[car_type]
        for label in labels:
            _check_limits(deadline, start, options.label_limit, len(next_labels), stats)
            for qty in range(max_q + 1):
                quantities = list(label.quantities)
                quantities[stage - 1] = qty
                residuals = list(_placement_residuals(label.residual, choices, qty, stats, deadline, start))
                if options.use_local_skyline:
                    residuals = local_skyline(residuals, stats)
                for residual in residuals:
                    next_labels.append(
                        ResidualLabel(
                            stage=stage,
                            reduced_cost=label.reduced_cost - options.unit_reward * qty,
                            quantities=tuple(quantities),
                            residual=residual,
                        )
                    )
                _check_limits(deadline, start, options.label_limit, len(next_labels), stats)
        stats.labels_created += len(next_labels)
        stats.stage_attempted.append(stats.placement_assignments_attempted - attempted_before)
        stats.stage_raw_feasible.append(stats.placement_assignments_feasible_raw - raw_before)
        stats.stage_created.append(len(next_labels))
        next_labels = apply_quantity_skyline(next_labels, stats)
        if options.use_dominance:
            next_labels = apply_label_dominance(next_labels, stats)
        stats.stage_kept.append(len(next_labels))
        labels = next_labels
        if not labels:
            break

    elapsed = time.perf_counter() - start
    profile_map: ProfileMap = {}
    for label in labels:
        if label.stage != len(car_types):
            continue
        key = quantity_key(car_types, label.quantities)
        if key:
            profile_map.setdefault(key, []).append(label.residual)
    profile_map = {key: skyline_profiles(values) for key, values in profile_map.items()}
    return profile_map, stats, elapsed


def local_skyline(residuals: Iterable[Residual], stats: ProfileLabelingStats, eps: float = 1e-9) -> List[Residual]:
    kept: List[Residual] = []
    seen = set()
    for residual in residuals:
        key = tuple(round(value, 8) for value in residual)
        if key in seen:
            stats.labels_pruned_by_local_skyline += 1
            continue
        seen.add(key)
        dominated = False
        remove_idx: List[int] = []
        for idx, old in enumerate(kept):
            if residual_weakly_dominates(old, residual, eps):
                dominated = True
                break
            if residual_dominates(residual, old, eps):
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


def skyline_profiles(profiles: Iterable[Residual]) -> List[Residual]:
    stats = ProfileLabelingStats()
    return local_skyline(profiles, stats)


def apply_quantity_skyline(labels: List[ResidualLabel], stats: ProfileLabelingStats) -> List[ResidualLabel]:
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
                if residual_weakly_dominates(old.residual, cand.residual):
                    dominated = True
                    break
                if residual_dominates(cand.residual, old.residual):
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


def apply_label_dominance(labels: List[ResidualLabel], stats: ProfileLabelingStats) -> List[ResidualLabel]:
    kept: List[ResidualLabel] = []
    for cand in sorted(labels, key=lambda label: label.reduced_cost):
        if any(label_dominates(old, cand) for old in kept):
            stats.labels_pruned_by_dominance += 1
            continue
        kept.append(cand)
    stats.labels_after_dominance += len(kept)
    return kept


def label_dominates(a: ResidualLabel, b: ResidualLabel, eps: float = 1e-9) -> bool:
    if a.reduced_cost > b.reduced_cost + eps:
        return False
    if not residual_weakly_dominates(a.residual, b.residual, eps):
        return False
    return a.reduced_cost < b.reduced_cost - eps or residual_dominates(a.residual, b.residual, eps)


def residual_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b)) and any(x > y + eps for x, y in zip(a, b))


def residual_weakly_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b))


def quantity_key(car_types: Sequence[int], quantities: Sequence[int]) -> QuantityKey:
    return tuple((car_type, qty) for car_type, qty in zip(car_types, quantities) if qty > 0)


def choice_count_vectors(total: int, num_choices: int) -> Iterator[Tuple[int, ...]]:
    if num_choices == 0:
        if total == 0:
            yield ()
        return
    if num_choices == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in choice_count_vectors(total - first, num_choices - 1):
            yield (first,) + rest


def _placement_residuals(
    residual: Residual,
    choices: Sequence[PlacementChoice],
    quantity: int,
    stats: ProfileLabelingStats,
    deadline: float | None,
    start: float,
) -> Iterator[Residual]:
    if quantity == 0:
        stats.placement_assignments_attempted += 1
        stats.placement_assignments_feasible_raw += 1
        yield residual
        return
    for counts in choice_count_vectors(quantity, len(choices)):
        _check_deadline(deadline, start, stats)
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


def _apply_choice(residual: Residual, choice: PlacementChoice, count: int, eps: float = 1e-9) -> Residual | None:
    updated = list(residual)
    amount = count * choice.load
    for idx in choice.hits:
        updated[idx] -= amount
        if updated[idx] < -eps:
            return None
    return tuple(updated)


def _check_limits(
    deadline: float | None,
    start: float,
    label_limit: int | None,
    label_count: int,
    stats: ProfileLabelingStats,
) -> None:
    _check_deadline(deadline, start, stats)
    if label_limit is not None and label_count > label_limit:
        raise LabelingLimitExceeded("label_limit", time.perf_counter() - start, stats)


def _check_deadline(deadline: float | None, start: float, stats: ProfileLabelingStats) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise LabelingLimitExceeded("labeling_time_limit", time.perf_counter() - start, stats)
