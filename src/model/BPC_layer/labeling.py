from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Set, Tuple

from src.utility.config import config as Config


@dataclass(frozen=True)
class DualValues:
    alpha: Dict[int, float]
    beta: Dict[int, float]
    gamma: float
    branch_a: Dict[int, float] = field(default_factory=dict)
    branch_q: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerSpec:
    """A single compartment subproblem instance.

    The geometric effect of shape/deck position is passed via shape_params,
    and consumed only by BS evaluator.
    """

    layer_id: str
    car_types: List[int]
    car_lengths: Dict[int, float]
    layer_length_limit: float
    car_heights: Dict[int, float] = field(default_factory=dict)
    shape_params: Dict[str, object] = field(default_factory=dict)
    max_quantity_by_type: Dict[int, int] = field(default_factory=dict)


CompartmentSpec = LayerSpec


@dataclass(frozen=True)
class LabelingOptions:
    use_dominance: bool = True
    use_cuts: bool = False
    use_rc_bound: bool = True
    use_residual_profile: bool = False
    use_height_order: bool = True
    use_local_residual_skyline: bool = True
    residual_profile_mode: str = "full"
    profile_generator_mode: str = "hyb"
    compute_reachable_types: bool = False
    max_units_per_type: int = Config.max_units_per_compartment
    dominance_support_types: Tuple[int, ...] = ()
    eps: float = 1e-9


@dataclass
class LabelingStats:
    labels_feasible: int = 0
    labels_pruned_by_bound: int = 0
    labels_pruned_by_dominance: int = 0
    labels_pruned_by_local_skyline: int = 0
    labels_after_dominance: int = 0
    reachability_probes: int = 0


@dataclass(frozen=True)
class BSResult:
    feasible: bool
    best_length: float
    # Optional optimization: if BS can return forward-reachable nodes directly.
    reachable_types: Optional[Set[int]] = None


class BSEvaluator(Protocol):
    def evaluate(self, layer: LayerSpec, quantities: Dict[int, int]) -> BSResult:
        """Return feasibility and best loading length for a partial pattern."""


class CutEvaluator(Protocol):
    def is_feasible(self, layer: LayerSpec, quantities: Dict[int, int]) -> bool:
        """Return whether a candidate partial pattern satisfies active cut filters."""

    def reduced_cost_shift(self, layer: LayerSpec, quantities: Dict[int, int], duals: DualValues) -> float:
        """Return additional reduced-cost contribution from active cuts."""


@dataclass
class Label:
    stage: int
    reduced_cost: float
    quantities: Dict[int, int]
    best_length: float
    reachable_types: Set[int]


@dataclass
class ResidualLabel:
    stage: int
    reduced_cost: float
    quantities: Dict[int, int]
    total_length: float
    residual: Tuple[float, ...]


@dataclass(frozen=True)
class LayerPattern:
    layer_id: str
    quantities: Dict[int, int]
    reduced_cost: float
    best_length: float
    shape_params: Dict[str, object]


CompartmentPattern = LayerPattern


import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC_layer.feasibility_check import (
    HierarchicalBSEvaluator,
    build_layer_resource_model,
)


def _same_support_on_types(a_q: Dict[int, int], b_q: Dict[int, int], support_types: Sequence[int]) -> bool:
    return all((a_q.get(i, 0) > 0) == (b_q.get(i, 0) > 0) for i in support_types)


def _label_dominates(
    a: Label,
    b: Label,
    ordered_types: List[int],
    eps: float,
    support_types: Sequence[int] = (),
) -> bool:
    if a.reduced_cost > b.reduced_cost + eps:
        return False
    if a.best_length > b.best_length + eps:
        return False
    if not a.reachable_types.issuperset(b.reachable_types):
        return False
    if support_types and not _same_support_on_types(a.quantities, b.quantities, support_types):
        return False

    for i in ordered_types:
        if a.quantities.get(i, 0) > b.quantities.get(i, 0):
            return False

    # Exclude exact equality.
    if (
        abs(a.reduced_cost - b.reduced_cost) <= eps
        and abs(a.best_length - b.best_length) <= eps
        and a.reachable_types == b.reachable_types
        and all(a.quantities.get(i, 0) == b.quantities.get(i, 0) for i in ordered_types)
    ):
        return False
    return True


def _apply_dominance(
    labels: List[Label],
    ordered_types: List[int],
    eps: float,
    support_types: Sequence[int] = (),
    stats: Optional[LabelingStats] = None,
) -> List[Label]:
    kept: List[Label] = []
    for cand in labels:
        dominated = False
        remove_idx: List[int] = []
        for idx, old in enumerate(kept):
            if _label_dominates(old, cand, ordered_types, eps, support_types=support_types):
                dominated = True
                break
            if _label_dominates(cand, old, ordered_types, eps, support_types=support_types):
                remove_idx.append(idx)
        if dominated:
            if stats is not None:
                stats.labels_pruned_by_dominance += 1
            continue
        if remove_idx:
            if stats is not None:
                stats.labels_pruned_by_dominance += len(remove_idx)
            kept = [v for i, v in enumerate(kept) if i not in set(remove_idx)]
        kept.append(cand)
    if stats is not None:
        stats.labels_after_dominance += len(kept)
    return kept


def _infer_reachable_types(
    layer: LayerSpec,
    current_q: Dict[int, int],
    next_types: Iterable[int],
    bs: BSEvaluator,
    options: LabelingOptions,
    stats: Optional[LabelingStats] = None,
) -> Set[int]:
    reachable: Set[int] = set()
    for t in next_types:
        max_q = min(options.max_units_per_type, int(layer.max_quantity_by_type.get(t, options.max_units_per_type)))
        if current_q.get(t, 0) >= max_q:
            continue
        probe_q = dict(current_q)
        probe_q[t] = probe_q.get(t, 0) + 1
        if stats is not None:
            stats.reachability_probes += 1
        probe = bs.evaluate(layer, probe_q)
        if probe.feasible:
            reachable.add(t)
    return reachable


def _future_reduced_cost_lower_bound(
    current_rc: float,
    remaining_types: Iterable[int],
    layer: LayerSpec,
    duals: DualValues,
    options: LabelingOptions,
) -> float:
    """Optimistic lower bound for any suffix extension, ignoring geometry.

    If this value is non-negative, no feasible suffix can make the label produce
    a negative reduced-cost column. The bound is intentionally optimistic, so
    pruning on it is safe.
    """

    lb = current_rc
    for t in remaining_types:
        max_q = min(options.max_units_per_type, int(layer.max_quantity_by_type.get(t, options.max_units_per_type)))
        coef = layer.car_lengths[t] + duals.alpha.get(t, 0.0) + duals.beta.get(t, 0.0) + duals.branch_q.get(t, 0.0)
        best_delta = 0.0
        for q in range(1, max_q + 1):
            delta = -coef * q - duals.branch_a.get(t, 0.0)
            if delta < best_delta:
                best_delta = delta
        lb += best_delta
    return lb


def _residual_label_dominates(
    a: ResidualLabel,
    b: ResidualLabel,
    ordered_types: List[int],
    eps: float,
    support_types: Sequence[int] = (),
) -> bool:
    if a.reduced_cost > b.reduced_cost + eps:
        return False
    if support_types and not _same_support_on_types(a.quantities, b.quantities, support_types):
        return False

    quantity_strict = False
    for i in ordered_types:
        if a.quantities.get(i, 0) > b.quantities.get(i, 0):
            return False
        if a.quantities.get(i, 0) < b.quantities.get(i, 0):
            quantity_strict = True

    profile_strict = False
    for a_res, b_res in zip(a.residual, b.residual):
        if a_res + eps < b_res:
            return False
        if a_res > b_res + eps:
            profile_strict = True

    return profile_strict or quantity_strict or a.reduced_cost < b.reduced_cost - eps


def _apply_residual_dominance(
    labels: List[ResidualLabel],
    eps: float,
    support_types: Sequence[int] = (),
    stats: Optional[LabelingStats] = None,
) -> List[ResidualLabel]:
    if not labels:
        return []

    import numpy as np

    ordered = sorted(labels, key=lambda label: label.reduced_cost)
    type_order = sorted({i for label in ordered for i in label.quantities})
    support_idx = [type_order.index(i) for i in support_types if i in type_order]
    dim = len(ordered[0].residual)
    kept_residuals = np.empty((len(ordered), dim), dtype=float)
    kept_costs = np.empty(len(ordered), dtype=float)
    kept_quantities = np.empty((len(ordered), len(type_order)), dtype=int)
    kept: List[ResidualLabel] = []

    for cand in ordered:
        dominated = False
        kept_count = len(kept)
        if kept_count:
            cand_residual = np.asarray(cand.residual, dtype=float)
            cand_quantities = np.asarray([cand.quantities.get(i, 0) for i in type_order], dtype=int)
            residual_ok = np.all(kept_residuals[:kept_count] + eps >= cand_residual, axis=1)
            quantity_ok = np.all(kept_quantities[:kept_count] <= cand_quantities, axis=1)
            if support_idx:
                support_ok = np.all((kept_quantities[:kept_count, support_idx] > 0) == (cand_quantities[support_idx] > 0), axis=1)
            else:
                support_ok = True
            if np.any(residual_ok & quantity_ok & support_ok):
                profile_strict = np.any(kept_residuals[:kept_count] > cand_residual + eps, axis=1)
                quantity_strict = np.any(kept_quantities[:kept_count] < cand_quantities, axis=1)
                cost_strict = kept_costs[:kept_count] < cand.reduced_cost - eps
                dominated = bool(np.any(residual_ok & quantity_ok & support_ok & (profile_strict | quantity_strict | cost_strict)))

        if dominated:
            if stats is not None:
                stats.labels_pruned_by_dominance += 1
            continue

        kept_residuals[kept_count, :] = cand.residual
        kept_costs[kept_count] = cand.reduced_cost
        kept_quantities[kept_count, :] = [cand.quantities.get(i, 0) for i in type_order]
        kept.append(cand)

    if stats is not None:
        stats.labels_after_dominance += len(kept)
    return kept


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


def _placement_consumptions(
    choices,
    quantity: int,
    unit_length: float,
    resource_count: int,
) -> Iterator[Tuple[float, ...]]:
    if quantity == 0:
        yield (0.0,) * resource_count
        return
    if not choices:
        return

    seen: Set[Tuple[float, ...]] = set()
    for counts in _choice_count_vectors(quantity, len(choices)):
        consumption = [0.0] * resource_count
        for count, choice in zip(counts, choices):
            if count == 0:
                continue
            amount = count * unit_length
            for idx in choice.hits:
                consumption[idx] += amount
        key = tuple(consumption)
        if key in seen:
            continue
        seen.add(key)
        yield key


def _residual_vector_dominates(a: Tuple[float, ...], b: Tuple[float, ...], eps: float) -> bool:
    strict = False
    for a_val, b_val in zip(a, b):
        if a_val + eps < b_val:
            return False
        if a_val > b_val + eps:
            strict = True
    return strict


def _local_residual_skyline(
    residuals: List[Tuple[float, ...]],
    eps: float,
    stats: Optional[LabelingStats] = None,
) -> List[Tuple[float, ...]]:
    kept: List[Tuple[float, ...]] = []
    for residual in residuals:
        dominated = False
        remove_idx: List[int] = []
        for idx, old in enumerate(kept):
            if _residual_vector_dominates(old, residual, eps):
                dominated = True
                break
            if _residual_vector_dominates(residual, old, eps):
                remove_idx.append(idx)
        if dominated:
            if stats is not None:
                stats.labels_pruned_by_local_skyline += 1
            continue
        if remove_idx:
            if stats is not None:
                stats.labels_pruned_by_local_skyline += len(remove_idx)
            kept = [value for idx, value in enumerate(kept) if idx not in set(remove_idx)]
        kept.append(residual)
    return kept


def _apply_consumption(
    residual: Tuple[float, ...],
    consumption: Tuple[float, ...],
    eps: float,
) -> Optional[Tuple[float, ...]]:
    new_residual = tuple(r - c for r, c in zip(residual, consumption))
    if any(v < -eps for v in new_residual):
        return None
    return new_residual


def _choice_weakly_dominates(a, b) -> bool:
    return set(a.hits).issubset(set(b.hits))


def _prune_dominated_choices(choices) -> Tuple:
    kept = []
    for choice in choices:
        if any(other != choice and _choice_weakly_dominates(other, choice) for other in choices):
            continue
        kept.append(choice)
    return tuple(kept)


def _choices_cover_all(original, reduced) -> bool:
    return all(any(_choice_weakly_dominates(choice, old) for choice in reduced) for old in original)


def _choices_for_profile_generator(resource_model, mode: str) -> Dict[int, Tuple]:
    normalized = mode.lower().strip()
    if normalized in {"exact", "ex", "e"}:
        return dict(resource_model.choices_by_type)
    if normalized not in {"gr", "hyb", "hybrid"}:
        raise ValueError(f"Unknown profile_generator_mode: {mode}")

    selected: Dict[int, Tuple] = {}
    for car_type, choices in resource_model.choices_by_type.items():
        reduced = _prune_dominated_choices(choices)
        if reduced and _choices_cover_all(choices, reduced):
            selected[car_type] = reduced
        else:
            selected[car_type] = choices
    return selected


def generate_layer_patterns_residual(
    layer: LayerSpec,
    duals: DualValues,
    options: LabelingOptions = LabelingOptions(use_residual_profile=True),
    cut_evaluator: Optional[CutEvaluator] = None,
    stats: Optional[LabelingStats] = None,
) -> List[LayerPattern]:
    """Generate layer patterns with residual-profile labels.

    The dynamic chunking geometry is preprocessed into monotone interval
    resources. Each label stores the remaining residual capacity vector, so
    extension feasibility is a componentwise residual-capacity check rather
    than a DFS feasibility search.
    """

    if options.use_cuts and cut_evaluator is None:
        raise ValueError("use_cuts=True requires a cut_evaluator.")
    if options.use_height_order:
        ordered_types = sorted(layer.car_types, key=lambda i: (-float(layer.car_heights.get(i, 0.0)), i))
    else:
        ordered_types = list(layer.car_types)
    resource_model = build_layer_resource_model(layer, interval_profile=options.residual_profile_mode)
    choices_by_type = _choices_for_profile_generator(resource_model, options.profile_generator_mode)
    resource_count = len(resource_model.capacities)

    root_quantities = {i: 0 for i in ordered_types}
    root_rc = -duals.gamma / 2.0
    if options.use_cuts and cut_evaluator is not None:
        root_rc += cut_evaluator.reduced_cost_shift(layer, root_quantities, duals)
    root_label = ResidualLabel(
        stage=0,
        reduced_cost=root_rc,
        quantities=root_quantities,
        total_length=0.0,
        residual=resource_model.capacities,
    )

    current_labels: List[ResidualLabel] = [root_label]

    for stage, car_type in enumerate(ordered_types, start=1):
        next_labels: List[ResidualLabel] = []
        choices = choices_by_type.get(car_type, ())
        max_q = min(options.max_units_per_type, int(layer.max_quantity_by_type.get(car_type, options.max_units_per_type)))
        unit_resource = layer.car_lengths[car_type] + resource_model.delta
        unit_length = layer.car_lengths[car_type]
        coef = unit_length + duals.alpha.get(car_type, 0.0) + duals.beta.get(car_type, 0.0) + duals.branch_q.get(car_type, 0.0)
        remaining = ordered_types[stage:]

        for lb in current_labels:
            for q in range(0, max_q + 1):
                q_new = dict(lb.quantities)
                q_new[car_type] = q

                rc = lb.reduced_cost - coef * q
                if q > 0:
                    rc -= duals.branch_a.get(car_type, 0.0)
                if options.use_cuts and cut_evaluator is not None:
                    rc += cut_evaluator.reduced_cost_shift(layer, q_new, duals) - cut_evaluator.reduced_cost_shift(
                        layer, lb.quantities, duals
                    )

                if options.use_rc_bound and not options.use_cuts:
                    rc_lb = _future_reduced_cost_lower_bound(rc, remaining, layer, duals, options)
                    if rc_lb >= -options.eps:
                        if stats is not None:
                            stats.labels_pruned_by_bound += 1
                        continue

                total_length = lb.total_length + unit_length * q
                residual_children: List[Tuple[float, ...]] = []
                for consumption in _placement_consumptions(choices, q, unit_resource, resource_count):
                    residual = _apply_consumption(lb.residual, consumption, options.eps)
                    if residual is None:
                        continue
                    residual_children.append(residual)

                if options.use_dominance and options.use_local_residual_skyline:
                    residual_children = _local_residual_skyline(residual_children, options.eps, stats=stats)

                for residual in residual_children:
                    if stats is not None:
                        stats.labels_feasible += 1
                    next_labels.append(
                        ResidualLabel(
                            stage=stage,
                            reduced_cost=rc,
                            quantities=q_new,
                            total_length=total_length,
                            residual=residual,
                        )
                    )

        if options.use_dominance:
            next_labels = _apply_residual_dominance(
                next_labels,
                options.eps,
                support_types=options.dominance_support_types,
                stats=stats,
            )

        current_labels = next_labels
        if not current_labels:
            break

    best_by_quantities: Dict[Tuple[Tuple[int, int], ...], LayerPattern] = {}
    for lb in current_labels:
        if lb.stage != len(ordered_types):
            continue
        if lb.total_length > layer.layer_length_limit + options.eps:
            continue
        key = tuple(sorted((i, q) for i, q in lb.quantities.items() if q > 0))
        pattern = LayerPattern(
            layer_id=layer.layer_id,
            quantities=dict(lb.quantities),
            reduced_cost=lb.reduced_cost,
            best_length=lb.total_length,
            shape_params=dict(layer.shape_params),
        )
        old = best_by_quantities.get(key)
        if old is None or pattern.reduced_cost < old.reduced_cost - options.eps:
            best_by_quantities[key] = pattern

    return list(best_by_quantities.values())


def generate_layer_patterns(
    layer: LayerSpec,
    duals: DualValues,
    bs: BSEvaluator,
    options: LabelingOptions = LabelingOptions(),
    cut_evaluator: Optional[CutEvaluator] = None,
    stats: Optional[LabelingStats] = None,
) -> List[LayerPattern]:
    """Generate feasible patterns for one layer using forward label extension.

    Workflow:
    1) Traverse car types in fixed order: first type -> ... -> last type.
    2) For each type i, enumerate loading quantity q_i in [0, 6] (or tighter limit).
    3) After each extension, call BS to get best length and feasibility.
    4) Maintain reachable type set for the remaining steps.
    5) Optionally apply dominance and optional cut correction in reduced cost.
    """

    if options.use_cuts and cut_evaluator is None:
        raise ValueError("use_cuts=True requires a cut_evaluator.")

    ordered_types = list(layer.car_types)

    root_quantities = {i: 0 for i in ordered_types}
    root_rc = -duals.gamma / 2.0
    if options.use_cuts and cut_evaluator is not None:
        root_rc += cut_evaluator.reduced_cost_shift(layer, root_quantities, duals)
    root_label = Label(
        stage=0,
        reduced_cost=root_rc,
        quantities=root_quantities,
        best_length=0.0,
        reachable_types=set(ordered_types),
    )

    current_labels: List[Label] = [root_label]

    for stage, car_type in enumerate(ordered_types, start=1):
        next_labels: List[Label] = []

        for lb in current_labels:
            max_q = min(options.max_units_per_type, int(layer.max_quantity_by_type.get(car_type, options.max_units_per_type)))
            for q in range(0, max_q + 1):
                q_new = dict(lb.quantities)
                q_new[car_type] = q

                total_cars = sum(q_new.values())
                length_sum = sum(layer.car_lengths[c] * qty for c, qty in q_new.items())
                
                # Check if the entire group fits inside the central undivided region (E for upper, C for lower)
                # Updated to match the new dynamic_segmentation central physical absolute flat lengths
                safe_len = 11400.0 if layer.shape_params.get("compartment", "lower") == "upper" else 10867.0
                if length_sum + 400.0 * total_cars <= safe_len:
                    bs_result = BSResult(feasible=True, best_length=length_sum, reachable_types=set(ordered_types))
                else:
                    bs_result = bs.evaluate(layer, q_new)
                    
                if not bs_result.feasible:
                    continue
                if stats is not None:
                    stats.labels_feasible += 1

                if options.use_cuts and cut_evaluator is not None and not cut_evaluator.is_feasible(layer, q_new):
                    continue

                # Add alpha (mandatory), beta (optional), branch_q (qty per car type)
                rc = lb.reduced_cost - (
                    layer.car_lengths[car_type] + 
                    duals.alpha.get(car_type, 0.0) + 
                    duals.beta.get(car_type, 0.0) + 
                    duals.branch_q.get(car_type, 0.0)
                ) * q
                
                # Add branch_a (wagon existence for car type)
                if q > 0:
                    rc -= duals.branch_a.get(car_type, 0.0)

                if options.use_cuts and cut_evaluator is not None:
                    rc += cut_evaluator.reduced_cost_shift(layer, q_new, duals) - cut_evaluator.reduced_cost_shift(
                        layer, lb.quantities, duals
                    )

                remaining = ordered_types[stage:]
                if options.use_rc_bound and not options.use_cuts:
                    rc_lb = _future_reduced_cost_lower_bound(rc, remaining, layer, duals, options)
                    if rc_lb >= -options.eps:
                        if stats is not None:
                            stats.labels_pruned_by_bound += 1
                        continue

                if options.compute_reachable_types and bs_result.reachable_types is not None:
                    reachable_types = set(v for v in bs_result.reachable_types if v in remaining)
                elif options.compute_reachable_types:
                    reachable_types = _infer_reachable_types(layer, q_new, remaining, bs, options, stats=stats)
                else:
                    reachable_types = set(remaining)

                next_labels.append(
                    Label(
                        stage=stage,
                        reduced_cost=rc,
                        quantities=q_new,
                        best_length=bs_result.best_length,
                        reachable_types=reachable_types,
                    )
                )

        if options.use_dominance:
            next_labels = _apply_dominance(
                next_labels,
                ordered_types,
                options.eps,
                support_types=options.dominance_support_types,
                stats=stats,
            )

        current_labels = next_labels
        if not current_labels:
            break

    patterns: List[LayerPattern] = []
    for lb in current_labels:
        if lb.stage == len(ordered_types) and lb.best_length <= layer.layer_length_limit + options.eps:
            patterns.append(
                LayerPattern(
                    layer_id=layer.layer_id,
                    quantities=dict(lb.quantities),
                    reduced_cost=lb.reduced_cost,
                    best_length=lb.best_length,
                    shape_params=dict(layer.shape_params),
                )
            )

    return patterns


def merge_layer_patterns_placeholder(*args, **kwargs):
    """Placeholder for merging upper/lower layer patterns into full columns.

    Future merging checks:
    1) Same workstation/deck position compatibility.
    2) Sum quantities by car type <= optional + mandatory upper bounds.
    3) Merged reduced cost < 0.
    """
    raise NotImplementedError("Merging will be implemented in the next step.")
