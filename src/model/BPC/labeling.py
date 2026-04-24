from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Set, Tuple


@dataclass(frozen=True)
class DualValues:
    alpha: Dict[int, float]
    beta: Dict[int, float]
    gamma: float


@dataclass(frozen=True)
class LayerSpec:
    """A single compartment/layer subproblem instance.

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


@dataclass(frozen=True)
class LabelingOptions:
    use_dominance: bool = True
    use_cuts: bool = False
    max_units_per_type: int = 6
    eps: float = 1e-9


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


@dataclass(frozen=True)
class LayerPattern:
    layer_id: str
    quantities: Dict[int, int]
    reduced_cost: float
    best_length: float
    shape_params: Dict[str, object]


import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.feasibility_check import HierarchicalBSEvaluator


def _label_dominates(a: Label, b: Label, ordered_types: List[int], eps: float) -> bool:
    if a.reduced_cost > b.reduced_cost + eps:
        return False
    if a.best_length > b.best_length + eps:
        return False
    if not a.reachable_types.issuperset(b.reachable_types):
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


def _apply_dominance(labels: List[Label], ordered_types: List[int], eps: float) -> List[Label]:
    kept: List[Label] = []
    for cand in labels:
        dominated = False
        remove_idx: List[int] = []
        for idx, old in enumerate(kept):
            if _label_dominates(old, cand, ordered_types, eps):
                dominated = True
                break
            if _label_dominates(cand, old, ordered_types, eps):
                remove_idx.append(idx)
        if dominated:
            continue
        if remove_idx:
            kept = [v for i, v in enumerate(kept) if i not in set(remove_idx)]
        kept.append(cand)
    return kept


def _infer_reachable_types(
    layer: LayerSpec,
    current_q: Dict[int, int],
    next_types: Iterable[int],
    bs: BSEvaluator,
    options: LabelingOptions,
) -> Set[int]:
    reachable: Set[int] = set()
    for t in next_types:
        max_q = min(options.max_units_per_type, int(layer.max_quantity_by_type.get(t, options.max_units_per_type)))
        if current_q.get(t, 0) >= max_q:
            continue
        probe_q = dict(current_q)
        probe_q[t] = probe_q.get(t, 0) + 1
        probe = bs.evaluate(layer, probe_q)
        if probe.feasible:
            reachable.add(t)
    return reachable


def generate_layer_patterns(
    layer: LayerSpec,
    duals: DualValues,
    bs: BSEvaluator,
    options: LabelingOptions = LabelingOptions(),
    cut_evaluator: Optional[CutEvaluator] = None,
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
    root_label = Label(
        stage=0,
        reduced_cost=-duals.gamma / 2.0,
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

                bs_result = bs.evaluate(layer, q_new)
                if not bs_result.feasible:
                    continue

                if options.use_cuts and cut_evaluator is not None and not cut_evaluator.is_feasible(layer, q_new):
                    continue

                rc = lb.reduced_cost - (layer.car_lengths[car_type] + duals.alpha.get(car_type, 0.0) + duals.beta.get(car_type, 0.0)) * q

                if options.use_cuts and cut_evaluator is not None:
                    rc += cut_evaluator.reduced_cost_shift(layer, q_new, duals)

                remaining = ordered_types[stage:]
                if bs_result.reachable_types is not None:
                    reachable_types = set(v for v in bs_result.reachable_types if v in remaining)
                else:
                    reachable_types = _infer_reachable_types(layer, q_new, remaining, bs, options)

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
            next_labels = _apply_dominance(next_labels, ordered_types, options.eps)

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
