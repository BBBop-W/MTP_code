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

    def evaluate(self, layer: LayerSpec, quantities: Dict[int, int]) -> BSResult:
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

    def _read_components(self, layer: LayerSpec) -> List[BSComponent]:
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
        layer: LayerSpec,
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

    def _one_step_reachable(self, layer: LayerSpec, quantities: Dict[int, int]) -> Set[int]:
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

    def evaluate_no_reachable(self, layer: LayerSpec, quantities: Dict[int, int]) -> BSResult:
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
