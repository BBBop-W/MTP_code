from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import floor
from typing import Dict, List, Tuple

from src.model.BPC.labeling import CutEvaluator, DualValues, LayerSpec


@dataclass(frozen=True)
class CutState:
    """Dual-like coefficients for optional pricing cuts.

    eta_sum: aggregated dual of rounded-capacity cuts.
    sigma_by_subset: dual values for 3-SR cuts, keyed by sorted type tuples.
    """

    eta_sum: float = 0.0
    sigma_by_subset: Dict[Tuple[int, int, int], float] = field(default_factory=dict)


class SimpleCutEvaluator(CutEvaluator):
    """Cut evaluator with runtime switches.

    Implemented behaviors:
    1) Feasibility filter (optional): cap total units in a subpattern.
    2) Reduced-cost correction terms (optional): RCC + 3-SR style adjustments.

    Notes:
    - This is pricing-side integration scaffolding. Dual values for cuts can be
      injected via CutState when those cuts are added to RMP.
    """

    def __init__(
        self,
        max_total_by_type: Dict[int, int],
        wagon_capacity: int | None = None,
        cut_state: CutState | None = None,
    ) -> None:
        self.max_total_by_type = {int(k): int(v) for k, v in max_total_by_type.items()}
        self.wagon_capacity = wagon_capacity
        self.cut_state = cut_state or CutState()

        # Default all 3-subsets ready for 3-SR dual injection.
        types = sorted(self.max_total_by_type.keys())
        self.default_triplets: List[Tuple[int, int, int]] = list(combinations(types, 3))

    def is_feasible(self, layer: LayerSpec, quantities: Dict[int, int]) -> bool:
        for i, q in quantities.items():
            if q < 0:
                return False
            if q > self.max_total_by_type.get(i, 0):
                return False

        if self.wagon_capacity is not None:
            if sum(max(0, int(v)) for v in quantities.values()) > int(self.wagon_capacity):
                return False

        return True

    def reduced_cost_shift(self, layer: LayerSpec, quantities: Dict[int, int], duals: DualValues) -> float:
        # Reduced cost adjustment:
        #   -sum eta_c - sum sigma_c * floor(0.5 * sum_{i in Ic'} delta_i(r))
        shift = -float(self.cut_state.eta_sum)

        sigma_items = self.cut_state.sigma_by_subset.items()
        if not sigma_items:
            return shift

        for subset, sigma in sigma_items:
            val = 0
            for i in subset:
                total_i = self.max_total_by_type.get(i, 0)
                if total_i <= 0:
                    continue
                if quantities.get(i, 0) > total_i / 2.0:
                    val += 1
            coeff = floor(0.5 * val)
            shift -= float(sigma) * coeff

        return shift
