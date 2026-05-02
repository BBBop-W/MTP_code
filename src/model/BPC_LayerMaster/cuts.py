from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import floor
from typing import Dict, List, Tuple

from src.model.BPC_LayerMaster.labeling import CutEvaluator, DualValues, LayerSpec


@dataclass(frozen=True)
class CutState:
    """Dual-like coefficients for optional pricing cuts.

    sigma_by_subset: dual values for 3-SR cuts, keyed by (compartment, sorted type tuple).
    compartment must be 'upper' or 'lower'.
    """

    eta_sum: float = 0.0
    sigma_by_subset: Dict[Tuple[str, Tuple[int, int, int]], float] = field(default_factory=dict)


class SimpleCutEvaluator(CutEvaluator):
    """Cut evaluator for LayerMaster.

    Implemented behaviors:
    1) Feasibility filter: No total max limits applied here since it's an unconstrained layer length check.
    2) Reduced-cost correction terms: 3-SR style adjustments based on compartment.
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

    def is_feasible(self, layer: LayerSpec, quantities: Dict[int, int]) -> bool:
        for i, q in quantities.items():
            if q < 0:
                return False
            # We don't artificially restrict the absolute limit here anymore
            # because the dynamic segments define exact packing capacity.

        return True

    def reduced_cost_shift(self, layer: LayerSpec, quantities: Dict[int, int], duals: DualValues) -> float:
        # Reduced cost adjustment:
        #   - sum sigma_c * floor(0.5 * sum_{i in Ic'} q_i)
        shift = 0.0

        sigma_items = self.cut_state.sigma_by_subset.items()
        if not sigma_items:
            return shift

        comp = layer.shape_params.get("compartment", "lower")

        for (cut_comp, subset), sigma in sigma_items:
            if cut_comp != comp:
                continue
                
            val = 0
            for i in subset:
                total_i = self.max_total_by_type.get(i, 0)
                if total_i <= 0:
                    continue
                # In standard 3-SR, we sum the quantities of cars in the subset and divide by 2.
                # We assume a car occupies >= half of the wagon/layer, so quantity > 0 means it's present.
                if quantities.get(i, 0) > 0:
                    val += quantities[i]
                    
            coeff = floor(0.5 * val)
            shift -= float(sigma) * coeff

        return shift
