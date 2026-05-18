from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import floor
from typing import Dict, List, Tuple

from src.model.BPC_compartment.labeling import CutEvaluator, DualValues, CompartmentSpec


@dataclass(frozen=True)
class CutState:
    """Dual-like coefficients for optional pricing cuts.

    eta_sum: dual of the generalized wagon-count lower-bound cut.
    sigma_by_subset: dual values for global 3-SR cuts, keyed by sorted type tuples.
    """

    eta_sum: float = 0.0
    sigma_by_subset: Dict[Tuple[int, int, int], float] = field(default_factory=dict)


class SimpleCutEvaluator(CutEvaluator):
    """Cut evaluator for compartment pricing."""

    def __init__(
        self,
        max_total_by_type: Dict[int, int],
        wagon_capacity: int | None = None,
        cut_state: CutState | None = None,
    ) -> None:
        self.max_total_by_type = {int(k): int(v) for k, v in max_total_by_type.items()}
        self.wagon_capacity = wagon_capacity
        self.cut_state = cut_state or CutState()

    def is_feasible(self, compartment_spec: CompartmentSpec, quantities: Dict[int, int]) -> bool:
        for i, q in quantities.items():
            if q < 0:
                return False
            # We don't artificially restrict the absolute limit here anymore
            # because the dynamic segments define exact packing capacity.

        return True

    def reduced_cost_shift(self, compartment_spec: CompartmentSpec, quantities: Dict[int, int], duals: DualValues) -> float:
        shift = 0.0
        if compartment_spec.shape_params.get("compartment", "lower") == "upper":
            shift -= float(self.cut_state.eta_sum)

        sigma_items = self.cut_state.sigma_by_subset.items()
        if not sigma_items:
            return shift

        for subset, sigma in sigma_items:
            val = 0
            for i in subset:
                total_i = self.max_total_by_type.get(i, 0)
                if total_i > 0 and quantities.get(i, 0) > total_i / 2.0:
                    val += 1

            coeff = floor(0.5 * val)
            shift -= float(sigma) * coeff

        return shift
