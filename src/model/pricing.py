from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.model.cuts import CutState, SimpleCutEvaluator
from src.model.labeling import DualValues, HierarchicalBSEvaluator, LabelingOptions, LayerPattern, generate_layer_patterns
from src.model.layer_specs import LayerRunItem, build_layer_sequence
from src.model.master import MasterLPSolution, MasterProblem, PatternColumn
from src.model.merge import MergedPattern, merge_first_feasible


@dataclass
class PricingStats:
    generated_subpatterns: int = 0
    merge_attempt_pairs: int = 0


class EarlyStopPricingEngine:
    """Pricing engine with ordered labeling + early-stop merge.

    Order:
      upper h-h, lower h-h -> try merge;
      upper m-m, lower m-m -> try merge;
      upper m-h, lower m-h -> try merge;
      upper h-m, lower h-m -> try merge.
    """

    def __init__(
        self,
        use_dominance: bool = True,
        use_cuts: bool = False,
        max_units_per_type: int = 6,
        wagon_capacity_cut: int | None = 10,
        verbose: bool = True,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.bs = HierarchicalBSEvaluator()
        self.options = LabelingOptions(use_dominance=use_dominance, use_cuts=use_cuts, max_units_per_type=max_units_per_type)
        self.wagon_capacity_cut = wagon_capacity_cut
        self.verbose = verbose
        self.logger = logger
        self._column_seq = 0
        self._seen_signatures: Set[Tuple[int, ...]] = set()
        self.stats = PricingStats()
        self.cut_state = CutState()

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        if self.logger is not None:
            self.logger(f"[Pricing] {msg}")
        else:
            print(f"[Pricing] {msg}")

    def generate_columns(self, lp_solution: MasterLPSolution, master: MasterProblem) -> List[PatternColumn]:
        if lp_solution.dual_gamma is None:
            self._log("Skip pricing because dual gamma is unavailable")
            return []

        self._log(
            f"Start pricing: alpha/beta size={len(lp_solution.dual_alpha)}/{len(lp_solution.dual_beta)}, "
            f"gamma={lp_solution.dual_gamma:.6f}, use_cuts={self.options.use_cuts}, use_dominance={self.options.use_dominance}"
        )

        duals = DualValues(alpha=lp_solution.dual_alpha, beta=lp_solution.dual_beta, gamma=lp_solution.dual_gamma)
        car_heights = {i: float(master.car_info.iloc[i - 1]["height"]) for i in master.I}
        cut_evaluator = None
        if self.options.use_cuts:
            cut_evaluator = SimpleCutEvaluator(
                max_total_by_type=master.U,
                wagon_capacity=self.wagon_capacity_cut,
                cut_state=self.cut_state,
            )

        sequence = build_layer_sequence(
            car_types=master.I,
            car_lengths=master.length,
            car_heights=car_heights,
            max_quantity_by_type=master.U,
            max_units_per_type=self.options.max_units_per_type,
        )

        subpatterns: Dict[Tuple[str, str], List[LayerPattern]] = {}
        for item in sequence:
            self._log(f"Solve subproblem: deck={item.deck}, compartment={item.compartment}, layer={item.layer.layer_id}")
            patterns = generate_layer_patterns(
                layer=item.layer,
                duals=duals,
                bs=self.bs,
                options=self.options,
                cut_evaluator=cut_evaluator,
            )
            self.stats.generated_subpatterns += len(patterns)
            self._log(f"Subproblem done: layer={item.layer.layer_id}, patterns={len(patterns)}")
            subpatterns[(item.deck, item.compartment)] = patterns

            # Merge right after a pair is available.
            if item.compartment != "lower":
                continue

            merged = self._try_merge_pair(
                deck=item.deck,
                upper_patterns=subpatterns.get((item.deck, "upper"), []),
                lower_patterns=patterns,
                max_total_by_type=master.U,
            )
            if merged is None:
                self._log(f"Merging failed for deck={item.deck}, continue next pair")
                continue

            column = self._to_column(merged, master)
            if column is None:
                self._log(f"Merging produced duplicate/signature-repeated column at deck={item.deck}")
                continue
            self._log(f"Merging success: deck={item.deck}, column={column.column_id}, rc={merged.reduced_cost:.6f}")
            return [column]

        self._log("Pricing finished: no negative reduced-cost column found")
        return []

    def _try_merge_pair(
        self,
        deck: str,
        upper_patterns: List[LayerPattern],
        lower_patterns: List[LayerPattern],
        max_total_by_type: Dict[int, int],
    ) -> Optional[MergedPattern]:
        self.stats.merge_attempt_pairs += 1
        self._log(
            f"Try merge pair: deck={deck}, upper_patterns={len(upper_patterns)}, lower_patterns={len(lower_patterns)}"
        )
        if not upper_patterns or not lower_patterns:
            return None
        merged = merge_first_feasible(
            upper_patterns=upper_patterns,
            lower_patterns=lower_patterns,
            max_total_by_type=max_total_by_type,
            require_negative_reduced_cost=True,
        )
        if merged is None:
            return None
        if merged.deck and merged.deck != deck:
            self._log(f"Merge rejected by deck mismatch: merged={merged.deck}, expected={deck}")
            return None
        return merged

    def _to_column(self, merged: MergedPattern, master: MasterProblem) -> Optional[PatternColumn]:
        signature = tuple(int(merged.quantities.get(i, 0)) for i in master.I)
        if signature in self._seen_signatures:
            self._log("Candidate column rejected due to repeated signature")
            return None
        self._seen_signatures.add(signature)

        self._column_seq += 1
        col_id = f"cg_{merged.deck}_{self._column_seq}"
        cost = -sum(master.length[i] * merged.quantities.get(i, 0) for i in master.I)
        return PatternColumn(
            column_id=col_id,
            q={i: int(merged.quantities.get(i, 0)) for i in master.I},
            cost=float(cost),
            metadata={
                "source": "pricing_label_merge",
                "deck": merged.deck,
                "reduced_cost": f"{merged.reduced_cost:.6f}",
                "upper": merged.upper.layer_id,
                "lower": merged.lower.layer_id,
            },
        )
