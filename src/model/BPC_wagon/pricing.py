from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.model.BPC_wagon.cuts import CutState, SimpleCutEvaluator
import src.model.BPC_layer.feasibility_check as layer_feasibility
from src.model.BPC_layer.feasibility_check import HierarchicalBSEvaluator
from src.model.BPC_layer.labeling import (
    DualValues,
    LabelingOptions,
    LabelingStats,
    LayerPattern,
    generate_layer_patterns,
    generate_layer_patterns_outer_inner,
    generate_layer_patterns_residual,
)
from src.model.BPC_wagon.layer_specs import LayerRunItem, build_layer_sequence
from src.model.BPC_wagon.CG import MasterLPSolution, MasterProblem, PatternColumn
from src.model.BPC_wagon.merge import MergedPattern, merge_feasible_patterns
from src.utility.config import config as Config


@dataclass
class PricingStats:
    generated_subpatterns: int = 0
    merge_attempt_pairs: int = 0
    labeling_time: float = 0.0
    bs_time: float = 0.0
    merge_time: float = 0.0
    labels_feasible: int = 0
    labels_pruned_by_bound: int = 0
    labels_pruned_by_dominance: int = 0
    labels_pruned_by_local_skyline: int = 0
    labels_after_dominance: int = 0
    reachability_probes: int = 0


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
        use_rc_bound: bool = False,
        use_residual_profile: bool = True,
        use_outer_inner_profile: bool = False,
        use_height_order: bool = True,
        use_local_residual_skyline: bool = True,
        residual_profile_mode: str = "full",
        compute_reachable_types: bool = False,
        num_splits: int = 1,
        independent_mode_split: bool = True,
        max_units_per_type: int = Config.max_units_per_compartment,
        wagon_capacity_cut: int | None = 10,
        max_columns_per_pricing: int = 20,
        verbose: bool = True,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.bs = HierarchicalBSEvaluator(compute_reachable_types=compute_reachable_types)
        self.options = LabelingOptions(
            use_dominance=use_dominance,
            use_cuts=use_cuts,
            use_rc_bound=use_rc_bound,
            use_residual_profile=use_residual_profile,
            use_outer_inner_profile=use_outer_inner_profile,
            use_height_order=use_height_order,
            use_local_residual_skyline=use_local_residual_skyline,
            residual_profile_mode=residual_profile_mode,
            compute_reachable_types=compute_reachable_types,
            max_units_per_type=max_units_per_type,
        )
        self.wagon_capacity_cut = wagon_capacity_cut
        self.max_columns_per_pricing = int(max_columns_per_pricing)
        self.num_splits = int(num_splits)
        self.independent_mode_split = bool(independent_mode_split)
        self.verbose = verbose
        self.logger = logger
        self._column_seq = 0
        self._seen_signatures: Set[Tuple[int, ...]] = set()
        self._dominance_support_types: Tuple[int, ...] = ()
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

        layer_feasibility.GLOBAL_NUM_SPLITS = self.num_splits
        layer_feasibility.GLOBAL_INDEP_MODE = self.independent_mode_split
        layer_feasibility._SEGMENTS_CACHE.clear()

        self._log(
            f"Start pricing: alpha/beta size={len(lp_solution.dual_alpha)}/{len(lp_solution.dual_beta)}, "
            f"gamma={lp_solution.dual_gamma:.6f}, use_cuts={self.options.use_cuts}, use_dominance={self.options.use_dominance}"
        )

        duals = DualValues(
            alpha=lp_solution.dual_alpha, 
            beta=lp_solution.dual_beta, 
            gamma=lp_solution.dual_gamma,
            # a_i belongs to a merged full-wagon column, so it is priced after merge.
            branch_a={},
            branch_q=lp_solution.dual_branch_q
        )
        self._dominance_support_types = tuple(
            sorted(i for i, value in lp_solution.dual_branch_a.items() if abs(value) > self.options.eps)
        )
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

        # Reset timers
        self.stats.labeling_time = 0.0
        self.stats.bs_time = 0.0
        self.stats.merge_time = 0.0
        self.stats.labels_feasible = 0
        self.stats.labels_pruned_by_bound = 0
        self.stats.labels_pruned_by_dominance = 0
        self.stats.labels_pruned_by_local_skyline = 0
        self.stats.labels_after_dominance = 0
        self.stats.reachability_probes = 0

        subpatterns: Dict[Tuple[str, str], List[LayerPattern]] = {}
        merged_candidates: List[MergedPattern] = []
        for item in sequence:
            self._log(f"Solve subproblem: deck={item.deck}, compartment={item.compartment}, layer={item.layer.layer_id}")
            
            t0 = time.time()
            lab_stats = LabelingStats()
            patterns = self._generate_aligned_layer_patterns(item, duals, cut_evaluator, lab_stats)
            t_lab = time.time() - t0
            self.stats.labeling_time += t_lab
            self._accumulate_labeling_stats(lab_stats)
            
            # Extract BS time collected inside HierarchicalBSEvaluator
            self.stats.bs_time += self.bs.accumulated_time
            self.stats.labeling_time -= self.bs.accumulated_time # Remove BS time from pure labeling time
            self.bs.accumulated_time = 0.0 # reset

            self.stats.generated_subpatterns += len(patterns)
            self._log(f"Subproblem done: layer={item.layer.layer_id}, patterns={len(patterns)}")
            subpatterns[(item.deck, item.compartment)] = patterns

            # Merge right after a pair is available.
            if item.compartment != "lower":
                continue

            t0 = time.time()
            merged = self._try_merge_pair(
                deck=item.deck,
                upper_patterns=subpatterns.get((item.deck, "upper"), []),
                lower_patterns=patterns,
                max_total_by_type=master.U,
                lp_solution=lp_solution,
                master=master,
            )
            self.stats.merge_time += (time.time() - t0)

            if merged is None:
                self._log(f"Merging failed for deck={item.deck}, continue next pair")
                continue

            merged_candidates.extend(merged)

        if not merged_candidates:
            self._log("Pricing finished: no negative reduced-cost column found")
            return []

        columns: List[PatternColumn] = []
        sorted_candidates = sorted(merged_candidates, key=lambda candidate: candidate.reduced_cost)
        for merged in sorted_candidates:
            column = self._to_column(merged, master)
            if column is None:
                continue
            columns.append(column)
            if len(columns) >= self.max_columns_per_pricing:
                break
        if not columns:
            self._log("Merging produced only duplicate/signature-repeated columns")
            return []
        self._log(f"Merging success: columns={len(columns)}, best_rc={sorted_candidates[0].reduced_cost:.6f}")
        return columns

    def _generate_aligned_layer_patterns(
        self,
        item: LayerRunItem,
        duals: DualValues,
        cut_evaluator: SimpleCutEvaluator | None,
        lab_stats: LabelingStats,
    ) -> List[LayerPattern]:
        options = self.options
        if self._dominance_support_types and options.use_dominance:
            options = replace(options, dominance_support_types=self._dominance_support_types)
        if self.options.use_outer_inner_profile and not self.options.use_cuts:
            return generate_layer_patterns_outer_inner(
                layer=item.layer,
                duals=duals,
                options=options,
                stats=lab_stats,
            )
        if options.use_residual_profile and not options.use_cuts:
            return generate_layer_patterns_residual(
                layer=item.layer,
                duals=duals,
                options=options,
                stats=lab_stats,
            )
        return generate_layer_patterns(
            layer=item.layer,
            duals=duals,
            bs=self.bs,
            options=options,
            cut_evaluator=cut_evaluator,
            stats=lab_stats,
        )

    def _accumulate_labeling_stats(self, lab_stats: LabelingStats) -> None:
        self.stats.labels_feasible += lab_stats.labels_feasible
        self.stats.labels_pruned_by_bound += lab_stats.labels_pruned_by_bound
        self.stats.labels_pruned_by_dominance += lab_stats.labels_pruned_by_dominance
        self.stats.labels_pruned_by_local_skyline += lab_stats.labels_pruned_by_local_skyline
        self.stats.labels_after_dominance += lab_stats.labels_after_dominance
        self.stats.reachability_probes += lab_stats.reachability_probes

    def _try_merge_pair(
        self,
        deck: str,
        upper_patterns: List[LayerPattern],
        lower_patterns: List[LayerPattern],
        max_total_by_type: Dict[int, int],
        lp_solution: MasterLPSolution,
        master: MasterProblem,
    ) -> List[MergedPattern]:
        self.stats.merge_attempt_pairs += 1
        self._log(
            f"Try merge pair: deck={deck}, upper_patterns={len(upper_patterns)}, lower_patterns={len(lower_patterns)}"
        )
        if not upper_patterns or not lower_patterns:
            return []
        merged = merge_feasible_patterns(
            upper_patterns=upper_patterns,
            lower_patterns=lower_patterns,
            max_total_by_type=max_total_by_type,
            require_negative_reduced_cost=True,
            reduced_cost_fn=lambda _u, _l, q: self._full_column_reduced_cost(q, lp_solution, master),
            forbidden_signatures=self._seen_signatures,
            signature_order=master.I,
            max_results=self.max_columns_per_pricing,
        )
        return [candidate for candidate in merged if not candidate.deck or candidate.deck == deck]

    def _full_column_reduced_cost(
        self,
        quantities: Dict[int, int],
        lp_solution: MasterLPSolution,
        master: MasterProblem,
    ) -> float:
        rc = -float(lp_solution.dual_gamma or 0.0)
        for i in master.I:
            q = int(quantities.get(i, 0))
            if q <= 0:
                continue
            rc -= (
                master.length[i]
                + lp_solution.dual_alpha.get(i, 0.0)
                + lp_solution.dual_beta.get(i, 0.0)
                + lp_solution.dual_branch_q.get(i, 0.0)
            ) * q
            rc -= lp_solution.dual_branch_a.get(i, 0.0)
        return rc

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
