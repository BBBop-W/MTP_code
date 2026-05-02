import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC_LayerMaster.CG import MasterProblem, MasterLPSolution, PatternColumn
from src.model.BPC_LayerMaster.labeling import (
    LayerSpec,
    generate_layer_patterns,
    DualValues,
    LabelingOptions,
)
from src.model.BPC_LayerMaster.feasibility_check import HierarchicalBSEvaluator
from src.utility.config import config as Config

@dataclass
class PricingStats:
    labeling_time: float = 0.0
    bs_time: float = 0.0

@dataclass
class PricingOptions:
    use_dominance: bool = True
    use_cuts: bool = False

class EarlyStopPricingEngine:
    def __init__(self, use_dominance: bool = True, use_cuts: bool = False, verbose: bool = False):
        self.options = PricingOptions(use_dominance=use_dominance, use_cuts=use_cuts)
        self.verbose = verbose
        self.evaluator = HierarchicalBSEvaluator()
        self.wagon_capacity_cut: Optional[int] = None
        self.stats = PricingStats()
        self._col_counter = 0

    def generate_columns(
        self, solution: MasterLPSolution, master: MasterProblem
    ) -> List[PatternColumn]:
        self.stats.labeling_time = 0.0
        self.stats.bs_time = 0.0
        self.evaluator.accumulated_time = 0.0

        modes = ["h-h", "h-m", "m-h", "m-m"]
        new_columns = []

        from src.model.BPC_LayerMaster.cuts import SimpleCutEvaluator, CutState
        cut_evaluator = None
        if self.options.use_cuts:
            cut_state = CutState(sigma_by_subset=solution.dual_sigma)
            max_totals = {i: 6 for i in master.I}
            cut_evaluator = SimpleCutEvaluator(max_total_by_type=max_totals, cut_state=cut_state)

        for p in modes:
            gamma_p = solution.dual_gamma.get(p, 0.0)
            kappa = solution.dual_kappa

            # UPPER
            layer_id_u = f"upper_{p}"
            spec_u = LayerSpec(
                layer_id=layer_id_u,
                car_types=master.I,
                car_lengths=master.length,
                car_heights={i: float(master.car_info.iloc[i-1]["height"]) for i in master.I},
                layer_length_limit=Config.top_len,
                shape_params={"compartment": "upper", "deck": p},
                max_quantity_by_type={i: 6 for i in master.I},
            )

            duals_u = DualValues(
                alpha=solution.dual_alpha,
                beta=solution.dual_beta,
                gamma=2.0 * (gamma_p + kappa), 
                branch_a={},
                branch_q=solution.dual_branch_q,
            )

            t_start = time.time()
            patterns_u = generate_layer_patterns(
                layer=spec_u,
                duals=duals_u,
                bs=self.evaluator,
                options=LabelingOptions(use_dominance=self.options.use_dominance, use_cuts=self.options.use_cuts),
                cut_evaluator=cut_evaluator
            )
            self.stats.labeling_time += (time.time() - t_start)

            for pat in patterns_u:
                if pat.reduced_cost < -1e-5:
                    self._col_counter += 1
                    col = PatternColumn(
                        column_id=f"new_{self._col_counter}",
                        q=pat.quantities,
                        cost=-sum(master.length[i]*qty for i, qty in pat.quantities.items()),
                        compartment="upper",
                        deck_mode=p,
                        metadata={"source": "pricing", "rc": pat.reduced_cost}
                    )
                    new_columns.append(col)

            # LOWER
            layer_id_l = f"lower_{p}"
            spec_l = LayerSpec(
                layer_id=layer_id_l,
                car_types=master.I,
                car_lengths=master.length,
                car_heights={i: float(master.car_info.iloc[i-1]["height"]) for i in master.I},
                layer_length_limit=Config.bottom_len,
                shape_params={"compartment": "lower", "deck": p},
                max_quantity_by_type={i: 6 for i in master.I},
            )

            # For lower: we want + gamma_p instead of - gamma_p. And no kappa.
            # So root RC needs to be + gamma_p.
            # `-duals.gamma / 2.0` = `gamma_p` => `duals.gamma` = `-2.0 * gamma_p`
            t_start = time.time()
            
            duals_l = DualValues(
                alpha=solution.dual_alpha,
                beta=solution.dual_beta,
                gamma=-2.0 * gamma_p,
                branch_a={},
                branch_q=solution.dual_branch_q,
            )
            
            patterns_l = generate_layer_patterns(
                layer=spec_l,
                duals=duals_l,
                bs=self.evaluator,
                options=LabelingOptions(use_dominance=self.options.use_dominance, use_cuts=self.options.use_cuts),
                cut_evaluator=cut_evaluator
            )
            self.stats.labeling_time += (time.time() - t_start)

            for pat in patterns_l:
                if pat.reduced_cost < -1e-5:
                    self._col_counter += 1
                    col = PatternColumn(
                        column_id=f"new_{self._col_counter}",
                        q=pat.quantities,
                        cost=-sum(master.length[i]*qty for i, qty in pat.quantities.items()),
                        compartment="lower",
                        deck_mode=p,
                        metadata={"source": "pricing", "rc": pat.reduced_cost}
                    )
                    new_columns.append(col)

        self.stats.bs_time = self.evaluator.accumulated_time
        new_columns.sort(key=lambda c: c.metadata["rc"]) 
        return new_columns
