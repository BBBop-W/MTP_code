from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.model.BPC_LayerMaster.feasibility_check as feasibility_check
from src.model.BPC_LayerMaster.BBtree import BBTree
from src.model.gurobi import build_and_solve


@dataclass(frozen=True)
class SmallCase:
    name: str
    cars: List[dict]
    carriage_num: int
    num_splits: int = 1


def _write_instance(root: Path, case: SmallCase) -> Path:
    instance_dir = root / case.name
    instance_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case.cars).to_csv(instance_dir / "cars.csv", index=False)
    pd.DataFrame({"carriage_num": [case.carriage_num]}).to_csv(instance_dir / "carriage.csv", index=False)
    return instance_dir


def _solve_layermaster(instance_dir: Path, num_splits: int):
    feasibility_check.GLOBAL_NUM_SPLITS = num_splits
    feasibility_check.GLOBAL_INDEP_MODE = True
    feasibility_check._SEGMENTS_CACHE.clear()
    tree = BBTree(
        instance_dir=instance_dir,
        output_root=Path("/private/tmp/mtp_layermaster_small_align"),
        max_nodes=100,
        max_cg_iters=500,
        log_to_console=False,
        use_dominance=True,
        use_cuts=False,
        use_rc_bound=True,
        use_residual_profile=True,
        use_outer_inner_profile=False,
        use_height_order=True,
        use_local_residual_skyline=True,
        residual_profile_mode="full",
        compute_reachable_types=False,
        print_bb_progress=False,
        print_subproblem_progress=False,
    )
    t0 = time.perf_counter()
    result = tree.solve()
    return result, time.perf_counter() - t0, tree


def run_case(root: Path, case: SmallCase) -> None:
    instance_dir = _write_instance(root, case)
    compact_out = Path("/private/tmp/mtp_compact_small_align") / case.name
    shutil.rmtree(compact_out, ignore_errors=True)

    t0 = time.perf_counter()
    compact = build_and_solve(
        instance_dir,
        compact_out,
        log_to_console=False,
        num_splits=case.num_splits,
        independent_mode_split=True,
        time_limit=60,
        mip_gap=1e-8,
    )
    compact_time = time.perf_counter() - t0

    layer_result, layer_time, tree = _solve_layermaster(instance_dir, case.num_splits)
    compact_obj = compact.get("obj_val")
    layer_obj = layer_result.best_objective
    aligned = compact_obj is not None and layer_obj is not None and abs(compact_obj + layer_obj) <= 1e-5

    print(f"\n=== {case.name} ===")
    print(
        "compact: "
        f"status={compact.get('status')}, obj={compact_obj}, bound={compact.get('obj_bound')}, "
        f"time={compact_time:.3f}s"
    )
    print(
        "layermaster: "
        f"obj={layer_obj}, nodes={layer_result.explored_nodes}, generated={layer_result.generated_columns}, "
        f"time={layer_time:.3f}s, labels={tree.cg_engine.stats.labeling_time:.3f}s"
    )
    print(f"aligned={aligned}")
    if not aligned:
        raise AssertionError(f"Objective mismatch for {case.name}: compact={compact_obj}, layermaster={layer_obj}")


def main() -> None:
    root = Path("/private/tmp/mtp_small_align_instances")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    cases = [
        SmallCase(
            name="tiny_2types_1wagon",
            carriage_num=1,
            num_splits=1,
            cars=[
                {"program": "A", "model": "a1", "length": 4500, "height": 1500, "optional": 1, "mandatory": 1},
                {"program": "B", "model": "b1", "length": 5200, "height": 1800, "optional": 1, "mandatory": 1},
            ],
        ),
        SmallCase(
            name="tiny_3types_2wagons",
            carriage_num=2,
            num_splits=2,
            cars=[
                {"program": "A", "model": "a1", "length": 4200, "height": 1450, "optional": 2, "mandatory": 1},
                {"program": "B", "model": "b1", "length": 5100, "height": 1700, "optional": 2, "mandatory": 1},
                {"program": "C", "model": "c1", "length": 5800, "height": 1980, "optional": 1, "mandatory": 1},
            ],
        ),
        SmallCase(
            name="tiny_4types_2wagons",
            carriage_num=2,
            num_splits=3,
            cars=[
                {"program": "A", "model": "a1", "length": 4100, "height": 1425, "optional": 2, "mandatory": 1},
                {"program": "B", "model": "b1", "length": 4700, "height": 1600, "optional": 2, "mandatory": 1},
                {"program": "C", "model": "c1", "length": 5350, "height": 1800, "optional": 1, "mandatory": 1},
                {"program": "D", "model": "d1", "length": 6100, "height": 2050, "optional": 1, "mandatory": 0},
            ],
        ),
    ]
    for case in cases:
        run_case(root, case)


if __name__ == "__main__":
    main()
