import sys
from pathlib import Path
from math import inf

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config
from src.model.BPC.labeling import (
    LayerSpec,
    DualValues,
    LabelingOptions,
    generate_layer_patterns,
)
from src.model.BPC.feasibility_check import HierarchicalBSEvaluator
import time

print("--- Testing Labeling / Layer Pattern Generation ---")

car_types = [101, 102, 103]
car_lengths = {101: 4300.0, 102: 4500.0, 103: 4800.0}
car_heights = {101: 1500.0, 102: 1750.0, 103: 1900.0}
max_qty = {101: 2, 102: 2, 103: 2}

layer = LayerSpec(
    layer_id="upper_m-m",
    car_types=car_types,
    car_lengths=car_lengths,
    layer_length_limit=Config.top_len,
    car_heights=car_heights,
    shape_params={"deck": "m-m", "compartment": "upper"},
    max_quantity_by_type=max_qty
)

duals = DualValues(
    alpha={101: 1000.0, 102: 1500.0, 103: 2000.0},
    beta={101: 0.0, 102: 0.0, 103: 0.0},
    gamma=5000.0 
)

bs_evaluator = HierarchicalBSEvaluator()
options = LabelingOptions(use_dominance=True, max_units_per_type=2)

t0 = time.time()
patterns = generate_layer_patterns(layer, duals, bs_evaluator, options)
t1 = time.time()

print(f"Generated {len(patterns)} patterns in {t1-t0:.4f} seconds.")

patterns.sort(key=lambda p: p.reduced_cost)

print("\nTop 5 Patterns (Best Reduced Cost):")
for i, p in enumerate(patterns[:5]):
    clean_q = {k: v for k, v in p.quantities.items() if v > 0}
    print(f"  {i+1}. RC: {p.reduced_cost:.2f}, Len: {p.best_length}, Qty: {clean_q}")
