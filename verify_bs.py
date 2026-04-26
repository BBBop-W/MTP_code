import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.feasibility_check import check_layer_bs, check_layer_gurobi
car_lengths = {2: 4715.0, 3: 5230.0, 4: 3905.0}
car_heights = {2: 1715.0, 3: 2070.0, 4: 1960.0}
q_top = {2: 3, 3: 1, 4: 1} 

res_grb = check_layer_gurobi("upper", "m-m", q_top, car_lengths, car_heights)
print(f"GRB: {res_grb}")

res_bs = check_layer_bs("upper", "m-m", q_top, car_lengths, car_heights)
print(f"BS: {res_bs}")

