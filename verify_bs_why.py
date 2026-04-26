import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.feasibility_check import check_layer_bs
car_lengths = {2: 4715.0, 3: 5230.0, 4: 3905.0}
car_heights = {2: 1715.0, 3: 2070.0, 4: 1960.0}
q_top = {2: 3, 3: 1, 4: 1} 

res_bs = check_layer_bs("upper", "m-m", q_top, car_lengths, car_heights)
print(f"Upper BS: {res_bs}")

car_lengths_bot = {1: 4700.0, 6: 4385.0}
car_heights_bot = {1: 1670.0, 6: 1835.0}
q_bot = {1: 2, 6: 3} 

res_bs_bot = check_layer_bs("lower", "m-m", q_bot, car_lengths_bot, car_heights_bot)
print(f"Lower BS: {res_bs_bot}")
