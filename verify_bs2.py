import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.feasibility_check import check_layer_bs, check_layer_gurobi
from src.utility.config import config as Config

car_lengths = {1: 4700.0, 6: 4385.0} # bottom cars
car_heights = {1: 1670.0, 6: 1835.0}
q_bot = {1: 2, 6: 3} 

# This is the failing bottom layer. Let's see what Gurobi says.
res_grb = check_layer_gurobi("lower", "m-m", q_bot, car_lengths, car_heights)
print(f"GRB: {res_grb}")

