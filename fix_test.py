import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.feasibility_check import check_layer_bs, check_layer_gurobi
from src.utility.config import config as Config

car_lengths = {2: 4715.0, 3: 5230.0, 4: 3905.0}
car_heights = {2: 1715.0, 3: 2070.0, 4: 1960.0}
q_top = {2: 3, 3: 1, 4: 1} 

# This route was feasible in C++, but Gurobi and _simple_check_layer_bs rejected it.
# Let's analyze the C++ code vs Python code for this specific case.
# C++ placed them at:
# 2: 0 -> 4715 (in D_left, valid since 1715 < 1720)
# 2: 5115 -> 9830 (in E, valid)
# 3: 10230 -> 15460 (in E, valid)
# 4: 15860 -> 19765 (in E, valid)
# 2: 20165 -> 24880 (in D_right, valid)

# Wait... the 4th car ends at 19765. E region ends at 19900.
# So Cars 2, 3, 4 are entirely inside E region!
# And the 5th car starts at 20165. D_right starts at 19900.
# So the 5th car is entirely in D_right.

# So 1 car is in D_left, 3 cars are in E, 1 car is in D_right.
# Let's check Gurobi model limits.
# L[E] = 14900.
# Length of 3 cars in E: (4715+400) + (5230+400) + (3905+400) = 5115 + 5630 + 4305 = 15050.
# Wait! 15050 > L[E] + 400 (15300)? No, 15050 < 15300.
# So Gurobi's E capacity constraint IS satisfied: 15050 <= 15300.

# Why did Gurobi reject it?
# Let's trace test_gurobi_c.py again.
