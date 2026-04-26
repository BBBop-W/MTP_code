import sys
from pathlib import Path
import gurobipy as gp

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config

car_lengths = {2: 4715.0, 3: 5230.0, 4: 3905.0}
car_heights = {2: 1715.0, 3: 2070.0, 4: 1960.0}
q_top = {2: 3, 3: 1, 4: 1} 

# Gurobi check adds Delta to everything, so it checks:
# sum_E = 3 * (Length + 400)
# sum_E = (4715+400) + (5230+400) + (3905+400)
L_E_used = (4715 + 400) + (5230 + 400) + (3905 + 400)
print(f"Gurobi thinks E uses: {L_E_used}")
print(f"Gurobi checks against: {Config.E_len + 400}")

