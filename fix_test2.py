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

# Let's recreate Gurobi exactly as in Python code:
# model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H6[k]) <= sum(L[h] for h in H6[k]))
# H6[left] = [D_left, E]
# Length constraint: sum_x(D_left + E) <= L[D_left] + L[E] = 5000 + 14900 = 19900.
# For our assignment: 1 in D_left, 3 in E.
# sum length: (4715+400) + 15050 = 5115 + 15050 = 20165.
# 20165 > 19900.
# Ah! Gurobi's constraint is violated!
print(f"H6[left] constraint uses: 20165")
print(f"H6[left] constraint limit: {Config.D_len + Config.E_len}")

