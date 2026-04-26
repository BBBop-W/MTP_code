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

model = gp.Model()
# model.Params.OutputFlag = 1

I = [2, 3, 4]
quantities = q_top

H = ["D_left", "D_right", "E"]
x = model.addVars(I, H, vtype=gp.GRB.INTEGER, lb=0, name="x")

model.addConstrs((gp.quicksum(x[i, h] for h in H) == quantities[i] for i in I), name="qty")

# heights
h_limits = {"D_left": Config.D_height_m, "D_right": Config.D_height_m, "E": Config.E_height}

for i in I:
    for h in H:
        if car_heights[i] > h_limits[h]:
            model.addConstr(x[i, h] == 0)

L = {"D_left": Config.D_len, "D_right": Config.D_len, "E": Config.E_len}
Delta = 400.0

# lengths
# total E length
model.addConstr(gp.quicksum(x[i, "E"] * (car_lengths[i] + Delta) for i in I) <= L["E"] + Delta)

# D left + E
model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in ["D_left", "E"]) <= L["D_left"] + L["E"])

# D right + E
model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in ["D_right", "E"]) <= L["D_right"] + L["E"])

# all
model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H) <= sum(L[h] for h in H) - Delta)

model.optimize()
if model.status == gp.GRB.OPTIMAL:
    print("Gurobi solution found:")
    for i in I:
        for h in H:
            if x[i,h].x > 0.5:
                print(f"Car {i} -> {h} : {x[i,h].x}")
else:
    print("Infeasible")
