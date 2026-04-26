import sys
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.feasibility_check import check_layer_bs
from src.utility.generate_instance import load_candidates

instance = "m6c6"
col_path = PROJECT_ROOT / "result" / instance / "VNS" / "column.csv"
info_path = PROJECT_ROOT / "result" / instance / "VNS" / "carriage_info.json"

if not col_path.exists() or not info_path.exists():
    print("VNS output not found.")
    sys.exit(0)

# Load car params
cars_path = PROJECT_ROOT / "data/Instance" / instance / "cars.csv"
car_info = pd.read_csv(cars_path)
col_map = {"Brand": "program", "Model": "model", "Length": "length", "Height": "height"}
car_info = car_info.rename(columns=col_map)
car_lengths = {i+1: float(car_info.iloc[i]["length"]) for i in range(len(car_info))}
car_heights = {i+1: float(car_info.iloc[i]["height"]) for i in range(len(car_info))}

car_type_map = {}
for i in range(len(car_info)):
    b = str(car_info.iloc[i]["program"]).strip()
    m = str(car_info.iloc[i]["model"]).strip()
    car_type_map[(b, m)] = i + 1

# Load columns
df = pd.read_csv(col_path)
carriage_cols = [c for c in df.columns if c not in ["brand", "model"]]

with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

for idx, col_name in enumerate(carriage_cols):
    q_dict = {}
    for _, row in df.iterrows():
        b = str(row["brand"]).strip()
        m = str(row["model"]).strip()
        qty = int(row[col_name])
        if qty > 0:
            q_dict[car_type_map[(b, m)]] = qty
            
    deck_mode = info["carriage"][idx]["position"]
    
    # Re-evaluate with BS
    # We don't know the exact split between upper/lower from column.csv alone,
    # but we can look at carriage_info.json to see the exact car names.
    top_cars = info["carriage"][idx]["top"]
    bottom_cars = info["carriage"][idx]["bottom"]
    
    q_top = {k: 0 for k in car_lengths.keys()}
    q_bot = {k: 0 for k in car_lengths.keys()}
    
    for cname in top_cars:
        for (b, m), c_id in car_type_map.items():
            if m == cname or (b + " " + m) == cname: # JSON has just model, or brand+model? 
                q_top[c_id] += 1
                break
    for cname in bottom_cars:
        for (b, m), c_id in car_type_map.items():
            if m == cname or (b + " " + m) == cname:
                q_bot[c_id] += 1
                break
                
    # Clean zeros
    q_top = {k: v for k, v in q_top.items() if v > 0}
    q_bot = {k: v for k, v in q_bot.items() if v > 0}
    
    print(f"--- Carriage {idx+1} ({deck_mode}) ---")
    print(f"Top: {q_top}")
    if q_top:
        res_top = check_layer_bs("upper", deck_mode, q_top, car_lengths, car_heights)
        print(f"  BS check upper: {res_top}")
    print(f"Bottom: {q_bot}")
    if q_bot:
        res_bot = check_layer_bs("lower", deck_mode, q_bot, car_lengths, car_heights)
        print(f"  BS check lower: {res_bot}")
