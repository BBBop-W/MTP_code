from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

from Conf import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC_layer.feasibility_check import dynamic_recursive_bs


def IsFeasible(c, p):
    return IsFeasible_Length(c, p)


def IsFeasible_Length(c, p):
    position = c.position
    spacing = c.spacing
    return IsFeasible_route(p, c.route[0], (position, spacing, 0)) and IsFeasible_route(p, c.route[1], (position, spacing, 1))


def IsFeasible_Height(c, p):
    return IsFeasible_Length(c, p)


def IsFeasible_route(p, route, info):
    return IsFeasible_Length_route(p, route, info)


def IsFeasible_Length_route(p, route, info):
    position, _spacing, floor = info
    if len(route) > Config.max_units_per_compartment:
        return False
    if not route:
        return True

    deck_mode = _deck_mode(position)
    compartment = "upper" if floor == 0 else "lower"
    quantities = _route_quantities(route)
    lengths, heights = _vehicle_geometry(p)
    return dynamic_recursive_bs(
        compartment=compartment,
        deck_mode=deck_mode,
        quantities=quantities,
        car_lengths=lengths,
        car_heights=heights,
    ) is not None


def IsFeasible_Height_route(p, route, info):
    return IsFeasible_Length_route(p, route, info)


def _deck_mode(position: int) -> str:
    return "h-h" if int(position) == 0 else "m-m"


def _route_quantities(route: List[int]) -> Dict[int, int]:
    quantities: Dict[int, int] = {}
    for vehicle_id in route:
        quantities[vehicle_id] = quantities.get(vehicle_id, 0) + 1
    return quantities


def _vehicle_geometry(p) -> Tuple[Dict[int, float], Dict[int, float]]:
    lengths = {v.id: float(v.length) for v in p.vehicle}
    heights = {v.id: float(v.height) for v in p.vehicle}
    return lengths, heights
