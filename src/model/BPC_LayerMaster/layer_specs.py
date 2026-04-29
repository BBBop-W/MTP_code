from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from model.BPC.labeling import LayerSpec
from src.utility.config import config as Config


@dataclass(frozen=True)
class LayerRunItem:
    deck: str
    compartment: str  # "upper" | "lower"
    layer: LayerSpec


def build_layer_sequence(
    car_types: List[int],
    car_lengths: Dict[int, float],
    car_heights: Dict[int, float],
    max_quantity_by_type: Dict[int, int],
    max_units_per_type: int = 6,
) -> List[LayerRunItem]:
    """Build the 8 layer runs in the required order.

    Order:
      upper h-h, lower h-h,
      upper m-m, lower m-m,
      upper m-h, lower m-h,
      upper h-m, lower h-m
    """

    seq: List[LayerRunItem] = []
    for deck in ["h-h", "m-m", "m-h", "h-m"]:
        seq.append(
            LayerRunItem(
                deck=deck,
                compartment="upper",
                layer=_build_upper_layer(deck, car_types, car_lengths, car_heights, max_quantity_by_type, max_units_per_type),
            )
        )
        seq.append(
            LayerRunItem(
                deck=deck,
                compartment="lower",
                layer=_build_lower_layer(deck, car_types, car_lengths, car_heights, max_quantity_by_type, max_units_per_type),
            )
        )
    return seq


def _deck_side_mode(deck: str) -> Dict[str, str]:
    if deck == "h-h":
        return {"left": "h", "right": "h"}
    if deck == "m-m":
        return {"left": "m", "right": "m"}
    if deck == "m-h":
        return {"left": "m", "right": "h"}
    if deck == "h-m":
        return {"left": "h", "right": "m"}
    raise ValueError(f"Unknown deck position: {deck}")


def _build_lower_layer(
    deck: str,
    car_types: List[int],
    car_lengths: Dict[int, float],
    car_heights: Dict[int, float],
    max_quantity_by_type: Dict[int, int],
    max_units_per_type: int,
) -> LayerSpec:
    mode = _deck_side_mode(deck)
    a_left_h = Config.A_height_h if mode["left"] == "h" else Config.A_height_m
    a_right_h = Config.A_height_h if mode["right"] == "h" else Config.A_height_m

    components = [
        {"component_id": "A_left", "height_limit": a_left_h, "lengths_by_k": {"left": Config.A_len}},
        {"component_id": "A_right", "height_limit": a_right_h, "lengths_by_k": {"right": Config.A_len}},
        {"component_id": "B_left", "height_limit": Config.B_height, "lengths_by_k": {"left": Config.B_len}},
        {"component_id": "B_right", "height_limit": Config.B_height, "lengths_by_k": {"right": Config.B_len}},
        {"component_id": "C", "height_limit": Config.C_height, "lengths_by_k": {"left": Config.C_len / 2.0, "right": Config.C_len / 2.0}},
    ]

    return LayerSpec(
        layer_id=f"lower_{deck}",
        car_types=list(car_types),
        car_lengths=dict(car_lengths),
        layer_length_limit=Config.bottom_len,
        car_heights=dict(car_heights),
        shape_params={"delta": 400.0, "deck": deck, "compartment": "lower", "components": components},
        max_quantity_by_type={i: min(max_units_per_type, int(max_quantity_by_type.get(i, max_units_per_type))) for i in car_types},
    )


def _build_upper_layer(
    deck: str,
    car_types: List[int],
    car_lengths: Dict[int, float],
    car_heights: Dict[int, float],
    max_quantity_by_type: Dict[int, int],
    max_units_per_type: int,
) -> LayerSpec:
    mode = _deck_side_mode(deck)
    d_left_h = Config.D_height_h if mode["left"] == "h" else Config.D_height_m
    d_right_h = Config.D_height_h if mode["right"] == "h" else Config.D_height_m

    components = [
        {"component_id": "D_left", "height_limit": d_left_h, "lengths_by_k": {"left": Config.D_len}},
        {"component_id": "D_right", "height_limit": d_right_h, "lengths_by_k": {"right": Config.D_len}},
        {"component_id": "E", "height_limit": Config.E_height, "lengths_by_k": {"left": Config.E_len / 2.0, "right": Config.E_len / 2.0}},
    ]

    return LayerSpec(
        layer_id=f"upper_{deck}",
        car_types=list(car_types),
        car_lengths=dict(car_lengths),
        layer_length_limit=Config.top_len,
        car_heights=dict(car_heights),
        shape_params={"delta": 400.0, "deck": deck, "compartment": "upper", "components": components},
        max_quantity_by_type={i: min(max_units_per_type, int(max_quantity_by_type.get(i, max_units_per_type))) for i in car_types},
    )
