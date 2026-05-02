import math
import numpy as np
import pandas as pd

class CarriageGeometry:
    def __init__(self):
        # 1. 车辆与车厢全局参数
        self.carriage_length = 25000.0
        self.center_x = self.carriage_length / 2.0  # 12500.0
        self.roof_height = 4340.0
        
        # 2. 上下层甲板(Deck)的关键参数
        self.deck_h_height = 2270.0
        # 中间位甲板参数
        self.deck_m_groove_length = 11400.0
        self.deck_m_groove_start = self.center_x - (self.deck_m_groove_length / 2.0) # 12500 - 5700 = 6800.0
        self.deck_m_end_height = 4340.0 - 1780.0  # 2560.0
        
        # 3. 下层地板(Floor)的关键参数
        self.floor_end_height = 680.0
        self.floor_groove_length = 10867.0
        self.floor_groove_start = self.center_x - (self.floor_groove_length / 2.0)   # 12500 - 5433.5 = 7066.5
        self.slope_angle_deg = 9.0
        self.slope_ratio = math.tan(math.radians(self.slope_angle_deg)) # ~0.15838
        self.floor_slope_start = self.floor_groove_start - (self.floor_end_height / self.slope_ratio) # 7066.5 - 4293.3 = 2773.2

    # ==========================
    # 绝对高度函数 (0 <= x <= center_x)
    # ==========================
    def _deck_height(self, x: float, mode: str) -> float:
        if mode == 'h':
            return self.deck_h_height
        else: # 'm'
            if x >= self.deck_m_groove_start:
                return self.deck_h_height
            else:
                # 线性插值
                ratio = x / self.deck_m_groove_start
                return self.deck_m_end_height - ratio * (self.deck_m_end_height - self.deck_h_height)

    def _floor_height(self, x: float) -> float:
        if x <= self.floor_slope_start:
            return self.floor_end_height
        elif x >= self.floor_groove_start:
            return 0.0
        else:
            # 线性下降
            return self.floor_end_height - (x - self.floor_slope_start) * self.slope_ratio
            
    def _upper_roof_height(self, x: float) -> float:
        return self.roof_height

    # ==========================
    # 净高函数 H(x)
    # ==========================
    def get_clearance(self, x: float, layer: str, mode: str) -> float:
        """获取指定 x 坐标处的绝对净高"""
        if x > self.center_x:
             x = self.carriage_length - x  # 利用对称性

        if layer == 'upper':
            return self._upper_roof_height(x) - self._deck_height(x, mode)
        elif layer == 'lower':
            return self._deck_height(x, mode) - self._floor_height(x)
        else:
            raise ValueError("layer must be 'upper' or 'lower'")

    # ==========================
    # 切分算法
    # ==========================
    def solve_x_for_height(self, target_h: float, layer: str, mode: str) -> float:
        """
        求解方程 H(x) = target_h，返回交点的 x 坐标 (左半边)
        如果目标高度大于最大可用高度，返回 None。
        """
        h_min = self.get_clearance(0, layer, mode)
        h_max = self.get_clearance(self.center_x, layer, mode)
        
        if target_h <= h_min + 0.1: 
            return 0.0
        if target_h > h_max + 0.1:
            return None

        left, right = 0.0, self.center_x
        for _ in range(50):
            mid = (left + right) / 2.0
            h_mid = self.get_clearance(mid, layer, mode)
            if h_mid < target_h:
                left = mid
            else:
                right = mid
        return right

    def generate_center_out_segments(self, input_targets: list, layer: str):
        """
        根据给定的 (高度, 模式) 列表，从中心向两端对称生成分块。
        """
        if layer == 'upper':
            central_len = self.deck_m_groove_length # 11400.0
        else:
            central_len = self.floor_groove_length  # 10867.0
            
        central_start_x = self.center_x - (central_len / 2.0)
        
        cut_points = [0.0, central_start_x]
        
        for h, mode in input_targets:
            x = self.solve_x_for_height(h, layer, mode)
            if x is not None and 0 < x < central_start_x:
                cut_points.append(x)
                
        cut_points = sorted(list(set([round(cp, 2) for cp in cut_points])))
        
        segments = []
        
        c_h = self.get_clearance(self.center_x, layer, 'h')
        c_m = self.get_clearance(self.center_x, layer, 'm')
        segments.append({
            "name": "central",
            "len": round(central_len, 2),
            "h_h": round(c_h, 2),
            "h_m": round(c_m, 2)
        })
        
        outward_idx = 1
        for i in range(len(cut_points) - 1, 0, -1):
            right_x = cut_points[i]
            left_x = cut_points[i-1]
            
            length = right_x - left_x
            if length <= 0: continue
            
            h_h = self.get_clearance(left_x, layer, 'h')
            h_m = self.get_clearance(left_x, layer, 'm')
            
            segments.append({
                "name": f"block_{outward_idx}",
                "len": round(length, 2),
                "h_h": round(h_h, 2),
                "h_m": round(h_m, 2)
            })
            outward_idx += 1
            
        return segments


def get_model_segments(car_info_df: pd.DataFrame, num_splits: int, independent_mode_split: bool) -> dict:
    """
    High-level API for Gurobi and BPC to get the dynamic segments based on car instance.
    """
    geom = CarriageGeometry()
    unique_heights = sorted(car_info_df["height"].unique().tolist())
    
    # 1. Select heights based on num_splits
    if num_splits > 0 and num_splits < len(unique_heights):
        indices = np.linspace(0, len(unique_heights) - 1, num_splits, dtype=int)
        selected_heights = [unique_heights[i] for i in indices]
    else:
        selected_heights = unique_heights

    # 2. Build input targets
    if independent_mode_split:
        input_targets = [(h, 'h') for h in selected_heights] + [(h, 'm') for h in selected_heights]
    else:
        input_targets = [(h, 'h') for h in selected_heights]

    # 3. Generate segments for both layers
    lower_segments = geom.generate_center_out_segments(input_targets, layer='lower')
    upper_segments = geom.generate_center_out_segments(input_targets, layer='upper')
    
    return {
        "lower": {
            "central": lower_segments[0],
            "blocks": lower_segments[1:]
        },
        "upper": {
            "central": upper_segments[0],
            "blocks": upper_segments[1:]
        }
    }

if __name__ == "__main__":
    # Test the API
    df = pd.DataFrame({"height": [1600, 1750, 1820, 1900, 2050]})
    res = get_model_segments(df, num_splits=2, independent_mode_split=True)
    import json
    print(json.dumps(res, indent=2))
