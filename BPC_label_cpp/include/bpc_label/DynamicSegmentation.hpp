#pragma once

#include <vector>

#include "bpc_label/Config.hpp"

namespace bpc_label {

struct Segment {
    double length = 0.0;
    double h_h = 0.0;
    double h_m = 0.0;
};

struct LayerSegments {
    Segment central;
    std::vector<Segment> blocks;
};

struct ModelSegments {
    LayerSegments lower;
    LayerSegments upper;
};

class CarriageGeometry {
public:
    double get_clearance(double x, CompartmentKind layer, char mode) const;
    double solve_x_for_height(double target_h, CompartmentKind layer, char mode) const;
    LayerSegments generate_center_out_segments(
        const std::vector<std::pair<double, char>>& targets,
        CompartmentKind layer
    ) const;
    LayerSegments generate_even_side_segments(int num_blocks, CompartmentKind layer) const;

private:
    static constexpr double carriage_length_ = 25000.0;
    static constexpr double center_x_ = carriage_length_ / 2.0;
    static constexpr double roof_height_ = 4340.0;
    static constexpr double deck_h_height_ = 2270.0;
    static constexpr double deck_m_groove_length_ = 11400.0;
    static constexpr double deck_m_groove_start_ = center_x_ - deck_m_groove_length_ / 2.0;
    static constexpr double deck_m_end_height_ = 4340.0 - 1780.0;
    static constexpr double floor_end_height_ = 680.0;
    static constexpr double floor_groove_length_ = 10867.0;
    static constexpr double floor_groove_start_ = center_x_ - floor_groove_length_ / 2.0;
    static constexpr double slope_angle_deg_ = 9.0;

    double deck_height(double x, char mode) const;
    double floor_height(double x) const;
    static double round2(double value);
};

ModelSegments get_model_segments(
    std::vector<double> car_heights,
    int num_splits,
    bool independent_mode_split
);

}  // namespace bpc_label
