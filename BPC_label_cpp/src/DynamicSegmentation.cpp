#include "bpc_label/DynamicSegmentation.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace bpc_label {

namespace {

double slope_ratio() {
    return std::tan(9.0 * std::acos(-1.0) / 180.0);
}

double central_length(CompartmentKind layer) {
    return layer == CompartmentKind::Upper ? 11400.0 : 10867.0;
}

double unique_round2(double value) {
    return std::round(value * 100.0) / 100.0;
}

}  // namespace

double CarriageGeometry::round2(double value) {
    return unique_round2(value);
}

double CarriageGeometry::deck_height(double x, char mode) const {
    if (mode == 'h') {
        return deck_h_height_;
    }
    if (x >= deck_m_groove_start_) {
        return deck_h_height_;
    }
    const double ratio = x / deck_m_groove_start_;
    return deck_m_end_height_ - ratio * (deck_m_end_height_ - deck_h_height_);
}

double CarriageGeometry::floor_height(double x) const {
    const double floor_slope_start = floor_groove_start_ - floor_end_height_ / slope_ratio();
    if (x <= floor_slope_start) {
        return floor_end_height_;
    }
    if (x >= floor_groove_start_) {
        return 0.0;
    }
    return floor_end_height_ - (x - floor_slope_start) * slope_ratio();
}

double CarriageGeometry::get_clearance(double x, CompartmentKind layer, char mode) const {
    if (x > center_x_) {
        x = carriage_length_ - x;
    }
    if (layer == CompartmentKind::Upper) {
        return roof_height_ - deck_height(x, mode);
    }
    return deck_height(x, mode) - floor_height(x);
}

double CarriageGeometry::solve_x_for_height(double target_h, CompartmentKind layer, char mode) const {
    const double h_min = get_clearance(0.0, layer, mode);
    const double h_max = get_clearance(center_x_, layer, mode);
    if (target_h <= h_min + 0.1) {
        return 0.0;
    }
    if (target_h > h_max + 0.1) {
        return -1.0;
    }

    double left = 0.0;
    double right = center_x_;
    for (int iter = 0; iter < 50; ++iter) {
        const double mid = (left + right) / 2.0;
        if (get_clearance(mid, layer, mode) < target_h) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return right;
}

LayerSegments CarriageGeometry::generate_center_out_segments(
    const std::vector<std::pair<double, char>>& targets,
    CompartmentKind layer
) const {
    const double c_len = central_length(layer);
    const double central_start_x = center_x_ - c_len / 2.0;

    std::set<double> cut_points;
    cut_points.insert(0.0);
    cut_points.insert(round2(central_start_x));

    for (const auto& [height, mode] : targets) {
        const double x = solve_x_for_height(height, layer, mode);
        if (x > 0.0 && x < central_start_x) {
            cut_points.insert(round2(x));
        }
    }

    std::vector<double> cuts(cut_points.begin(), cut_points.end());
    std::sort(cuts.begin(), cuts.end());

    LayerSegments out;
    out.central = Segment{
        round2(c_len),
        round2(get_clearance(center_x_, layer, 'h')),
        round2(get_clearance(center_x_, layer, 'm')),
    };

    for (int i = static_cast<int>(cuts.size()) - 1; i > 0; --i) {
        const double right_x = cuts[static_cast<std::size_t>(i)];
        const double left_x = cuts[static_cast<std::size_t>(i - 1)];
        const double length = right_x - left_x;
        if (length <= 0.0) {
            continue;
        }
        out.blocks.push_back(Segment{
            round2(length),
            round2(get_clearance(left_x, layer, 'h')),
            round2(get_clearance(left_x, layer, 'm')),
        });
    }
    return out;
}

LayerSegments CarriageGeometry::generate_even_side_segments(int num_blocks, CompartmentKind layer) const {
    if (num_blocks <= 0) {
        throw std::invalid_argument("num_blocks must be positive");
    }
    const double c_len = central_length(layer);
    const double central_start_x = center_x_ - c_len / 2.0;

    LayerSegments out;
    out.central = Segment{
        round2(c_len),
        round2(get_clearance(center_x_, layer, 'h')),
        round2(get_clearance(center_x_, layer, 'm')),
    };

    std::vector<double> cuts;
    cuts.reserve(static_cast<std::size_t>(num_blocks + 1));
    for (int i = 0; i <= num_blocks; ++i) {
        cuts.push_back(central_start_x * static_cast<double>(i) / static_cast<double>(num_blocks));
    }

    for (int i = static_cast<int>(cuts.size()) - 1; i > 0; --i) {
        const double right_x = cuts[static_cast<std::size_t>(i)];
        const double left_x = cuts[static_cast<std::size_t>(i - 1)];
        out.blocks.push_back(Segment{
            round2(right_x - left_x),
            round2(get_clearance(left_x, layer, 'h')),
            round2(get_clearance(left_x, layer, 'm')),
        });
    }
    return out;
}

ModelSegments get_model_segments(
    std::vector<double> car_heights,
    int num_splits,
    bool independent_mode_split
) {
    if (num_splits <= 0) {
        throw std::invalid_argument("num_splits must be positive");
    }

    std::sort(car_heights.begin(), car_heights.end());
    car_heights.erase(std::unique(car_heights.begin(), car_heights.end()), car_heights.end());

    CarriageGeometry geom;
    if (!independent_mode_split) {
        return ModelSegments{
            geom.generate_even_side_segments(num_splits, CompartmentKind::Lower),
            geom.generate_even_side_segments(num_splits, CompartmentKind::Upper),
        };
    }

    std::vector<double> selected_heights;
    if (num_splits > 0 && static_cast<std::size_t>(num_splits) < car_heights.size()) {
        if (num_splits == 1) {
            selected_heights.push_back(car_heights.front());
        } else {
            const double last = static_cast<double>(car_heights.size() - 1);
            for (int i = 0; i < num_splits; ++i) {
                const int idx = static_cast<int>(std::floor(last * i / static_cast<double>(num_splits - 1)));
                selected_heights.push_back(car_heights[static_cast<std::size_t>(idx)]);
            }
        }
    } else {
        selected_heights = car_heights;
    }

    std::vector<std::pair<double, char>> targets;
    targets.reserve(selected_heights.size() * 2);
    for (double h : selected_heights) {
        targets.push_back({h, 'h'});
    }
    for (double h : selected_heights) {
        targets.push_back({h, 'm'});
    }

    return ModelSegments{
        geom.generate_center_out_segments(targets, CompartmentKind::Lower),
        geom.generate_center_out_segments(targets, CompartmentKind::Upper),
    };
}

}  // namespace bpc_label
