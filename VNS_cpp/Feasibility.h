#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <iostream>
#include "Problem.h"
#include "Carriage.h"
#include "Conf.h"

struct DynamicSegment {
    double length;
    double h_h;
    double h_m;
};

struct DynamicInterval {
    int l_idx;
    int r_idx;
    double cap;
};

class DynamicGeometry {
private:
    std::vector<DynamicSegment> lower_blocks;
    std::vector<DynamicSegment> upper_blocks;
    std::vector<DynamicInterval> lower_intervals;
    std::vector<DynamicInterval> upper_intervals;
<<<<<<< HEAD
    static constexpr double kPi = 3.14159265358979323846;
=======
    static constexpr double PI = 3.14159265358979323846;
>>>>>>> origin/multiChunking
    
    // Geometry math from Python dynamic_segmentation.py
    double carriage_length = 25000.0;
    double center_x = 12500.0;
    double roof_height = 4340.0;
    
    double deck_h_height = 2270.0;
    double deck_m_groove_length = 11400.0;
    double deck_m_groove_start = 12500.0 - 5700.0;
    double deck_m_end_height = 4340.0 - 1780.0;
    
    double floor_end_height = 680.0;
    double floor_groove_length = 10867.0;
    double floor_groove_start = 12500.0 - 5433.5;
<<<<<<< HEAD
    double slope_ratio = std::tan(9.0 * kPi / 180.0);
    double floor_slope_start = (12500.0 - 5433.5) - (680.0 / std::tan(9.0 * kPi / 180.0));
=======
    double slope_ratio = std::tan(9.0 * PI / 180.0);
    double floor_slope_start = (12500.0 - 5433.5) - (680.0 / std::tan(9.0 * PI / 180.0));
>>>>>>> origin/multiChunking

    double _deck_height(double x, bool is_m) {
        if (!is_m) return deck_h_height;
        if (x >= deck_m_groove_start) return deck_h_height;
        double ratio = x / deck_m_groove_start;
        return deck_m_end_height - ratio * (deck_m_end_height - deck_h_height);
    }
    
    double _floor_height(double x) {
        if (x <= floor_slope_start) return floor_end_height;
        if (x >= floor_groove_start) return 0.0;
        return floor_end_height - (x - floor_slope_start) * slope_ratio;
    }
    
    double get_clearance(double x, bool is_upper, bool is_m) {
        if (x > center_x) x = carriage_length - x;
        if (is_upper) return roof_height - _deck_height(x, is_m);
        else return _deck_height(x, is_m) - _floor_height(x);
    }
    
    double solve_x(double target_h, bool is_upper, bool is_m) {
        double h_min = get_clearance(0, is_upper, is_m);
        double h_max = get_clearance(center_x, is_upper, is_m);
        if (target_h <= h_min + 0.1) return 0.0;
        if (target_h > h_max + 0.1) return -1.0;
        
        double left = 0.0, right = center_x;
        for (int i=0; i<50; ++i) {
            double mid = (left + right) / 2.0;
            if (get_clearance(mid, is_upper, is_m) < target_h) left = mid;
            else right = mid;
        }
        return right;
    }
    
    void build_layer(bool is_upper, const std::vector<double>& heights, int num_splits, bool indep) {
        std::set<double> unique_h(heights.begin(), heights.end());
        std::vector<double> h_vec(unique_h.begin(), unique_h.end());
        std::sort(h_vec.begin(), h_vec.end());
        
        std::vector<double> sel_h;
        if (num_splits > 0 && num_splits < h_vec.size()) {
            for (int i=0; i<num_splits; ++i) {
                double raw_idx = (num_splits == 1)
                    ? 0.0
                    : i * (h_vec.size() - 1.0) / (num_splits - 1);
                int idx = static_cast<int>(raw_idx);
                sel_h.push_back(h_vec[idx]);
            }
        } else {
            sel_h = h_vec;
        }
        
        double central_len = is_upper ? deck_m_groove_length : floor_groove_length;
        double central_start_x = center_x - central_len / 2.0;
        
        std::vector<double> cut_points = {0.0, central_start_x};
        for (double h : sel_h) {
            double x_h = solve_x(h, is_upper, false);
            if (x_h > 0 && x_h < central_start_x) cut_points.push_back(x_h);
            if (indep) {
                double x_m = solve_x(h, is_upper, true);
                if (x_m > 0 && x_m < central_start_x) cut_points.push_back(x_m);
            }
        }
        
        std::sort(cut_points.begin(), cut_points.end());
        auto last = std::unique(cut_points.begin(), cut_points.end(), [](double a, double b){ return std::abs(a-b)<0.01; });
        cut_points.erase(last, cut_points.end());
        
        std::vector<DynamicSegment>& blocks = is_upper ? upper_blocks : lower_blocks;
        blocks.clear();
        
        blocks.push_back({central_len, get_clearance(center_x, is_upper, false), get_clearance(center_x, is_upper, true)});
        
        for (int i = cut_points.size() - 1; i > 0; --i) {
            double rx = cut_points[i];
            double lx = cut_points[i-1];
            if (rx - lx <= 0.01) continue;
            blocks.push_back({rx - lx, get_clearance(lx, is_upper, false), get_clearance(lx, is_upper, true)});
        }
        
        // Build intervals
        std::vector<DynamicInterval>& intervals = is_upper ? upper_intervals : lower_intervals;
        intervals.clear();
        int N = blocks.size() - 1;
        for (int l = 0; l <= N; ++l) {
            for (int r = 0; r <= N; ++r) {
                double mod = 400.0;
                if (l == N && r == N) mod = -400.0;
                else if (l == N || r == N) mod = 0.0;
                
                double cap = blocks[0].length + mod;
                for (int i=1; i<=l; ++i) cap += blocks[i].length;
                for (int i=1; i<=r; ++i) cap += blocks[i].length;
                intervals.push_back({l, r, cap});
            }
        }
    }

public:
    int num_splits = 1;
    bool independent_mode = true;
    bool initialized = false;
    
    void init(Problem* p) {
        if (initialized) return;
        std::vector<double> heights;
        for (int i=0; i<p->vehicle_types; ++i) heights.push_back(p->vehicle[i].height);
        build_layer(false, heights, num_splits, independent_mode);
        build_layer(true, heights, num_splits, independent_mode);
        initialized = true;
    }
    
    const std::vector<DynamicSegment>& get_blocks(bool is_upper) const { return is_upper ? upper_blocks : lower_blocks; }
    const std::vector<DynamicInterval>& get_intervals(bool is_upper) const { return is_upper ? upper_intervals : lower_intervals; }
};

extern DynamicGeometry GLOBAL_GEOM;

inline bool dfs_check(int car_idx, const std::vector<std::vector<std::vector<int>>>& car_choice_hits, 
                      const std::vector<double>& car_lens, const std::vector<double>& caps, 
                      std::vector<double>& current_usage) {
    if (car_idx == car_lens.size()) return true;
    double clen = car_lens[car_idx];
    
    for (const auto& hits : car_choice_hits[car_idx]) {
        bool valid = true;
        for (int j : hits) {
            if (current_usage[j] + clen > caps[j] + 1e-5) {
                valid = false;
                break;
            }
        }
        if (valid) {
            for (int j : hits) current_usage[j] += clen;
            if (dfs_check(car_idx + 1, car_choice_hits, car_lens, caps, current_usage)) return true;
            for (int j : hits) current_usage[j] -= clen;
        }
    }
    return false;
}

inline bool IsFeasible_Floor(const std::vector<int>& route, Problem* p, int mode_left, int mode_right, int floor, int spacing) {
    if (route.empty()) return true;
    if (route.size() > Config::max_units_per_compartment) return false;
    
    GLOBAL_GEOM.init(p);
    
    bool is_upper = (floor == 0);
    const auto& blocks = GLOBAL_GEOM.get_blocks(is_upper);
    const auto& intervals = GLOBAL_GEOM.get_intervals(is_upper);
    int N_blocks = blocks.size() - 1;
    
    bool pi_left = (mode_left == 1);
    bool pi_right = (mode_right == 1);
    
    std::vector<double> limit_left;
    std::vector<double> limit_right;
    for (const auto& b : blocks) {
        limit_left.push_back(pi_left ? b.h_m : b.h_h);
        limit_right.push_back(pi_right ? b.h_m : b.h_h);
    }
    double actual_limit_central = std::min(limit_left[0], limit_right[0]);
    
    std::vector<int> sorted_route = route;
    std::sort(sorted_route.begin(), sorted_route.end(), [&](int a, int b) {
        return p->GetVehicle(a)->height > p->GetVehicle(b)->height;
    });
    
    std::vector<std::vector<std::vector<int>>> car_choices; // For each car, a list of choices, where each choice is a list of hit interval indices
    std::vector<double> caps;
    
    std::vector<DynamicInterval> active_intervals;
    for (const auto& inter : intervals) {
        active_intervals.push_back(inter);
        caps.push_back(inter.cap);
    }
    
    std::vector<double> car_lens_w_delta;
    
    for (int v_id : sorted_route) {
        double h = p->GetVehicle(v_id)->height;
        car_lens_w_delta.push_back(p->GetVehicle(v_id)->length + spacing);
        
        int max_l = -1, max_r = -1;
        for (int idx = N_blocks; idx > 0; --idx) {
            if (h <= limit_left[idx]) { max_l = idx; break; }
        }
        if (max_l == -1 && h <= actual_limit_central) max_l = 0;
        
        for (int idx = N_blocks; idx > 0; --idx) {
            if (h <= limit_right[idx]) { max_r = idx; break; }
        }
        if (max_r == -1 && h <= actual_limit_central) max_r = 0;
        
        if (max_l == -1 && max_r == -1) return false;
        
        std::vector<std::pair<std::string, int>> raw_choices;
        if (max_l == 0 && max_r == 0) {
            raw_choices.push_back({"central", 0});
        } else {
            if (max_l >= 0) raw_choices.push_back({"left", max_l});
            if (max_r >= 0) raw_choices.push_back({"right", max_r});
        }
        
        std::vector<std::vector<int>> hit_choices;
        for (const auto& ch : raw_choices) {
            std::vector<int> hits;
            for (size_t j = 0; j < active_intervals.size(); ++j) {
                bool inside = false;
                if (ch.first == "central") inside = true;
                else if (ch.first == "left" && ch.second <= active_intervals[j].l_idx) inside = true;
                else if (ch.first == "right" && ch.second <= active_intervals[j].r_idx) inside = true;
                if (inside) hits.push_back(j);
            }
            hit_choices.push_back(hits);
        }
        car_choices.push_back(hit_choices);
    }
    
    std::vector<double> current_usage(active_intervals.size(), 0.0);
    return dfs_check(0, car_choices, car_lens_w_delta, caps, current_usage);
}

inline bool IsFeasible_Length(const Carriage& c, Problem* p) {
    return IsFeasible_Floor(c.route[0], p, c.mode_left, c.mode_right, 0, c.spacing) &&
           IsFeasible_Floor(c.route[1], p, c.mode_left, c.mode_right, 1, c.spacing);
}

inline bool IsFeasible_Height(const Carriage& c, Problem* p) {
    return true; 
}

inline bool IsFeasible(const Carriage& c, Problem* p) {
    return IsFeasible_Length(c, p);
}

struct RouteInfo {
    int mode_left;
    int mode_right;
    int spacing;
    int floor;
};

inline bool IsFeasible_Length_route(Problem* p, const std::vector<int>& route, const RouteInfo& info) {
    return IsFeasible_Floor(route, p, info.mode_left, info.mode_right, info.floor, info.spacing);
}

inline bool IsFeasible_Height_route(Problem* p, const std::vector<int>& route, const RouteInfo& info) {
    return true;
}

inline bool IsFeasible_route(Problem* p, const std::vector<int>& route, const RouteInfo& info) {
    return IsFeasible_Floor(route, p, info.mode_left, info.mode_right, info.floor, info.spacing);
}
