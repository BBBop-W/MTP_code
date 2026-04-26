#pragma once
#include <vector>
#include <algorithm>
#include "Problem.h"
#include "Carriage.h"
#include "Conf.h"

struct Region {
    double start, end, height;
};

inline bool IsFeasible_Floor(const std::vector<int>& route, Problem* p, int mode_left, int mode_right, int floor, int spacing) {
    if (route.empty()) return true;

    std::vector<Region> regions;
    double total_len = 0.0;
    
    if (floor == 0) { // Top floor: D1, E, D2
        double d_left_height = (mode_left == 0) ? Config::D_height_h : Config::D_height_m;
        double d_right_height = (mode_right == 0) ? Config::D_height_h : Config::D_height_m;
        
        regions.push_back({0.0, Config::D_len, d_left_height});
        regions.push_back({Config::D_len, Config::D_len + Config::E_len, Config::E_height});
        regions.push_back({Config::D_len + Config::E_len, Config::D_len * 2 + Config::E_len, d_right_height});
        total_len = Config::D_len * 2 + Config::E_len;
    } else { // Bottom floor: A1, B1, C, B2, A2
        double a_left_height = (mode_left == 0) ? Config::A_height_h : Config::A_height_m;
        double a_right_height = (mode_right == 0) ? Config::A_height_h : Config::A_height_m;
        
        double a_end = Config::A_len;
        double b_end = a_end + Config::B_len;
        double c_end = b_end + Config::C_len;
        double b2_end = c_end + Config::B_len;
        double a2_end = b2_end + Config::A_len;
        
        regions.push_back({0.0, a_end, a_left_height});
        regions.push_back({a_end, b_end, Config::B_height});
        regions.push_back({b_end, c_end, Config::C_height});
        regions.push_back({c_end, b2_end, Config::B_height});
        regions.push_back({b2_end, a2_end, a_right_height});
        total_len = a2_end;
    }

    double x = 0.0;

    for (int v_id : route) {
        Vehicle* v = p->GetVehicle(v_id);
        bool placed = false;
        
        while (x + v->length <= total_len + 1e-5) {
            bool valid = true;
            double jump_to = x;
            
            for (const auto& r : regions) {
                if (std::max(x, r.start) < std::min(x + v->length, r.end) - 1e-5) {
                    if (r.height < v->height) {
                        valid = false;
                        jump_to = r.end;
                        break; 
                    }
                }
            }
            
            if (valid) {
                x += v->length + spacing;
                placed = true;
                break;
            } else {
                x = jump_to; 
            }
        }
        
        if (!placed) {
            return false; 
        }
    }

    // Explicitly enforce the Gurobi capacity cuts (sum of lengths + delta)
    double delta = spacing;
    if (floor == 0) {
        double h_d_left = (mode_left == 1) ? Config::D_height_m : Config::D_height_h;
        double h_d_right = (mode_right == 1) ? Config::D_height_m : Config::D_height_h;
        
        double sum_E = 0, sum_D_L = 0, sum_D_R = 0;
        for (int v_id : route) {
            double h = p->GetVehicle(v_id)->height;
            double l = p->GetVehicle(v_id)->length + delta;
            
            if (h > h_d_left && h > h_d_right) {
                sum_E += l;
            } else if (h > h_d_left) {
                sum_E += l; // Push to E or D_right. Let's conservatively add to E for bounds checking.
            } else if (h > h_d_right) {
                sum_E += l; 
            } else {
                // If it fits anywhere, we don't strictly need to assign it here unless we are solving the bin packing perfectly.
                // The continuous check already verified it packs. 
                // However, Gurobi explicitly checks: sum_{h in H6[k]} x*(L+Delta) <= L_D + L_E.
                // We'll just enforce the global bound to catch the exact case where continuous straddles boundaries.
            }
        }
        
        double total_car_len_with_delta = 0.0;
        for (int v_id : route) {
            total_car_len_with_delta += p->GetVehicle(v_id)->length + delta;
        }
        
        if (total_car_len_with_delta > 2 * Config::D_len + Config::E_len - delta + 1e-5) return false;
        
        if (mode_left == 1) {
            double left_e_used = 0.0;
            for (int v_id : route) {
                double h = p->GetVehicle(v_id)->height;
                double l = p->GetVehicle(v_id)->length + delta;
                // Cars that CANNOT go to D_right MUST go to D_left or E
                if (h > h_d_right || h <= h_d_left) { 
                    left_e_used += l;
                }
            }
            // A conservative check: if ALL cars that could possibly go to D_left + E exceed the limit.
            // A better way is to call a simplified checker. But let's just stick to the known boundary gap:
            // if sum(L + delta) for all cars > D_len + E_len when mode_left=1?
            // Actually, if we just check the total length of the route up to a certain point:
        }
        
        // Exact Python/Gurobi logic replication for upper deck (mode_left=1 -> pi_left=1)
        if (mode_left == 1 || mode_right == 1) {
            std::vector<int> cars(route.begin(), route.end());
            // Brute force 2^N left/right assignment just like Python _simple_check_layer_bs
            // This is O(2^N) but N <= 8, so it's virtually instantaneous (max 256 iterations)
            bool global_feasible = false;
            int n_cars = cars.size();
            for (int mask = 0; mask < (1 << n_cars); ++mask) {
                double temp_E = 0, temp_D_L = 0, temp_D_R = 0;
                bool valid_assignment = true;
                
                for (int i = 0; i < n_cars; ++i) {
                    int v_id = cars[i];
                    double h = p->GetVehicle(v_id)->height;
                    double l = p->GetVehicle(v_id)->length + delta;
                    
                    if (h > Config::E_height) { valid_assignment = false; break; }
                    
                    if ((mask & (1 << i)) == 0) { // left
                        if (h > h_d_left) temp_E += l;
                        else temp_D_L += l;
                    } else { // right
                        if (h > h_d_right) temp_E += l;
                        else temp_D_R += l;
                    }
                }
                
                if (!valid_assignment) continue;
                if (temp_E + temp_D_L + temp_D_R > Config::E_len + 2*Config::D_len - delta + 1e-5) continue;
                
                bool feasible = true;
                if (mode_left == 1) {
                    if (temp_E > Config::E_len + delta + 1e-5) feasible = false;
                    if (temp_E + temp_D_L > Config::E_len + Config::D_len + 1e-5) feasible = false;
                }
                if (mode_right == 1) {
                    if (temp_E > Config::E_len + delta + 1e-5) feasible = false;
                    if (temp_E + temp_D_R > Config::E_len + Config::D_len + 1e-5) feasible = false;
                }
                
                if (feasible) {
                    global_feasible = true;
                    break;
                }
            }
            if (!global_feasible) return false;
        }
        
    } else { // bottom floor
        double total_car_len_with_delta = 0.0;
        for (int v_id : route) {
            total_car_len_with_delta += p->GetVehicle(v_id)->length + delta;
        }
        if (total_car_len_with_delta > 2 * Config::A_len + 2 * Config::B_len + Config::C_len - delta + 1e-5) return false;
        
        if (mode_left == 1 || mode_right == 1) {
            double h_a_left = (mode_left == 1) ? Config::A_height_m : Config::A_height_h;
            double h_a_right = (mode_right == 1) ? Config::A_height_m : Config::A_height_h;
            double h_b = Config::B_height;
            double h_c = Config::C_height;
            
            std::vector<int> cars(route.begin(), route.end());
            bool global_feasible = false;
            int n_cars = cars.size();
            
            for (int mask = 0; mask < (1 << n_cars); ++mask) {
                double temp_C = 0, temp_B_L = 0, temp_B_R = 0, temp_A_L = 0, temp_A_R = 0;
                bool valid_assignment = true;
                
                for (int i = 0; i < n_cars; ++i) {
                    int v_id = cars[i];
                    double h = p->GetVehicle(v_id)->height;
                    double l = p->GetVehicle(v_id)->length + delta;
                    
                    if (h > h_c) { valid_assignment = false; break; }
                    
                    if (h > h_b) {
                        temp_C += l;
                    } else if ((mask & (1 << i)) == 0) { // left
                        if (h > h_a_left) temp_B_L += l;
                        else temp_A_L += l;
                    } else { // right
                        if (h > h_a_right) temp_B_R += l;
                        else temp_A_R += l;
                    }
                }
                
                if (!valid_assignment) continue;
                
                if (temp_C > Config::C_len + delta + 1e-5) continue;
                if (temp_C + temp_B_L + temp_B_R > Config::C_len + 2*Config::B_len + delta + 1e-5) continue;
                if (temp_C + temp_B_L + temp_B_R + temp_A_L + temp_A_R > Config::C_len + 2*Config::B_len + 2*Config::A_len - delta + 1e-5) continue;
                
                bool feasible = true;
                if (mode_left == 1) {
                    if (temp_C + temp_B_L > Config::C_len + Config::B_len + delta + 1e-5) feasible = false;
                    if (temp_C + temp_B_L + temp_A_L > Config::C_len + Config::B_len + Config::A_len + 1e-5) feasible = false;
                }
                if (mode_right == 1) {
                    if (temp_C + temp_B_R > Config::C_len + Config::B_len + delta + 1e-5) feasible = false;
                    if (temp_C + temp_B_R + temp_A_R > Config::C_len + Config::B_len + Config::A_len + 1e-5) feasible = false;
                }
                
                if (feasible) {
                    global_feasible = true;
                    break;
                }
            }
            if (!global_feasible) return false;
        }
    }

    return true;
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
