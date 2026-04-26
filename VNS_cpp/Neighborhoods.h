#pragma once
#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
#include "Solution.h"
#include "Problem.h"
#include "Feasibility.h"
#include "Insert.h"
#include "Conf.h"

class Neighborhoods {
public:
    int try_num;

    Neighborhoods() : try_num(100) {}

    bool RelocateBasic(Solution& result, int c1_id, int old_place, int c2_id, int new_place, int old_floor, int new_floor, Problem* p) {
        if (c1_id == c2_id && (old_place == new_place || old_place == new_place + 1 || old_place + 1 == new_place) && old_floor == new_floor) {
            return false;
        }
        // Bounds check
        if (old_place >= result.carriage[c1_id].route[old_floor].size() || new_place > result.carriage[c2_id].route[new_floor].size()) {
            return false;
        }
        int v_id = result.carriage[c1_id].route[old_floor][old_place];
        Vehicle* vehicle = p->GetVehicle(v_id);
        
        // Adjust new_place if we are removing an element before it in the same vector
        int actual_new_place = new_place;
        if (c1_id == c2_id && old_floor == new_floor && old_place < new_place) {
            actual_new_place--;
        }

        if (!EraseVehicle(result.carriage[c1_id], old_place, p, old_floor)) {
            return false;
        }
        
        if (actual_new_place > result.carriage[c2_id].route[new_floor].size()) {
            printf("CRITICAL BUG: actual_new_place=%d, size=%zu\n", actual_new_place, result.carriage[c2_id].route[new_floor].size());
        }
        
        if (!InsertCustomer(result.carriage[c2_id], vehicle, actual_new_place, p, false, new_floor)) {
            InsertCustomer(result.carriage[c1_id], vehicle, old_place, p, false, old_floor);
            return false;
        }
        result.CalculateSolutionObj(p);
        return true;
    }

    bool SwapBasic(Solution& result, int c1_id, int c2_id, int place1, int place2, Problem* p, int floor1, int floor2) {
        if (c1_id == c2_id && (place1 == place2 || place1 + 1 == place2 || place1 == place2 + 1) && floor1 == floor2) {
            return false;
        }
        // Bounds check
        if (place1 >= result.carriage[c1_id].route[floor1].size() || place2 >= result.carriage[c2_id].route[floor2].size()) {
            return false;
        }
        Solution result_tmp;
        result_tmp.copy_construct(result);
        int tmp = result_tmp.carriage[c1_id].route[floor1][place1];
        result_tmp.carriage[c1_id].route[floor1][place1] = result_tmp.carriage[c2_id].route[floor2][place2];
        result_tmp.carriage[c2_id].route[floor2][place2] = tmp;

        bool is_success = IsFeasible(result_tmp.carriage[c1_id], p) && IsFeasible(result_tmp.carriage[c2_id], p);
        if (is_success) {
            int tmp2 = result.carriage[c1_id].route[floor1][place1];
            result.carriage[c1_id].route[floor1][place1] = result.carriage[c2_id].route[floor2][place2];
            result.carriage[c2_id].route[floor2][place2] = tmp2;
            result.CalculateSolutionObj(p);
            return true;
        }
        return false;
    }

    bool Reposition(Solution& result, int c_id, int side, Problem* p) {
        Solution result_tmp;
        result_tmp.copy_construct(result);
        
        Carriage& c = result_tmp.carriage[c_id];
        
        if (side == 0) { // left
            c.mode_left = 1 - c.mode_left; // toggle
        } else { // right
            c.mode_right = 1 - c.mode_right; // toggle
        }
        
        // Remove cars greedily from the ends if infeasible
        for (int f : {0, 1}) {
            while (!c.route[f].empty() && !IsFeasible_Floor(c.route[f], p, c.mode_left, c.mode_right, f, c.spacing)) {
                // If left side changed, we might want to remove from the left (index 0).
                // If right side changed, remove from right (index size-1).
                int remove_idx = (side == 0) ? 0 : c.route[f].size() - 1;
                int v_id = c.route[f][remove_idx];
                EraseVehicle(c, remove_idx, p, f);
                p->GetVehicle(v_id)->UpdateParameter_Removing();
            }
        }
        
        c.CalculateCarriageObj(p);
        result_tmp.CalculateSolutionObj(p);
        
        // If it's feasible overall (it should be since we erased until feasible)
        if (IsFeasible(c, p)) {
            result.copy_construct(result_tmp);
            return true;
        }
        return false;
    }

    bool CrossBasic(Solution& result, Problem* p, int c1_id, int c2_id, int place1, int place2, int floor1, int floor2) {
        int len1 = result.carriage[c1_id].route[floor1].size();
        int len2 = result.carriage[c2_id].route[floor2].size();

        if (len1 <= 2 || len2 <= 2) return false;
        if (place1 > len1 - 1 || place2 > len2 - 1) return false;
        if (c1_id == c2_id && floor1 == floor2) return false;

        std::vector<int> route1 = result.carriage[c1_id].route[floor1];
        int mode_left1 = result.carriage[c1_id].mode_left;
        int mode_right1 = result.carriage[c1_id].mode_right;
        int spacing1 = result.carriage[c1_id].spacing;
        RouteInfo info1 = {mode_left1, mode_right1, spacing1, floor1};

        std::vector<int> route2 = result.carriage[c2_id].route[floor2];
        int mode_left2 = result.carriage[c2_id].mode_left;
        int mode_right2 = result.carriage[c2_id].mode_right;
        int spacing2 = result.carriage[c2_id].spacing;
        RouteInfo info2 = {mode_left2, mode_right2, spacing2, floor2};

        std::vector<int> new_route1(route1.begin(), route1.begin() + place1);
        new_route1.insert(new_route1.end(), route2.begin() + place2, route2.end());

        std::vector<int> new_route2(route2.begin(), route2.begin() + place2);
        new_route2.insert(new_route2.end(), route1.begin() + place1, route1.end());

        bool is_success1 = IsFeasible_route(p, new_route1, info1);
        bool is_success2 = IsFeasible_route(p, new_route2, info2);

        if (is_success1 && is_success2) {
            result.carriage[c1_id].route[floor1] = new_route1;
            result.carriage[c2_id].route[floor2] = new_route2;
            result.CalculateSolutionObj(p);
            return true;
        }
        return false;
    }

    bool InterRelocateBest(Solution& result, Problem* p) {
        Solution best;
        Solution old;
        best.copy_construct(result);
        old.copy_construct(result);
        bool improved = false;
        int length = result.carriage_num;
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int f1 : {0, 1}) {
                    for (int f2 : {0, 1}) {
                        for (size_t m = 0; m < result.carriage[i].route[f1].size(); ++m) {
                            for (size_t n = 0; n <= result.carriage[j].route[f2].size(); ++n) {
                                if (RelocateBasic(result, i, m, j, n, f1, f2, p) && result.obj > best.obj + Config::eps) {
                                    best.copy_construct(result);
                                    improved = true;
                                }
                                result.copy_construct(old);
                            }
                        }
                    }
                }
            }
        }
        result.copy_construct(best);
        return improved;
    }

    bool InterSwapBest(Solution& result, Problem* p) {
        Solution best;
        Solution old;
        best.copy_construct(result);
        old.copy_construct(result);
        bool improved = false;
        int length = result.carriage_num;
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int f1 : {0, 1}) {
                    for (int f2 : {0, 1}) {
                        for (size_t m = 0; m < result.carriage[i].route[f1].size(); ++m) {
                            for (size_t n = 0; n < result.carriage[j].route[f2].size(); ++n) {
                                if (SwapBasic(result, i, j, m, n, p, f1, f2) && result.obj > best.obj + Config::eps) {
                                    best.copy_construct(result);
                                    improved = true;
                                }
                                result.copy_construct(old);
                            }
                        }
                    }
                }
            }
        }
        result.copy_construct(best);
        return improved;
    }

    bool InterOptBest(Solution& result, Problem* p) {
        Solution best;
        Solution old;
        best.copy_construct(result);
        old.copy_construct(result);
        bool improved = false;
        int length = result.carriage_num;
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int f1 : {0, 1}) {
                    for (int f2 : {0, 1}) {
                        for (size_t m = 0; m < result.carriage[i].route[f1].size(); ++m) {
                            for (size_t n = 0; n < result.carriage[j].route[f2].size(); ++n) {
                                if (CrossBasic(result, p, i, j, m, n, f1, f2) && result.obj > best.obj + Config::eps) {
                                    best.copy_construct(result);
                                    improved = true;
                                }
                                result.copy_construct(old);
                            }
                        }
                    }
                }
            }
        }
        result.copy_construct(best);
        return improved;
    }

    bool InterRelocateRandom(Solution& result, Problem* p) {
        Solution old;
        old.copy_construct(result);
        std::vector<std::tuple<int, int, int, int, int, int>> moves;
        int length = result.carriage_num;
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int f1 : {0, 1}) {
                    for (int f2 : {0, 1}) {
                        for (size_t m = 0; m < result.carriage[i].route[f1].size(); ++m) {
                            for (size_t n = 0; n <= result.carriage[j].route[f2].size(); ++n) {
                                moves.push_back(std::make_tuple(i, m, j, n, f1, f2));
                            }
                        }
                    }
                }
            }
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(moves.begin(), moves.end(), g);
        
        for (const auto& move : moves) {
            if (RelocateBasic(result, std::get<0>(move), std::get<1>(move), std::get<2>(move), std::get<3>(move), std::get<4>(move), std::get<5>(move), p)) {
                if (result.obj > old.obj + Config::eps) {
                    return true;
                }
            }
            result.copy_construct(old);
        }
        return false;
    }

    bool InterSwapRandom(Solution& result, Problem* p) {
        Solution old;
        old.copy_construct(result);
        std::vector<std::tuple<int, int, int, int, int, int>> moves;
        int length = result.carriage_num;
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int f1 : {0, 1}) {
                    for (int f2 : {0, 1}) {
                        for (size_t m = 0; m < result.carriage[i].route[f1].size(); ++m) {
                            for (size_t n = 0; n < result.carriage[j].route[f2].size(); ++n) {
                                moves.push_back(std::make_tuple(i, m, j, n, f1, f2));
                            }
                        }
                    }
                }
            }
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(moves.begin(), moves.end(), g);
        
        for (const auto& move : moves) {
            if (SwapBasic(result, std::get<0>(move), std::get<2>(move), std::get<1>(move), std::get<3>(move), p, std::get<4>(move), std::get<5>(move))) {
                if (result.obj > old.obj + Config::eps) {
                    return true;
                }
            }
            result.copy_construct(old);
        }
        return false;
    }

    bool InterOptRandom(Solution& result, Problem* p) {
        Solution old;
        old.copy_construct(result);
        std::vector<std::tuple<int, int, int, int, int, int>> moves;
        int length = result.carriage_num;
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int f1 : {0, 1}) {
                    for (int f2 : {0, 1}) {
                        for (size_t m = 0; m < result.carriage[i].route[f1].size(); ++m) {
                            for (size_t n = 0; n < result.carriage[j].route[f2].size(); ++n) {
                                moves.push_back(std::make_tuple(i, j, m, n, f1, f2));
                            }
                        }
                    }
                }
            }
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(moves.begin(), moves.end(), g);
        
        for (const auto& move : moves) {
            if (CrossBasic(result, p, std::get<0>(move), std::get<1>(move), std::get<2>(move), std::get<3>(move), std::get<4>(move), std::get<5>(move))) {
                if (result.obj > old.obj + Config::eps) {
                    return true;
                }
            }
            result.copy_construct(old);
        }
        return false;
    }
};