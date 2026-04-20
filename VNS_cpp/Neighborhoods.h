#pragma once
#include <vector>
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
        int v_id = result.carriage[c1_id].route[old_floor][old_place];
        Vehicle* vehicle = p->GetVehicle(v_id);
        if (!EraseVehicle(result.carriage[c1_id], old_place, p, old_floor)) {
            return false;
        }
        if (!InsertCustomer(result.carriage[c2_id], vehicle, new_place, p, false, new_floor)) {
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

    bool Reposition(Solution& result, int c_id, Problem* p) {
        Solution result_tmp;
        result_tmp.copy_construct(result);
        
        if (result_tmp.carriage[c_id].position == 0) {
            int length = result_tmp.carriage[c_id].route[0].size();
            if (length >= 4) {
                int v1_id = result_tmp.carriage[c_id].route[0][0];
                int v2_id = result_tmp.carriage[c_id].route[0][length - 1];
                EraseVehicle(result_tmp.carriage[c_id], length - 1, p, 0);
                EraseVehicle(result_tmp.carriage[c_id], 0, p, 0);
                p->GetVehicle(v1_id)->UpdateParameter_Removing();
                p->GetVehicle(v2_id)->UpdateParameter_Removing();
            }
            result_tmp.carriage[c_id].position = 1;
        } else if (result_tmp.carriage[c_id].position == 1) {
            int length = result_tmp.carriage[c_id].route[1].size();
            if (length >= 4) {
                int v1_id = result_tmp.carriage[c_id].route[1][0];
                int v2_id = result_tmp.carriage[c_id].route[1][length - 1];
                EraseVehicle(result_tmp.carriage[c_id], length - 1, p, 1);
                EraseVehicle(result_tmp.carriage[c_id], 0, p, 1);
                p->GetVehicle(v1_id)->UpdateParameter_Removing();
                p->GetVehicle(v2_id)->UpdateParameter_Removing();
            }
            result_tmp.carriage[c_id].position = 0;
        }
        result_tmp.CalculateSolutionObj(p);
        bool is_success = IsFeasible(result_tmp.carriage[c_id], p);
        if (is_success) {
            result.copy_construct(result_tmp);
        }
        return is_success;
    }

    bool CrossBasic(Solution& result, Problem* p, int c1_id, int c2_id, int place1, int place2, int floor1, int floor2) {
        int len1 = result.carriage[c1_id].route[floor1].size();
        int len2 = result.carriage[c2_id].route[floor2].size();

        if (len1 <= 2 || len2 <= 2) return false;
        if (place1 > len1 - 1 || place2 > len2 - 1) return false;
        if (c1_id == c2_id && floor1 == floor2) return false;

        std::vector<int> route1 = result.carriage[c1_id].route[floor1];
        int position1 = result.carriage[c1_id].position;
        int spacing1 = result.carriage[c1_id].spacing;
        RouteInfo info1 = {position1, spacing1, floor1};

        std::vector<int> route2 = result.carriage[c2_id].route[floor2];
        int position2 = result.carriage[c2_id].position;
        int spacing2 = result.carriage[c2_id].spacing;
        RouteInfo info2 = {position2, spacing2, floor2};

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
                            for (size_t n = 0; n < result.carriage[j].route[f2].size(); ++n) {
                                if (RelocateBasic(result, i, m, j, n, f1, f2, p) && result.obj > best.obj) {
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
                                if (SwapBasic(result, i, j, m, n, p, f1, f2) && result.obj > best.obj) {
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
                                if (CrossBasic(result, p, i, j, m, n, f1, f2) && result.obj > best.obj) {
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
};