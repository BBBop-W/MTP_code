#pragma once
#include <vector>
#include "Problem.h"
#include "Carriage.h"
#include "Feasibility.h"

bool InsertCustomer(Carriage& c, Vehicle* v, int place, Problem* p, bool BothFloor = true, int floor = 0) {
    c.route[floor].insert(c.route[floor].begin() + place, v->id);
    if (IsFeasible(c, p)) {
        c.CalculateCarriageObj(p);
        return true;
    } else {
        c.route[floor].erase(c.route[floor].begin() + place);
        if (BothFloor) {
            c.route[1 - floor].insert(c.route[1 - floor].begin() + place, v->id);
            if (IsFeasible(c, p)) {
                c.CalculateCarriageObj(p);
                return true;
            } else {
                c.route[1 - floor].erase(c.route[1 - floor].begin() + place);
            }
        }
        c.CalculateCarriageObj(p);
        return false;
    }
}

bool EraseVehicle(Carriage& c, int place, Problem* p, int floor = 0) {
    if (c.length(floor) == 0) {
        return false;
    }
    int v_id = c.route[floor][place];
    c.route[floor].erase(c.route[floor].begin() + place);
    if (IsFeasible(c, p)) {
        c.CalculateCarriageObj(p);
        return true;
    } else {
        c.route[floor].insert(c.route[floor].begin() + place, v_id);
        return false;
    }
}

struct BestRouteResult {
    double max_obj;
    int best_floor;
    int best_place;
};

BestRouteResult BestToRoute(Carriage& c, Vehicle* v, Problem* p) {
    double max_obj = 0;
    int best_place = -1;
    int best_floor = -1;
    
    if (c.length(0) == 0) {
        return {max_obj, 0, 0};
    }
    if (c.length(1) == 0) {
        return {max_obj, 1, 0};
    }
    for (int f : {0, 1}) {
        for (size_t i = 0; i < c.route[f].size(); ++i) {
            double obj_temp = c.obj;
            if (InsertCustomer(c, v, i, p, false, f)) {
                double obj = c.obj - obj_temp;
                if (obj > max_obj) {
                    max_obj = obj;
                    best_place = i;
                    best_floor = f;
                }
                EraseVehicle(c, i, p, f);
            }
        }
    }
    return {max_obj, best_floor, best_place};
}