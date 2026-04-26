#pragma once
#include <vector>
#include "Problem.h"
#include "Conf.h"

class Carriage {
public:
    int id;
    int spacing;
    std::vector<int> route[2];
    double obj;
    double actual_length;
    int num;
    int mode_left;  // 0 for horizontal (h), 1 for middle (m)
    int mode_right; // 0 for horizontal (h), 1 for middle (m)

    Carriage() : id(0), spacing(400), obj(0.0), actual_length(0.0), num(0), mode_left(0), mode_right(0) {}

    int length(int floor = -1) const {
        if (floor == -1) {
            return route[0].size() + route[1].size();
        } else {
            return route[floor].size();
        }
    }

    double CalculateCarriageObj(Problem* p) {
        int Num1 = route[0].size();
        int Num2 = route[1].size();
        
        double length1 = 0;
        for (size_t i = 0; i < route[0].size(); ++i) {
            length1 += p->vehicle[route[0][i]].length;
        }
        double length2 = 0;
        for (size_t i = 0; i < route[1].size(); ++i) {
            length2 += p->vehicle[route[1][i]].length;
        }
        
        actual_length = length1 + length2;
        
        obj = (length1 * length1 + length2 * length2) / 100000.0;
        return obj;
    }

    void copy_construct(const Carriage& origin) {
        id = origin.id;
        spacing = origin.spacing;
        route[0] = origin.route[0];
        route[1] = origin.route[1];
        obj = origin.obj;
        actual_length = origin.actual_length;
        num = origin.num;
        mode_left = origin.mode_left;
        mode_right = origin.mode_right;
    }
};