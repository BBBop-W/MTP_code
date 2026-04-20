#pragma once
#include <vector>
#include "Problem.h"
#include "Carriage.h"
#include "Conf.h"

struct Region {
    double max_len;
    double max_height;
};

inline bool IsFeasible_Floor(const std::vector<int>& route, Problem* p, int position, int floor, int spacing) {
    if (route.empty()) return true;

    std::vector<Region> regions;
    if (floor == 0) { // Top floor: D1, E, D2
        double d_height = (position == 0) ? Config::D_height_h : Config::D_height_m;
        regions.push_back({Config::D_len, d_height});
        regions.push_back({Config::E_len, Config::E_height});
        regions.push_back({Config::D_len, d_height});
    } else { // Bottom floor: A1, B1, C, B2, A2
        double a_height = (position == 0) ? Config::A_height_h : Config::A_height_m;
        regions.push_back({Config::A_len, a_height});
        regions.push_back({Config::B_len, Config::B_height});
        regions.push_back({Config::C_len, Config::C_height});
        regions.push_back({Config::B_len, Config::B_height});
        regions.push_back({Config::A_len, a_height});
    }

    size_t curr_region = 0;
    double current_used_len = 0;

    for (int v_id : route) {
        Vehicle* v = p->GetVehicle(v_id);
        bool placed = false;
        
        while (curr_region < regions.size()) {
            // 检查当前车是否满足当前区域的高度限制，以及加上这辆车后是否超过该区域的最大长度
            // 注意：判断放入时只加 v->length，放入成功后 current_used_len 再加上 v->length + spacing
            if (v->height <= regions[curr_region].max_height && 
                current_used_len + v->length <= regions[curr_region].max_len) {
                
                current_used_len += v->length + spacing;
                placed = true;
                break; // 成功放入当前区域
            } else {
                // 当前区域放不下（高度超限或长度不够），移动到下一个区域，前一个区域的剩余长度作废
                curr_region++;
                current_used_len = 0;
            }
        }
        
        if (!placed) {
            return false; // 所有剩下的区域都放不下这辆车
        }
    }

    return true;
}

inline bool IsFeasible_Length(const Carriage& c, Problem* p) {
    return IsFeasible_Floor(c.route[0], p, c.position, 0, c.spacing) &&
           IsFeasible_Floor(c.route[1], p, c.position, 1, c.spacing);
}

inline bool IsFeasible_Height(const Carriage& c, Problem* p) {
    // 高度检查已经合并到 IsFeasible_Floor 的区域逻辑中
    return true; 
}

inline bool IsFeasible(const Carriage& c, Problem* p) {
    return IsFeasible_Length(c, p);
}

struct RouteInfo {
    int position;
    int spacing;
    int floor;
};

inline bool IsFeasible_Length_route(Problem* p, const std::vector<int>& route, const RouteInfo& info) {
    return IsFeasible_Floor(route, p, info.position, info.floor, info.spacing);
}

inline bool IsFeasible_Height_route(Problem* p, const std::vector<int>& route, const RouteInfo& info) {
    return true;
}

inline bool IsFeasible_route(Problem* p, const std::vector<int>& route, const RouteInfo& info) {
    return IsFeasible_Floor(route, p, info.position, info.floor, info.spacing);
}
