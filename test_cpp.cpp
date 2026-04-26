#include "VNS_cpp/Feasibility.h"
#include "VNS_cpp/Problem.h"
#include <iostream>

int main() {
    Problem p;
    p.vehicle_types = 3;
    Vehicle v2; v2.id = 2; v2.length = 4715.0; v2.height = 1715.0;
    Vehicle v3; v3.id = 3; v3.length = 5230.0; v3.height = 2070.0;
    Vehicle v4; v4.id = 4; v4.length = 3905.0; v4.height = 1960.0;
    p.vehicle.push_back(v2);
    p.vehicle.push_back(v3);
    p.vehicle.push_back(v4);
    
    std::vector<int> cars = {2, 2, 3, 4, 2}; // Feasible route from earlier output
    
    std::vector<Region> regions;
    regions.push_back({0.0, 5000.0, 1720.0});
    regions.push_back({5000.0, 19900.0, 2070.0});
    regions.push_back({19900.0, 24900.0, 1720.0});
    
    double x = 0.0;
    for (int v_id : cars) {
        Vehicle* v = p.GetVehicle(v_id);
        bool placed = false;
        std::cout << "Trying to place car " << v_id << " (L=" << v->length << ", H=" << v->height << ") starting at x=" << x << "\n";
        
        while (x + v->length <= 24900.0 + 1e-5) {
            bool valid = true;
            double jump_to = x;
            
            for (const auto& r : regions) {
                if (std::max(x, r.start) < std::min(x + v->length, r.end) - 1e-5) {
                    if (r.height < v->height) {
                        valid = false;
                        jump_to = r.end;
                        std::cout << "  Collision with region [" << r.start << ", " << r.end << "] (H=" << r.height << "). Jumping to " << jump_to << "\n";
                        break;
                    }
                }
            }
            
            if (valid) {
                std::cout << "  Placed at x=" << x << " to " << (x + v->length) << "\n";
                x += v->length + 400.0;
                placed = true;
                break;
            } else {
                x = jump_to; 
            }
        }
        
        if (!placed) {
            std::cout << "  FAILED TO PLACE\n";
            return 1;
        }
    }
    std::cout << "ALL PLACED SUCCESSFULLY\n";
    return 0;
}
