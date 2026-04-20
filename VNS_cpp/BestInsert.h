#pragma once
#include "Solution.h"
#include "Problem.h"
#include "Insert.h"
#include <algorithm>

class BestInsert {
public:
    int tryNumber;

    BestInsert() {
        tryNumber = 10000;
    }

    bool Construct(Solution& result, Problem* p) {
        std::vector<Vehicle*> vnode;
        for(auto& v : p->vehicle) vnode.push_back(&v);
        
        std::sort(vnode.begin(), vnode.end(), [](Vehicle* a, Vehicle* b) {
            return a->var_mandatory > b->var_mandatory;
        });
        
        std::vector<Carriage*> cnode;
        for(auto& c : result.carriage) cnode.push_back(&c);
        std::sort(cnode.begin(), cnode.end(), [](Carriage* a, Carriage* b) {
            return a->length() > b->length();
        });

        for (auto v : vnode) {
            for (int k = 0; k < v->var_mandatory; ++k) {
                for (auto c : cnode) {
                    BestRouteResult best_insert = BestToRoute(*c, v, p);
                    if (best_insert.best_place == -1) continue;
                    InsertCustomer(*c, v, best_insert.best_place, p, false, best_insert.best_floor);
                    v->var_mandatory -= 1;
                    v->var_optional -= 1;
                    result.CalculateSolutionObj(p);
                    break;
                }
            }
        }

        std::sort(cnode.begin(), cnode.end(), [](Carriage* a, Carriage* b) {
            return a->length() < b->length();
        });
        std::sort(vnode.begin(), vnode.end(), [](Vehicle* a, Vehicle* b) {
            return a->var_optional > b->var_optional;
        });

        for (auto v : vnode) {
            for (int k = 0; k < v->var_optional; ++k) {
                for (auto c : cnode) {
                    BestRouteResult best_insert = BestToRoute(*c, v, p);
                    if (best_insert.best_place == -1) continue;
                    InsertCustomer(*c, v, best_insert.best_place, p, false, best_insert.best_floor);
                    v->var_optional -= 1;
                    result.CalculateSolutionObj(p);
                    break;
                }
            }
        }
        return true;
    }

    Solution* Solve(Problem* p) {
        Solution* result = new Solution();
        bool success = Construct(*result, p);
        if (!success) {
            std::cout << "Cannot generate the feasible solution!" << std::endl;
            delete result;
            return nullptr;
        }
        std::cout << "Generate the initial Solution" << std::endl;
        return result;
    }
};