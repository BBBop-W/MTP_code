#pragma once
#include "Solution.h"
#include "Problem.h"
#include "Insert.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

class BestInsert {
public:
    int tryNumber;

    BestInsert() {
        tryNumber = 10000;
    }

private:
    enum class OrderStrategy {
        ScarceTallLong,
        TallLong,
        LongTall,
        CountTallLong,
        CountOnly
    };

    enum class PlacementStrategy {
        Pack,
        Spread,
        Balance
    };

    struct InsertChoice {
        bool feasible = false;
        Carriage* carriage = nullptr;
        int floor = -1;
        int place = -1;
        int mode_left = 0;
        int mode_right = 0;
        double delta_obj = -std::numeric_limits<double>::infinity();
        double placement_score = 0.0;
        int mode_changes = 0;
    };

    int EmptyFeasiblePositions(Vehicle* v, Problem* p) {
        int count = 0;
        for (int ml : {0, 1}) {
            for (int mr : {0, 1}) {
                for (int f : {0, 1}) {
                    Carriage empty;
                    empty.mode_left = ml;
                    empty.mode_right = mr;
                    if (InsertCustomer(empty, v, 0, p, false, f)) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    std::vector<Vehicle*> OrderedVehicles(Problem* p, OrderStrategy strategy) {
        std::vector<Vehicle*> vnode;
        for (auto& v : p->vehicle) vnode.push_back(&v);

        std::stable_sort(vnode.begin(), vnode.end(), [&](Vehicle* a, Vehicle* b) {
            if (strategy == OrderStrategy::ScarceTallLong) {
                int da = EmptyFeasiblePositions(a, p);
                int db = EmptyFeasiblePositions(b, p);
                if (da != db) return da < db;
                if (a->height != b->height) return a->height > b->height;
                if (a->length != b->length) return a->length > b->length;
                return a->var_mandatory > b->var_mandatory;
            }
            if (strategy == OrderStrategy::TallLong) {
                if (a->height != b->height) return a->height > b->height;
                if (a->length != b->length) return a->length > b->length;
                return a->var_mandatory > b->var_mandatory;
            }
            if (strategy == OrderStrategy::LongTall) {
                if (a->length != b->length) return a->length > b->length;
                if (a->height != b->height) return a->height > b->height;
                return a->var_mandatory > b->var_mandatory;
            }
            if (strategy == OrderStrategy::CountTallLong) {
                if (a->var_mandatory != b->var_mandatory) return a->var_mandatory > b->var_mandatory;
                if (a->height != b->height) return a->height > b->height;
                return a->length > b->length;
            }
            return a->var_mandatory > b->var_mandatory;
        });
        return vnode;
    }

    InsertChoice BestInsertion(Carriage& c, Vehicle* v, Problem* p, bool allow_mode_change, PlacementStrategy placement) {
        InsertChoice best;
        std::vector<std::pair<int, int>> modes;
        if (allow_mode_change) {
            modes = {{c.mode_left, c.mode_right}, {0, 0}, {0, 1}, {1, 0}, {1, 1}};
        } else {
            modes = {{c.mode_left, c.mode_right}};
        }

        for (const auto& mode : modes) {
            for (int f : {0, 1}) {
                for (size_t i = 0; i <= c.route[f].size(); ++i) {
                    Carriage temp;
                    temp.copy_construct(c);
                    temp.mode_left = mode.first;
                    temp.mode_right = mode.second;
                    double before = temp.obj;
                    if (!InsertCustomer(temp, v, static_cast<int>(i), p, false, f)) {
                        continue;
                    }

                    double delta = temp.obj - before;
                    double score = delta;
                    if (placement == PlacementStrategy::Balance) {
                        score = temp.actual_length;
                    }
                    int mode_changes = (mode.first != c.mode_left ? 1 : 0) + (mode.second != c.mode_right ? 1 : 0);
                    bool better = false;
                    if (!best.feasible) {
                        better = true;
                    } else if (placement == PlacementStrategy::Pack) {
                        better = score > best.placement_score + 1e-9;
                    } else {
                        better = score < best.placement_score - 1e-9;
                    }
                    if (!better && std::abs(score - best.placement_score) <= 1e-9) {
                        better = mode_changes < best.mode_changes;
                    }
                    if (better) {
                        best.feasible = true;
                        best.floor = f;
                        best.place = static_cast<int>(i);
                        best.mode_left = mode.first;
                        best.mode_right = mode.second;
                        best.delta_obj = delta;
                        best.placement_score = score;
                        best.mode_changes = mode_changes;
                    }
                }
            }
        }
        return best;
    }

    InsertChoice BestInsertion(std::vector<Carriage*>& cnode, Vehicle* v, Problem* p, bool allow_mode_change, PlacementStrategy placement) {
        InsertChoice best;
        for (auto c : cnode) {
            InsertChoice choice = BestInsertion(*c, v, p, allow_mode_change, placement);
            if (!choice.feasible) continue;
            choice.carriage = c;
            bool better = false;
            if (!best.feasible) {
                better = true;
            } else if (placement == PlacementStrategy::Pack) {
                better = choice.placement_score > best.placement_score + 1e-9;
            } else {
                better = choice.placement_score < best.placement_score - 1e-9;
            }
            if (!better && std::abs(choice.placement_score - best.placement_score) <= 1e-9) {
                better = choice.mode_changes < best.mode_changes;
            }
            if (better) {
                best = choice;
            }
        }
        return best;
    }

    bool ApplyInsertion(const InsertChoice& choice, Vehicle* v, Problem* p) {
        if (!choice.feasible || choice.carriage == nullptr) return false;
        choice.carriage->mode_left = choice.mode_left;
        choice.carriage->mode_right = choice.mode_right;
        return InsertCustomer(*choice.carriage, v, choice.place, p, false, choice.floor);
    }

    bool InsertMandatory(Solution& result, Problem* p, OrderStrategy strategy, PlacementStrategy placement) {
        std::vector<Vehicle*> vnode = OrderedVehicles(p, strategy);

        std::vector<Carriage*> cnode;
        for(auto& c : result.carriage) cnode.push_back(&c);

        for (auto v : vnode) {
            while (v->var_mandatory > 0) {
                InsertChoice best_insert = BestInsertion(cnode, v, p, true, placement);
                if (!ApplyInsertion(best_insert, v, p)) {
                    return false;
                }
                v->var_mandatory -= 1;
                result.CalculateSolutionObj(p);
            }
        }
        return true;
    }

    int CountFeasibleInsertions(std::vector<Carriage*>& cnode, Vehicle* v, Problem* p) {
        int count = 0;
        for (auto c : cnode) {
            for (int ml : {0, 1}) {
                for (int mr : {0, 1}) {
                    for (int f : {0, 1}) {
                        for (size_t pos = 0; pos <= c->route[f].size(); ++pos) {
                            Carriage temp;
                            temp.copy_construct(*c);
                            temp.mode_left = ml;
                            temp.mode_right = mr;
                            if (InsertCustomer(temp, v, static_cast<int>(pos), p, false, f)) {
                                count++;
                            }
                        }
                    }
                }
            }
        }
        return count;
    }

    bool InsertMandatoryDynamic(Solution& result, Problem* p, PlacementStrategy placement) {
        std::vector<Carriage*> cnode;
        for(auto& c : result.carriage) cnode.push_back(&c);

        int remaining = 0;
        for (const auto& v : p->vehicle) remaining += v.var_mandatory;

        while (remaining > 0) {
            Vehicle* selected = nullptr;
            int best_count = std::numeric_limits<int>::max();
            for (auto& vehicle : p->vehicle) {
                if (vehicle.var_mandatory <= 0) continue;
                int feasible_count = CountFeasibleInsertions(cnode, &vehicle, p);
                if (feasible_count <= 0) {
                    return false;
                }
                bool better = selected == nullptr || feasible_count < best_count;
                if (!better && feasible_count == best_count) {
                    if (vehicle.height != selected->height) {
                        better = vehicle.height > selected->height;
                    } else if (vehicle.length != selected->length) {
                        better = vehicle.length > selected->length;
                    } else {
                        better = vehicle.var_mandatory > selected->var_mandatory;
                    }
                }
                if (better) {
                    selected = &vehicle;
                    best_count = feasible_count;
                }
            }

            if (selected == nullptr) return false;
            InsertChoice best_insert = BestInsertion(cnode, selected, p, true, placement);
            if (!ApplyInsertion(best_insert, selected, p)) {
                return false;
            }
            selected->var_mandatory -= 1;
            remaining--;
            result.CalculateSolutionObj(p);
        }
        return true;
    }

    void InsertOptional(Solution& result, Problem* p) {
        std::vector<Vehicle*> vnode;
        for(auto& v : p->vehicle) vnode.push_back(&v);

        std::sort(vnode.begin(), vnode.end(), [](Vehicle* a, Vehicle* b) {
            return a->var_optional > b->var_optional;
        });

        std::vector<Carriage*> cnode;
        for(auto& c : result.carriage) cnode.push_back(&c);

        for (auto v : vnode) {
            while (v->var_optional > 0) {
                InsertChoice best_insert = BestInsertion(cnode, v, p, true, PlacementStrategy::Pack);
                if (!ApplyInsertion(best_insert, v, p)) {
                    break;
                }
                v->var_optional -= 1;
                result.CalculateSolutionObj(p);
            }
        }
    }

public:
    bool RepairRemaining(Solution& result, Problem* p) {
        std::vector<Problem::VehicleState> initial_state = p->BackupState();
        Solution original;
        original.copy_construct(result);

        std::vector<OrderStrategy> strategies = {
            OrderStrategy::ScarceTallLong,
            OrderStrategy::TallLong,
            OrderStrategy::LongTall,
            OrderStrategy::CountTallLong,
            OrderStrategy::CountOnly,
        };

        std::vector<PlacementStrategy> placement_strategies = {
            PlacementStrategy::Pack,
            PlacementStrategy::Spread,
            PlacementStrategy::Balance,
        };

        for (OrderStrategy strategy : strategies) {
            for (PlacementStrategy placement : placement_strategies) {
                p->RestoreState(initial_state);
                Solution candidate;
                candidate.copy_construct(original);
                if (!InsertMandatory(candidate, p, strategy, placement)) {
                    continue;
                }
                InsertOptional(candidate, p);
                candidate.CalculateSolutionObj(p);
                result.copy_construct(candidate);
                return true;
            }
        }

        for (PlacementStrategy placement : placement_strategies) {
            p->RestoreState(initial_state);
            Solution candidate;
            candidate.copy_construct(original);
            if (!InsertMandatoryDynamic(candidate, p, placement)) {
                continue;
            }
            InsertOptional(candidate, p);
            candidate.CalculateSolutionObj(p);
            result.copy_construct(candidate);
            return true;
        }

        p->RestoreState(initial_state);
        result.copy_construct(original);
        return false;
    }

    bool Construct(Solution& result, Problem* p) {
        std::vector<Problem::VehicleState> initial_state = p->BackupState();
        std::vector<OrderStrategy> strategies = {
            OrderStrategy::ScarceTallLong,
            OrderStrategy::TallLong,
            OrderStrategy::LongTall,
            OrderStrategy::CountTallLong,
            OrderStrategy::CountOnly,
        };

        std::vector<PlacementStrategy> placement_strategies = {
            PlacementStrategy::Pack,
            PlacementStrategy::Spread,
            PlacementStrategy::Balance,
        };
        std::vector<std::pair<int, int>> initial_modes = {
            {0, 0},
            {1, 1},
            {0, 1},
            {1, 0},
        };

        for (const auto& initial_mode : initial_modes) {
            for (OrderStrategy strategy : strategies) {
                for (PlacementStrategy placement : placement_strategies) {
                    p->RestoreState(initial_state);
                    Solution candidate(p->carriage_num);
                    for (auto& c : candidate.carriage) {
                        c.mode_left = initial_mode.first;
                        c.mode_right = initial_mode.second;
                    }
                    if (!InsertMandatory(candidate, p, strategy, placement)) {
                        continue;
                    }
                    InsertOptional(candidate, p);
                    candidate.CalculateSolutionObj(p);
                    result.copy_construct(candidate);
                    return true;
                }
            }
        }

        for (const auto& initial_mode : initial_modes) {
            for (PlacementStrategy placement : placement_strategies) {
                p->RestoreState(initial_state);
                Solution candidate(p->carriage_num);
                for (auto& c : candidate.carriage) {
                    c.mode_left = initial_mode.first;
                    c.mode_right = initial_mode.second;
                }
                if (!InsertMandatoryDynamic(candidate, p, placement)) {
                    continue;
                }
                InsertOptional(candidate, p);
                candidate.CalculateSolutionObj(p);
                result.copy_construct(candidate);
                return true;
            }
        }

        p->RestoreState(initial_state);
        return false;
    }

    Solution* Solve(Problem* p) {
        Solution* result = new Solution(p->carriage_num);
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
