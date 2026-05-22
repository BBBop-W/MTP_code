#include "Problem.h"
#include "BestInsert.h"
#include "Insert.h"
#include "Feasibility.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct InsertChoice {
    bool feasible = false;
    int carriage_idx = -1;
    int floor = -1;
    int place = -1;
    int mode_left = 0;
    int mode_right = 0;
    double delta_obj = 0.0;
};

struct AttemptCounts {
    int current_mode_positions = 0;
    int any_mode_positions = 0;
    int empty_hh_positions = 0;
    int empty_any_mode_positions = 0;
};

static std::string mode_name(int left, int right) {
    return std::string(left ? "m" : "h") + "-" + (right ? "m" : "h");
}

static std::string vehicle_name(const Vehicle* v) {
    std::ostringstream oss;
    oss << v->id << " " << v->brand << " " << v->model
        << " L=" << v->length << " H=" << v->height;
    return oss.str();
}

static InsertChoice best_to_route_any_mode(Carriage& c, Vehicle* v, Problem* p, bool allow_mode_change) {
    InsertChoice best;
    double best_delta = -1e100;
    std::vector<std::pair<int, int>> modes;
    if (allow_mode_change) {
        modes = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
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
                if (!best.feasible || delta > best_delta) {
                    best.feasible = true;
                    best.floor = f;
                    best.place = static_cast<int>(i);
                    best.mode_left = mode.first;
                    best.mode_right = mode.second;
                    best.delta_obj = delta;
                    best_delta = delta;
                }
            }
        }
    }
    return best;
}

static AttemptCounts count_attempts(
    const std::vector<Carriage*>& cnode,
    Vehicle* v,
    Problem* p
) {
    AttemptCounts counts;
    for (size_t ci = 0; ci < cnode.size(); ++ci) {
        Carriage* c = cnode[ci];
        for (int f : {0, 1}) {
            for (size_t place = 0; place <= c->route[f].size(); ++place) {
                Carriage temp;
                temp.copy_construct(*c);
                if (InsertCustomer(temp, v, static_cast<int>(place), p, false, f)) {
                    counts.current_mode_positions += 1;
                }
            }
        }
        for (int ml : {0, 1}) {
            for (int mr : {0, 1}) {
                for (int f : {0, 1}) {
                    for (size_t place = 0; place <= c->route[f].size(); ++place) {
                        Carriage temp;
                        temp.copy_construct(*c);
                        temp.mode_left = ml;
                        temp.mode_right = mr;
                        if (InsertCustomer(temp, v, static_cast<int>(place), p, false, f)) {
                            counts.any_mode_positions += 1;
                        }
                    }
                }
            }
        }
    }

    for (int ml : {0, 1}) {
        for (int mr : {0, 1}) {
            for (int f : {0, 1}) {
                Carriage empty;
                empty.mode_left = ml;
                empty.mode_right = mr;
                if (InsertCustomer(empty, v, 0, p, false, f)) {
                    counts.empty_any_mode_positions += 1;
                    if (ml == 0 && mr == 0) {
                        counts.empty_hh_positions += 1;
                    }
                }
            }
        }
    }
    return counts;
}

static void print_carriage_loads(const std::vector<Carriage*>& cnode, Problem* p) {
    for (size_t idx = 0; idx < cnode.size(); ++idx) {
        Carriage* c = cnode[idx];
        double len0 = 0.0;
        double len1 = 0.0;
        for (int id : c->route[0]) len0 += p->vehicle[id].length + c->spacing;
        for (int id : c->route[1]) len1 += p->vehicle[id].length + c->spacing;
        std::cout << "    carriage " << idx + 1
                  << " mode=" << mode_name(c->mode_left, c->mode_right)
                  << " upper_n=" << c->route[0].size() << " upper_len+space=" << len0
                  << " lower_n=" << c->route[1].size() << " lower_len+space=" << len1
                  << "\n";
    }
}

static bool construct_variant(Problem* p, bool allow_mode_change, bool verbose_failure) {
    std::vector<Vehicle*> vnode;
    for (auto& v : p->vehicle) vnode.push_back(&v);
    std::sort(vnode.begin(), vnode.end(), [](Vehicle* a, Vehicle* b) {
        return a->var_mandatory > b->var_mandatory;
    });

    Solution result(p->carriage_num);
    std::vector<Carriage*> cnode;
    for (auto& c : result.carriage) cnode.push_back(&c);
    std::sort(cnode.begin(), cnode.end(), [](Carriage* a, Carriage* b) {
        return a->length() > b->length();
    });

    int inserted_mandatory = 0;
    for (auto v : vnode) {
        int original_count = v->var_mandatory;
        while (v->var_mandatory > 0) {
            bool inserted = false;
            for (size_t ci = 0; ci < cnode.size(); ++ci) {
                InsertChoice choice = best_to_route_any_mode(*cnode[ci], v, p, allow_mode_change);
                if (!choice.feasible) continue;

                cnode[ci]->mode_left = choice.mode_left;
                cnode[ci]->mode_right = choice.mode_right;
                if (!InsertCustomer(*cnode[ci], v, choice.place, p, false, choice.floor)) {
                    continue;
                }
                v->var_mandatory -= 1;
                inserted_mandatory += 1;
                result.CalculateSolutionObj(p);
                inserted = true;
                break;
            }
            if (!inserted) {
                if (verbose_failure) {
                    AttemptCounts counts = count_attempts(cnode, v, p);
                    std::cout << "  fail_vehicle=" << vehicle_name(v) << "\n";
                    std::cout << "  original_type_mandatory=" << original_count
                              << " remaining_of_type=" << v->var_mandatory
                              << " inserted_mandatory_total=" << inserted_mandatory << "\n";
                    std::cout << "  feasible_positions_current_modes="
                              << counts.current_mode_positions
                              << " feasible_positions_any_mode="
                              << counts.any_mode_positions << "\n";
                    std::cout << "  vehicle_alone_empty_hh_positions="
                              << counts.empty_hh_positions
                              << " vehicle_alone_empty_any_mode_positions="
                              << counts.empty_any_mode_positions << "\n";
                    print_carriage_loads(cnode, p);
                }
                return false;
            }
        }
    }
    return true;
}

static void run_case(const std::string& instance_name, int num_splits, bool independent_mode) {
    GLOBAL_GEOM.num_splits = num_splits;
    GLOBAL_GEOM.independent_mode = independent_mode;
    GLOBAL_GEOM.initialized = false;

    Problem original_problem;
    original_problem.LoadVRPTW(instance_name);
    GLOBAL_GEOM.init(&original_problem);

    const auto& lower_blocks = GLOBAL_GEOM.get_blocks(false);
    const auto& upper_blocks = GLOBAL_GEOM.get_blocks(true);

    std::cout << "\n== " << instance_name
              << " num_splits=" << num_splits
              << " independent_mode=" << (independent_mode ? 1 : 0)
              << " vehicle_types=" << original_problem.vehicle_types
              << " carriage_num=" << original_problem.carriage_num << " ==\n";
    std::cout << "  geometry lower_side_blocks=" << (lower_blocks.size() - 1)
              << " upper_side_blocks=" << (upper_blocks.size() - 1) << "\n";
    std::cout << "  lower_block_lengths:";
    for (const auto& block : lower_blocks) std::cout << " " << block.length;
    std::cout << "\n";
    std::cout << "  upper_block_lengths:";
    for (const auto& block : upper_blocks) std::cout << " " << block.length;
    std::cout << "\n";

    Problem improved_problem = original_problem;
    GLOBAL_GEOM.initialized = false;
    Solution improved_solution(improved_problem.carriage_num);
    BestInsert improved_bi;
    bool improved_ok = improved_bi.Construct(improved_solution, &improved_problem);
    improved_solution.CalculateSolutionObj(&improved_problem);
    std::cout << "  improved_BI=" << (improved_ok ? "OK" : "FAIL")
              << " loaded_length=" << (improved_ok ? improved_solution.actual_length : 0.0)
              << "\n";

    Problem hh_problem = original_problem;
    GLOBAL_GEOM.initialized = false;
    bool hh_ok = construct_variant(&hh_problem, false, true);
    std::cout << "  original_hh_only_BI=" << (hh_ok ? "OK" : "FAIL") << "\n";

    Problem mode_problem = original_problem;
    GLOBAL_GEOM.initialized = false;
    bool mode_ok = construct_variant(&mode_problem, true, true);
    std::cout << "  allow_mode_change_BI=" << (mode_ok ? "OK" : "FAIL") << "\n";
}

int main(int argc, char* argv[]) {
    int num_splits = 3;
    bool independent_mode = false;
    std::vector<std::string> instances;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num-splits" && i + 1 < argc) {
            num_splits = std::stoi(argv[++i]);
        } else if (arg == "--independent-mode" && i + 1 < argc) {
            independent_mode = std::stoi(argv[++i]) != 0;
        } else {
            instances.push_back(arg);
        }
    }

    if (instances.empty()) {
        for (int i = 1; i <= 10; ++i) {
            std::ostringstream oss;
            oss << "real_case_2026-05-21/case_" << std::setw(2) << std::setfill('0') << i;
            instances.push_back(oss.str());
        }
    }

    for (const auto& instance : instances) {
        run_case(instance, num_splits, independent_mode);
    }
    return 0;
}
