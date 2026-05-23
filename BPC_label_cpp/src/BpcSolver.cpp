#include "bpc_label/BpcSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "gurobi_c++.h"
#include "bpc_label/SolverPricing.hpp"
#include "bpc_label/Warmstart.hpp"

namespace bpc_label {

namespace {

constexpr double INF = 1e90;
std::size_t deck_index(DeckMode mode) {
    switch (mode) {
        case DeckMode::HH:
            return 0;
        case DeckMode::HM:
            return 1;
        case DeckMode::MH:
            return 2;
        case DeckMode::MM:
            return 3;
    }
    return 0;
}

double now_sec() {
    return monotonic_seconds();
}

bool finite_lower(double value) {
    return value > -INF / 2.0;
}

bool finite_upper(double value) {
    return value < INF / 2.0;
}

bool acceptable_status(int status, const GRBModel& model) {
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {
        return true;
    }
    if (status == GRB_TIME_LIMIT) {
        return model.get(GRB_IntAttr_SolCount) > 0;
    }
    return false;
}

void configure_model(GRBModel& model, double time_limit, int threads, bool log_to_console) {
    model.set(GRB_IntParam_OutputFlag, log_to_console ? 1 : 0);
    if (threads > 0) {
        model.set(GRB_IntParam_Threads, threads);
    }
    if (time_limit > 0.0 && time_limit < INF / 2.0) {
        model.set(GRB_DoubleParam_TimeLimit, time_limit);
    }
}

double safe_obj_val(const GRBModel& model) {
    return model.get(GRB_DoubleAttr_ObjVal);
}

std::string seq_id(const std::string& prefix, int seq) {
    std::ostringstream oss;
    oss << prefix << "_" << seq;
    return oss.str();
}

void accumulate_pricing_stats(PricingStats& dst, const PricingStats& src) {
    dst.generated_subpatterns += src.generated_subpatterns;
    dst.merge_attempt_pairs += src.merge_attempt_pairs;
    dst.labeling_stats.labels_generated_raw += src.labeling_stats.labels_generated_raw;
    dst.labeling_stats.labels_feasible += src.labeling_stats.labels_feasible;
    dst.labeling_stats.labels_pruned_by_bound += src.labeling_stats.labels_pruned_by_bound;
    dst.labeling_stats.labels_pruned_by_dominance += src.labeling_stats.labels_pruned_by_dominance;
    dst.labeling_stats.labels_after_dominance += src.labeling_stats.labels_after_dominance;
    dst.labeling_stats.labels_avoided_by_d2 += src.labeling_stats.labels_avoided_by_d2;
    dst.labeling_stats.hybrid_calls += src.labeling_stats.hybrid_calls;
    dst.labeling_stats.hybrid_ordered_type_sum += src.labeling_stats.hybrid_ordered_type_sum;
    dst.labeling_stats.hybrid_ordered_type_max = std::max(
        dst.labeling_stats.hybrid_ordered_type_max,
        src.labeling_stats.hybrid_ordered_type_max
    );
    dst.labeling_stats.hybrid_ordered_quantity_sum += src.labeling_stats.hybrid_ordered_quantity_sum;
    dst.labeling_stats.hybrid_total_quantity_sum += src.labeling_stats.hybrid_total_quantity_sum;
}

int sr_coeff_for_column(
    const PatternColumn& column,
    const std::array<int, 3>& subset,
    const InstanceData& instance
) {
    int val = 0;
    for (int type_id : subset) {
        auto it = std::find(instance.car_types.begin(), instance.car_types.end(), type_id);
        if (it == instance.car_types.end()) {
            continue;
        }
        const std::size_t idx = static_cast<std::size_t>(std::distance(instance.car_types.begin(), it));
        if (column.quantities[idx] > static_cast<double>(instance.upper_bounds[idx]) / 2.0) {
            val += 1;
        }
    }
    return static_cast<int>(std::floor(0.5 * static_cast<double>(val)));
}

}  // namespace

std::string method_name(BpcMethod method) {
    return method == BpcMethod::WagonLabel ? "wagon" : "compartment";
}

BpcMethod parse_method(const std::string& method) {
    if (method == "wagon" || method == "wagon_label" || method == "bpc_wagon_label") {
        return BpcMethod::WagonLabel;
    }
    if (method == "compartment" || method == "compartment_label" || method == "bpc_compartment_label") {
        return BpcMethod::CompartmentLabel;
    }
    throw std::invalid_argument("unknown method: " + method);
}

std::string pricing_backend_name(PricingBackend backend) {
    return backend == PricingBackend::Label ? "label" : "solver";
}

PricingBackend parse_pricing_backend(const std::string& backend) {
    if (backend == "label" || backend == "labeling" || backend == "merge" || backend == "merging") {
        return PricingBackend::Label;
    }
    if (backend == "solver" || backend == "mip" || backend == "gurobi") {
        return PricingBackend::Solver;
    }
    throw std::invalid_argument("unknown pricing backend: " + backend);
}

MasterProblem::MasterProblem(const InstanceData& instance, BpcMethod method)
    : instance_(instance), method_(method) {}

void MasterProblem::seed_initial_columns() {
    if (method_ == BpcMethod::WagonLabel) {
        for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
            std::vector<int> q(instance_.car_types.size(), 0);
            q[i] = 1;
            columns_.push_back(PatternColumn{
                "seed_i" + std::to_string(instance_.car_types[i]),
                q,
                -instance_.lengths[i],
                CompartmentKind::Lower,
                DeckMode::HH,
                false,
            });
        }
        return;
    }

    for (DeckMode deck : compartment_deck_order()) {
        for (CompartmentKind compartment : {CompartmentKind::Upper, CompartmentKind::Lower}) {
            columns_.push_back(PatternColumn{
                "empty_" + compartment_name(compartment) + "_" + deck_name(deck),
                std::vector<int>(instance_.car_types.size(), 0),
                0.0,
                compartment,
                deck,
                true,
            });
        }
    }
}

bool MasterProblem::has_equivalent_column(const PatternColumn& column) const {
    for (const auto& old : columns_) {
        if (method_ == BpcMethod::WagonLabel) {
            if (old.quantities == column.quantities) {
                return true;
            }
        } else {
            if (old.is_compartment_column == column.is_compartment_column &&
                old.compartment == column.compartment &&
                old.deck == column.deck &&
                old.quantities == column.quantities) {
                return true;
            }
        }
    }
    return false;
}

bool MasterProblem::add_column(const PatternColumn& column) {
    if (has_equivalent_column(column)) {
        return false;
    }
    columns_.push_back(column);
    return true;
}

MasterLPSolution MasterProblem::solve_lp(
    const std::map<int, std::pair<double, double>>& branch_a_bounds,
    const std::map<int, std::pair<double, double>>& branch_q_bounds,
    const std::set<std::array<int, 3>>& active_sr_cuts,
    bool use_capacity_cut,
    double time_limit,
    int threads,
    bool log_to_console
) const {
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    configure_model(model, time_limit, threads, log_to_console);

    std::vector<GRBVar> theta;
    theta.reserve(columns_.size());
    for (const auto& col : columns_) {
        theta.push_back(model.addVar(0.0, GRB_INFINITY, col.cost, GRB_CONTINUOUS, col.id));
    }

    std::vector<GRBVar> unmet;
    unmet.reserve(instance_.car_types.size());
    for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
        unmet.push_back(model.addVar(0.0, GRB_INFINITY, penalty_unmet_, GRB_CONTINUOUS, "unmet"));
    }

    std::vector<GRBConstr> mandatory_constr(instance_.car_types.size());
    std::vector<GRBConstr> optional_constr(instance_.car_types.size());
    for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
        GRBLinExpr expr = unmet[i];
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const int q = columns_[c].quantities[i];
            if (q != 0) {
                expr += static_cast<double>(q) * theta[c];
            }
        }
        mandatory_constr[i] = model.addConstr(expr >= instance_.mandatory[i], "mandatory");

        GRBLinExpr opt_expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const int q = columns_[c].quantities[i];
            if (q != 0) {
                opt_expr += static_cast<double>(q) * theta[c];
            }
        }
        optional_constr[i] = model.addConstr(opt_expr <= instance_.upper_bounds[i], "optional");
    }

    GRBConstr wagon_constr;
    std::array<GRBConstr, 4> map_constr{};
    if (method_ == BpcMethod::WagonLabel) {
        GRBLinExpr wagon_expr = 0.0;
        for (GRBVar& var : theta) {
            wagon_expr += var;
        }
        wagon_constr = model.addConstr(wagon_expr <= instance_.carriage_num, "wagon_limit");
    } else {
        for (DeckMode deck : compartment_deck_order()) {
            GRBLinExpr expr = 0.0;
            for (std::size_t c = 0; c < columns_.size(); ++c) {
                if (columns_[c].deck != deck) {
                    continue;
                }
                expr += (columns_[c].compartment == CompartmentKind::Upper ? 1.0 : -1.0) * theta[c];
            }
            map_constr[deck_index(deck)] = model.addConstr(expr == 0.0, "wagon_map");
        }
        GRBLinExpr wagon_expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            if (columns_[c].compartment == CompartmentKind::Upper) {
                wagon_expr += theta[c];
            }
        }
        wagon_constr = model.addConstr(wagon_expr <= instance_.carriage_num, "wagon_limit");
    }

    std::vector<std::pair<int, GRBConstr>> branch_a_constr;
    std::vector<std::pair<int, GRBConstr>> branch_q_constr;
    for (const auto& [type_idx, bounds] : branch_a_bounds) {
        GRBLinExpr expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            if (columns_[c].quantities[static_cast<std::size_t>(type_idx)] > 0) {
                expr += theta[c];
            }
        }
        if (finite_lower(bounds.first)) {
            branch_a_constr.push_back({type_idx, model.addConstr(expr >= bounds.first, "branch_a_lb")});
        }
        if (finite_upper(bounds.second)) {
            branch_a_constr.push_back({type_idx, model.addConstr(expr <= bounds.second, "branch_a_ub")});
        }
    }

    for (const auto& [type_idx, bounds] : branch_q_bounds) {
        GRBLinExpr expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const int q = columns_[c].quantities[static_cast<std::size_t>(type_idx)];
            if (q != 0) {
                expr += static_cast<double>(q) * theta[c];
            }
        }
        if (finite_lower(bounds.first)) {
            branch_q_constr.push_back({type_idx, model.addConstr(expr >= bounds.first, "branch_q_lb")});
        }
        if (finite_upper(bounds.second)) {
            branch_q_constr.push_back({type_idx, model.addConstr(expr <= bounds.second, "branch_q_ub")});
        }
    }

    GRBConstr capacity_constr;
    bool has_capacity_constr = false;
    if (use_capacity_cut) {
        const int total_mandatory = std::accumulate(instance_.mandatory.begin(), instance_.mandatory.end(), 0);
        const int denominator = 2 * Config::max_units_per_compartment;
        const int min_wagons = static_cast<int>(std::ceil(static_cast<double>(total_mandatory) / denominator));

        GRBLinExpr expr = 0.0;
        if (method_ == BpcMethod::WagonLabel) {
            for (GRBVar& var : theta) {
                expr += var;
            }
        } else {
            for (std::size_t c = 0; c < columns_.size(); ++c) {
                if (columns_[c].compartment == CompartmentKind::Upper) {
                    expr += theta[c];
                }
            }
        }
        capacity_constr = model.addConstr(expr >= min_wagons, "capacity_cut");
        has_capacity_constr = true;
    }

    std::vector<std::pair<std::array<int, 3>, GRBConstr>> sr_constr;
    for (const auto& subset : active_sr_cuts) {
        GRBLinExpr expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const int coeff = sr_coeff_for_column(columns_[c], subset, instance_);
            if (coeff > 0) {
                expr += static_cast<double>(coeff) * theta[c];
            }
        }
        sr_constr.push_back({subset, model.addConstr(expr <= 1.0, "sr_cut")});
    }

    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model.optimize();

    const int status = model.get(GRB_IntAttr_Status);
    MasterLPSolution sol;
    sol.status = status;
    if (!acceptable_status(status, model)) {
        return sol;
    }

    sol.has_solution = true;
    sol.objective = safe_obj_val(model);
    sol.theta.resize(columns_.size(), 0.0);
    for (std::size_t c = 0; c < columns_.size(); ++c) {
        sol.theta[c] = theta[c].get(GRB_DoubleAttr_X);
    }
    sol.unmet.resize(instance_.car_types.size(), 0.0);
    for (std::size_t i = 0; i < unmet.size(); ++i) {
        sol.unmet[i] = unmet[i].get(GRB_DoubleAttr_X);
    }

    sol.dual_alpha.resize(instance_.car_types.size(), 0.0);
    sol.dual_beta.resize(instance_.car_types.size(), 0.0);
    for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
        sol.dual_alpha[i] = mandatory_constr[i].get(GRB_DoubleAttr_Pi);
        sol.dual_beta[i] = optional_constr[i].get(GRB_DoubleAttr_Pi);
    }
    if (method_ == BpcMethod::WagonLabel) {
        sol.dual_gamma = wagon_constr.get(GRB_DoubleAttr_Pi);
    } else {
        for (DeckMode deck : compartment_deck_order()) {
            sol.dual_gamma_by_deck[deck_index(deck)] = map_constr[deck_index(deck)].get(GRB_DoubleAttr_Pi);
        }
        sol.dual_kappa = wagon_constr.get(GRB_DoubleAttr_Pi);
    }

    sol.dual_branch_a.resize(instance_.car_types.size(), 0.0);
    sol.dual_branch_q.resize(instance_.car_types.size(), 0.0);
    for (const auto& [idx, constr] : branch_a_constr) {
        sol.dual_branch_a[static_cast<std::size_t>(idx)] += constr.get(GRB_DoubleAttr_Pi);
    }
    for (const auto& [idx, constr] : branch_q_constr) {
        sol.dual_branch_q[static_cast<std::size_t>(idx)] += constr.get(GRB_DoubleAttr_Pi);
    }
    if (has_capacity_constr) {
        sol.dual_eta = capacity_constr.get(GRB_DoubleAttr_Pi);
    }
    for (const auto& [subset, constr] : sr_constr) {
        sol.dual_sigma.push_back(TripletSigma{subset, constr.get(GRB_DoubleAttr_Pi)});
    }
    return sol;
}

MasterIPSolution MasterProblem::solve_restricted_ip(
    const std::map<int, std::pair<double, double>>& branch_a_bounds,
    const std::map<int, std::pair<double, double>>& branch_q_bounds,
    double time_limit,
    int threads,
    bool log_to_console
) const {
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    configure_model(model, time_limit, threads, log_to_console);

    std::vector<GRBVar> theta;
    theta.reserve(columns_.size());
    for (const auto& col : columns_) {
        theta.push_back(model.addVar(0.0, GRB_INFINITY, col.cost, GRB_INTEGER, col.id));
    }

    std::vector<GRBVar> unmet;
    unmet.reserve(instance_.car_types.size());
    for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
        unmet.push_back(model.addVar(0.0, GRB_INFINITY, penalty_unmet_, GRB_CONTINUOUS, "unmet"));
    }

    for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
        GRBLinExpr expr = unmet[i];
        GRBLinExpr opt_expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const int q = columns_[c].quantities[i];
            if (q != 0) {
                expr += static_cast<double>(q) * theta[c];
                opt_expr += static_cast<double>(q) * theta[c];
            }
        }
        model.addConstr(expr >= instance_.mandatory[i], "mandatory");
        model.addConstr(opt_expr <= instance_.upper_bounds[i], "optional");
    }

    if (method_ == BpcMethod::WagonLabel) {
        GRBLinExpr wagon_expr = 0.0;
        for (GRBVar& var : theta) {
            wagon_expr += var;
        }
        model.addConstr(wagon_expr <= instance_.carriage_num, "wagon_limit");
    } else {
        for (DeckMode deck : compartment_deck_order()) {
            GRBLinExpr expr = 0.0;
            for (std::size_t c = 0; c < columns_.size(); ++c) {
                if (columns_[c].deck != deck) {
                    continue;
                }
                expr += (columns_[c].compartment == CompartmentKind::Upper ? 1.0 : -1.0) * theta[c];
            }
            model.addConstr(expr == 0.0, "wagon_map");
        }
        GRBLinExpr wagon_expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            if (columns_[c].compartment == CompartmentKind::Upper) {
                wagon_expr += theta[c];
            }
        }
        model.addConstr(wagon_expr <= instance_.carriage_num, "wagon_limit");
    }

    for (const auto& [type_idx, bounds] : branch_a_bounds) {
        GRBLinExpr expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            if (columns_[c].quantities[static_cast<std::size_t>(type_idx)] > 0) {
                expr += theta[c];
            }
        }
        if (finite_lower(bounds.first)) {
            model.addConstr(expr >= bounds.first, "branch_a_lb");
        }
        if (finite_upper(bounds.second)) {
            model.addConstr(expr <= bounds.second, "branch_a_ub");
        }
    }
    for (const auto& [type_idx, bounds] : branch_q_bounds) {
        GRBLinExpr expr = 0.0;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const int q = columns_[c].quantities[static_cast<std::size_t>(type_idx)];
            if (q != 0) {
                expr += static_cast<double>(q) * theta[c];
            }
        }
        if (finite_lower(bounds.first)) {
            model.addConstr(expr >= bounds.first, "branch_q_lb");
        }
        if (finite_upper(bounds.second)) {
            model.addConstr(expr <= bounds.second, "branch_q_ub");
        }
    }

    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model.optimize();

    const int status = model.get(GRB_IntAttr_Status);
    MasterIPSolution sol;
    sol.status = status;
    if (!acceptable_status(status, model)) {
        return sol;
    }
    sol.has_solution = true;
    sol.objective = safe_obj_val(model);
    sol.theta.resize(columns_.size(), 0.0);
    for (std::size_t c = 0; c < columns_.size(); ++c) {
        sol.theta[c] = theta[c].get(GRB_DoubleAttr_X);
    }
    sol.unmet.resize(instance_.car_types.size(), 0.0);
    for (std::size_t i = 0; i < unmet.size(); ++i) {
        sol.unmet[i] = unmet[i].get(GRB_DoubleAttr_X);
    }
    return sol;
}

bool MasterProblem::has_unmet_demand(const MasterLPSolution& solution, double eps) const {
    return std::any_of(solution.unmet.begin(), solution.unmet.end(), [eps](double v) { return v > eps; });
}

bool MasterProblem::has_unmet_demand(const MasterIPSolution& solution, double eps) const {
    return std::any_of(solution.unmet.begin(), solution.unmet.end(), [eps](double v) { return v > eps; });
}

std::tuple<char, int, double> MasterProblem::choose_branch_var(const MasterLPSolution& solution, double eps) const {
    std::vector<double> a_sums(instance_.car_types.size(), 0.0);
    std::vector<double> q_sums(instance_.car_types.size(), 0.0);

    for (std::size_t c = 0; c < columns_.size(); ++c) {
        const double value = c < solution.theta.size() ? solution.theta[c] : 0.0;
        if (value <= eps) {
            continue;
        }
        for (std::size_t i = 0; i < instance_.car_types.size(); ++i) {
            const int q = columns_[c].quantities[i];
            if (q > 0) {
                a_sums[i] += value;
                q_sums[i] += static_cast<double>(q) * value;
            }
        }
    }

    for (std::size_t i = 0; i < a_sums.size(); ++i) {
        if (std::abs(a_sums[i] - std::round(a_sums[i])) > eps) {
            return {'a', static_cast<int>(i), a_sums[i]};
        }
    }
    for (std::size_t i = 0; i < q_sums.size(); ++i) {
        if (std::abs(q_sums[i] - std::round(q_sums[i])) > eps) {
            return {'q', static_cast<int>(i), q_sums[i]};
        }
    }
    return {'-', -1, 0.0};
}

bool MasterProblem::has_branch_candidate(const MasterLPSolution& solution, double eps) const {
    return std::get<0>(choose_branch_var(solution, eps)) != '-';
}

bool MasterProblem::is_integral(const MasterLPSolution& solution, double eps) const {
    if (!solution.has_solution || has_unmet_demand(solution, eps)) {
        return false;
    }
    for (double value : solution.theta) {
        if (value > eps && std::abs(value - std::round(value)) > eps) {
            return false;
        }
    }
    return !has_branch_candidate(solution, eps);
}

MasterSnapshot MasterProblem::snapshot_from_solution(const MasterLPSolution& solution) const {
    MasterSnapshot snapshot;
    snapshot.car_types = instance_.car_types;
    snapshot.car_lengths = instance_.lengths;
    snapshot.car_heights = instance_.heights;
    snapshot.max_total_by_type = instance_.upper_bounds;
    snapshot.dual_alpha = solution.dual_alpha;
    snapshot.dual_beta = solution.dual_beta;
    snapshot.dual_branch_a = solution.dual_branch_a;
    snapshot.dual_branch_q = solution.dual_branch_q;
    snapshot.dual_gamma = solution.dual_gamma;
    snapshot.dual_gamma_by_deck = solution.dual_gamma_by_deck;
    snapshot.dual_kappa = solution.dual_kappa;
    snapshot.dual_eta = solution.dual_eta;
    snapshot.dual_sigma = solution.dual_sigma;
    return snapshot;
}

std::vector<std::array<int, 3>> MasterProblem::separate_3sr_cuts(
    const MasterLPSolution& solution,
    const std::set<std::array<int, 3>>& active_sr_cuts,
    double eps
) const {
    std::vector<std::array<int, 3>> violated;
    const auto& types = instance_.car_types;
    for (std::size_t a = 0; a < types.size(); ++a) {
        for (std::size_t b = a + 1; b < types.size(); ++b) {
            for (std::size_t c = b + 1; c < types.size(); ++c) {
                const std::array<int, 3> subset{types[a], types[b], types[c]};
                if (active_sr_cuts.find(subset) != active_sr_cuts.end()) {
                    continue;
                }
                double lhs = 0.0;
                for (std::size_t col_idx = 0; col_idx < columns_.size() && col_idx < solution.theta.size(); ++col_idx) {
                    const double value = solution.theta[col_idx];
                    if (value <= eps) {
                        continue;
                    }
                    const int coeff = sr_coeff_for_column(columns_[col_idx], subset, instance_);
                    if (coeff > 0) {
                        lhs += static_cast<double>(coeff) * value;
                    }
                }
                if (lhs > 1.0 + eps) {
                    violated.push_back(subset);
                }
            }
        }
    }
    return violated;
}

std::set<std::vector<int>> MasterProblem::existing_wagon_signatures() const {
    std::set<std::vector<int>> out;
    for (const auto& col : columns_) {
        out.insert(col.quantities);
    }
    return out;
}

std::map<std::pair<CompartmentKind, DeckMode>, std::set<std::vector<int>>>
MasterProblem::existing_compartment_signatures() const {
    std::map<std::pair<CompartmentKind, DeckMode>, std::set<std::vector<int>>> out;
    for (const auto& col : columns_) {
        if (col.is_compartment_column) {
            out[{col.compartment, col.deck}].insert(col.quantities);
        }
    }
    return out;
}

BpcSolver::BpcSolver(InstanceData instance, BpcOptions options)
    : master_(instance, options.method), options_(std::move(options)) {
    options_.pricing.use_cuts = options_.use_cuts;
    options_.pricing.labeling.use_cuts = options_.use_cuts;
    options_.pricing.threads = options_.threads;
    master_.seed_initial_columns();
    if (!options_.warmstarts.empty()) {
        warmstart_ = load_warmstart_columns(
            master_,
            options_.warmstarts,
            options_.pricing,
            options_.log_progress
        );
    }
}

std::vector<PricingColumn> BpcSolver::price(
    const MasterLPSolution& solution,
    PricingStats* stats,
    double deadline
) {
    MasterSnapshot snapshot = master_.snapshot_from_solution(solution);
    PricingOptions pricing_options = options_.pricing;
    pricing_options.labeling.deadline = deadline;
    if (options_.pricing_backend == PricingBackend::Solver) {
        if (options_.method == BpcMethod::WagonLabel) {
            return price_wagon_columns_solver(
                snapshot,
                pricing_options,
                master_.existing_wagon_signatures(),
                stats
            );
        }
        return price_compartment_columns_solver(
            snapshot,
            pricing_options,
            master_.existing_compartment_signatures(),
            stats
        );
    }
    if (options_.method == BpcMethod::WagonLabel) {
        return price_wagon_columns(snapshot, pricing_options, master_.existing_wagon_signatures(), stats);
    }
    return price_compartment_columns(snapshot, pricing_options, stats);
}

int BpcSolver::add_pricing_columns(const std::vector<PricingColumn>& pricing_columns) {
    int added = 0;
    for (const auto& priced : pricing_columns) {
        PatternColumn col;
        col.id = seq_id("cpp_col", ++column_seq_);
        col.quantities = priced.quantities;
        col.cost = priced.cost;
        col.compartment = priced.compartment;
        col.deck = priced.deck;
        col.is_compartment_column = options_.method == BpcMethod::CompartmentLabel;
        if (master_.add_column(col)) {
            ++added;
            ++generated_columns_;
        }
    }
    return added;
}

MasterLPSolution BpcSolver::run_column_generation(const Node& node, double deadline) {
    double remaining = deadline - now_sec();
    if (remaining <= 0.0) {
        return MasterLPSolution{};
    }

    double t = now_sec();
    MasterLPSolution solution = master_.solve_lp(
        node.branch_a_bounds,
        node.branch_q_bounds,
        active_sr_cuts_,
        options_.use_cuts,
        remaining,
        options_.threads,
        false
    );
    master_time_ += now_sec() - t;
    if (!solution.has_solution) {
        return solution;
    }

    int it = 1;
    while (options_.max_cg_iters <= 0 || it <= options_.max_cg_iters) {
        remaining = deadline - now_sec();
        if (remaining <= 0.0) {
            break;
        }

        PricingStats pricing_stats;
        t = now_sec();
        const auto new_columns = price(solution, &pricing_stats, deadline);
        pricing_time_ += now_sec() - t;
        accumulate_pricing_stats(pricing_stats_total_, pricing_stats);

        const int added = add_pricing_columns(new_columns);
        if (added == 0) {
            if (options_.use_cuts) {
                const auto cuts = master_.separate_3sr_cuts(solution, active_sr_cuts_);
                if (!cuts.empty()) {
                    for (const auto& cut : cuts) {
                        active_sr_cuts_.insert(cut);
                    }
                    remaining = deadline - now_sec();
                    if (remaining <= 0.0) {
                        break;
                    }
                    t = now_sec();
                    solution = master_.solve_lp(
                        node.branch_a_bounds,
                        node.branch_q_bounds,
                        active_sr_cuts_,
                        options_.use_cuts,
                        remaining,
                        options_.threads,
                        false
                    );
                    master_time_ += now_sec() - t;
                    if (!solution.has_solution) {
                        break;
                    }
                    ++it;
                    continue;
                }
            }
            break;
        }

        t = now_sec();
        solution = master_.solve_lp(
            node.branch_a_bounds,
            node.branch_q_bounds,
            active_sr_cuts_,
            options_.use_cuts,
            remaining,
            options_.threads,
            false
        );
        master_time_ += now_sec() - t;
        if (!solution.has_solution) {
            break;
        }
        ++it;
    }
    return solution;
}

double BpcSolver::gap(double best_obj, double best_bound) {
    if (best_bound <= -INF / 2.0) {
        return 1.0;
    }
    return std::max(0.0, best_obj - best_bound) / (std::abs(best_obj) + 1e-10);
}

double BpcSolver::final_bound(const std::vector<Node>& queue) const {
    if (!has_incumbent_) {
        return 0.0;
    }
    double bound = best_obj_;
    for (const auto& node : queue) {
        if (node.lower_bound <= -INF / 2.0) {
            return -INF;
        }
        bound = std::min(bound, node.lower_bound);
    }
    for (double unresolved : unresolved_bounds_) {
        bound = std::min(bound, unresolved);
    }
    return bound;
}

BpcResult BpcSolver::solve() {
    const double start = now_sec();
    const double deadline = start + options_.time_limit;
    std::vector<Node> queue;
    queue.push_back(Node{});
    int next_node_id = 0;
    int explored = 0;

    if (options_.log_progress) {
        std::cout << "[CPP-BPC] method=" << method_name(options_.method)
                  << " pricing_backend=" << pricing_backend_name(options_.pricing_backend)
                  << " max_nodes=" << options_.max_nodes
                  << " max_cg_iters=" << options_.max_cg_iters
                  << " threads=" << options_.threads << "\n";
    }

    if (warmstart_.added > 0) {
        const double remaining = deadline - now_sec();
        if (remaining > 0.0) {
            const double t = now_sec();
            MasterIPSolution ip = master_.solve_restricted_ip({}, {}, remaining, options_.threads, false);
            master_time_ += now_sec() - t;
            if (ip.has_solution && !master_.has_unmet_demand(ip)) {
                warmstart_incumbent_ = true;
                warmstart_objective_ = ip.objective;
                has_incumbent_ = true;
                best_obj_ = ip.objective;
                best_theta_ = ip.theta;
            }
        }
    }

    while (!queue.empty() && (options_.max_nodes <= 0 || explored < options_.max_nodes)) {
        if (now_sec() >= deadline) {
            break;
        }

        std::sort(queue.begin(), queue.end(), [](const Node& a, const Node& b) {
            return a.lower_bound < b.lower_bound;
        });
        Node node = queue.front();
        queue.erase(queue.begin());
        ++explored;

        MasterLPSolution lp = run_column_generation(node, deadline);
        if (!lp.has_solution) {
            continue;
        }
        node.lower_bound = lp.objective;

        if (has_incumbent_ && lp.objective >= best_obj_ - 1e-6) {
            continue;
        }

        if (master_.is_integral(lp)) {
            if (!has_incumbent_ || lp.objective < best_obj_) {
                has_incumbent_ = true;
                best_obj_ = lp.objective;
                best_theta_ = lp.theta;
            }
            if (gap(best_obj_, final_bound(queue)) <= options_.mip_gap_tol) {
                break;
            }
            continue;
        }

        if (!master_.has_branch_candidate(lp)) {
            const double remaining = options_.time_limit - (now_sec() - start);
            const double t = now_sec();
            MasterIPSolution ip = master_.solve_restricted_ip(
                node.branch_a_bounds,
                node.branch_q_bounds,
                remaining,
                options_.threads,
                false
            );
            master_time_ += now_sec() - t;
            if (ip.has_solution && !master_.has_unmet_demand(ip)) {
                if (!has_incumbent_ || ip.objective < best_obj_) {
                    has_incumbent_ = true;
                    best_obj_ = ip.objective;
                    best_theta_ = ip.theta;
                }
                if (ip.objective <= lp.objective + 1e-6) {
                    continue;
                }
            }
            unresolved_bounds_.push_back(lp.objective);
            continue;
        }

        const auto [branch_type, type_idx, value] = master_.choose_branch_var(lp);
        const double floor_v = std::floor(value);
        const double ceil_v = std::ceil(value);

        Node left = node;
        Node right = node;
        left.id = ++next_node_id;
        right.id = ++next_node_id;
        left.depth = node.depth + 1;
        right.depth = node.depth + 1;
        left.lower_bound = lp.objective;
        right.lower_bound = lp.objective;

        auto& left_map = branch_type == 'a' ? left.branch_a_bounds : left.branch_q_bounds;
        auto& right_map = branch_type == 'a' ? right.branch_a_bounds : right.branch_q_bounds;
        if (left_map.find(type_idx) == left_map.end()) {
            left_map[type_idx] = {-INF, INF};
        }
        if (right_map.find(type_idx) == right_map.end()) {
            right_map[type_idx] = {-INF, INF};
        }
        left_map[type_idx].second = std::min(left_map[type_idx].second, floor_v);
        right_map[type_idx].first = std::max(right_map[type_idx].first, ceil_v);

        queue.push_back(std::move(left));
        queue.push_back(std::move(right));

        if (has_incumbent_ && gap(best_obj_, final_bound(queue)) <= options_.mip_gap_tol) {
            break;
        }
    }

    const double wall = now_sec() - start;
    const double bound = has_incumbent_ ? final_bound(queue) : 0.0;
    BpcResult result;
    result.best_objective = best_obj_;
    result.has_incumbent = has_incumbent_;
    result.best_bound = bound;
    result.gap = has_incumbent_ ? gap(best_obj_, bound) : 1.0;
    result.explored_nodes = explored;
    result.generated_columns = generated_columns_;
    result.total_columns = static_cast<int>(master_.columns().size());
    if (has_incumbent_) {
        const auto& columns = master_.columns();
        double vehicle_count = 0.0;
        for (std::size_t c = 0; c < columns.size() && c < best_theta_.size(); ++c) {
            const int multiplicity = static_cast<int>(std::llround(best_theta_[c]));
            if (multiplicity <= 0) {
                continue;
            }
            result.selected_columns.push_back(SelectedPatternColumn{columns[c], multiplicity});
            const int column_count = std::accumulate(
                columns[c].quantities.begin(),
                columns[c].quantities.end(),
                0
            );
            vehicle_count += static_cast<double>(multiplicity * column_count);
        }
        result.loaded_vehicle_count = static_cast<int>(std::llround(vehicle_count));
    }
    result.wall_time = wall;
    result.master_time = master_time_;
    result.pricing_time = pricing_time_;
    result.generated_subpatterns = pricing_stats_total_.generated_subpatterns;
    result.merge_attempt_pairs = pricing_stats_total_.merge_attempt_pairs;
    result.labels_generated_raw = pricing_stats_total_.labeling_stats.labels_generated_raw;
    result.labels_feasible = pricing_stats_total_.labeling_stats.labels_feasible;
    result.labels_pruned_by_bound = pricing_stats_total_.labeling_stats.labels_pruned_by_bound;
    result.labels_pruned_by_dominance = pricing_stats_total_.labeling_stats.labels_pruned_by_dominance;
    result.labels_after_dominance = pricing_stats_total_.labeling_stats.labels_after_dominance;
    result.labels_pruned_total = result.labels_pruned_by_bound +
        result.labels_pruned_by_dominance;
    result.labels_avoided_by_d2 = pricing_stats_total_.labeling_stats.labels_avoided_by_d2;
    result.hybrid_calls = pricing_stats_total_.labeling_stats.hybrid_calls;
    result.hybrid_ordered_type_sum = pricing_stats_total_.labeling_stats.hybrid_ordered_type_sum;
    result.hybrid_ordered_type_max = pricing_stats_total_.labeling_stats.hybrid_ordered_type_max;
    result.hybrid_ordered_quantity_sum = pricing_stats_total_.labeling_stats.hybrid_ordered_quantity_sum;
    result.hybrid_total_quantity_sum = pricing_stats_total_.labeling_stats.hybrid_total_quantity_sum;
    result.warmstart_incumbent = warmstart_incumbent_;
    result.warmstart_objective = warmstart_objective_;
    result.warmstart = warmstart_;
    return result;
}

}  // namespace bpc_label
