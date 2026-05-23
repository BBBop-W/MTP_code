#include "bpc_label/SolverPricing.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "gurobi_c++.h"

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

double at_or_zero(const std::vector<double>& values, std::size_t idx) {
    return idx < values.size() ? values[idx] : 0.0;
}

int at_or_default(const std::vector<int>& values, std::size_t idx, int fallback) {
    return idx < values.size() ? values[idx] : fallback;
}

double remaining_time(double deadline) {
    if (deadline <= 0.0) {
        return INF;
    }
    return std::max(0.0, deadline - monotonic_seconds());
}

void configure_pricing_model(GRBModel& model, double deadline, int threads) {
    model.set(GRB_IntParam_OutputFlag, 0);
    if (threads > 0) {
        model.set(GRB_IntParam_Threads, threads);
    }
    const double rem = remaining_time(deadline);
    if (rem > 0.0 && rem < INF / 2.0) {
        model.set(GRB_DoubleParam_TimeLimit, rem);
    }
}

bool acceptable_pricing_status(const GRBModel& model) {
    const int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {
        return model.get(GRB_IntAttr_SolCount) > 0;
    }
    if (status == GRB_TIME_LIMIT) {
        return model.get(GRB_IntAttr_SolCount) > 0;
    }
    return false;
}

double column_cost(const MasterSnapshot& master, const std::vector<int>& quantities) {
    double cost = 0.0;
    for (std::size_t i = 0; i < quantities.size(); ++i) {
        cost -= master.car_lengths[i] * static_cast<double>(quantities[i]);
    }
    return cost;
}

std::unordered_map<int, std::size_t> type_index(const std::vector<int>& car_types) {
    std::unordered_map<int, std::size_t> out;
    for (std::size_t i = 0; i < car_types.size(); ++i) {
        out[car_types[i]] = i;
    }
    return out;
}

bool choice_hits_interval(const PlacementChoice& choice, int interval_idx) {
    return std::find(choice.hits.begin(), choice.hits.end(), interval_idx) != choice.hits.end();
}

void add_forbidden_signature_constraints(
    GRBModel& model,
    const std::vector<GRBLinExpr>& q_expr,
    const std::vector<int>& car_types,
    const std::set<std::vector<int>>& forbidden,
    int big_m
) {
    int forbid_idx = 0;
    for (const auto& signature : forbidden) {
        if (signature.size() != car_types.size()) {
            continue;
        }
        std::vector<GRBVar> y_pos;
        std::vector<GRBVar> y_neg;
        y_pos.reserve(car_types.size());
        y_neg.reserve(car_types.size());
        for (int type_id : car_types) {
            y_pos.push_back(model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "forbid_pos"));
            y_neg.push_back(model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "forbid_neg"));
            (void)type_id;
        }
        GRBLinExpr any_diff = 0.0;
        for (std::size_t i = 0; i < car_types.size(); ++i) {
            model.addConstr(q_expr[i] - signature[i] >= 1.0 - big_m * (1.0 - y_pos[i]));
            model.addConstr(signature[i] - q_expr[i] >= 1.0 - big_m * (1.0 - y_neg[i]));
            any_diff += y_pos[i] + y_neg[i];
        }
        model.addConstr(any_diff >= 1.0, "forbid_signature_" + std::to_string(++forbid_idx));
    }
}

void add_sr_cut_terms(
    GRBModel& model,
    const MasterSnapshot& master,
    const std::vector<GRBLinExpr>& q_expr,
    const std::vector<int>& max_q_by_type,
    std::vector<std::pair<double, GRBVar>>& sr_terms
) {
    const auto index_by_type = type_index(master.car_types);
    int cut_idx = 0;
    for (const auto& item : master.dual_sigma) {
        if (std::abs(item.sigma) <= 1e-12) {
            continue;
        }
        std::vector<GRBVar> high_vars;
        for (int type_id : item.type_ids) {
            auto it = index_by_type.find(type_id);
            if (it == index_by_type.end()) {
                continue;
            }
            const std::size_t idx = it->second;
            GRBVar high = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "sr_high");
            const int threshold = static_cast<int>(master.max_total_by_type[idx] / 2 + 1);
            const int max_q = at_or_default(max_q_by_type, idx, Config::max_units_per_compartment);
            if (threshold <= 0 || max_q < threshold) {
                model.addConstr(high == 0.0);
            } else {
                model.addConstr(q_expr[idx] >= static_cast<double>(threshold) * high);
                model.addConstr(q_expr[idx] <= static_cast<double>(threshold - 1) + static_cast<double>(max_q) * high);
            }
            high_vars.push_back(high);
        }
        if (high_vars.empty()) {
            continue;
        }
        GRBVar coeff = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "sr_coeff");
        GRBLinExpr count = 0.0;
        for (GRBVar& high : high_vars) {
            count += high;
        }
        model.addConstr(count >= 2.0 * coeff, "sr_coeff_lb_" + std::to_string(cut_idx));
        model.addConstr(count <= 1.0 + 2.0 * coeff, "sr_coeff_ub_" + std::to_string(cut_idx));
        sr_terms.push_back({item.sigma, coeff});
        ++cut_idx;
    }
}

struct CompartmentMipResult {
    bool found = false;
    CompartmentKind compartment = CompartmentKind::Lower;
    DeckMode deck = DeckMode::HH;
    std::vector<int> quantities;
    double reduced_cost = 0.0;
    double node_count = 0.0;
};

CompartmentMipResult solve_compartment_mip(
    const MasterSnapshot& master,
    const CompartmentSpec& spec,
    const DualValues& duals,
    const PricingOptions& options,
    const CutConfig* cuts,
    const std::set<std::vector<int>>& forbidden
) {
    CompartmentMipResult result;
    result.compartment = spec.compartment;
    result.deck = spec.deck;
    result.quantities.assign(master.car_types.size(), 0);
    if (deadline_reached(options.labeling.deadline)) {
        return result;
    }

    ResourceModel resource = build_compartment_resource_model(spec, "full");
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    configure_pricing_model(model, options.labeling.deadline, options.threads);

    const int n = static_cast<int>(master.car_types.size());
    const int big_m = 2 * Config::max_units_per_compartment + 1;
    std::vector<std::vector<GRBVar>> x(static_cast<std::size_t>(n));
    std::vector<GRBLinExpr> q_expr(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        const auto& choices = resource.choices_by_type[static_cast<std::size_t>(i)];
        for (std::size_t k = 0; k < choices.size(); ++k) {
            GRBVar var = model.addVar(0.0, Config::max_units_per_compartment, 0.0, GRB_INTEGER, "x");
            x[static_cast<std::size_t>(i)].push_back(var);
            q_expr[static_cast<std::size_t>(i)] += var;
        }
    }

    std::vector<GRBVar> a;
    a.reserve(static_cast<std::size_t>(n));
    std::vector<int> max_q_by_type(static_cast<std::size_t>(n), Config::max_units_per_compartment);
    GRBLinExpr total_units = 0.0;
    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const int max_q = std::min(
            Config::max_units_per_compartment,
            at_or_default(spec.max_quantity_by_type, idx, Config::max_units_per_compartment)
        );
        max_q_by_type[idx] = max_q;
        GRBVar active = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "a");
        a.push_back(active);
        model.addConstr(q_expr[idx] <= static_cast<double>(max_q));
        model.addConstr(q_expr[idx] <= static_cast<double>(Config::max_units_per_compartment) * active);
        model.addConstr(q_expr[idx] >= active);
        total_units += q_expr[idx];
    }
    model.addConstr(total_units <= Config::max_units_per_compartment, "compartment_quantity");
    model.addConstr(total_units >= 1.0, "nonempty");
    add_forbidden_signature_constraints(model, q_expr, master.car_types, forbidden, big_m);

    for (std::size_t interval = 0; interval < resource.capacities.size(); ++interval) {
        GRBLinExpr usage = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& choices = resource.choices_by_type[static_cast<std::size_t>(i)];
            for (std::size_t k = 0; k < choices.size(); ++k) {
                if (choice_hits_interval(choices[k], static_cast<int>(interval))) {
                    usage += choices[k].resource_length * x[static_cast<std::size_t>(i)][k];
                }
            }
        }
        model.addConstr(usage <= resource.capacities[interval], "interval");
    }

    std::vector<std::pair<double, GRBVar>> sr_terms;
    if (cuts != nullptr && !cuts->sigma_by_subset.empty()) {
        MasterSnapshot cut_master = master;
        cut_master.dual_sigma = cuts->sigma_by_subset;
        add_sr_cut_terms(model, cut_master, q_expr, max_q_by_type, sr_terms);
    }

    GRBLinExpr rc = -duals.gamma / 2.0;
    if (cuts != nullptr && (!cuts->eta_upper_only || spec.compartment == CompartmentKind::Upper)) {
        rc -= cuts->eta_sum;
    }
    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double coef = master.car_lengths[idx] +
            at_or_zero(duals.alpha, idx) +
            at_or_zero(duals.beta, idx) +
            at_or_zero(duals.branch_q, idx);
        rc -= coef * q_expr[idx];
        rc -= at_or_zero(duals.branch_a, idx) * a[idx];
    }
    for (const auto& [sigma, coeff] : sr_terms) {
        rc -= sigma * coeff;
    }
    model.setObjective(rc, GRB_MINIMIZE);
    model.optimize();
    result.node_count = acceptable_pricing_status(model) ? model.get(GRB_DoubleAttr_NodeCount) : 0.0;
    if (!acceptable_pricing_status(model) || model.get(GRB_DoubleAttr_ObjVal) >= -1e-5) {
        return result;
    }

    result.found = true;
    result.reduced_cost = model.get(GRB_DoubleAttr_ObjVal);
    for (int i = 0; i < n; ++i) {
        int qty = 0;
        for (const GRBVar& var : x[static_cast<std::size_t>(i)]) {
            qty += static_cast<int>(std::llround(var.get(GRB_DoubleAttr_X)));
        }
        result.quantities[static_cast<std::size_t>(i)] = qty;
    }
    return result;
}

std::string make_column_id(const std::string& prefix, DeckMode deck, int seq) {
    std::ostringstream oss;
    oss << prefix << "_" << deck_name(deck) << "_" << seq;
    return oss.str();
}

DualValues make_base_duals(const MasterSnapshot& master, double gamma) {
    return DualValues{
        master.dual_alpha,
        master.dual_beta,
        gamma,
        master.dual_branch_a,
        master.dual_branch_q,
    };
}

CutConfig make_cut_config(const MasterSnapshot& master, bool eta_upper_only) {
    return CutConfig{
        master.dual_eta,
        master.dual_sigma,
        master.max_total_by_type,
        eta_upper_only,
    };
}

CompartmentSpec make_spec_for(
    const MasterSnapshot& master,
    CompartmentKind compartment,
    DeckMode deck,
    const PricingOptions& options
) {
    return make_compartment_spec(
        compartment,
        deck,
        master.car_types,
        master.car_lengths,
        master.car_heights,
        master.max_total_by_type,
        options.num_splits,
        options.independent_mode_split,
        options.labeling.max_units_per_type
    );
}

struct WagonMipResult {
    bool found = false;
    DeckMode deck = DeckMode::HH;
    std::vector<int> quantities;
    double reduced_cost = 0.0;
    double node_count = 0.0;
};

void add_resource_constraints(
    GRBModel& model,
    const MasterSnapshot& master,
    const ResourceModel& resource,
    const std::vector<std::vector<GRBVar>>& x
) {
    for (std::size_t interval = 0; interval < resource.capacities.size(); ++interval) {
        GRBLinExpr usage = 0.0;
        for (std::size_t i = 0; i < master.car_types.size(); ++i) {
            const auto& choices = resource.choices_by_type[i];
            for (std::size_t k = 0; k < choices.size(); ++k) {
                if (choice_hits_interval(choices[k], static_cast<int>(interval))) {
                    usage += choices[k].resource_length * x[i][k];
                }
            }
        }
        model.addConstr(usage <= resource.capacities[interval], "interval");
    }
}

[[maybe_unused]] WagonMipResult solve_fixed_deck_wagon_mip(
    const MasterSnapshot& master,
    DeckMode deck,
    const PricingOptions& options,
    const std::set<std::vector<int>>& forbidden
) {
    WagonMipResult result;
    result.deck = deck;
    result.quantities.assign(master.car_types.size(), 0);
    if (deadline_reached(options.labeling.deadline)) {
        return result;
    }

    auto upper_spec = make_spec_for(master, CompartmentKind::Upper, deck, options);
    auto lower_spec = make_spec_for(master, CompartmentKind::Lower, deck, options);
    ResourceModel upper_resource = build_compartment_resource_model(upper_spec, "full");
    ResourceModel lower_resource = build_compartment_resource_model(lower_spec, "full");

    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    configure_pricing_model(model, options.labeling.deadline, options.threads);

    const int n = static_cast<int>(master.car_types.size());
    const int big_m = 2 * Config::max_units_per_compartment + 1;
    std::vector<std::vector<GRBVar>> x_upper(static_cast<std::size_t>(n));
    std::vector<std::vector<GRBVar>> x_lower(static_cast<std::size_t>(n));
    std::vector<GRBLinExpr> q_expr(static_cast<std::size_t>(n), 0.0);
    GRBLinExpr upper_units = 0.0;
    GRBLinExpr lower_units = 0.0;

    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        for (std::size_t k = 0; k < upper_resource.choices_by_type[idx].size(); ++k) {
            GRBVar var = model.addVar(0.0, Config::max_units_per_compartment, 0.0, GRB_INTEGER, "xu");
            x_upper[idx].push_back(var);
            q_expr[idx] += var;
            upper_units += var;
        }
        for (std::size_t k = 0; k < lower_resource.choices_by_type[idx].size(); ++k) {
            GRBVar var = model.addVar(0.0, Config::max_units_per_compartment, 0.0, GRB_INTEGER, "xl");
            x_lower[idx].push_back(var);
            q_expr[idx] += var;
            lower_units += var;
        }
    }

    std::vector<GRBVar> a;
    a.reserve(static_cast<std::size_t>(n));
    std::vector<int> max_q_by_type(static_cast<std::size_t>(n), 2 * Config::max_units_per_compartment);
    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const int max_q = at_or_default(master.max_total_by_type, idx, 2 * Config::max_units_per_compartment);
        max_q_by_type[idx] = max_q;
        GRBVar active = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "a");
        a.push_back(active);
        model.addConstr(q_expr[idx] <= static_cast<double>(max_q));
        model.addConstr(q_expr[idx] <= static_cast<double>(2 * Config::max_units_per_compartment) * active);
        model.addConstr(q_expr[idx] >= active);
    }
    model.addConstr(upper_units <= Config::max_units_per_compartment, "upper_quantity");
    model.addConstr(lower_units <= Config::max_units_per_compartment, "lower_quantity");
    model.addConstr(upper_units + lower_units >= 1.0, "nonempty");
    add_forbidden_signature_constraints(model, q_expr, master.car_types, forbidden, big_m);
    add_resource_constraints(model, master, upper_resource, x_upper);
    add_resource_constraints(model, master, lower_resource, x_lower);

    std::vector<std::pair<double, GRBVar>> sr_terms;
    if (options.use_cuts && !master.dual_sigma.empty()) {
        add_sr_cut_terms(model, master, q_expr, max_q_by_type, sr_terms);
    }

    GRBLinExpr rc = -master.dual_gamma;
    if (options.use_cuts) {
        rc -= master.dual_eta;
    }
    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double coef = master.car_lengths[idx] +
            at_or_zero(master.dual_alpha, idx) +
            at_or_zero(master.dual_beta, idx) +
            at_or_zero(master.dual_branch_q, idx);
        rc -= coef * q_expr[idx];
        rc -= at_or_zero(master.dual_branch_a, idx) * a[idx];
    }
    for (const auto& [sigma, coeff] : sr_terms) {
        rc -= sigma * coeff;
    }
    model.setObjective(rc, GRB_MINIMIZE);
    model.optimize();
    result.node_count = acceptable_pricing_status(model) ? model.get(GRB_DoubleAttr_NodeCount) : 0.0;
    if (!acceptable_pricing_status(model) || model.get(GRB_DoubleAttr_ObjVal) >= -1e-5) {
        return result;
    }

    result.found = true;
    result.reduced_cost = model.get(GRB_DoubleAttr_ObjVal);
    for (int i = 0; i < n; ++i) {
        int qty = 0;
        for (const GRBVar& var : x_upper[static_cast<std::size_t>(i)]) {
            qty += static_cast<int>(std::llround(var.get(GRB_DoubleAttr_X)));
        }
        for (const GRBVar& var : x_lower[static_cast<std::size_t>(i)]) {
            qty += static_cast<int>(std::llround(var.get(GRB_DoubleAttr_X)));
        }
        result.quantities[static_cast<std::size_t>(i)] = qty;
    }
    return result;
}

WagonMipResult solve_full_wagon_mip(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::set<std::vector<int>>& forbidden
) {
    WagonMipResult result;
    result.quantities.assign(master.car_types.size(), 0);
    if (deadline_reached(options.labeling.deadline)) {
        return result;
    }

    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    configure_pricing_model(model, options.labeling.deadline, options.threads);

    const int n = static_cast<int>(master.car_types.size());
    const int big_m = 2 * Config::max_units_per_compartment + 1;

    std::map<DeckMode, GRBVar> z;
    GRBLinExpr deck_choice = 0.0;
    for (DeckMode deck : wagon_deck_order()) {
        z[deck] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "z_" + deck_name(deck));
        deck_choice += z[deck];
    }
    model.addConstr(deck_choice == 1.0, "deck_position");

    std::vector<GRBLinExpr> q_expr(static_cast<std::size_t>(n), 0.0);
    GRBLinExpr upper_units = 0.0;
    GRBLinExpr lower_units = 0.0;

    std::map<DeckMode, std::vector<std::vector<GRBVar>>> x_upper_by_deck;
    std::map<DeckMode, std::vector<std::vector<GRBVar>>> x_lower_by_deck;

    for (DeckMode deck : wagon_deck_order()) {
        auto upper_spec = make_spec_for(master, CompartmentKind::Upper, deck, options);
        auto lower_spec = make_spec_for(master, CompartmentKind::Lower, deck, options);
        ResourceModel upper_resource = build_compartment_resource_model(upper_spec, "full");
        ResourceModel lower_resource = build_compartment_resource_model(lower_spec, "full");

        std::vector<std::vector<GRBVar>> x_upper(static_cast<std::size_t>(n));
        std::vector<std::vector<GRBVar>> x_lower(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            for (std::size_t k = 0; k < upper_resource.choices_by_type[idx].size(); ++k) {
                GRBVar var = model.addVar(0.0, Config::max_units_per_compartment, 0.0, GRB_INTEGER, "xu");
                model.addConstr(var <= static_cast<double>(Config::max_units_per_compartment) * z[deck]);
                x_upper[idx].push_back(var);
                q_expr[idx] += var;
                upper_units += var;
            }
            for (std::size_t k = 0; k < lower_resource.choices_by_type[idx].size(); ++k) {
                GRBVar var = model.addVar(0.0, Config::max_units_per_compartment, 0.0, GRB_INTEGER, "xl");
                model.addConstr(var <= static_cast<double>(Config::max_units_per_compartment) * z[deck]);
                x_lower[idx].push_back(var);
                q_expr[idx] += var;
                lower_units += var;
            }
        }
        add_resource_constraints(model, master, upper_resource, x_upper);
        add_resource_constraints(model, master, lower_resource, x_lower);
        x_upper_by_deck[deck] = std::move(x_upper);
        x_lower_by_deck[deck] = std::move(x_lower);
    }

    std::vector<GRBVar> a;
    a.reserve(static_cast<std::size_t>(n));
    std::vector<int> max_q_by_type(static_cast<std::size_t>(n), 2 * Config::max_units_per_compartment);
    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const int max_q = at_or_default(master.max_total_by_type, idx, 2 * Config::max_units_per_compartment);
        max_q_by_type[idx] = max_q;
        GRBVar active = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "a");
        a.push_back(active);
        model.addConstr(q_expr[idx] <= static_cast<double>(max_q));
        model.addConstr(q_expr[idx] <= static_cast<double>(2 * Config::max_units_per_compartment) * active);
        model.addConstr(q_expr[idx] >= active);
    }

    model.addConstr(upper_units <= Config::max_units_per_compartment, "upper_quantity");
    model.addConstr(lower_units <= Config::max_units_per_compartment, "lower_quantity");
    model.addConstr(upper_units + lower_units >= 1.0, "nonempty");
    add_forbidden_signature_constraints(model, q_expr, master.car_types, forbidden, big_m);

    std::vector<std::pair<double, GRBVar>> sr_terms;
    if (options.use_cuts && !master.dual_sigma.empty()) {
        add_sr_cut_terms(model, master, q_expr, max_q_by_type, sr_terms);
    }

    GRBLinExpr rc = -master.dual_gamma;
    if (options.use_cuts) {
        rc -= master.dual_eta;
    }
    for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double coef = master.car_lengths[idx] +
            at_or_zero(master.dual_alpha, idx) +
            at_or_zero(master.dual_beta, idx) +
            at_or_zero(master.dual_branch_q, idx);
        rc -= coef * q_expr[idx];
        rc -= at_or_zero(master.dual_branch_a, idx) * a[idx];
    }
    for (const auto& [sigma, coeff] : sr_terms) {
        rc -= sigma * coeff;
    }

    model.setObjective(rc, GRB_MINIMIZE);
    model.optimize();
    result.node_count = acceptable_pricing_status(model) ? model.get(GRB_DoubleAttr_NodeCount) : 0.0;
    if (!acceptable_pricing_status(model) || model.get(GRB_DoubleAttr_ObjVal) >= -1e-5) {
        return result;
    }

    result.found = true;
    result.reduced_cost = model.get(GRB_DoubleAttr_ObjVal);
    for (DeckMode deck : wagon_deck_order()) {
        if (z[deck].get(GRB_DoubleAttr_X) > 0.5) {
            result.deck = deck;
            break;
        }
    }
    for (int i = 0; i < n; ++i) {
        int qty = 0;
        const std::size_t idx = static_cast<std::size_t>(i);
        for (DeckMode deck : wagon_deck_order()) {
            for (const GRBVar& var : x_upper_by_deck[deck][idx]) {
                qty += static_cast<int>(std::llround(var.get(GRB_DoubleAttr_X)));
            }
            for (const GRBVar& var : x_lower_by_deck[deck][idx]) {
                qty += static_cast<int>(std::llround(var.get(GRB_DoubleAttr_X)));
            }
        }
        result.quantities[idx] = qty;
    }
    return result;
}

}  // namespace

std::vector<PricingColumn> price_compartment_columns_solver(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::map<std::pair<CompartmentKind, DeckMode>, std::set<std::vector<int>>>& forbidden_signatures,
    PricingStats* stats
) {
    std::vector<PricingColumn> columns;
    int seq = 0;
    CutConfig cut_config = make_cut_config(master, true);
    const CutConfig* cuts = options.use_cuts ? &cut_config : nullptr;

    for (DeckMode deck : compartment_deck_order()) {
        if (deadline_reached(options.labeling.deadline)) {
            break;
        }
        const double gamma_p = master.dual_gamma_by_deck[deck_index(deck)];
        for (CompartmentKind compartment : {CompartmentKind::Upper, CompartmentKind::Lower}) {
            if (deadline_reached(options.labeling.deadline)) {
                break;
            }
            const double gamma = compartment == CompartmentKind::Upper
                ? 2.0 * (gamma_p + master.dual_kappa)
                : -2.0 * gamma_p;
            auto spec = make_spec_for(master, compartment, deck, options);
            auto duals = make_base_duals(master, gamma);

            std::set<std::vector<int>> forbidden;
            auto it = forbidden_signatures.find({compartment, deck});
            if (it != forbidden_signatures.end()) {
                forbidden = it->second;
            }

            const int max_cols = std::max(1, options.max_columns_per_subproblem);
            for (int col_idx = 0; col_idx < max_cols; ++col_idx) {
                auto result = solve_compartment_mip(master, spec, duals, options, cuts, forbidden);
                if (stats != nullptr) {
                    stats->generated_subpatterns += result.found ? 1 : 0;
                }
                if (!result.found) {
                    break;
                }
                forbidden.insert(result.quantities);
                ++seq;
                columns.push_back(PricingColumn{
                    make_column_id("solver_comp", deck, seq),
                    deck,
                    compartment,
                    master.car_types,
                    result.quantities,
                    result.reduced_cost,
                    column_cost(master, result.quantities),
                    "solver_compartment_mip",
                });
            }
        }
    }

    std::sort(columns.begin(), columns.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });
    return columns;
}

std::vector<PricingColumn> price_wagon_columns_solver(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::set<std::vector<int>>& forbidden_signatures,
    PricingStats* stats
) {
    std::vector<PricingColumn> columns;
    std::set<std::vector<int>> forbidden = forbidden_signatures;
    const int max_cols = std::max(1, options.max_columns_per_pricing);

    for (int col_idx = 0; col_idx < max_cols; ++col_idx) {
        if (deadline_reached(options.labeling.deadline)) {
            break;
        }
        WagonMipResult best = solve_full_wagon_mip(master, options, forbidden);
        if (stats != nullptr) {
            stats->merge_attempt_pairs += 1;
        }
        if (!best.found) {
            break;
        }
        forbidden.insert(best.quantities);
        columns.push_back(PricingColumn{
            make_column_id("solver_wagon", best.deck, static_cast<int>(columns.size()) + 1),
            best.deck,
            CompartmentKind::Lower,
            master.car_types,
            best.quantities,
            best.reduced_cost,
            column_cost(master, best.quantities),
            "solver_wagon_mip",
        });
        if (stats != nullptr) {
            stats->generated_subpatterns += 1;
        }
    }

    std::sort(columns.begin(), columns.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });
    return columns;
}

}  // namespace bpc_label
