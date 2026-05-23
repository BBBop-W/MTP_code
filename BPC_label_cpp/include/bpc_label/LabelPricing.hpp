#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "bpc_label/Config.hpp"
#include "bpc_label/DynamicSegmentation.hpp"

namespace bpc_label {

struct DualValues {
    std::vector<double> alpha;
    std::vector<double> beta;
    double gamma = 0.0;
    std::vector<double> branch_a;
    std::vector<double> branch_q;
};

struct TripletSigma {
    std::array<int, 3> type_ids{};
    double sigma = 0.0;
};

struct CutConfig {
    double eta_sum = 0.0;
    std::vector<TripletSigma> sigma_by_subset;
    std::vector<int> max_total_by_type;
    bool eta_upper_only = true;
};

struct CompartmentSpec {
    std::string compartment_id;
    CompartmentKind compartment = CompartmentKind::Lower;
    DeckMode deck = DeckMode::HH;
    std::vector<int> car_types;
    std::vector<double> car_lengths;
    std::vector<double> car_heights;
    std::vector<int> max_quantity_by_type;
    double compartment_length_limit = 0.0;
    int num_splits = 1;
    bool independent_mode_split = true;
};

struct LabelingOptions {
    bool use_dominance = true;
    bool use_cuts = false;
    bool use_rc_bound = true;
    bool use_height_order = true;
    bool use_local_d1_pruning = true;
    std::string residual_profile_mode = "full";
    std::string profile_generator_mode = "hyb";
    std::string order_dominance_scope = "profile";
    int max_units_per_type = Config::max_units_per_compartment;
    double component_length_perturbation_min_mm = 50.0;
    double component_length_perturbation_max_mm = 0.0;
    int component_length_perturbation_type_period = 2;
    std::vector<int> dominance_support_types;
    double eps = Config::eps;
    double deadline = 0.0;
};

struct LabelingStats {
    std::size_t labels_generated_raw = 0;
    std::size_t labels_feasible = 0;
    std::size_t labels_pruned_by_bound = 0;
    std::size_t labels_pruned_by_dominance = 0;
    std::size_t labels_pruned_by_order = 0;
    std::size_t labels_after_dominance = 0;
    std::size_t placements_skipped_by_order = 0;
    std::size_t labels_avoided_by_order = 0;
};

struct PlacementChoice {
    std::string side;
    int block = 0;
    // Resource consumption in every region containing this component: l_i^h + Delta.
    double resource_length = 0.0;
    std::vector<int> hits;
    double height_limit = 0.0;
    double component_capacity = 0.0;
};

struct ResourceModel {
    std::vector<double> capacities;
    std::vector<std::array<double, 3>> intervals;
    std::vector<std::vector<PlacementChoice>> choices_by_type;
    double delta = Config::safety_clearance_delta;
    int full_region_index = -1;
};

struct ResidualLabel {
    int stage = 0;
    double reduced_cost = 0.0;
    std::vector<int> quantities;
    double total_length = 0.0;
    std::vector<double> residual;
    // Generation-only state for D2/D3: future ordered choices must contain these regions.
    std::vector<int> required_hits;
};

struct CompartmentPattern {
    std::string compartment_id;
    CompartmentKind compartment = CompartmentKind::Lower;
    DeckMode deck = DeckMode::HH;
    std::vector<int> car_types;
    std::vector<int> quantities;
    double reduced_cost = 0.0;
    double best_length = 0.0;
};

ResourceModel build_compartment_resource_model(
    const CompartmentSpec& spec,
    const std::string& interval_profile
);

bool is_compartment_feasible(
    const CompartmentSpec& spec,
    const std::vector<int>& quantities,
    const std::string& interval_profile = "full",
    double eps = Config::eps
);

std::vector<CompartmentPattern> generate_compartment_patterns_residual(
    const CompartmentSpec& spec,
    const DualValues& duals,
    const LabelingOptions& options,
    const CutConfig* cuts,
    LabelingStats* stats
);

CompartmentSpec make_compartment_spec(
    CompartmentKind compartment,
    DeckMode deck,
    const std::vector<int>& car_types,
    const std::vector<double>& car_lengths,
    const std::vector<double>& car_heights,
    const std::vector<int>& max_quantity_by_type,
    int num_splits,
    bool independent_mode_split,
    int max_units_per_type = Config::max_units_per_compartment
);

}  // namespace bpc_label
