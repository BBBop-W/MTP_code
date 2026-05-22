#pragma once

#include <set>
#include <string>
#include <vector>

#include "bpc_label/LabelPricing.hpp"

namespace bpc_label {

struct MasterSnapshot {
    std::vector<int> car_types;
    std::vector<double> car_lengths;
    std::vector<double> car_heights;
    std::vector<int> max_total_by_type;
    std::vector<double> dual_alpha;
    std::vector<double> dual_beta;
    std::vector<double> dual_branch_a;
    std::vector<double> dual_branch_q;
    double dual_gamma = 0.0;
    std::array<double, 4> dual_gamma_by_deck{0.0, 0.0, 0.0, 0.0};
    double dual_kappa = 0.0;
    double dual_eta = 0.0;
    std::vector<TripletSigma> dual_sigma;
};

struct PricingColumn {
    std::string column_id;
    DeckMode deck = DeckMode::HH;
    CompartmentKind compartment = CompartmentKind::Lower;
    std::vector<int> car_types;
    std::vector<int> quantities;
    double reduced_cost = 0.0;
    double cost = 0.0;
    std::string source;
};

struct PricingOptions {
    LabelingOptions labeling;
    int num_splits = 1;
    bool independent_mode_split = true;
    bool use_cuts = false;
    int max_columns_per_pricing = Config::max_wagon_pricing_columns;
    int max_columns_per_subproblem = Config::max_compartment_pricing_columns_per_subproblem;
};

struct PricingStats {
    std::size_t generated_subpatterns = 0;
    std::size_t merge_attempt_pairs = 0;
    LabelingStats labeling_stats;
};

std::vector<PricingColumn> price_compartment_columns(
    const MasterSnapshot& master,
    const PricingOptions& options,
    PricingStats* stats = nullptr
);

std::vector<PricingColumn> price_wagon_columns(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::set<std::vector<int>>& forbidden_signatures = {},
    PricingStats* stats = nullptr
);

}  // namespace bpc_label
