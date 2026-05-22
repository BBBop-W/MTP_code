#include <iostream>
#include <set>

#include "bpc_label/WagonPricing.hpp"

using namespace bpc_label;

namespace {

MasterSnapshot smoke_master() {
    MasterSnapshot master;
    master.car_types = {1, 2, 3, 4, 5};
    master.car_lengths = {4300.0, 4700.0, 5100.0, 5400.0, 5900.0};
    master.car_heights = {1580.0, 1700.0, 1780.0, 1870.0, 2020.0};
    master.max_total_by_type = {3, 3, 2, 2, 1};

    master.dual_alpha = {-100.0, -120.0, -80.0, -60.0, -40.0};
    master.dual_beta = {-3000.0, -3200.0, -3600.0, -4100.0, -4500.0};
    master.dual_branch_a = {0.0, 0.0, 0.0, 0.0, 0.0};
    master.dual_branch_q = {0.0, 0.0, 0.0, 0.0, 0.0};
    master.dual_gamma = 1000.0;
    master.dual_gamma_by_deck = {50.0, 20.0, 20.0, 50.0};
    master.dual_kappa = 0.0;
    return master;
}

void print_columns(const std::string& title, const std::vector<PricingColumn>& columns) {
    std::cout << title << ": " << columns.size() << " columns\n";
    const std::size_t limit = std::min<std::size_t>(columns.size(), 5);
    for (std::size_t k = 0; k < limit; ++k) {
        const auto& col = columns[k];
        std::cout << "  " << col.column_id
                  << " deck=" << deck_name(col.deck)
                  << " rc=" << col.reduced_cost
                  << " q=(";
        for (std::size_t i = 0; i < col.quantities.size(); ++i) {
            if (i) {
                std::cout << ",";
            }
            std::cout << col.quantities[i];
        }
        std::cout << ")\n";
    }
}

}  // namespace

int main() {
    MasterSnapshot master = smoke_master();

    PricingOptions options;
    options.num_splits = 2;
    options.independent_mode_split = true;
    options.use_cuts = false;
    options.labeling.use_dominance = true;
    options.labeling.use_rc_bound = false;
    options.labeling.use_height_order = true;
    options.labeling.use_local_residual_skyline = true;
    options.labeling.residual_profile_mode = "full";
    options.labeling.profile_generator_mode = "hyb";
    options.labeling.max_units_per_type = Config::max_units_per_compartment;

    PricingStats compartment_stats;
    const auto compartment_columns = price_compartment_columns(master, options, &compartment_stats);
    print_columns("compartment", compartment_columns);
    std::cout << "  labels_feasible=" << compartment_stats.labeling_stats.labels_feasible
              << " dominated=" << compartment_stats.labeling_stats.labels_pruned_by_dominance
              << " local_skyline=" << compartment_stats.labeling_stats.labels_pruned_by_local_skyline
              << "\n";

    PricingStats wagon_stats;
    const auto wagon_columns = price_wagon_columns(master, options, std::set<std::vector<int>>{}, &wagon_stats);
    print_columns("wagon", wagon_columns);
    std::cout << "  subpatterns=" << wagon_stats.generated_subpatterns
              << " merge_pairs=" << wagon_stats.merge_attempt_pairs
              << " labels_feasible=" << wagon_stats.labeling_stats.labels_feasible
              << "\n";

    return 0;
}
