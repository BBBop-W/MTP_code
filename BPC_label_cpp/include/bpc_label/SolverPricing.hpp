#pragma once

#include <set>
#include <string>
#include <map>
#include <vector>

#include "bpc_label/WagonPricing.hpp"

namespace bpc_label {

std::vector<PricingColumn> price_compartment_columns_solver(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::map<std::pair<CompartmentKind, DeckMode>, std::set<std::vector<int>>>& forbidden_signatures,
    PricingStats* stats = nullptr
);

std::vector<PricingColumn> price_wagon_columns_solver(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::set<std::vector<int>>& forbidden_signatures,
    PricingStats* stats = nullptr
);

}  // namespace bpc_label
