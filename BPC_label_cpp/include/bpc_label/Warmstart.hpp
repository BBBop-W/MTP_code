#pragma once

#include <string>
#include <utility>
#include <vector>

#include "bpc_label/BpcSolver.hpp"

namespace bpc_label {

WarmstartLoadResult load_warmstart_columns(
    MasterProblem& master,
    const std::vector<std::pair<std::string, std::string>>& warmstarts,
    const PricingOptions& pricing_options,
    bool log_progress
);

}  // namespace bpc_label
