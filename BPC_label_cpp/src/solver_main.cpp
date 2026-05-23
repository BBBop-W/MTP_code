#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bpc_label/BpcSolver.hpp"
#include "bpc_label/InstanceIO.hpp"
#include "gurobi_c++.h"

using namespace bpc_label;

namespace {

bool parse_bool(const std::string& value) {
    return value == "1" || value == "true" || value == "True" || value == "yes" || value == "on";
}

std::string arg_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    }
    ++i;
    return argv[i];
}

void print_usage() {
    std::cout
        << "Usage: bpc_label_solver --instance DIR --method wagon|compartment [options]\n"
        << "Options:\n"
        << "  --max-nodes N\n"
        << "  --max-cg-iters N\n"
        << "  --max-columns-per-pricing N\n"
        << "  --max-columns-per-subproblem N\n"
        << "  --time-limit SEC\n"
        << "  --mip-gap GAP\n"
        << "  --num-splits N\n"
        << "  --independent-mode-split true|false\n"
        << "  --use-cuts true|false\n"
        << "  --use-dominance true|false\n"
        << "  --use-local-d1-pruning true|false\n"
        << "  --pricing-backend label|solver\n"
        << "  --profile-generator-mode ex|gr|hyb|d2\n"
        << "  --residual-profile-mode full|fans_diag\n"
        << "  --order-dominance-scope profile|rho_h\n"
        << "  --component-length-perturbation-mm MM\n"
        << "  --component-length-perturbation-min-mm MM\n"
        << "  --component-length-perturbation-type-period N\n"
        << "  --warmstart-json PREFIX:PATH\n"
        << "  --quiet\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string instance_dir;
        BpcOptions options;
        options.method = BpcMethod::WagonLabel;
        options.pricing.num_splits = 1;
        options.pricing.independent_mode_split = true;
        options.pricing.labeling.profile_generator_mode = "hyb";
        options.pricing.labeling.residual_profile_mode = "full";
        options.pricing.labeling.use_dominance = true;
        options.pricing.labeling.use_rc_bound = true;
        options.pricing.labeling.use_height_order = true;
        options.pricing.labeling.use_local_d1_pruning = true;
        bool user_set_rc_bound = false;
        bool user_set_max_columns = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else if (arg == "--instance") {
                instance_dir = arg_value(i, argc, argv);
            } else if (arg == "--method") {
                options.method = parse_method(arg_value(i, argc, argv));
            } else if (arg == "--max-nodes") {
                options.max_nodes = std::stoi(arg_value(i, argc, argv));
            } else if (arg == "--max-cg-iters") {
                options.max_cg_iters = std::stoi(arg_value(i, argc, argv));
            } else if (arg == "--max-columns-per-pricing") {
                options.pricing.max_columns_per_pricing = std::stoi(arg_value(i, argc, argv));
                user_set_max_columns = true;
            } else if (arg == "--max-columns-per-subproblem") {
                options.pricing.max_columns_per_subproblem = std::stoi(arg_value(i, argc, argv));
                user_set_max_columns = true;
            } else if (arg == "--time-limit") {
                options.time_limit = std::stod(arg_value(i, argc, argv));
            } else if (arg == "--mip-gap") {
                options.mip_gap_tol = std::stod(arg_value(i, argc, argv));
            } else if (arg == "--num-splits") {
                options.pricing.num_splits = std::stoi(arg_value(i, argc, argv));
            } else if (arg == "--independent-mode-split") {
                options.pricing.independent_mode_split = parse_bool(arg_value(i, argc, argv));
            } else if (arg == "--use-cuts") {
                options.use_cuts = parse_bool(arg_value(i, argc, argv));
            } else if (arg == "--use-dominance") {
                options.pricing.labeling.use_dominance = parse_bool(arg_value(i, argc, argv));
            } else if (arg == "--use-local-d1-pruning") {
                options.pricing.labeling.use_local_d1_pruning = parse_bool(arg_value(i, argc, argv));
            } else if (arg == "--pricing-backend" || arg == "--pricing-method") {
                options.pricing_backend = parse_pricing_backend(arg_value(i, argc, argv));
            } else if (arg == "--use-rc-bound") {
                options.pricing.labeling.use_rc_bound = parse_bool(arg_value(i, argc, argv));
                user_set_rc_bound = true;
            } else if (arg == "--profile-generator-mode") {
                options.pricing.labeling.profile_generator_mode = arg_value(i, argc, argv);
            } else if (arg == "--residual-profile-mode") {
                options.pricing.labeling.residual_profile_mode = arg_value(i, argc, argv);
            } else if (arg == "--order-dominance-scope") {
                options.pricing.labeling.order_dominance_scope = arg_value(i, argc, argv);
            } else if (arg == "--component-length-perturbation-mm" ||
                       arg == "--component-length-perturbation-max-mm") {
                options.pricing.labeling.component_length_perturbation_max_mm =
                    std::stod(arg_value(i, argc, argv));
            } else if (arg == "--component-length-perturbation-min-mm") {
                options.pricing.labeling.component_length_perturbation_min_mm =
                    std::stod(arg_value(i, argc, argv));
            } else if (arg == "--component-length-perturbation-type-period") {
                options.pricing.labeling.component_length_perturbation_type_period =
                    std::stoi(arg_value(i, argc, argv));
            } else if (arg == "--warmstart-json") {
                const std::string raw = arg_value(i, argc, argv);
                const std::size_t sep = raw.find(':');
                if (sep == std::string::npos) {
                    options.warmstarts.push_back({"WS", raw});
                } else {
                    options.warmstarts.push_back({raw.substr(0, sep), raw.substr(sep + 1)});
                }
            } else if (arg == "--quiet") {
                options.log_progress = false;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        if (instance_dir.empty()) {
            print_usage();
            return 2;
        }

        if (!user_set_rc_bound && options.method == BpcMethod::WagonLabel) {
            options.pricing.labeling.use_rc_bound = false;
        }
        if (!user_set_max_columns && options.pricing_backend == PricingBackend::Solver) {
            options.pricing.max_columns_per_pricing = 1;
            options.pricing.max_columns_per_subproblem = 1;
        }
        options.pricing.use_cuts = options.use_cuts;
        options.pricing.labeling.use_cuts = options.use_cuts;
        if (options.pricing.labeling.order_dominance_scope != "profile" &&
            options.pricing.labeling.order_dominance_scope != "rho_h") {
            throw std::invalid_argument("unknown order-dominance-scope: " +
                options.pricing.labeling.order_dominance_scope);
        }

        InstanceData instance = load_instance(instance_dir);
        BpcSolver solver(instance, options);
        BpcResult result = solver.solve();

        std::cout << "{\n"
                  << "  \"method\": \"" << method_name(options.method) << "\",\n"
                  << "  \"pricing_backend\": \"" << pricing_backend_name(options.pricing_backend) << "\",\n"
                  << "  \"profile_generator_mode\": \""
                  << options.pricing.labeling.profile_generator_mode << "\",\n"
                  << "  \"has_incumbent\": " << (result.has_incumbent ? "true" : "false") << ",\n"
                  << "  \"best_objective\": " << (result.has_incumbent ? result.best_objective : 0.0) << ",\n"
                  << "  \"loaded_length_mm\": " << (result.has_incumbent ? -result.best_objective : 0.0) << ",\n"
                  << "  \"best_bound\": " << result.best_bound << ",\n"
                  << "  \"gap\": " << result.gap << ",\n"
                  << "  \"explored_nodes\": " << result.explored_nodes << ",\n"
                  << "  \"generated_columns\": " << result.generated_columns << ",\n"
                  << "  \"total_columns\": " << result.total_columns << ",\n"
                  << "  \"loaded_vehicle_count\": " << result.loaded_vehicle_count << ",\n"
                  << "  \"generated_subpatterns\": " << result.generated_subpatterns << ",\n"
                  << "  \"merge_attempt_pairs\": " << result.merge_attempt_pairs << ",\n"
                  << "  \"labels_generated_raw\": " << result.labels_generated_raw << ",\n"
                  << "  \"labels_feasible\": " << result.labels_feasible << ",\n"
                  << "  \"labels_pruned_by_bound\": " << result.labels_pruned_by_bound << ",\n"
                  << "  \"labels_pruned_by_dominance\": " << result.labels_pruned_by_dominance << ",\n"
                  << "  \"labels_pruned_by_order\": " << result.labels_pruned_by_order << ",\n"
                  << "  \"labels_after_dominance\": " << result.labels_after_dominance << ",\n"
                  << "  \"labels_pruned_total\": " << result.labels_pruned_total << ",\n"
                  << "  \"placements_skipped_by_order\": " << result.placements_skipped_by_order << ",\n"
                  << "  \"labels_avoided_by_order\": " << result.labels_avoided_by_order << ",\n"
                  << "  \"order_dominance_scope\": \""
                  << options.pricing.labeling.order_dominance_scope << "\",\n"
                  << "  \"component_length_perturbation_min_mm\": "
                  << options.pricing.labeling.component_length_perturbation_min_mm << ",\n"
                  << "  \"component_length_perturbation_max_mm\": "
                  << options.pricing.labeling.component_length_perturbation_max_mm << ",\n"
                  << "  \"component_length_perturbation_type_period\": "
                  << options.pricing.labeling.component_length_perturbation_type_period << ",\n"
                  << "  \"warmstart_incumbent\": " << (result.warmstart_incumbent ? "true" : "false") << ",\n"
                  << "  \"warmstart_objective\": " << (result.warmstart_incumbent ? result.warmstart_objective : 0.0) << ",\n"
                  << "  \"warmstart_loaded_length_mm\": " << (result.warmstart_incumbent ? -result.warmstart_objective : 0.0) << ",\n"
                  << "  \"warmstart_added\": " << result.warmstart.added << ",\n"
                  << "  \"warmstart_skipped_infeasible\": " << result.warmstart.skipped_infeasible << ",\n"
                  << "  \"warmstart_skipped_duplicate\": " << result.warmstart.skipped_duplicate << ",\n"
                  << "  \"warmstart_skipped_empty\": " << result.warmstart.skipped_empty << ",\n"
                  << "  \"warmstart_skipped_unknown\": " << result.warmstart.skipped_unknown << ",\n"
                  << "  \"warmstart_skipped_over_limit\": " << result.warmstart.skipped_over_limit << ",\n"
                  << "  \"wall_time\": " << result.wall_time << ",\n"
                  << "  \"master_time\": " << result.master_time << ",\n"
                  << "  \"pricing_time\": " << result.pricing_time << ",\n"
                  << "  \"threads\": 0,\n"
                  << "  \"selected_columns\": [\n";
        for (std::size_t c = 0; c < result.selected_columns.size(); ++c) {
            const auto& item = result.selected_columns[c];
            std::cout << "    {\"id\": \"" << item.column.id
                      << "\", \"multiplicity\": " << item.multiplicity
                      << ", \"deck\": \"" << deck_name(item.column.deck)
                      << "\", \"compartment\": \""
                      << (item.column.compartment == CompartmentKind::Upper ? "upper" : "lower")
                      << "\", \"quantities\": [";
            for (std::size_t i = 0; i < item.column.quantities.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << item.column.quantities[i];
            }
            std::cout << "]}";
            if (c + 1 < result.selected_columns.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "  ]\n"
                  << "}\n";
        return result.has_incumbent ? 0 : 1;
    } catch (const GRBException& exc) {
        std::cerr << "Gurobi error " << exc.getErrorCode() << ": " << exc.getMessage() << "\n";
        return 3;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 2;
    }
}
