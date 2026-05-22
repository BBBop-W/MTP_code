#pragma once

#include <array>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "bpc_label/InstanceIO.hpp"
#include "bpc_label/WagonPricing.hpp"

namespace bpc_label {

enum class BpcMethod { WagonLabel, CompartmentLabel };
enum class PricingBackend { Label, Solver };

struct PatternColumn {
    std::string id;
    std::vector<int> quantities;
    double cost = 0.0;
    CompartmentKind compartment = CompartmentKind::Lower;
    DeckMode deck = DeckMode::HH;
    bool is_compartment_column = false;
};

struct MasterLPSolution {
    int status = 0;
    double objective = 0.0;
    bool has_solution = false;
    std::vector<double> theta;
    std::vector<double> unmet;
    std::vector<double> dual_alpha;
    std::vector<double> dual_beta;
    double dual_gamma = 0.0;
    std::array<double, 4> dual_gamma_by_deck{0.0, 0.0, 0.0, 0.0};
    double dual_kappa = 0.0;
    std::vector<double> dual_branch_a;
    std::vector<double> dual_branch_q;
    double dual_eta = 0.0;
    std::vector<TripletSigma> dual_sigma;
};

struct MasterIPSolution {
    int status = 0;
    double objective = 0.0;
    bool has_solution = false;
    std::vector<double> theta;
    std::vector<double> unmet;
};

struct SelectedPatternColumn {
    PatternColumn column;
    int multiplicity = 0;
};

struct BpcOptions {
    BpcMethod method = BpcMethod::WagonLabel;
    PricingBackend pricing_backend = PricingBackend::Label;
    PricingOptions pricing;
    std::vector<std::pair<std::string, std::string>> warmstarts;
    int max_nodes = 5000;
    int max_cg_iters = 3000;
    double time_limit = 300.0;
    double mip_gap_tol = 5e-6;
    bool use_cuts = false;
    bool log_progress = true;
};

struct WarmstartLoadResult {
    int added = 0;
    int skipped_infeasible = 0;
    int skipped_duplicate = 0;
    int skipped_empty = 0;
    int skipped_unknown = 0;
    int skipped_over_limit = 0;

    WarmstartLoadResult& operator+=(const WarmstartLoadResult& other);
};

struct BpcResult {
    double best_objective = 0.0;
    bool has_incumbent = false;
    double best_bound = 0.0;
    double gap = 0.0;
    int explored_nodes = 0;
    int generated_columns = 0;
    int total_columns = 0;
    int loaded_vehicle_count = 0;
    double wall_time = 0.0;
    double master_time = 0.0;
    double pricing_time = 0.0;
    double labeling_time_proxy = 0.0;
    std::size_t labels_feasible = 0;
    std::size_t labels_pruned_by_dominance = 0;
    bool warmstart_incumbent = false;
    double warmstart_objective = 0.0;
    WarmstartLoadResult warmstart;
    std::vector<SelectedPatternColumn> selected_columns;
};

class MasterProblem {
public:
    MasterProblem(const InstanceData& instance, BpcMethod method);

    void seed_initial_columns();
    bool add_column(const PatternColumn& column);

    MasterLPSolution solve_lp(
        const std::map<int, std::pair<double, double>>& branch_a_bounds,
        const std::map<int, std::pair<double, double>>& branch_q_bounds,
        const std::set<std::array<int, 3>>& active_sr_cuts,
        bool use_capacity_cut,
        double time_limit,
        bool log_to_console
    ) const;

    MasterIPSolution solve_restricted_ip(
        const std::map<int, std::pair<double, double>>& branch_a_bounds,
        const std::map<int, std::pair<double, double>>& branch_q_bounds,
        double time_limit,
        bool log_to_console
    ) const;

    bool is_integral(const MasterLPSolution& solution, double eps = 1e-5) const;
    bool has_unmet_demand(const MasterLPSolution& solution, double eps = 1e-5) const;
    bool has_unmet_demand(const MasterIPSolution& solution, double eps = 1e-5) const;
    std::tuple<char, int, double> choose_branch_var(const MasterLPSolution& solution, double eps = 1e-5) const;
    bool has_branch_candidate(const MasterLPSolution& solution, double eps = 1e-5) const;

    MasterSnapshot snapshot_from_solution(const MasterLPSolution& solution) const;
    std::vector<std::array<int, 3>> separate_3sr_cuts(
        const MasterLPSolution& solution,
        const std::set<std::array<int, 3>>& active_sr_cuts,
        double eps = 1e-4
    ) const;
    std::set<std::vector<int>> existing_wagon_signatures() const;
    std::map<std::pair<CompartmentKind, DeckMode>, std::set<std::vector<int>>> existing_compartment_signatures() const;
    bool has_equivalent_column(const PatternColumn& column) const;

    const InstanceData& instance() const { return instance_; }
    BpcMethod method() const { return method_; }
    const std::vector<PatternColumn>& columns() const { return columns_; }

private:
    InstanceData instance_;
    BpcMethod method_;
    std::vector<PatternColumn> columns_;

    double penalty_unmet_ = 1e6;
};

class BpcSolver {
public:
    BpcSolver(InstanceData instance, BpcOptions options);
    BpcResult solve();

private:
    struct Node {
        int id = 0;
        int depth = 0;
        std::map<int, std::pair<double, double>> branch_a_bounds;
        std::map<int, std::pair<double, double>> branch_q_bounds;
        double lower_bound = -1e100;
    };

    std::vector<PricingColumn> price(const MasterLPSolution& solution, PricingStats* stats, double deadline);
    int add_pricing_columns(const std::vector<PricingColumn>& pricing_columns);
    MasterLPSolution run_column_generation(const Node& node, double deadline);
    double final_bound(const std::vector<Node>& queue) const;
    static double gap(double best_obj, double best_bound);

    MasterProblem master_;
    BpcOptions options_;
    int column_seq_ = 0;
    int generated_columns_ = 0;
    double master_time_ = 0.0;
    double pricing_time_ = 0.0;
    double best_obj_ = 0.0;
    bool has_incumbent_ = false;
    bool warmstart_incumbent_ = false;
    double warmstart_objective_ = 0.0;
    std::vector<double> best_theta_;
    std::vector<double> unresolved_bounds_;
    WarmstartLoadResult warmstart_;
    std::set<std::array<int, 3>> active_sr_cuts_;
};

std::string method_name(BpcMethod method);
BpcMethod parse_method(const std::string& method);
std::string pricing_backend_name(PricingBackend backend);
PricingBackend parse_pricing_backend(const std::string& backend);

}  // namespace bpc_label
