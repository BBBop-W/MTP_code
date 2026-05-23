#include "bpc_label/LabelPricing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace bpc_label {

namespace {

double at_or_zero(const std::vector<double>& values, std::size_t idx) {
    return idx < values.size() ? values[idx] : 0.0;
}

int at_or_default(const std::vector<int>& values, std::size_t idx, int fallback) {
    return idx < values.size() ? values[idx] : fallback;
}

bool keep_interval_for_profile(int l_idx, int r_idx, int n_blocks, const std::string& profile) {
    if (profile == "full") {
        return true;
    }
    if (profile == "fans_diag") {
        return l_idx == n_blocks || r_idx == n_blocks || l_idx == r_idx;
    }
    throw std::invalid_argument("unknown residual_profile_mode: " + profile);
}

bool vector_subset(const std::vector<int>& a, const std::vector<int>& b) {
    std::size_t ia = 0;
    std::size_t ib = 0;
    while (ia < a.size() && ib < b.size()) {
        if (a[ia] == b[ib]) {
            ++ia;
            ++ib;
        } else if (a[ia] > b[ib]) {
            ++ib;
        } else {
            return false;
        }
    }
    return ia == a.size();
}

std::size_t saturated_add(std::size_t a, std::size_t b) {
    const std::size_t max_value = static_cast<std::size_t>(-1);
    if (max_value - a < b) {
        return max_value;
    }
    return a + b;
}

std::size_t saturated_mul(std::size_t a, std::size_t b) {
    const std::size_t max_value = static_cast<std::size_t>(-1);
    if (a != 0 && b > max_value / a) {
        return max_value;
    }
    return a * b;
}

std::size_t count_compositions(int total, int parts) {
    if (total < 0 || parts < 0) {
        return 0;
    }
    if (parts == 0) {
        return total == 0 ? 1 : 0;
    }
    if (parts == 1) {
        return 1;
    }

    const int n = total + parts - 1;
    const int k = std::min(parts - 1, total);
    std::size_t result = 1;
    for (int j = 1; j <= k; ++j) {
        result = saturated_mul(result, static_cast<std::size_t>(n - k + j));
        result /= static_cast<std::size_t>(j);
    }
    return result;
}

struct ResidualChild {
    std::vector<double> residual;
};

enum class GeneratorMode {
    Exact,
    Hybrid,
    D2,
};

GeneratorMode parse_generator_mode(const std::string& mode) {
    if (mode == "exact" || mode == "ex" || mode == "e") {
        return GeneratorMode::Exact;
    }
    if (mode == "hyb" || mode == "hybrid") {
        return GeneratorMode::Hybrid;
    }
    if (mode == "d2") {
        return GeneratorMode::D2;
    }
    throw std::invalid_argument("unknown profile_generator_mode: " + mode);
}

void enumerate_consumption_rec(
    const std::vector<PlacementChoice>& choices,
    int choice_idx,
    int remaining,
    std::vector<double>& consumption,
    std::set<std::vector<double>>& seen,
    std::vector<std::vector<double>>& out,
    double deadline
) {
    if (deadline_reached(deadline)) {
        return;
    }
    if (choice_idx == static_cast<int>(choices.size()) - 1) {
        const int count = remaining;
        if (count > 0) {
            const double amount = static_cast<double>(count) *
                choices[static_cast<std::size_t>(choice_idx)].resource_length;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] += amount;
            }
        }
        if (seen.insert(consumption).second) {
            out.push_back(consumption);
        }
        if (count > 0) {
            const double amount = static_cast<double>(count) *
                choices[static_cast<std::size_t>(choice_idx)].resource_length;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] -= amount;
            }
        }
        return;
    }

    for (int count = 0; count <= remaining; ++count) {
        if (count > 0) {
            const double amount = static_cast<double>(count) *
                choices[static_cast<std::size_t>(choice_idx)].resource_length;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] += amount;
            }
        }
        enumerate_consumption_rec(
            choices,
            choice_idx + 1,
            remaining - count,
            consumption,
            seen,
            out,
            deadline
        );
        if (count > 0) {
            const double amount = static_cast<double>(count) *
                choices[static_cast<std::size_t>(choice_idx)].resource_length;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] -= amount;
            }
        }
    }
}

std::vector<std::vector<double>> placement_consumptions(
    const std::vector<PlacementChoice>& choices,
    int quantity,
    std::size_t resource_count,
    double deadline = 0.0
) {
    if (quantity == 0) {
        return {std::vector<double>(resource_count, 0.0)};
    }
    if (choices.empty()) {
        return {};
    }

    std::vector<std::vector<double>> out;
    std::set<std::vector<double>> seen;
    std::vector<double> consumption(resource_count, 0.0);
    enumerate_consumption_rec(choices, 0, quantity, consumption, seen, out, deadline);
    return out;
}

int choice_side_rank(const PlacementChoice& choice) {
    if (choice.side == "left") {
        return 0;
    }
    if (choice.side == "right") {
        return 1;
    }
    return 2;
}

std::vector<int> greedy_choice_order(const std::vector<PlacementChoice>& choices) {
    std::vector<int> order(choices.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const auto& a = choices[static_cast<std::size_t>(lhs)];
        const auto& b = choices[static_cast<std::size_t>(rhs)];
        if (a.hits.size() != b.hits.size()) {
            return a.hits.size() < b.hits.size();
        }
        if (a.block != b.block) {
            return a.block > b.block;
        }
        const int side_a = choice_side_rank(a);
        const int side_b = choice_side_rank(b);
        if (side_a != side_b) {
            return side_a < side_b;
        }
        return lhs < rhs;
    });
    return order;
}

int max_fit_count(
    const std::vector<double>& residual,
    const PlacementChoice& choice,
    double eps
) {
    if (choice.hits.empty()) {
        return std::numeric_limits<int>::max();
    }
    int fit = std::numeric_limits<int>::max();
    for (int hit : choice.hits) {
        const double available = residual[static_cast<std::size_t>(hit)] + eps;
        const int local_fit = static_cast<int>(std::floor(available / choice.resource_length));
        fit = std::min(fit, std::max(0, local_fit));
    }
    return fit;
}

std::vector<int> greedy_counts_for_residual(
    const std::vector<PlacementChoice>& choices,
    int quantity,
    const std::vector<double>& residual,
    double eps
) {
    std::vector<int> counts(choices.size(), 0);
    std::vector<double> current = residual;
    int remaining = quantity;
    for (int idx_int : greedy_choice_order(choices)) {
        if (remaining <= 0) {
            break;
        }
        const std::size_t idx = static_cast<std::size_t>(idx_int);
        const auto& choice = choices[idx];
        const int take = std::min(remaining, max_fit_count(current, choice, eps));
        if (take <= 0) {
            continue;
        }
        counts[idx] = take;
        const double amount = static_cast<double>(take) * choice.resource_length;
        for (int hit : choice.hits) {
            current[static_cast<std::size_t>(hit)] -= amount;
        }
        remaining -= take;
    }
    if (remaining > 0) {
        return {};
    }
    return counts;
}

bool apply_consumption(
    const std::vector<double>& residual,
    const std::vector<double>& consumption,
    double eps,
    std::vector<double>& out
);

std::vector<std::vector<double>> apply_local_d1_dominance(
    const std::vector<std::vector<double>>& residuals,
    double eps,
    LabelingStats* stats
);

bool greedy_child_for_residual(
    const std::vector<PlacementChoice>& choices,
    int quantity,
    const std::vector<double>& residual,
    double eps,
    ResidualChild& child
) {
    if (quantity == 0) {
        child = ResidualChild{residual};
        return true;
    }
    const std::vector<int> counts = greedy_counts_for_residual(choices, quantity, residual, eps);
    if (counts.empty()) {
        return false;
    }

    std::vector<double> next = residual;
    for (std::size_t idx = 0; idx < choices.size(); ++idx) {
        const int count = counts[idx];
        if (count <= 0) {
            continue;
        }
        const auto& choice = choices[idx];
        const double amount = static_cast<double>(count) * choice.resource_length;
        for (int hit : choice.hits) {
            next[static_cast<std::size_t>(hit)] -= amount;
            if (next[static_cast<std::size_t>(hit)] < -eps) {
                return false;
            }
        }
    }
    child = ResidualChild{std::move(next)};
    return true;
}

bool residual_vector_dominates(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double eps
) {
    bool strict = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] + eps < b[i]) {
            return false;
        }
        if (a[i] > b[i] + eps) {
            strict = true;
        }
    }
    return strict;
}

bool residual_vector_weakly_dominates(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double eps,
    bool& strict
) {
    strict = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] + eps < b[i]) {
            return false;
        }
        if (a[i] > b[i] + eps) {
            strict = true;
        }
    }
    return true;
}

bool residual_child_dominates(
    const ResidualChild& a,
    const ResidualChild& b,
    double eps
) {
    bool residual_strict = false;
    return residual_vector_weakly_dominates(a.residual, b.residual, eps, residual_strict) &&
        residual_strict;
}

std::vector<std::vector<double>> apply_local_d1_dominance(
    const std::vector<std::vector<double>>& residuals,
    double eps,
    LabelingStats* stats
) {
    std::vector<std::vector<double>> kept;
    for (const auto& residual : residuals) {
        bool dominated = false;
        std::vector<std::size_t> remove_idx;
        for (std::size_t i = 0; i < kept.size(); ++i) {
            if (residual_vector_dominates(kept[i], residual, eps)) {
                dominated = true;
                break;
            }
            if (residual_vector_dominates(residual, kept[i], eps)) {
                remove_idx.push_back(i);
            }
        }
        if (dominated) {
            if (stats != nullptr) {
                stats->labels_pruned_by_dominance += 1;
            }
            continue;
        }
        if (!remove_idx.empty()) {
            if (stats != nullptr) {
                stats->labels_pruned_by_dominance += remove_idx.size();
            }
            std::vector<std::vector<double>> next;
            for (std::size_t i = 0; i < kept.size(); ++i) {
                if (std::find(remove_idx.begin(), remove_idx.end(), i) == remove_idx.end()) {
                    next.push_back(std::move(kept[i]));
                }
            }
            kept = std::move(next);
        }
        kept.push_back(residual);
    }
    return kept;
}

std::vector<ResidualChild> apply_local_d1_dominance(
    const std::vector<ResidualChild>& children,
    double eps,
    LabelingStats* stats
) {
    std::vector<ResidualChild> kept;
    for (const auto& child : children) {
        bool dominated = false;
        std::vector<std::size_t> remove_idx;
        for (std::size_t i = 0; i < kept.size(); ++i) {
            if (residual_child_dominates(kept[i], child, eps)) {
                dominated = true;
                break;
            }
            if (residual_child_dominates(child, kept[i], eps)) {
                remove_idx.push_back(i);
            }
        }
        if (dominated) {
            if (stats != nullptr) {
                stats->labels_pruned_by_dominance += 1;
            }
            continue;
        }
        if (!remove_idx.empty()) {
            if (stats != nullptr) {
                stats->labels_pruned_by_dominance += remove_idx.size();
            }
            std::vector<ResidualChild> next;
            for (std::size_t i = 0; i < kept.size(); ++i) {
                if (std::find(remove_idx.begin(), remove_idx.end(), i) == remove_idx.end()) {
                    next.push_back(std::move(kept[i]));
                }
            }
            kept = std::move(next);
        }
        kept.push_back(child);
    }
    return kept;
}

bool apply_consumption(
    const std::vector<double>& residual,
    const std::vector<double>& consumption,
    double eps,
    std::vector<double>& out
) {
    out.resize(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        out[i] = residual[i] - consumption[i];
        if (out[i] < -eps) {
            return false;
        }
    }
    return true;
}

bool same_support_on_types(
    const std::vector<int>& a,
    const std::vector<int>& b,
    const std::vector<std::size_t>& support_idx
) {
    for (std::size_t idx : support_idx) {
        if ((a[idx] > 0) != (b[idx] > 0)) {
            return false;
        }
    }
    return true;
}

bool residual_label_dominates(
    const ResidualLabel& a,
    const ResidualLabel& b,
    double eps,
    const std::vector<std::size_t>& support_idx
) {
    if (a.reduced_cost > b.reduced_cost + eps) {
        return false;
    }
    if (!support_idx.empty() && !same_support_on_types(a.quantities, b.quantities, support_idx)) {
        return false;
    }

    bool quantity_strict = false;
    for (std::size_t i = 0; i < a.quantities.size(); ++i) {
        if (a.quantities[i] > b.quantities[i]) {
            return false;
        }
        if (a.quantities[i] < b.quantities[i]) {
            quantity_strict = true;
        }
    }

    bool profile_strict = false;
    for (std::size_t i = 0; i < a.residual.size(); ++i) {
        if (a.residual[i] + eps < b.residual[i]) {
            return false;
        }
        if (a.residual[i] > b.residual[i] + eps) {
            profile_strict = true;
        }
    }

    return profile_strict ||
        quantity_strict ||
        a.reduced_cost < b.reduced_cost - eps;
}

std::vector<ResidualLabel> apply_residual_dominance(
    std::vector<ResidualLabel> labels,
    const std::vector<int>& car_types,
    double eps,
    const std::vector<int>& support_types,
    LabelingStats* stats
) {
    std::sort(labels.begin(), labels.end(), [](const ResidualLabel& a, const ResidualLabel& b) {
        return a.reduced_cost < b.reduced_cost;
    });

    std::vector<std::size_t> support_idx;
    for (int type_id : support_types) {
        auto it = std::find(car_types.begin(), car_types.end(), type_id);
        if (it != car_types.end()) {
            support_idx.push_back(static_cast<std::size_t>(std::distance(car_types.begin(), it)));
        }
    }

    std::vector<ResidualLabel> kept;
    for (const auto& cand : labels) {
        bool dominated = false;
        for (const auto& old : kept) {
            if (residual_label_dominates(old, cand, eps, support_idx)) {
                dominated = true;
                break;
            }
        }
        if (dominated) {
            if (stats != nullptr) {
                stats->labels_pruned_by_dominance += 1;
            }
            continue;
        }
        kept.push_back(cand);
    }
    if (stats != nullptr) {
        stats->labels_after_dominance += kept.size();
    }
    return kept;
}

double reduced_cost_shift(
    const CompartmentSpec& spec,
    const std::vector<int>& quantities,
    const CutConfig* cuts
) {
    if (cuts == nullptr) {
        return 0.0;
    }

    double shift = 0.0;
    if (!cuts->eta_upper_only || spec.compartment == CompartmentKind::Upper) {
        shift -= cuts->eta_sum;
    }

    if (cuts->sigma_by_subset.empty()) {
        return shift;
    }

    std::unordered_map<int, std::size_t> index_by_type;
    for (std::size_t i = 0; i < spec.car_types.size(); ++i) {
        index_by_type[spec.car_types[i]] = i;
    }

    for (const auto& item : cuts->sigma_by_subset) {
        int val = 0;
        for (int type_id : item.type_ids) {
            auto it = index_by_type.find(type_id);
            if (it == index_by_type.end()) {
                continue;
            }
            const std::size_t idx = it->second;
            const int total_i = at_or_default(cuts->max_total_by_type, idx, 0);
            if (total_i > 0 && quantities[idx] > static_cast<double>(total_i) / 2.0) {
                val += 1;
            }
        }
        const int coeff = static_cast<int>(std::floor(0.5 * static_cast<double>(val)));
        shift -= item.sigma * static_cast<double>(coeff);
    }
    return shift;
}

double future_reduced_cost_lower_bound(
    double current_rc,
    const std::vector<int>& remaining_indices,
    const CompartmentSpec& spec,
    const DualValues& duals,
    const LabelingOptions& options
) {
    double lb = current_rc;
    for (int idx_int : remaining_indices) {
        const std::size_t idx = static_cast<std::size_t>(idx_int);
        const int max_q = std::min(
            options.max_units_per_type,
            at_or_default(spec.max_quantity_by_type, idx, options.max_units_per_type)
        );
        const double coef = spec.car_lengths[idx] +
            at_or_zero(duals.alpha, idx) +
            at_or_zero(duals.beta, idx) +
            at_or_zero(duals.branch_q, idx);
        double best_delta = 0.0;
        for (int q = 1; q <= max_q; ++q) {
            double delta = -coef * q - at_or_zero(duals.branch_a, idx);
            if (delta < best_delta) {
                best_delta = delta;
            }
        }
        lb += best_delta;
    }
    return lb;
}

std::vector<int> ordered_type_indices(const CompartmentSpec& spec, bool use_height_order) {
    std::vector<int> order(spec.car_types.size());
    std::iota(order.begin(), order.end(), 0);
    if (use_height_order) {
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const double ha = spec.car_heights[static_cast<std::size_t>(a)];
            const double hb = spec.car_heights[static_cast<std::size_t>(b)];
            if (std::abs(ha - hb) > 1e-9) {
                return ha > hb;
            }
            return spec.car_types[static_cast<std::size_t>(a)] < spec.car_types[static_cast<std::size_t>(b)];
        });
    }
    return order;
}

bool has_no_more_restrictive_choice(
    const std::vector<PlacementChoice>& choices,
    const PlacementChoice& target,
    double eps
) {
    return std::any_of(choices.begin(), choices.end(), [&](const PlacementChoice& choice) {
        return choice.hits == target.hits &&
            choice.resource_length <= target.resource_length + eps;
    });
}

void add_choice_usage(
    std::vector<double>& usage,
    const PlacementChoice& choice
) {
    for (int hit : choice.hits) {
        usage[static_cast<std::size_t>(hit)] += choice.resource_length;
    }
}

bool pair_usage_no_larger(
    const PlacementChoice& new_a,
    const PlacementChoice& new_b,
    const PlacementChoice& old_a,
    const PlacementChoice& old_b,
    std::size_t resource_count,
    double eps
) {
    std::vector<double> old_usage(resource_count, 0.0);
    std::vector<double> new_usage(resource_count, 0.0);
    add_choice_usage(old_usage, old_a);
    add_choice_usage(old_usage, old_b);
    add_choice_usage(new_usage, new_a);
    add_choice_usage(new_usage, new_b);
    for (std::size_t idx = 0; idx < resource_count; ++idx) {
        if (new_usage[idx] > old_usage[idx] + eps) {
            return false;
        }
    }
    return true;
}

bool pairwise_order_replacement_certified(
    const std::vector<PlacementChoice>& a_choices,
    const std::vector<PlacementChoice>& b_choices,
    std::size_t resource_count,
    double eps
) {
    // Certify the exchange argument behind D2 for the discretized component set.
    for (const auto& old_a : a_choices) {
        for (const auto& old_b : b_choices) {
            bool covered = false;
            for (const auto& new_a : a_choices) {
                for (const auto& new_b : b_choices) {
                    if (!vector_subset(new_a.hits, new_b.hits)) {
                        continue;
                    }
                    if (pair_usage_no_larger(
                            new_a,
                            new_b,
                            old_a,
                            old_b,
                            resource_count,
                            eps)) {
                        covered = true;
                        break;
                    }
                }
                if (covered) {
                    break;
                }
            }
            if (!covered) {
                return false;
            }
        }
    }
    return true;
}

bool same_region_choice_length(
    const std::vector<PlacementChoice>& choices,
    const std::vector<int>& hits,
    double& length
) {
    bool found = false;
    for (const auto& choice : choices) {
        if (choice.hits == hits && (!found || choice.resource_length < length)) {
            found = true;
            length = choice.resource_length;
        }
    }
    return found;
}

bool ordered_exchange_certified(
    const std::vector<PlacementChoice>& a_choices,
    const std::vector<PlacementChoice>& b_choices,
    double eps
) {
    for (const auto& b_outer : b_choices) {
        for (const auto& a_inner : a_choices) {
            if (!vector_subset(b_outer.hits, a_inner.hits)) {
                continue;
            }

            double a_outer_length = 0.0;
            double b_inner_length = 0.0;
            if (!same_region_choice_length(a_choices, b_outer.hits, a_outer_length) ||
                !same_region_choice_length(b_choices, a_inner.hits, b_inner_length)) {
                return false;
            }
            if (a_outer_length + b_inner_length >
                b_outer.resource_length + a_inner.resource_length + eps) {
                return false;
            }
        }
    }
    return true;
}

bool type_less_restrictive(
    const CompartmentSpec& spec,
    const ResourceModel& resource_model,
    int a,
    int b,
    double eps,
    bool require_exchange_certificate,
    bool require_profile_certificate
) {
    const std::size_t ia = static_cast<std::size_t>(a);
    const std::size_t ib = static_cast<std::size_t>(b);
    if (spec.car_lengths[ia] > spec.car_lengths[ib] + eps) {
        return false;
    }
    const auto& a_choices = resource_model.choices_by_type[ia];
    const auto& b_choices = resource_model.choices_by_type[ib];
    for (const auto& b_choice : b_choices) {
        if (!has_no_more_restrictive_choice(a_choices, b_choice, eps)) {
            return false;
        }
    }
    if (require_exchange_certificate &&
        !ordered_exchange_certified(a_choices, b_choices, eps)) {
        return false;
    }
    if (require_profile_certificate) {
        return pairwise_order_replacement_certified(
            a_choices,
            b_choices,
            resource_model.capacities.size(),
            eps
        );
    }
    return true;
}

std::vector<int> longest_order_compatible_chain(
    const CompartmentSpec& spec,
    const ResourceModel& resource_model,
    double eps,
    bool require_exchange_certificate,
    bool require_profile_certificate
) {
    std::vector<int> candidates(spec.car_types.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        const std::size_t ia = static_cast<std::size_t>(a);
        const std::size_t ib = static_cast<std::size_t>(b);
        if (std::abs(spec.car_lengths[ia] - spec.car_lengths[ib]) > eps) {
            return spec.car_lengths[ia] < spec.car_lengths[ib];
        }
        if (std::abs(spec.car_heights[ia] - spec.car_heights[ib]) > eps) {
            return spec.car_heights[ia] < spec.car_heights[ib];
        }
        return spec.car_types[ia] < spec.car_types[ib];
    });

    const std::size_t n = candidates.size();
    if (n == 0) {
        return {};
    }
    std::vector<int> dp(n, 1);
    std::vector<int> prev(n, -1);
    int best_pos = 0;
    for (std::size_t pos = 0; pos < n; ++pos) {
        for (std::size_t before = 0; before < pos; ++before) {
            if (!type_less_restrictive(
                    spec,
                    resource_model,
                    candidates[before],
                    candidates[pos],
                    eps,
                    require_exchange_certificate,
                    require_profile_certificate)) {
                continue;
            }
            if (dp[before] + 1 > dp[pos]) {
                dp[pos] = dp[before] + 1;
                prev[pos] = static_cast<int>(before);
            }
        }
        if (dp[pos] > dp[static_cast<std::size_t>(best_pos)]) {
            best_pos = static_cast<int>(pos);
        }
    }

    std::vector<int> chain;
    for (int pos = best_pos; pos >= 0; pos = prev[static_cast<std::size_t>(pos)]) {
        chain.push_back(candidates[static_cast<std::size_t>(pos)]);
        if (prev[static_cast<std::size_t>(pos)] < 0) {
            break;
        }
    }
    std::reverse(chain.begin(), chain.end());
    return chain;
}

std::vector<int> article_search_order(
    const CompartmentSpec& spec,
    const ResourceModel& resource_model,
    bool use_height_order,
    GeneratorMode generator_mode,
    double eps,
    std::vector<bool>& ordered_mask
) {
    ordered_mask.assign(spec.car_types.size(), false);
    if (generator_mode == GeneratorMode::Exact) {
        return ordered_type_indices(spec, use_height_order);
    }

    const bool require_exchange_certificate = true;
    const bool require_profile_certificate = true;
    const std::vector<int> ordered_chain = longest_order_compatible_chain(
        spec,
        resource_model,
        eps,
        require_exchange_certificate,
        require_profile_certificate
    );
    if (generator_mode == GeneratorMode::D2 &&
        ordered_chain.size() != spec.car_types.size()) {
        return ordered_type_indices(spec, use_height_order);
    }
    for (int idx : ordered_chain) {
        ordered_mask[static_cast<std::size_t>(idx)] = true;
    }

    if (generator_mode == GeneratorMode::D2) {
        return ordered_chain;
    }

    // D3 uses one order-compatible chain and treats all other types as conflicts.
    std::vector<int> search_order;
    const std::vector<int> conflict_order = ordered_type_indices(spec, use_height_order);
    for (int idx : conflict_order) {
        if (!ordered_mask[static_cast<std::size_t>(idx)]) {
            search_order.push_back(idx);
        }
    }
    search_order.insert(search_order.end(), ordered_chain.begin(), ordered_chain.end());
    return search_order;
}

std::vector<double> unique_heights_from_spec(const CompartmentSpec& spec) {
    std::vector<double> heights = spec.car_heights;
    std::sort(heights.begin(), heights.end());
    heights.erase(std::unique(heights.begin(), heights.end()), heights.end());
    return heights;
}

bool perturb_type_enabled(
    std::size_t type_idx,
    double perturbation_max_mm,
    int type_period
) {
    return perturbation_max_mm > 0.0 &&
        type_period > 0 &&
        static_cast<int>(type_idx % static_cast<std::size_t>(type_period)) == 0;
}

double deterministic_length_perturbation(
    std::size_t type_idx,
    const std::string& side,
    int block,
    const LabelingOptions& options
) {
    if (!perturb_type_enabled(
            type_idx,
            options.component_length_perturbation_max_mm,
            options.component_length_perturbation_type_period)) {
        return 0.0;
    }

    const double lo = std::max(0.0, std::min(
        options.component_length_perturbation_min_mm,
        options.component_length_perturbation_max_mm
    ));
    const double hi = std::max(
        options.component_length_perturbation_min_mm,
        options.component_length_perturbation_max_mm
    );
    if (hi <= 0.0) {
        return 0.0;
    }
    const int side_code = choice_side_rank(PlacementChoice{side, block, 0.0, {}});
    const int raw = static_cast<int>(
        ((type_idx + 1) * 53 + static_cast<std::size_t>(block + 3) * 97 +
         static_cast<std::size_t>(side_code + 1) * 193) % 1009
    );
    const double fraction = static_cast<double>(raw % 1000) / 999.0;
    const double magnitude = lo + (hi - lo) * fraction;
    const bool increase = (raw % 2) == 0;
    return increase ? magnitude : -magnitude;
}

ResourceModel build_compartment_resource_model_impl(
    const CompartmentSpec& spec,
    const std::string& interval_profile,
    const LabelingOptions& options
) {
    const auto sides = deck_sides(spec.deck);
    const bool pi_left = sides[0] == 'm';
    const bool pi_right = sides[1] == 'm';

    const ModelSegments segments = get_model_segments(
        unique_heights_from_spec(spec),
        spec.num_splits,
        spec.independent_mode_split
    );
    const LayerSegments& layer = spec.compartment == CompartmentKind::Upper ? segments.upper : segments.lower;
    const int n_blocks = static_cast<int>(layer.blocks.size());

    std::vector<double> limit_left{pi_left ? layer.central.h_m : layer.central.h_h};
    std::vector<double> limit_right{pi_right ? layer.central.h_m : layer.central.h_h};
    for (const Segment& block : layer.blocks) {
        limit_left.push_back(pi_left ? block.h_m : block.h_h);
        limit_right.push_back(pi_right ? block.h_m : block.h_h);
    }

    const double actual_limit_central = std::min(limit_left[0], limit_right[0]);
    std::vector<double> lengths{layer.central.length};
    for (const Segment& block : layer.blocks) {
        lengths.push_back(block.length);
    }

    ResourceModel out;
    out.delta = Config::safety_clearance_delta;
    for (int l_idx = 0; l_idx <= n_blocks; ++l_idx) {
        for (int r_idx = 0; r_idx <= n_blocks; ++r_idx) {
            double mod = 0.0;
            if (l_idx == n_blocks && r_idx == n_blocks) {
                mod = -out.delta;
            } else if (l_idx == n_blocks || r_idx == n_blocks) {
                mod = 0.0;
            } else {
                mod = out.delta;
            }

            double cap = lengths[0] + mod;
            for (int i = 1; i <= l_idx; ++i) {
                cap += lengths[static_cast<std::size_t>(i)];
            }
            for (int i = 1; i <= r_idx; ++i) {
                cap += lengths[static_cast<std::size_t>(i)];
            }
            if (keep_interval_for_profile(l_idx, r_idx, n_blocks, interval_profile)) {
                const int interval_idx = static_cast<int>(out.intervals.size());
                out.intervals.push_back({static_cast<double>(l_idx), static_cast<double>(r_idx), cap});
                out.capacities.push_back(cap);
                if (l_idx == n_blocks && r_idx == n_blocks) {
                    out.full_region_index = interval_idx;
                }
            }
        }
    }

    auto hits_for = [&](const std::string& side, int block_idx) {
        std::vector<int> hits;
        for (std::size_t idx = 0; idx < out.intervals.size(); ++idx) {
            const int l_int = static_cast<int>(out.intervals[idx][0]);
            const int r_int = static_cast<int>(out.intervals[idx][1]);
            bool inside = false;
            if (side == "central") {
                inside = true;
            } else if (side == "left" && block_idx <= l_int) {
                inside = true;
            } else if (side == "right" && block_idx <= r_int) {
                inside = true;
            }
            if (inside) {
                hits.push_back(static_cast<int>(idx));
            }
        }
        return hits;
    };

    out.choices_by_type.resize(spec.car_types.size());
    for (std::size_t t = 0; t < spec.car_types.size(); ++t) {
        const double height = spec.car_heights[t];
        std::vector<PlacementChoice> choices;
        if (height <= actual_limit_central) {
            const double resource_length = std::max(
                1.0,
                spec.car_lengths[t] +
                    deterministic_length_perturbation(t, "central", 0, options) +
                    out.delta
            );
            choices.push_back(PlacementChoice{"central", 0, resource_length, hits_for("central", 0)});
        }
        for (int idx = 1; idx <= n_blocks; ++idx) {
            if (height <= limit_left[static_cast<std::size_t>(idx)]) {
                const double resource_length = std::max(
                    1.0,
                    spec.car_lengths[t] +
                        deterministic_length_perturbation(t, "left", idx, options) +
                        out.delta
                );
                choices.push_back(PlacementChoice{"left", idx, resource_length, hits_for("left", idx)});
            }
            if (height <= limit_right[static_cast<std::size_t>(idx)]) {
                const double resource_length = std::max(
                    1.0,
                    spec.car_lengths[t] +
                        deterministic_length_perturbation(t, "right", idx, options) +
                        out.delta
                );
                choices.push_back(PlacementChoice{"right", idx, resource_length, hits_for("right", idx)});
            }
        }

        std::map<std::vector<int>, PlacementChoice> dedup;
        for (const auto& choice : choices) {
            auto it = dedup.find(choice.hits);
            if (it == dedup.end() ||
                choice.resource_length < it->second.resource_length) {
                dedup[choice.hits] = choice;
            }
        }
        for (const auto& kv : dedup) {
            out.choices_by_type[t].push_back(kv.second);
        }
    }

    return out;
}

}  // namespace

ResourceModel build_compartment_resource_model(
    const CompartmentSpec& spec,
    const std::string& interval_profile
) {
    return build_compartment_resource_model_impl(spec, interval_profile, LabelingOptions{});
}

bool is_compartment_feasible(
    const CompartmentSpec& spec,
    const std::vector<int>& quantities,
    const std::string& interval_profile,
    double eps
) {
    if (quantities.size() != spec.car_types.size() ||
        spec.car_types.size() != spec.car_lengths.size() ||
        spec.car_types.size() != spec.car_heights.size()) {
        return false;
    }

    int total_units = 0;
    double total_length = 0.0;
    for (std::size_t i = 0; i < quantities.size(); ++i) {
        const int q = quantities[i];
        if (q < 0) {
            return false;
        }
        if (q > at_or_default(spec.max_quantity_by_type, i, Config::max_units_per_compartment)) {
            return false;
        }
        total_units += q;
        total_length += spec.car_lengths[i] * static_cast<double>(q);
    }
    if (total_units > Config::max_units_per_compartment) {
        return false;
    }
    if (total_length > spec.compartment_length_limit + eps) {
        return false;
    }
    if (total_units == 0) {
        return true;
    }

    ResourceModel resource_model = build_compartment_resource_model(spec, interval_profile);
    std::vector<std::vector<double>> current_residuals{resource_model.capacities};
    const std::vector<int> order = ordered_type_indices(spec, true);

    for (int type_idx_int : order) {
        const std::size_t type_idx = static_cast<std::size_t>(type_idx_int);
        const int q = quantities[type_idx];
        if (q == 0) {
            continue;
        }
        const auto consumptions = placement_consumptions(
            resource_model.choices_by_type[type_idx],
            q,
            resource_model.capacities.size()
        );
        if (consumptions.empty()) {
            return false;
        }

        std::vector<std::vector<double>> next_residuals;
        for (const auto& residual : current_residuals) {
            for (const auto& consumption : consumptions) {
                std::vector<double> child;
                if (apply_consumption(residual, consumption, eps, child)) {
                    next_residuals.push_back(std::move(child));
                }
            }
        }
        current_residuals = apply_local_d1_dominance(next_residuals, eps, nullptr);
        if (current_residuals.empty()) {
            return false;
        }
    }
    return true;
}

std::vector<CompartmentPattern> generate_compartment_patterns_residual(
    const CompartmentSpec& spec,
    const DualValues& duals,
    const LabelingOptions& options,
    const CutConfig* cuts,
    LabelingStats* stats
) {
    if (spec.car_types.size() != spec.car_lengths.size() ||
        spec.car_types.size() != spec.car_heights.size()) {
        throw std::invalid_argument("car_types, car_lengths, and car_heights must have the same length");
    }

    ResourceModel resource_model = build_compartment_resource_model_impl(
        spec,
        options.residual_profile_mode,
        options
    );
    GeneratorMode generator_mode = parse_generator_mode(options.profile_generator_mode);
    const std::vector<std::vector<PlacementChoice>>& choices_by_type = resource_model.choices_by_type;
    std::vector<bool> ordered_type_mask;
    std::vector<int> order = article_search_order(
        spec,
        resource_model,
        options.use_height_order,
        generator_mode,
        options.eps,
        ordered_type_mask
    );
    if (stats != nullptr && generator_mode == GeneratorMode::Hybrid) {
        std::size_t ordered_count = 0;
        std::size_t ordered_quantity = 0;
        std::size_t total_quantity = 0;
        for (std::size_t idx = 0; idx < spec.car_types.size(); ++idx) {
            const int max_q = std::max(0, at_or_default(
                spec.max_quantity_by_type,
                idx,
                options.max_units_per_type
            ));
            total_quantity += static_cast<std::size_t>(max_q);
            if (idx < ordered_type_mask.size() && ordered_type_mask[idx]) {
                ++ordered_count;
                ordered_quantity += static_cast<std::size_t>(max_q);
            }
        }
        stats->hybrid_calls += 1;
        stats->hybrid_ordered_type_sum += ordered_count;
        stats->hybrid_ordered_type_max = std::max(stats->hybrid_ordered_type_max, ordered_count);
        stats->hybrid_ordered_quantity_sum += ordered_quantity;
        stats->hybrid_total_quantity_sum += total_quantity;
    }
    std::vector<int> root_quantities(spec.car_types.size(), 0);
    double root_rc = -duals.gamma / 2.0;
    if (options.use_cuts) {
        root_rc += reduced_cost_shift(spec, root_quantities, cuts);
    }

    std::vector<ResidualLabel> current_labels;
    current_labels.push_back(ResidualLabel{
        0,
        root_rc,
        root_quantities,
        0.0,
        resource_model.capacities,
    });
    for (std::size_t stage_pos = 0; stage_pos < order.size(); ++stage_pos) {
        const int type_idx_int = order[stage_pos];
        const std::size_t type_idx = static_cast<std::size_t>(type_idx_int);
        std::vector<ResidualLabel> next_labels;

        const auto& choices = choices_by_type[type_idx];
        const int max_q = std::min(
            options.max_units_per_type,
            at_or_default(spec.max_quantity_by_type, type_idx, options.max_units_per_type)
        );
        const double unit_length = spec.car_lengths[type_idx];
        const double coef = unit_length +
            at_or_zero(duals.alpha, type_idx) +
            at_or_zero(duals.beta, type_idx) +
            at_or_zero(duals.branch_q, type_idx);

        std::vector<int> remaining;
        for (std::size_t j = stage_pos + 1; j < order.size(); ++j) {
            remaining.push_back(order[j]);
        }
        const bool hybrid_ordered_stage =
            generator_mode == GeneratorMode::Hybrid &&
            ordered_type_mask[static_cast<std::size_t>(type_idx)];

        for (const auto& label : current_labels) {
            if (deadline_reached(options.deadline)) {
                return {};
            }
            const double old_shift = options.use_cuts ? reduced_cost_shift(spec, label.quantities, cuts) : 0.0;
            for (int q = 0; q <= max_q; ++q) {
                if (deadline_reached(options.deadline)) {
                    return {};
                }
                std::vector<int> q_new = label.quantities;
                q_new[type_idx] = q;

                double rc = label.reduced_cost - coef * static_cast<double>(q);
                if (q > 0) {
                    rc -= at_or_zero(duals.branch_a, type_idx);
                }
                if (options.use_cuts) {
                    rc += reduced_cost_shift(spec, q_new, cuts) - old_shift;
                }

                if (options.use_rc_bound && !options.use_cuts) {
                    const double rc_lb = future_reduced_cost_lower_bound(rc, remaining, spec, duals, options);
                    if (rc_lb >= -options.eps) {
                        if (stats != nullptr) {
                            stats->labels_pruned_by_bound += 1;
                        }
                        continue;
                    }
                }

                const double total_length = label.total_length + unit_length * static_cast<double>(q);
                std::vector<ResidualChild> residual_children;
                const bool use_d2_extension =
                    (generator_mode == GeneratorMode::D2 &&
                     ordered_type_mask[static_cast<std::size_t>(type_idx)]) ||
                    hybrid_ordered_stage;
                if (use_d2_extension) {
                    ResidualChild child;
                    if (greedy_child_for_residual(choices, q, label.residual, options.eps, child)) {
                        if (stats != nullptr) {
                            stats->labels_generated_raw += 1;
                            const std::size_t all_placements =
                                count_compositions(q, static_cast<int>(choices.size()));
                            if (all_placements > 1) {
                                stats->labels_avoided_by_d2 = saturated_add(
                                    stats->labels_avoided_by_d2,
                                    all_placements - 1
                                );
                            }
                        }
                        residual_children.push_back(std::move(child));
                    }
                } else {
                    const auto consumptions = placement_consumptions(
                        choices,
                        q,
                        resource_model.capacities.size(),
                        options.deadline
                    );
                    for (const auto& consumption : consumptions) {
                        std::vector<double> residual;
                        if (apply_consumption(label.residual, consumption, options.eps, residual)) {
                            if (stats != nullptr) {
                                stats->labels_generated_raw += 1;
                            }
                            residual_children.push_back(ResidualChild{
                                std::move(residual),
                            });
                        }
                    }
                }

                if (options.use_dominance && options.use_local_d1_pruning) {
                    residual_children = apply_local_d1_dominance(residual_children, options.eps, stats);
                }

                for (auto& child : residual_children) {
                    if (stats != nullptr) {
                        stats->labels_feasible += 1;
                    }
                    next_labels.push_back(ResidualLabel{
                        static_cast<int>(stage_pos + 1),
                        rc,
                        q_new,
                        total_length,
                        std::move(child.residual),
                    });
                }
            }
        }

        if (options.use_dominance) {
            next_labels = apply_residual_dominance(
                std::move(next_labels),
                spec.car_types,
                options.eps,
                options.dominance_support_types,
                stats
            );
        }
        current_labels = std::move(next_labels);
        if (current_labels.empty()) {
            break;
        }
    }

    std::map<std::vector<int>, CompartmentPattern> best_by_quantities;
    for (const auto& label : current_labels) {
        if (label.stage != static_cast<int>(order.size())) {
            continue;
        }
        if (label.total_length > spec.compartment_length_limit + options.eps) {
            continue;
        }
        std::vector<int> key = label.quantities;
        auto pattern = CompartmentPattern{
            spec.compartment_id,
            spec.compartment,
            spec.deck,
            spec.car_types,
            label.quantities,
            label.reduced_cost,
            label.total_length,
        };
        auto it = best_by_quantities.find(key);
        if (it == best_by_quantities.end() ||
            pattern.reduced_cost < it->second.reduced_cost - options.eps) {
            best_by_quantities[key] = std::move(pattern);
        }
    }

    std::vector<CompartmentPattern> patterns;
    patterns.reserve(best_by_quantities.size());
    for (auto& kv : best_by_quantities) {
        patterns.push_back(std::move(kv.second));
    }
    return patterns;
}

StructuralChainDiagnostic diagnose_structural_chain(
    const CompartmentSpec& spec,
    const LabelingOptions& options
) {
    ResourceModel resource_model = build_compartment_resource_model_impl(
        spec,
        options.residual_profile_mode,
        options
    );
    const GeneratorMode generator_mode = parse_generator_mode(options.profile_generator_mode);
    std::vector<bool> ordered_type_mask;
    const std::vector<int> search_order = article_search_order(
        spec,
        resource_model,
        options.use_height_order,
        generator_mode,
        options.eps,
        ordered_type_mask
    );

    StructuralChainDiagnostic diagnostic;
    diagnostic.compartment_id = spec.compartment_id;
    diagnostic.compartment = spec.compartment;
    diagnostic.deck = spec.deck;
    diagnostic.profile_generator_mode = options.profile_generator_mode;
    diagnostic.residual_profile_mode = options.residual_profile_mode;
    diagnostic.type_count = spec.car_types.size();

    for (int idx_int : search_order) {
        const std::size_t idx = static_cast<std::size_t>(idx_int);
        diagnostic.search_order_indices.push_back(idx_int);
        diagnostic.search_order_type_ids.push_back(spec.car_types[idx]);
        if (idx < ordered_type_mask.size() && ordered_type_mask[idx]) {
            diagnostic.ordered_type_indices.push_back(idx_int);
            diagnostic.ordered_type_ids.push_back(spec.car_types[idx]);
        }
    }

    for (std::size_t idx = 0; idx < spec.car_types.size(); ++idx) {
        const int max_q = std::max(0, std::min(
            options.max_units_per_type,
            at_or_default(spec.max_quantity_by_type, idx, options.max_units_per_type)
        ));
        diagnostic.total_quantity_sum += static_cast<std::size_t>(max_q);
        if (idx < ordered_type_mask.size() && ordered_type_mask[idx]) {
            diagnostic.ordered_quantity_sum += static_cast<std::size_t>(max_q);
        }
    }
    diagnostic.d2_full_certificate =
        diagnostic.ordered_type_indices.size() == spec.car_types.size();
    return diagnostic;
}

CompartmentSpec make_compartment_spec(
    CompartmentKind compartment,
    DeckMode deck,
    const std::vector<int>& car_types,
    const std::vector<double>& car_lengths,
    const std::vector<double>& car_heights,
    const std::vector<int>& max_quantity_by_type,
    int num_splits,
    bool independent_mode_split,
    int max_units_per_type
) {
    std::vector<int> max_q = max_quantity_by_type;
    if (max_q.size() < car_types.size()) {
        max_q.resize(car_types.size(), max_units_per_type);
    }
    for (int& value : max_q) {
        value = std::min(value, max_units_per_type);
    }

    const std::string cid = compartment_name(compartment) + "_" + deck_name(deck);
    return CompartmentSpec{
        cid,
        compartment,
        deck,
        car_types,
        car_lengths,
        car_heights,
        max_q,
        compartment == CompartmentKind::Upper ? Config::top_len : Config::bottom_len,
        num_splits,
        independent_mode_split,
    };
}

}  // namespace bpc_label
