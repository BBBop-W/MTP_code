#include "bpc_label/LabelPricing.hpp"

#include <algorithm>
#include <cmath>
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

bool choice_weakly_dominates(const PlacementChoice& a, const PlacementChoice& b) {
    return vector_subset(a.hits, b.hits);
}

std::vector<PlacementChoice> prune_dominated_choices(const std::vector<PlacementChoice>& choices) {
    std::vector<PlacementChoice> kept;
    for (const auto& choice : choices) {
        bool dominated = false;
        for (const auto& other : choices) {
            if (&choice == &other) {
                continue;
            }
            if (choice_weakly_dominates(other, choice)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            kept.push_back(choice);
        }
    }
    return kept;
}

bool choices_cover_all(
    const std::vector<PlacementChoice>& original,
    const std::vector<PlacementChoice>& reduced
) {
    for (const auto& old : original) {
        bool covered = false;
        for (const auto& choice : reduced) {
            if (choice_weakly_dominates(choice, old)) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<PlacementChoice>> choices_for_profile_generator(
    const ResourceModel& resource_model,
    const std::string& mode
) {
    if (mode == "exact" || mode == "ex" || mode == "e") {
        return resource_model.choices_by_type;
    }
    if (mode != "gr" && mode != "hyb" && mode != "hybrid") {
        throw std::invalid_argument("unknown profile_generator_mode: " + mode);
    }

    std::vector<std::vector<PlacementChoice>> out = resource_model.choices_by_type;
    for (std::size_t i = 0; i < out.size(); ++i) {
        auto reduced = prune_dominated_choices(out[i]);
        if (!reduced.empty() && choices_cover_all(out[i], reduced)) {
            out[i] = std::move(reduced);
        }
    }
    return out;
}

void enumerate_consumption_rec(
    const std::vector<PlacementChoice>& choices,
    int choice_idx,
    int remaining,
    double unit_resource,
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
            const double amount = static_cast<double>(count) * unit_resource;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] += amount;
            }
        }
        if (seen.insert(consumption).second) {
            out.push_back(consumption);
        }
        if (count > 0) {
            const double amount = static_cast<double>(count) * unit_resource;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] -= amount;
            }
        }
        return;
    }

    for (int count = 0; count <= remaining; ++count) {
        if (count > 0) {
            const double amount = static_cast<double>(count) * unit_resource;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] += amount;
            }
        }
        enumerate_consumption_rec(
            choices,
            choice_idx + 1,
            remaining - count,
            unit_resource,
            consumption,
            seen,
            out,
            deadline
        );
        if (count > 0) {
            const double amount = static_cast<double>(count) * unit_resource;
            for (int hit : choices[static_cast<std::size_t>(choice_idx)].hits) {
                consumption[static_cast<std::size_t>(hit)] -= amount;
            }
        }
    }
}

std::vector<std::vector<double>> placement_consumptions(
    const std::vector<PlacementChoice>& choices,
    int quantity,
    double unit_resource,
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
    enumerate_consumption_rec(choices, 0, quantity, unit_resource, consumption, seen, out, deadline);
    return out;
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

std::vector<std::vector<double>> local_residual_skyline(
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
                stats->labels_pruned_by_local_skyline += 1;
            }
            continue;
        }
        if (!remove_idx.empty()) {
            if (stats != nullptr) {
                stats->labels_pruned_by_local_skyline += remove_idx.size();
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

    return profile_strict || quantity_strict || a.reduced_cost < b.reduced_cost - eps;
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

std::vector<double> unique_heights_from_spec(const CompartmentSpec& spec) {
    std::vector<double> heights = spec.car_heights;
    std::sort(heights.begin(), heights.end());
    heights.erase(std::unique(heights.begin(), heights.end()), heights.end());
    return heights;
}

}  // namespace

ResourceModel build_compartment_resource_model(
    const CompartmentSpec& spec,
    const std::string& interval_profile
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
                out.intervals.push_back({static_cast<double>(l_idx), static_cast<double>(r_idx), cap});
                out.capacities.push_back(cap);
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
            choices.push_back(PlacementChoice{"central", 0, hits_for("central", 0)});
        }
        for (int idx = 1; idx <= n_blocks; ++idx) {
            if (height <= limit_left[static_cast<std::size_t>(idx)]) {
                choices.push_back(PlacementChoice{"left", idx, hits_for("left", idx)});
            }
            if (height <= limit_right[static_cast<std::size_t>(idx)]) {
                choices.push_back(PlacementChoice{"right", idx, hits_for("right", idx)});
            }
        }

        std::map<std::vector<int>, PlacementChoice> dedup;
        for (const auto& choice : choices) {
            dedup.emplace(choice.hits, choice);
        }
        for (const auto& kv : dedup) {
            out.choices_by_type[t].push_back(kv.second);
        }
    }

    return out;
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
            spec.car_lengths[type_idx] + resource_model.delta,
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
        current_residuals = local_residual_skyline(next_residuals, eps, nullptr);
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

    const std::vector<int> order = ordered_type_indices(spec, options.use_height_order);
    ResourceModel resource_model = build_compartment_resource_model(spec, options.residual_profile_mode);
    std::vector<std::vector<PlacementChoice>> choices_by_type =
        choices_for_profile_generator(resource_model, options.profile_generator_mode);

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
        const double unit_resource = spec.car_lengths[type_idx] + resource_model.delta;
        const double unit_length = spec.car_lengths[type_idx];
        const double coef = unit_length +
            at_or_zero(duals.alpha, type_idx) +
            at_or_zero(duals.beta, type_idx) +
            at_or_zero(duals.branch_q, type_idx);

        std::vector<int> remaining;
        for (std::size_t j = stage_pos + 1; j < order.size(); ++j) {
            remaining.push_back(order[j]);
        }

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
                std::vector<std::vector<double>> residual_children;
                const auto consumptions = placement_consumptions(
                    choices,
                    q,
                    unit_resource,
                    resource_model.capacities.size(),
                    options.deadline
                );
                for (const auto& consumption : consumptions) {
                    std::vector<double> residual;
                    if (apply_consumption(label.residual, consumption, options.eps, residual)) {
                        residual_children.push_back(std::move(residual));
                    }
                }

                if (options.use_dominance && options.use_local_residual_skyline) {
                    residual_children = local_residual_skyline(residual_children, options.eps, stats);
                }

                for (auto& residual : residual_children) {
                    if (stats != nullptr) {
                        stats->labels_feasible += 1;
                    }
                    next_labels.push_back(ResidualLabel{
                        static_cast<int>(stage_pos + 1),
                        rc,
                        q_new,
                        total_length,
                        std::move(residual),
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
