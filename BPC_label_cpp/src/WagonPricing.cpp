#include "bpc_label/WagonPricing.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>

namespace bpc_label {

namespace {

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

int at_or_zero(const std::vector<int>& values, std::size_t idx) {
    return idx < values.size() ? values[idx] : 0;
}

double column_cost(const MasterSnapshot& master, const std::vector<int>& quantities) {
    double cost = 0.0;
    for (std::size_t i = 0; i < master.car_types.size(); ++i) {
        cost -= master.car_lengths[i] * static_cast<double>(quantities[i]);
    }
    return cost;
}

void accumulate(LabelingStats& dst, const LabelingStats& src) {
    dst.labels_generated_raw += src.labels_generated_raw;
    dst.labels_feasible += src.labels_feasible;
    dst.labels_pruned_by_bound += src.labels_pruned_by_bound;
    dst.labels_pruned_by_dominance += src.labels_pruned_by_dominance;
    dst.labels_after_dominance += src.labels_after_dominance;
    dst.labels_avoided_by_d2 += src.labels_avoided_by_d2;
    dst.hybrid_calls += src.hybrid_calls;
    dst.hybrid_ordered_type_sum += src.hybrid_ordered_type_sum;
    dst.hybrid_ordered_type_max = std::max(dst.hybrid_ordered_type_max, src.hybrid_ordered_type_max);
    dst.hybrid_ordered_quantity_sum += src.hybrid_ordered_quantity_sum;
    dst.hybrid_total_quantity_sum += src.hybrid_total_quantity_sum;
}

std::vector<CompartmentPattern> negative_top_k(
    std::vector<CompartmentPattern> patterns,
    int k
) {
    patterns.erase(
        std::remove_if(
            patterns.begin(),
            patterns.end(),
            [](const CompartmentPattern& pattern) { return pattern.reduced_cost >= -1e-5; }
        ),
        patterns.end()
    );
    std::sort(patterns.begin(), patterns.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });
    if (k > 0 && static_cast<std::size_t>(k) < patterns.size()) {
        patterns.resize(static_cast<std::size_t>(k));
    }
    return patterns;
}

CutConfig make_cut_config(const MasterSnapshot& master, bool eta_upper_only) {
    return CutConfig{
        master.dual_eta,
        master.dual_sigma,
        master.max_total_by_type,
        eta_upper_only,
    };
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

std::vector<CompartmentPattern> generate_subpatterns(
    const MasterSnapshot& master,
    CompartmentKind compartment,
    DeckMode deck,
    double gamma,
    const PricingOptions& options,
    const CutConfig* cuts,
    LabelingStats* stats
) {
    auto spec = make_compartment_spec(
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
    auto duals = make_base_duals(master, gamma);
    return generate_compartment_patterns_residual(spec, duals, options.labeling, cuts, stats);
}

std::vector<int> merge_quantities(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = at_or_zero(a, i) + at_or_zero(b, i);
    }
    return out;
}

double full_wagon_reduced_cost(
    const std::vector<int>& quantities,
    const MasterSnapshot& master,
    bool use_cuts
) {
    double rc = -master.dual_gamma;
    for (std::size_t i = 0; i < master.car_types.size(); ++i) {
        const int q = at_or_zero(quantities, i);
        if (q <= 0) {
            continue;
        }
        rc -= (
            master.car_lengths[i] +
            at_or_zero(master.dual_alpha, i) +
            at_or_zero(master.dual_beta, i) +
            at_or_zero(master.dual_branch_q, i)
        ) * static_cast<double>(q);
        rc -= at_or_zero(master.dual_branch_a, i);
    }

    if (use_cuts) {
        rc -= master.dual_eta;
        std::unordered_map<int, std::size_t> index_by_type;
        for (std::size_t i = 0; i < master.car_types.size(); ++i) {
            index_by_type[master.car_types[i]] = i;
        }
        for (const auto& item : master.dual_sigma) {
            int val = 0;
            for (int type_id : item.type_ids) {
                auto it = index_by_type.find(type_id);
                if (it == index_by_type.end()) {
                    continue;
                }
                const std::size_t idx = it->second;
                if (master.max_total_by_type[idx] > 0 &&
                    quantities[idx] > static_cast<double>(master.max_total_by_type[idx]) / 2.0) {
                    val += 1;
                }
            }
            const int coeff = static_cast<int>(std::floor(0.5 * static_cast<double>(val)));
            rc -= item.sigma * static_cast<double>(coeff);
        }
    }

    return rc;
}

struct MergedPattern {
    DeckMode deck = DeckMode::HH;
    std::vector<int> quantities;
    double reduced_cost = 0.0;
    CompartmentPattern upper;
    CompartmentPattern lower;
};

std::vector<std::pair<std::size_t, std::size_t>> pair_order(std::size_t n_up, std::size_t n_low) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    if (n_up == 0 || n_low == 0) {
        return out;
    }
    out.push_back({0, 0});
    for (std::size_t i = 0; i < n_up; ++i) {
        for (std::size_t j = 0; j < n_low; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }
            out.push_back({i, j});
        }
    }
    return out;
}

std::vector<MergedPattern> merge_feasible_patterns(
    DeckMode deck,
    std::vector<CompartmentPattern> upper,
    std::vector<CompartmentPattern> lower,
    const MasterSnapshot& master,
    bool use_cuts,
    const std::set<std::vector<int>>& forbidden_signatures,
    int max_results
) {
    if (upper.empty() || lower.empty()) {
        return {};
    }

    std::sort(upper.begin(), upper.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });
    std::sort(lower.begin(), lower.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });

    std::map<std::vector<int>, MergedPattern> best_by_signature;
    for (const auto& [i, j] : pair_order(upper.size(), lower.size())) {
        const auto& u = upper[i];
        const auto& l = lower[j];
        if (u.deck != l.deck || u.deck != deck) {
            continue;
        }
        std::vector<int> q = merge_quantities(u.quantities, l.quantities);
        if (forbidden_signatures.find(q) != forbidden_signatures.end()) {
            continue;
        }
        const double rc = full_wagon_reduced_cost(q, master, use_cuts);
        if (rc >= -1e-5) {
            continue;
        }
        MergedPattern candidate{deck, q, rc, u, l};
        auto it = best_by_signature.find(q);
        if (it == best_by_signature.end() || candidate.reduced_cost < it->second.reduced_cost) {
            best_by_signature[q] = std::move(candidate);
        }
    }

    std::vector<MergedPattern> out;
    for (auto& kv : best_by_signature) {
        out.push_back(std::move(kv.second));
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });
    if (max_results > 0 && static_cast<std::size_t>(max_results) < out.size()) {
        out.resize(static_cast<std::size_t>(max_results));
    }
    return out;
}

std::string make_column_id(const std::string& prefix, DeckMode deck, int seq) {
    std::ostringstream oss;
    oss << prefix << "_" << deck_name(deck) << "_" << seq;
    return oss.str();
}

}  // namespace

std::vector<PricingColumn> price_compartment_columns(
    const MasterSnapshot& master,
    const PricingOptions& options,
    PricingStats* stats
) {
    std::vector<PricingColumn> columns;
    int seq = 0;

    CutConfig cut_config = make_cut_config(master, true);
    const CutConfig* cuts = options.use_cuts ? &cut_config : nullptr;
    PricingOptions local_options = options;
    local_options.labeling.use_cuts = options.use_cuts;
    local_options.labeling.dominance_support_types.clear();
    for (std::size_t i = 0; i < master.dual_branch_a.size() && i < master.car_types.size(); ++i) {
        if (std::abs(master.dual_branch_a[i]) > local_options.labeling.eps) {
            local_options.labeling.dominance_support_types.push_back(master.car_types[i]);
        }
    }

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

            LabelingStats local_stats;
            auto patterns = generate_subpatterns(
                master,
                compartment,
                deck,
                gamma,
                local_options,
                cuts,
                &local_stats
            );
            if (stats != nullptr) {
                stats->generated_subpatterns += patterns.size();
                accumulate(stats->labeling_stats, local_stats);
            }

            for (const auto& pattern : negative_top_k(patterns, options.max_columns_per_subproblem)) {
                if (pattern.reduced_cost >= -1e-5) {
                    continue;
                }
                ++seq;
                columns.push_back(PricingColumn{
                    make_column_id("comp", deck, seq),
                    deck,
                    compartment,
                    master.car_types,
                    pattern.quantities,
                    pattern.reduced_cost,
                    column_cost(master, pattern.quantities),
                    "compartment_label",
                });
            }
        }
    }

    std::sort(columns.begin(), columns.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });
    return columns;
}

std::vector<PricingColumn> price_wagon_columns(
    const MasterSnapshot& master,
    const PricingOptions& options,
    const std::set<std::vector<int>>& forbidden_signatures,
    PricingStats* stats
) {
    std::map<std::pair<DeckMode, CompartmentKind>, std::vector<CompartmentPattern>> subpatterns;
    std::vector<MergedPattern> merged_candidates;

    LabelingOptions wagon_labeling = options.labeling;
    wagon_labeling.use_cuts = options.use_cuts;
    wagon_labeling.dominance_support_types.clear();
    for (std::size_t i = 0; i < master.dual_branch_a.size() && i < master.car_types.size(); ++i) {
        if (std::abs(master.dual_branch_a[i]) > wagon_labeling.eps) {
            wagon_labeling.dominance_support_types.push_back(master.car_types[i]);
        }
    }

    PricingOptions local_options = options;
    local_options.labeling = wagon_labeling;

    DualValues subproblem_duals{
        master.dual_alpha,
        master.dual_beta,
        master.dual_gamma,
        std::vector<double>(master.car_types.size(), 0.0),
        master.dual_branch_q,
    };

    int sequence_pairs = 0;
    for (DeckMode deck : wagon_deck_order()) {
        if (deadline_reached(options.labeling.deadline)) {
            break;
        }
        for (CompartmentKind compartment : {CompartmentKind::Upper, CompartmentKind::Lower}) {
            if (deadline_reached(options.labeling.deadline)) {
                break;
            }
            auto spec = make_compartment_spec(
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

            LabelingStats local_stats;
            auto patterns = generate_compartment_patterns_residual(
                spec,
                subproblem_duals,
                local_options.labeling,
                nullptr,
                &local_stats
            );
            if (stats != nullptr) {
                stats->generated_subpatterns += patterns.size();
                accumulate(stats->labeling_stats, local_stats);
            }

            subpatterns[{deck, compartment}] = std::move(patterns);
            if (compartment == CompartmentKind::Lower) {
                if (stats != nullptr) {
                    stats->merge_attempt_pairs += 1;
                }
                ++sequence_pairs;
                (void)sequence_pairs;
                auto merged = merge_feasible_patterns(
                    deck,
                    subpatterns[{deck, CompartmentKind::Upper}],
                    subpatterns[{deck, CompartmentKind::Lower}],
                    master,
                    options.use_cuts,
                    forbidden_signatures,
                    options.max_columns_per_pricing
                );
                merged_candidates.insert(
                    merged_candidates.end(),
                    std::make_move_iterator(merged.begin()),
                    std::make_move_iterator(merged.end())
                );
            }
        }
    }

    std::sort(merged_candidates.begin(), merged_candidates.end(), [](const auto& a, const auto& b) {
        return a.reduced_cost < b.reduced_cost;
    });

    std::vector<PricingColumn> columns;
    std::set<std::vector<int>> emitted = forbidden_signatures;
    int seq = 0;
    for (const auto& merged : merged_candidates) {
        if (emitted.find(merged.quantities) != emitted.end()) {
            continue;
        }
        emitted.insert(merged.quantities);
        ++seq;
        columns.push_back(PricingColumn{
            make_column_id("wagon", merged.deck, seq),
            merged.deck,
            CompartmentKind::Lower,
            master.car_types,
            merged.quantities,
            merged.reduced_cost,
            column_cost(master, merged.quantities),
            "wagon_label_merge",
        });
        if (options.max_columns_per_pricing > 0 &&
            static_cast<int>(columns.size()) >= options.max_columns_per_pricing) {
            break;
        }
    }
    return columns;
}

}  // namespace bpc_label
