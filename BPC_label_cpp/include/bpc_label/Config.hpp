#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace bpc_label {

struct Config {
    static constexpr double eps = 1e-9;
    static constexpr int max_units_per_compartment = 10;
    // Nonpositive values mean "unlimited" for label-pricing column updates.
    static constexpr int max_wagon_pricing_columns = 0;
    static constexpr int max_compartment_pricing_columns_per_subproblem = 0;
    static constexpr double safety_clearance_delta = 50.0;

    static constexpr double A_len = 4300.0;
    static constexpr double B_len = 2000.0;
    static constexpr double C_len = 12400.0;
    static constexpr double D_len = 5000.0;
    static constexpr double E_len = 14900.0;

    static constexpr double A_height_h = 1700.0;
    static constexpr double A_height_m = 1780.0;
    static constexpr double B_height = 2100.0;
    static constexpr double C_height = 2270.0;
    static constexpr double E_height = 2070.0;
    static constexpr double D_height_h = 2070.0;
    static constexpr double D_height_m = 1720.0;

    static constexpr double bottom_len = 2.0 * A_len + 2.0 * B_len + C_len;
    static constexpr double top_len = 2.0 * D_len + E_len;
};

inline double monotonic_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline bool deadline_reached(double deadline) {
    return deadline > 0.0 && monotonic_seconds() >= deadline;
}

enum class CompartmentKind { Upper, Lower };
enum class DeckMode { HH, HM, MH, MM };

inline std::string deck_name(DeckMode mode) {
    switch (mode) {
        case DeckMode::HH:
            return "h-h";
        case DeckMode::HM:
            return "h-m";
        case DeckMode::MH:
            return "m-h";
        case DeckMode::MM:
            return "m-m";
    }
    return "h-h";
}

inline std::array<char, 2> deck_sides(DeckMode mode) {
    switch (mode) {
        case DeckMode::HH:
            return {'h', 'h'};
        case DeckMode::HM:
            return {'h', 'm'};
        case DeckMode::MH:
            return {'m', 'h'};
        case DeckMode::MM:
            return {'m', 'm'};
    }
    return {'h', 'h'};
}

inline std::vector<DeckMode> compartment_deck_order() {
    return {DeckMode::HH, DeckMode::HM, DeckMode::MH, DeckMode::MM};
}

inline std::vector<DeckMode> wagon_deck_order() {
    return {DeckMode::HH, DeckMode::MM, DeckMode::MH, DeckMode::HM};
}

inline std::string compartment_name(CompartmentKind kind) {
    return kind == CompartmentKind::Upper ? "upper" : "lower";
}

}  // namespace bpc_label
