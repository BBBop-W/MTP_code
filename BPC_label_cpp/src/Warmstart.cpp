#include "bpc_label/Warmstart.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace bpc_label {

namespace {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }
};

void append_utf8(std::string& out, int cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    throw std::runtime_error("invalid JSON unicode escape");
}

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    JsonValue parse() {
        JsonValue value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            throw std::runtime_error("trailing bytes in JSON");
        }
        return value;
    }

private:
    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("unexpected end of JSON");
        }
        const char ch = text_[pos_];
        if (ch == '{') return parse_object();
        if (ch == '[') return parse_array();
        if (ch == '"') return JsonValue{JsonValue::Type::String, false, 0.0, parse_string(), {}, {}};
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) return parse_number();
        if (match("true")) return JsonValue{JsonValue::Type::Bool, true, 0.0, "", {}, {}};
        if (match("false")) return JsonValue{JsonValue::Type::Bool, false, 0.0, "", {}, {}};
        if (match("null")) return JsonValue{};
        throw std::runtime_error("invalid JSON value");
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        skip_ws();
        if (peek('}')) {
            expect('}');
            return value;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            value.object.emplace(std::move(key), parse_value());
            skip_ws();
            if (peek('}')) {
                expect('}');
                break;
            }
            expect(',');
        }
        return value;
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        skip_ws();
        if (peek(']')) {
            expect(']');
            return value;
        }
        while (true) {
            value.array.push_back(parse_value());
            skip_ws();
            if (peek(']')) {
                expect(']');
                break;
            }
            expect(',');
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < text_.size()) {
            const char ch = text_[pos_++];
            if (ch == '"') {
                return out;
            }
            if (ch != '\\') {
                out.push_back(ch);
                continue;
            }
            if (pos_ >= text_.size()) {
                throw std::runtime_error("unterminated JSON escape");
            }
            const char esc = text_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': append_utf8(out, parse_hex4()); break;
                default: throw std::runtime_error("invalid JSON escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    int parse_hex4() {
        if (pos_ + 4 > text_.size()) {
            throw std::runtime_error("short JSON unicode escape");
        }
        int cp = 0;
        for (int i = 0; i < 4; ++i) {
            cp = cp * 16 + hex_value(text_[pos_++]);
        }
        return cp;
    }

    JsonValue parse_number() {
        const std::size_t begin = pos_;
        if (text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number = std::stod(text_.substr(begin, pos_ - begin));
        return value;
    }

    bool match(const char* literal) {
        const std::string word(literal);
        if (text_.compare(pos_, word.size(), word) == 0) {
            pos_ += word.size();
            return true;
        }
        return false;
    }

    bool peek(char ch) const {
        return pos_ < text_.size() && text_[pos_] == ch;
    }

    void expect(char ch) {
        skip_ws();
        if (pos_ >= text_.size() || text_[pos_] != ch) {
            throw std::runtime_error("unexpected JSON token");
        }
        ++pos_;
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    std::string text_;
    std::size_t pos_ = 0;
};

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

const JsonValue* object_get(const JsonValue& value, const std::string& key) {
    if (!value.is_object()) {
        return nullptr;
    }
    auto it = value.object.find(key);
    return it == value.object.end() ? nullptr : &it->second;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

int rounded_count(double value) {
    return static_cast<int>(std::llround(value));
}

DeckMode parse_deck_mode(const std::string& raw) {
    const std::string value = trim(raw);
    if (value == "h-h" || value == "HH") return DeckMode::HH;
    if (value == "h-m" || value == "HM") return DeckMode::HM;
    if (value == "m-h" || value == "MH") return DeckMode::MH;
    if (value == "m-m" || value == "MM") return DeckMode::MM;
    throw std::runtime_error("unknown deck mode: " + raw);
}

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open warmstart JSON: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::unordered_map<std::string, int> build_name_map(const InstanceData& instance) {
    std::map<std::string, std::set<int>> candidates;
    for (std::size_t i = 0; i < instance.car_types.size(); ++i) {
        const int idx = static_cast<int>(i);
        const std::string program = i < instance.programs.size() ? trim(instance.programs[i]) : "";
        const std::string model = i < instance.models.size() ? trim(instance.models[i]) : "";
        if (!program.empty() && !model.empty()) {
            candidates[program + " " + model].insert(idx);
            candidates[program + "-" + model].insert(idx);
        }
        if (!model.empty()) {
            candidates[model].insert(idx);
        }
    }

    std::unordered_map<std::string, int> out;
    for (const auto& [name, ids] : candidates) {
        if (ids.size() == 1) {
            out[name] = *ids.begin();
        }
    }
    return out;
}

void add_named_quantity(
    const std::string& raw_name,
    int count,
    const std::unordered_map<std::string, int>& name_map,
    std::vector<int>& quantities,
    int& unknown
) {
    if (count <= 0) {
        return;
    }
    const std::string name = trim(raw_name);
    auto it = name_map.find(name);
    if (it == name_map.end()) {
        unknown += count;
        return;
    }
    quantities[static_cast<std::size_t>(it->second)] += count;
}

void collect_array_names(
    const JsonValue* value,
    const std::unordered_map<std::string, int>& name_map,
    std::vector<int>& quantities,
    int& unknown
) {
    if (value == nullptr || !value->is_array()) {
        return;
    }
    for (const JsonValue& item : value->array) {
        if (item.is_string()) {
            add_named_quantity(item.string, 1, name_map, quantities, unknown);
        } else {
            ++unknown;
        }
    }
}

void collect_object_counts(
    const JsonValue* value,
    const std::unordered_map<std::string, int>& name_map,
    std::vector<int>& quantities,
    int& unknown
) {
    if (value == nullptr || !value->is_object()) {
        return;
    }
    for (const auto& [name, count_value] : value->object) {
        const int count = count_value.is_number() ? rounded_count(count_value.number) : 1;
        add_named_quantity(name, count, name_map, quantities, unknown);
    }
}

std::vector<int> compartment_quantities(
    const JsonValue& carriage,
    const std::string& array_key,
    const std::string& object_prefix,
    const InstanceData& instance,
    const std::unordered_map<std::string, int>& name_map,
    int& unknown
) {
    std::vector<int> quantities(instance.car_types.size(), 0);
    collect_array_names(object_get(carriage, array_key), name_map, quantities, unknown);
    if (carriage.is_object()) {
        for (const auto& [key, value] : carriage.object) {
            if (starts_with(key, object_prefix)) {
                collect_object_counts(&value, name_map, quantities, unknown);
            }
        }
    }
    return quantities;
}

bool is_empty_quantities(const std::vector<int>& quantities) {
    return std::all_of(quantities.begin(), quantities.end(), [](int q) { return q == 0; });
}

int total_units(const std::vector<int>& quantities) {
    return std::accumulate(quantities.begin(), quantities.end(), 0);
}

std::vector<int> add_quantities(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = (i < a.size() ? a[i] : 0) + (i < b.size() ? b[i] : 0);
    }
    return out;
}

bool within_upper_bounds(const InstanceData& instance, const std::vector<int>& quantities) {
    for (std::size_t i = 0; i < quantities.size(); ++i) {
        if (quantities[i] > instance.upper_bounds[i]) {
            return false;
        }
    }
    return true;
}

double column_cost(const InstanceData& instance, const std::vector<int>& quantities) {
    double cost = 0.0;
    for (std::size_t i = 0; i < quantities.size(); ++i) {
        cost -= instance.lengths[i] * static_cast<double>(quantities[i]);
    }
    return cost;
}

bool valid_compartment(
    const InstanceData& instance,
    const PricingOptions& pricing_options,
    CompartmentKind compartment,
    DeckMode deck,
    const std::vector<int>& quantities
) {
    if (is_empty_quantities(quantities)) {
        return true;
    }
    if (total_units(quantities) > Config::max_units_per_compartment) {
        return false;
    }
    if (!within_upper_bounds(instance, quantities)) {
        return false;
    }
    auto spec = make_compartment_spec(
        compartment,
        deck,
        instance.car_types,
        instance.lengths,
        instance.heights,
        instance.upper_bounds,
        pricing_options.num_splits,
        pricing_options.independent_mode_split,
        pricing_options.labeling.max_units_per_type
    );
    return is_compartment_feasible(spec, quantities, "full");
}

std::string make_warmstart_id(const std::string& prefix, int carriage_idx, const std::string& suffix) {
    std::ostringstream oss;
    oss << prefix << "_c" << carriage_idx;
    if (!suffix.empty()) {
        oss << "_" << suffix;
    }
    return oss.str();
}

void print_result(
    const std::string& path,
    const std::string& prefix,
    const WarmstartLoadResult& result,
    BpcMethod method
) {
    std::cout << "[Warmstart] " << method_name(method) << " " << prefix
              << " from " << path
              << ": added=" << result.added
              << ", infeasible=" << result.skipped_infeasible
              << ", duplicate=" << result.skipped_duplicate
              << ", empty=" << result.skipped_empty
              << ", unknown=" << result.skipped_unknown
              << ", over_limit=" << result.skipped_over_limit << "\n";
}

WarmstartLoadResult load_one_json(
    MasterProblem& master,
    const std::string& prefix,
    const std::string& path,
    const PricingOptions& pricing_options
) {
    const InstanceData& instance = master.instance();
    const JsonValue root = JsonParser(read_file(path)).parse();
    const JsonValue* carriage_array = object_get(root, "carriage");
    if (carriage_array == nullptr || !carriage_array->is_array()) {
        throw std::runtime_error("warmstart JSON has no carriage array: " + path);
    }

    WarmstartLoadResult result;
    const auto name_map = build_name_map(instance);

    for (std::size_t idx = 0; idx < carriage_array->array.size(); ++idx) {
        const JsonValue& carriage = carriage_array->array[idx];
        if (!carriage.is_object()) {
            result.skipped_unknown += 1;
            continue;
        }

        DeckMode deck = DeckMode::HH;
        try {
            const JsonValue* position = object_get(carriage, "position");
            deck = position != nullptr && position->is_string()
                ? parse_deck_mode(position->string)
                : DeckMode::HH;
        } catch (const std::exception&) {
            result.skipped_unknown += 1;
            continue;
        }

        int unknown = 0;
        std::vector<int> upper = compartment_quantities(
            carriage,
            "top",
            "upper_",
            instance,
            name_map,
            unknown
        );
        std::vector<int> lower = compartment_quantities(
            carriage,
            "bottom",
            "lower_",
            instance,
            name_map,
            unknown
        );
        result.skipped_unknown += unknown;

        if (master.method() == BpcMethod::WagonLabel) {
            if (is_empty_quantities(upper) && is_empty_quantities(lower)) {
                result.skipped_empty += 1;
                continue;
            }
            if (!valid_compartment(instance, pricing_options, CompartmentKind::Upper, deck, upper) ||
                !valid_compartment(instance, pricing_options, CompartmentKind::Lower, deck, lower)) {
                result.skipped_infeasible += 1;
                continue;
            }

            std::vector<int> q = add_quantities(upper, lower);
            if (!within_upper_bounds(instance, q)) {
                result.skipped_over_limit += 1;
                continue;
            }

            PatternColumn column;
            column.id = make_warmstart_id(prefix, static_cast<int>(idx), "");
            column.quantities = std::move(q);
            column.cost = column_cost(instance, column.quantities);
            column.compartment = CompartmentKind::Lower;
            column.deck = deck;
            column.is_compartment_column = false;
            if (master.add_column(column)) {
                result.added += 1;
            } else {
                result.skipped_duplicate += 1;
            }
            continue;
        }

        struct CompartmentCandidate {
            CompartmentKind compartment;
            const std::vector<int>* quantities;
        };
        for (const CompartmentCandidate item : {
                 CompartmentCandidate{CompartmentKind::Upper, &upper},
                 CompartmentCandidate{CompartmentKind::Lower, &lower},
             }) {
            const CompartmentKind compartment = item.compartment;
            const std::vector<int>& q = *item.quantities;
            if (is_empty_quantities(q)) {
                result.skipped_empty += 1;
                continue;
            }
            if (total_units(q) > Config::max_units_per_compartment || !within_upper_bounds(instance, q)) {
                result.skipped_over_limit += 1;
                continue;
            }
            if (!valid_compartment(instance, pricing_options, compartment, deck, q)) {
                result.skipped_infeasible += 1;
                continue;
            }

            PatternColumn column;
            column.id = make_warmstart_id(
                prefix,
                static_cast<int>(idx),
                compartment == CompartmentKind::Upper ? "upper" : "lower"
            );
            column.quantities = q;
            column.cost = column_cost(instance, column.quantities);
            column.compartment = compartment;
            column.deck = deck;
            column.is_compartment_column = true;
            if (master.add_column(column)) {
                result.added += 1;
            } else {
                result.skipped_duplicate += 1;
            }
        }
    }

    return result;
}

}  // namespace

WarmstartLoadResult& WarmstartLoadResult::operator+=(const WarmstartLoadResult& other) {
    added += other.added;
    skipped_infeasible += other.skipped_infeasible;
    skipped_duplicate += other.skipped_duplicate;
    skipped_empty += other.skipped_empty;
    skipped_unknown += other.skipped_unknown;
    skipped_over_limit += other.skipped_over_limit;
    return *this;
}

WarmstartLoadResult load_warmstart_columns(
    MasterProblem& master,
    const std::vector<std::pair<std::string, std::string>>& warmstarts,
    const PricingOptions& pricing_options,
    bool log_progress
) {
    WarmstartLoadResult total;
    for (const auto& [prefix, path] : warmstarts) {
        WarmstartLoadResult single = load_one_json(master, prefix, path, pricing_options);
        total += single;
        if (log_progress) {
            print_result(path, prefix, single, master.method());
        }
    }
    return total;
}

}  // namespace bpc_label
