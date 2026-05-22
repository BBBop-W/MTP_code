#include "bpc_label/InstanceIO.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace bpc_label {

namespace {

std::string trim(std::string value) {
    while (!value.empty() && static_cast<unsigned char>(value.front()) == 0xEF) {
        value.erase(value.begin());
    }
    if (value.size() >= 2 &&
        static_cast<unsigned char>(value[0]) == 0xBB &&
        static_cast<unsigned char>(value[1]) == 0xBF) {
        value.erase(value.begin(), value.begin() + 2);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    bool in_quote = false;
    for (char ch : line) {
        if (ch == '"') {
            in_quote = !in_quote;
            continue;
        }
        if (ch == ',' && !in_quote) {
            cells.push_back(trim(cell));
            cell.clear();
        } else {
            cell.push_back(ch);
        }
    }
    cells.push_back(trim(cell));
    return cells;
}

std::string normalize_header(std::string header) {
    header = trim(std::move(header));
    if (header == "Length") return "length";
    if (header == "Height") return "height";
    if (header == "Optional#") return "optional";
    if (header == "Mandatory#") return "mandatory";
    std::transform(header.begin(), header.end(), header.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return header;
}

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

int find_col(const std::unordered_map<std::string, int>& header, const std::string& name) {
    auto it = header.find(name);
    if (it == header.end()) {
        throw std::runtime_error("cars.csv missing required column: " + name);
    }
    return it->second;
}

int optional_col(const std::unordered_map<std::string, int>& header, const std::string& name) {
    auto it = header.find(name);
    return it == header.end() ? -1 : it->second;
}

}  // namespace

InstanceData load_instance(const std::string& instance_dir) {
    InstanceData data;

    const std::string cars_path = join_path(instance_dir, "cars.csv");
    std::ifstream cars(cars_path);
    if (!cars) {
        throw std::runtime_error("cannot open " + cars_path);
    }

    std::string line;
    if (!std::getline(cars, line)) {
        throw std::runtime_error("empty cars.csv");
    }
    auto headers = split_csv_line(line);
    std::unordered_map<std::string, int> header;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        header[normalize_header(headers[i])] = static_cast<int>(i);
    }

    const int length_idx = find_col(header, "length");
    const int height_idx = find_col(header, "height");
    const int optional_idx = find_col(header, "optional");
    const int mandatory_idx = find_col(header, "mandatory");
    const int program_idx = optional_col(header, "program");
    const int model_idx = optional_col(header, "model");

    int type_id = 1;
    while (std::getline(cars, line)) {
        if (trim(line).empty()) {
            continue;
        }
        auto cells = split_csv_line(line);
        const int required = std::max({length_idx, height_idx, optional_idx, mandatory_idx});
        if (static_cast<int>(cells.size()) <= required) {
            continue;
        }
        const double length = std::stod(cells[static_cast<std::size_t>(length_idx)]);
        const double height = std::stod(cells[static_cast<std::size_t>(height_idx)]);
        const int optional = std::stoi(cells[static_cast<std::size_t>(optional_idx)]);
        const int mandatory = std::stoi(cells[static_cast<std::size_t>(mandatory_idx)]);
        const std::string program = program_idx >= 0 && static_cast<int>(cells.size()) > program_idx
            ? cells[static_cast<std::size_t>(program_idx)]
            : "";
        const std::string model = model_idx >= 0 && static_cast<int>(cells.size()) > model_idx
            ? cells[static_cast<std::size_t>(model_idx)]
            : "";

        data.car_types.push_back(type_id++);
        data.programs.push_back(program);
        data.models.push_back(model);
        data.lengths.push_back(length);
        data.heights.push_back(height);
        data.mandatory.push_back(mandatory);
        data.upper_bounds.push_back(mandatory + optional);
    }

    const std::string carriage_path = join_path(instance_dir, "carriage.csv");
    std::ifstream carriage(carriage_path);
    if (!carriage) {
        throw std::runtime_error("cannot open " + carriage_path);
    }
    std::getline(carriage, line);
    if (!std::getline(carriage, line)) {
        throw std::runtime_error("carriage.csv missing carriage count");
    }
    auto cells = split_csv_line(line);
    if (cells.empty()) {
        throw std::runtime_error("carriage.csv missing carriage count");
    }
    data.carriage_num = std::stoi(cells.front());
    return data;
}

}  // namespace bpc_label
