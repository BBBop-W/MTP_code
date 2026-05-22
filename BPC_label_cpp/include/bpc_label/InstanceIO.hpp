#pragma once

#include <string>
#include <vector>

namespace bpc_label {

struct InstanceData {
    std::vector<int> car_types;
    std::vector<std::string> programs;
    std::vector<std::string> models;
    std::vector<double> lengths;
    std::vector<double> heights;
    std::vector<int> mandatory;
    std::vector<int> upper_bounds;
    int carriage_num = 0;
};

InstanceData load_instance(const std::string& instance_dir);

}  // namespace bpc_label
