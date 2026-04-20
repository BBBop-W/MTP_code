#pragma once
#include <string>

class Vehicle {
public:
    int id;
    std::string brand;
    std::string model;
    double length;
    double height;

    int num_optional;
    int num_mandatory;

    int var_mandatory;
    int var_optional;

    int limit_length[2];

    Vehicle() : id(0), brand(""), model(""), length(0.0), height(0.0),
                num_optional(0), num_mandatory(0), var_mandatory(0), var_optional(0) {
        limit_length[0] = 0;
        limit_length[1] = 0;
    }

    void UpdateParameter_Removing() {
        if (var_mandatory > 0) {
            var_mandatory += 1;
        } else if (var_optional == num_optional - num_mandatory) {
            var_mandatory += 1;
        } else {
            var_optional += 1;
        }
    }
};