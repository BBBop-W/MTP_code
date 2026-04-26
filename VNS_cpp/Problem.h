#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Vehicle.h"

class Problem {
public:
    int vehicle_types;
    std::vector<Vehicle> vehicle;
    int mandatory_sum;
    int optional_sum;
    std::string instance_name;
    int carriage_num;

    Problem() : vehicle_types(0), mandatory_sum(0), optional_sum(0), instance_name(""), carriage_num(0) {}

    Vehicle* GetVehicle(int id) {
        for (int i = 0; i < vehicle_types; ++i) {
            if (vehicle[i].id == id) {
                return &vehicle[i];
            }
        }
        return nullptr;
    }

    void LoadVRPTW(const std::string& file_name) {
        instance_name = file_name;
        std::string inputfile = "../data/Instance/" + file_name + "/cars.csv";
        std::ifstream fin(inputfile);
        if (!fin.is_open()) {
            inputfile = "../data/Instance/cars.csv";
            fin.open(inputfile);
            if (!fin.is_open()) {
                std::cerr << "Cannot open " << inputfile << std::endl;
                return;
            }
        }
        std::string line;
        std::getline(fin, line); // Skip header
        
        int i = 0;
        while (std::getline(fin, line)) {
            if (line.empty()) break;
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> tokens;
            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }
            if (tokens.size() < 6) continue;
            
            Vehicle V;
            V.id = i;
            V.brand = tokens[0];
            V.model = tokens[1];
            V.length = std::stod(tokens[2]);
            V.height = std::stod(tokens[3]);
            V.num_optional = std::stoi(tokens[4]);
            V.num_mandatory = std::stoi(tokens[5]);
            V.limit_length[0] = 0; // Default or unused
            V.limit_length[1] = 0; // Default or unused
            V.var_optional = V.num_optional;
            V.var_mandatory = V.num_mandatory;
            vehicle.push_back(V);
            i++;
            mandatory_sum += V.num_mandatory;
            optional_sum += V.num_optional;
        }
        vehicle_types = i;
        fin.close();

        std::string carriage_file = "../data/Instance/" + file_name + "/carriage.csv";
        std::ifstream c_fin(carriage_file);
        if (c_fin.is_open()) {
            std::string c_line;
            std::getline(c_fin, c_line); // Skip header
            if (std::getline(c_fin, c_line)) {
                std::stringstream c_ss(c_line);
                std::string c_item;
                if (std::getline(c_ss, c_item, ',')) {
                    carriage_num = std::stoi(c_item);
                }
            }
            c_fin.close();
        } else {
            carriage_num = 5; // Fallback
        }
    }

    void Summarize(const std::string& dir_path) {
        std::ofstream fout(dir_path + "/car_sol.csv");
        fout << "brand,model,num_chosen\n";
        for (const auto& v : vehicle) {
            fout << v.brand << "," << v.model << "," << (v.num_mandatory - v.var_mandatory + v.num_optional - v.var_optional) << "\n";
        }
        fout.close();
    }

    struct VehicleState {
        int var_mandatory;
        int var_optional;
    };

    std::vector<VehicleState> BackupState() {
        std::vector<VehicleState> state(vehicle_types);
        for (int i = 0; i < vehicle_types; ++i) {
            state[i] = {vehicle[i].var_mandatory, vehicle[i].var_optional};
        }
        return state;
    }

    void RestoreState(const std::vector<VehicleState>& state) {
        for (int i = 0; i < vehicle_types; ++i) {
            vehicle[i].var_mandatory = state[i].var_mandatory;
            vehicle[i].var_optional = state[i].var_optional;
        }
    }
};