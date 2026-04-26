#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include "Carriage.h"
#include "Problem.h"
#include "Conf.h"

class Solution {
public:
    int carriage_num;
    std::vector<Carriage> carriage;
    double obj;
    double actual_length;

    Solution() {
        carriage_num = 0;
        obj = 0;
        actual_length = 0;
    }

    Solution(int num) {
        carriage_num = num;
        carriage.resize(carriage_num);
        for (int i = 0; i < carriage_num; ++i) {
            carriage[i].id = i;
        }
        obj = 0;
        actual_length = 0;
    }

    double CalculateSolutionObj(Problem* p) {
        double objective = 0;
        double length_sum = 0;
        for (int i = 0; i < carriage_num; ++i) {
            carriage[i].CalculateCarriageObj(p);
            objective += carriage[i].obj;
            length_sum += carriage[i].actual_length;
        }
        obj = objective;
        actual_length = length_sum;
        return obj;
    }

    void output() {
        for (int i = 0; i < carriage_num; ++i) {
            std::cout << "carriage " << i << std::endl;
            std::cout << "up: [";
            for (size_t j = 0; j < carriage[i].route[0].size(); ++j) std::cout << carriage[i].route[0][j] << (j + 1 == carriage[i].route[0].size() ? "" : ", ");
            std::cout << "]\n";
            std::cout << "down: [";
            for (size_t j = 0; j < carriage[i].route[1].size(); ++j) std::cout << carriage[i].route[1][j] << (j + 1 == carriage[i].route[1].size() ? "" : ", ");
            std::cout << "]\n";
            std::cout << "obj: " << carriage[i].obj << ", actual loaded length: " << carriage[i].actual_length << "\n\n";
        }
    }

    void copy_construct(const Solution& origin) {
        obj = origin.obj;
        actual_length = origin.actual_length;
        carriage_num = origin.carriage_num;
        carriage.resize(carriage_num);
        for (int i = 0; i < carriage_num; ++i) {
            carriage[i].id = i;
            carriage[i].copy_construct(origin.carriage[i]);
        }
    }

    void Summarize(Problem* p, const std::string& dir_path) {
        std::ofstream fout(dir_path + "/carriage_info.json");
        fout << "{\n    \"carriage\": [\n";
        for (size_t i = 0; i < carriage.size(); ++i) {
            fout << "        {\n";
            fout << "            \"position\": \"" << (carriage[i].mode_left == 0 ? "h" : "m") << "-" << (carriage[i].mode_right == 0 ? "h" : "m") << "\",\n";
            
            fout << "            \"top\": [\n";
            for (size_t j = 0; j < carriage[i].route[0].size(); ++j) {
                Vehicle* v = p->GetVehicle(carriage[i].route[0][j]);
                fout << "                \"" << v->model << "\"" << (j + 1 == carriage[i].route[0].size() ? "" : ",") << "\n";
            }
            fout << "            ],\n";

            fout << "            \"bottom\": [\n";
            for (size_t j = 0; j < carriage[i].route[1].size(); ++j) {
                Vehicle* v = p->GetVehicle(carriage[i].route[1][j]);
                fout << "                \"" << v->model << "\"" << (j + 1 == carriage[i].route[1].size() ? "" : ",") << "\n";
            }
            fout << "            ]\n";
            fout << "        }" << (i + 1 == carriage.size() ? "" : ",") << "\n";
        }
        fout << "    ]\n}\n";
        fout.close();
    }

    void GenerateColumnCSV(Problem* p, const std::string& dir_path) {
        std::ofstream fout(dir_path + "/column.csv");
        fout << "brand,model";
        for (int i = 0; i < carriage_num; ++i) {
            fout << "," << (i + 1);
        }
        fout << "\n";
        
        for (const auto& v : p->vehicle) {
            fout << v.brand << "," << v.model;
            for (int i = 0; i < carriage_num; ++i) {
                int count = 0;
                for (int f = 0; f < 2; ++f) {
                    for (int id : carriage[i].route[f]) {
                        if (id == v.id) count++;
                    }
                }
                fout << "," << count;
            }
            fout << "\n";
        }
        fout.close();
    }
};