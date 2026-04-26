#include "Problem.h"
#include "BestInsert.h"
#include "VNS.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    std::string instance_name = "m6c6"; // default
    if (argc > 1) {
        instance_name = argv[1];
    }

    Problem p;
    p.LoadVRPTW(instance_name);

    auto bi_start = std::chrono::steady_clock::now();
    BestInsert b;
    Solution* result = b.Solve(&p);
    auto bi_end = std::chrono::steady_clock::now();
    double bi_duration = std::chrono::duration<double>(bi_end - bi_start).count();
    
    if (result) {
        std::cout << "\n\t==============================" << std::endl;
        std::cout << "\tInitial Solution (BI) Generated!" << std::endl;
        std::cout << "\tBI Objective: " << result->obj << std::endl;
        std::cout << "\tBI LOADED LENGTH: " << result->actual_length << std::endl;
        std::cout << "\tBI Execution Time: " << bi_duration << " s" << std::endl;
        std::cout << "\t==============================\n" << std::endl;

        std::string out_dir = "../result/" + p.instance_name + "/BI";
        std::string cmd = "mkdir -p " + out_dir;
        system(cmd.c_str());
        
        p.Summarize(out_dir);
        result->Summarize(&p, out_dir);
        result->GenerateColumnCSV(&p, out_dir);

        VNS V;
        auto res = V.Optimization(*result, &p, 20);
        
        std::string vns_dir = "../result/" + p.instance_name + "/VNS";
        std::string cmd_vns = "mkdir -p " + vns_dir;
        system(cmd_vns.c_str());
        res.second.Summarize(&p, vns_dir);
        res.second.GenerateColumnCSV(&p, vns_dir);

        delete result;
    }

    return 0;
}