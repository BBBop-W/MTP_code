#include "Problem.h"
#include "BestInsert.h"
#include "VNS.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    Problem p;
    p.LoadVRPTW("m10c10");

    BestInsert b;
    Solution* result = b.Solve(&p);
    
    if (result) {
        std::cout << "\n\t==============================" << std::endl;
        std::cout << "\tInitial Solution (BI) Generated!" << std::endl;
        std::cout << "\tBI Objective: " << result->obj << std::endl;
        std::cout << "\tBI LOADED LENGTH: " << result->actual_length << std::endl;
        std::cout << "\t==============================\n" << std::endl;

        std::string out_dir = "../result/" + p.instance_name + "/BI";
        std::string cmd = "mkdir -p " + out_dir;
        system(cmd.c_str());
        
        p.Summarize(out_dir);
        result->Summarize(&p, out_dir);
        result->GenerateColumnCSV(&p, out_dir);

        VNS V;
        auto res = V.Optimization(*result, &p, 20);
        delete result;
    }

    return 0;
}