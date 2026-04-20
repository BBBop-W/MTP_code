#pragma once
#include "Solution.h"
#include "Problem.h"
#include "BestInsert.h"
#include "Neighborhoods.h"
#include "Perturb.h"
#include "Conf.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

class VNS {
public:
    int max_nonImp;
    double best_time;
    int max_shake_strength;
    std::string m_fileName;

    VNS() : max_nonImp(0), best_time(0.0), max_shake_strength(0), m_fileName("") {}

    std::pair<bool, Solution> Solve(Solution& result, Problem* p) {
        std::cout << "Begin to solve by VNS..." << std::endl;
        BestInsert bi;
        bool success = bi.Construct(result, p);
        if (!success) {
            std::cout << "ERROR!!! -- Cannot generate feasible initial solution via BestInsert Method!" << std::endl;
        }
        bool changed = RandomVNDBest(result, p);
        return {changed, result};
    }

    bool RandomVNDBest(Solution& result, Problem* p) {
        std::cout << "Start BestRandomVND | VNS_Obj: " << result.obj << " | Actual Length: " << result.actual_length << std::endl;
        bool changed = false;
        Neighborhoods ns;
        int method_number = 3;
        std::vector<int> method = {0, 1, 2};
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(method.begin(), method.end(), g);
        
        Solution new_result;
        for (int iter = 0; iter < 10; ++iter) {
            for (int m = 0; m < 3; ++m) {
                bool flag = false;
                new_result.copy_construct(result);
                if (method[m] == 0) {
                    flag = ns.InterRelocateBest(new_result, p);
                } else if (method[m] == 1) {
                    flag = ns.InterSwapBest(new_result, p);
                } else if (method[m] == 2) {
                    flag = ns.InterOptBest(new_result, p);
                }
                if (flag) {
                    m = -1; 
                    std::shuffle(method.begin(), method.end(), g);
                    result.copy_construct(new_result);
                    changed = true;
                }
            }
        }
        std::cout << "End BestRandomVND | VNS_Obj: " << result.obj << " | Actual Length: " << result.actual_length << std::endl;
        return changed;
    }

    std::pair<bool, Solution> Optimization(Solution& result, Problem* p, int num_nonImprove) {
        auto tic = std::chrono::steady_clock::now();
        std::cout << "\tStart Optimization: " << num_nonImprove << std::endl;
        std::cout << "non-improve iteration = " << num_nonImprove << std::endl;

        BestInsert bi;
        Solution best;
        Solution local_best;
        Solution now;
        best.copy_construct(result);
        bool not_stop = true;
        
        while (not_stop) {
            int nonImprove = -1;
            while (nonImprove <= num_nonImprove && not_stop) {
                local_best.copy_construct(result);
                nonImprove += 1;
                for (int i = 0; i < result.carriage_num; ++i) {
                    now.copy_construct(result);
                    RuinRebuid(now, nonImprove, p);
                    RandomVNDBest(now, p);
                    std::cout << "\t------new local optimal solution | VNS_Obj: " << now.obj << " | Actual Length: " << now.actual_length << std::endl;
                    if (now.obj + Config::eps > local_best.obj) {
                        local_best.copy_construct(now);
                    }
                }

                result.copy_construct(local_best);
                if (result.obj + Config::eps > best.obj) {
                    best.copy_construct(result);
                    nonImprove = -1;
                    std::cout << "Found solution! VNS_Obj: " << best.obj << " | Actual Length: " << best.actual_length << std::endl;
                }

                auto toc = std::chrono::steady_clock::now();
                double duration = std::chrono::duration<double>(toc - tic).count();
                if (duration > Config::timelimit) {
                    not_stop = false;
                    break;
                }
            }

            if (not_stop) {
                if (result.obj > best.obj) {
                    result.copy_construct(best);
                }
                std::cout << "Perturb by RuinRebuid: " << nonImprove << ", best VNS_Obj = " << best.obj << " | Actual Length: " << best.actual_length << std::endl;
                int strength = p->mandatory_sum * 0.1 + nonImprove;
                RuinRebuid(result, strength, p);
            }
        }

        result.copy_construct(best);
        std::cout << "\n\t==============================" << std::endl;
        std::cout << "\tEnd Optimization!" << std::endl;
        std::cout << "\tBest VNS Objective: " << result.obj << std::endl;
        std::cout << "\tOPTIMAL MAXIMUM LOADED LENGTH: " << result.actual_length << std::endl;
        std::cout << "\t==============================\n" << std::endl;
        
        std::string out_dir = "../result/" + p->instance_name + "/VNS";
        std::string cmd = "mkdir -p " + out_dir;
        system(cmd.c_str());
        
        p->Summarize(out_dir);
        result.Summarize(p, out_dir);
        result.GenerateColumnCSV(p, out_dir);
        
        return {true, result};
    }
};