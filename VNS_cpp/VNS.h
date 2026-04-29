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
        auto tic = std::chrono::steady_clock::now();
        bool changed = RandomVNDBest(result, p, tic, 0.0);
        return {changed, result};
    }

    bool RandomVNDBest(Solution& result, Problem* p, std::chrono::steady_clock::time_point global_tic, double last_improve_time) {
        bool overall_changed = false;
        bool changed = true;
        Neighborhoods ns;
        // Only use Random (First Improvement) operators to heavily speed up the search for large instances
        // 3=RelocateRandom, 4=SwapRandom, 5=OptRandom
        std::vector<int> method = {3, 4, 5};
        
        auto vnd_tic = std::chrono::steady_clock::now();
        
        while (changed) {
            auto toc = std::chrono::steady_clock::now();
            double global_duration = std::chrono::duration<double>(toc - global_tic).count();
            if (global_duration > Config::timelimit || (global_duration - last_improve_time) > Config::non_improve_timelimit) {
                break; // Hard abort if global time limit or non-improve limit reached
            }
            double vnd_duration = std::chrono::duration<double>(toc - vnd_tic).count();
            if (vnd_duration > 15.0) {
                break; // Hard abort to prevent single VND local search from taking too long
            }

            changed = false;
            std::shuffle(method.begin(), method.end(), get_generator());
            for (int m = 0; m < method.size(); ++m) {
                bool flag = false;
                if (method[m] == 3) {
                    flag = ns.InterRelocateRandom(result, p);
                } else if (method[m] == 4) {
                    flag = ns.InterSwapRandom(result, p);
                } else if (method[m] == 5) {
                    flag = ns.InterOptRandom(result, p);
                }
                
                if (flag) {
                    changed = true;
                    overall_changed = true;
                    break; // break inner loop, restart VND
                }
            }
        }
        return overall_changed;
    }

    std::pair<bool, Solution> Optimization(Solution& result, Problem* p, int num_nonImprove) {
        auto tic = std::chrono::steady_clock::now();
        std::cout << "\n\t==============================" << std::endl;
        std::cout << "\tStart VNS Optimization" << std::endl;
        
        // We multiply max iterations by carriage_num to match the old search effort
        int max_iters = num_nonImprove * std::max(5, result.carriage_num); 
        std::cout << "\tMax non-improve iterations: " << max_iters << std::endl;

        // Ensure result is well-constructed if it wasn't
        if (result.actual_length == 0) {
            BestInsert bi;
            bi.Construct(result, p);
        }

        Solution global_best;
        global_best.copy_construct(result);
        double last_improve_time = 0.0;
        RandomVNDBest(global_best, p, tic, last_improve_time);
        
        Solution incumbent;
        incumbent.copy_construct(global_best);
        
        int nonImprove = 0;
        int total_iters = 0;
        
        std::cout << "\tInitial VNS_Obj: " << global_best.obj << " | Actual Length: " << global_best.actual_length << std::endl;

        while (nonImprove < max_iters) {
            total_iters++;
            auto toc = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(toc - tic).count();
            if (duration > Config::timelimit) {
                std::cout << "\tOverall time limit reached (" << Config::timelimit << "s)." << std::endl;
                break;
            }
            if (duration - last_improve_time > Config::non_improve_timelimit) {
                std::cout << "\tMax time without improvement reached (" << Config::non_improve_timelimit << "s)." << std::endl;
                break;
            }
            
            Solution shaken;
            shaken.copy_construct(incumbent);
            
            // Dynamic strength based on nonImprove to diversify more when stuck
            int strength = 1 + (nonImprove / 10); 
            bool success = RuinRebuild(shaken, strength, p);
            if (success) {
                
                if (shaken.obj > incumbent.obj + Config::eps) {
                    incumbent.copy_construct(shaken);
                    if (incumbent.obj > global_best.obj + Config::eps) {
                        global_best.copy_construct(incumbent);
                        nonImprove = 0;
                        last_improve_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - tic).count();
                        std::cout << "\tIter " << total_iters << " | New Global Best VNS_Obj: " << global_best.obj << " | Actual Length: " << global_best.actual_length << std::endl;
                    } else {
                        nonImprove++;
                    }
                } else {
                    nonImprove++;
                }
            } else {
                nonImprove++;
            }
            
            if (total_iters % 100 == 0) {
            }
        }

        auto final_toc = std::chrono::steady_clock::now();
        double final_duration = std::chrono::duration<double>(final_toc - tic).count();

        result.copy_construct(global_best);
        std::cout << "\n\t==============================" << std::endl;
        std::cout << "\tEnd VNS Optimization!" << std::endl;
        std::cout << "\tBest VNS Objective: " << result.obj << std::endl;
        std::cout << "\tOPTIMAL MAXIMUM LOADED LENGTH: " << result.actual_length << std::endl;
        std::cout << "\tVNS Execution Time: " << final_duration << " s" << std::endl;
        std::cout << "\t==============================\n" << std::endl;
        
        return {true, result};
    }
};