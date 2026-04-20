#pragma once
#include <random>
#include <algorithm>
#include <iostream>
#include "Solution.h"
#include "Problem.h"
#include "Neighborhoods.h"
#include "BestInsert.h"

bool RuinRebuid(Solution& result, int strength, Problem* p) {
    Solution old;
    old.copy_construct(result);
    strength = std::min(strength, result.carriage_num / 2 + 1);
    Neighborhoods Ns;
    
    std::random_device rd;
    std::mt19937 generator(rd());
    
    for (int i = 0; i < strength; ++i) {
        int r = generator() % result.carriage_num;
        bool is_success = Ns.Reposition(result, r, p);
        if (!is_success) {
            std::cout << "shake false!!!" << std::endl;
            return false;
        }
    }

    BestInsert bi;
    if (bi.Construct(result, p)) {
        std::cout << "======Reconstruct Solution Success!!!  " << std::endl;
        return true;
    }
    result.copy_construct(old);
    return false;
}