#pragma once
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "Solution.h"
#include "Problem.h"
#include "Neighborhoods.h"
#include "BestInsert.h"
#include "Insert.h"

bool RuinRebuild(Solution& result, int strength, Problem* p) {
    Solution old;
    old.copy_construct(result);

    std::mt19937& generator = get_generator();

    Neighborhoods Ns;

    // 1. Randomly alter deck positions
    int num_repos = std::min(strength / 2 + 1, result.carriage_num / 2);
    if (num_repos < 1) num_repos = 1;

    std::vector<int> carriages(result.carriage_num);
    std::iota(carriages.begin(), carriages.end(), 0);
    std::shuffle(carriages.begin(), carriages.end(), generator);

    for (int i = 0; i < num_repos; ++i) {
        int r = carriages[i];
        int side = generator() % 2; // 0 for left, 1 for right
        Ns.Reposition(result, r, side, p);
    }

    // 2. Randomly remove a subset of vehicles
    int num_remove = strength * 2; 
    int removed = 0;
    std::shuffle(carriages.begin(), carriages.end(), generator);
    for (int c_id : carriages) {
        for (int f : {0, 1}) {
            int len = result.carriage[c_id].route[f].size();
            while (len > 0) {
                int pos = generator() % len;
                int v_id = result.carriage[c_id].route[f][pos];
                bool success = EraseVehicle(result.carriage[c_id], pos, p, f);
                if (success) {
                    p->GetVehicle(v_id)->UpdateParameter_Removing();
                    removed++;
                    len--;
                } else {
                    break;
                }
                if (removed >= num_remove) break;
            }
            if (removed >= num_remove) break;
        }
        if (removed >= num_remove) break;
    }

    // 3. Repair the partial solution using BestInsert.
    BestInsert bi;
    if (bi.RepairRemaining(result, p)) {
        return true;
    }

    // If it fails to insert all mandatory vehicles, revert
    result.copy_construct(old);
    old.SyncProblemState(p);
    return false;
}
