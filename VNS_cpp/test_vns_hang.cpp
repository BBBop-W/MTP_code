#include "VNS.h"
#include "Problem.h"
int main() {
    Problem p;
    p.LoadVRPTW("m11c11");
    VNS v;
    Solution sol(p.carriage_num);
    v.Optimization(sol, &p, 20);
    return 0;
}
