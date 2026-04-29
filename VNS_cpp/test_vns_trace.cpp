#include "VNS.h"
#include "Problem.h"
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
void alarm_handler(int sig) {
    std::cerr << "Timeout alarm triggered. Exiting.\n";
    exit(1);
}
int main() {
    signal(SIGALRM, alarm_handler);
    alarm(20); // 20s timeout
    Problem p;
    p.LoadVRPTW("m11c11");
    VNS v;
    Solution sol(p.carriage_num);
    v.Optimization(sol, &p, 20);
    return 0;
}
