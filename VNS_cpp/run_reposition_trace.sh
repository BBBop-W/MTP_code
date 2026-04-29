sed -i '' 's/bool Reposition(Solution& result, int c_id, int side, Problem* p) {/bool Reposition(Solution\& result, int c_id, int side, Problem* p) { std::cout << "  [Repo] Start c_id=" << c_id << std::endl;/g' Neighborhoods.h
sed -i '' 's/for (int f : {0, 1}) {/std::cout << "  [Repo] Entering f loop" << std::endl; for (int f : {0, 1}) {/g' Neighborhoods.h
sed -i '' 's/c.CalculateCarriageObj(p);/std::cout << "  [Repo] Leaving f loop" << std::endl; c.CalculateCarriageObj(p);/g' Neighborhoods.h
g++ -O3 -std=c++11 test_vns_trace.cpp -o test_vns_trace && ./test_vns_trace
