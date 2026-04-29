sed -i '' 's/int num_remove = strength \* 2;/std::cout << "  Start erase loop" << std::endl; int num_remove = strength * 2;/g' Perturb.h
sed -i '' 's/BestInsert bi;/std::cout << "  Start BestInsert" << std::endl; BestInsert bi;/g' Perturb.h
sed -i '' 's/return false; \/\//std::cout << "  Failed BestInsert" << std::endl; return false; \/\//g' BestInsert.h
g++ -O3 -std=c++11 test_vns_trace.cpp -o test_vns_trace && ./test_vns_trace
