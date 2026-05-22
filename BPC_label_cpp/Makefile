CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic
GUROBI_HOME ?= /Library/gurobi1301/macos_universal2
CPPFLAGS ?= -Iinclude
GUROBI_CPPFLAGS := -I$(GUROBI_HOME)/include
GUROBI_LDFLAGS := -L$(GUROBI_HOME)/lib -Wl,-rpath,$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi130

PRICING_SRC := \
	src/DynamicSegmentation.cpp \
	src/LabelPricing.cpp \
	src/WagonPricing.cpp

SMOKE_SRC := \
	$(PRICING_SRC) \
	src/main.cpp

SOLVER_SRC := \
	$(PRICING_SRC) \
	src/InstanceIO.cpp \
	src/Warmstart.cpp \
	src/SolverPricing.cpp \
	src/BpcSolver.cpp \
	src/solver_main.cpp

SMOKE_OBJ := $(SMOKE_SRC:.cpp=.o)
SOLVER_OBJ := $(SOLVER_SRC:.cpp=.solver.o)
DEP := $(SMOKE_OBJ:.o=.d) $(SOLVER_OBJ:.solver.o=.solver.d)

.PHONY: all clean

all: bpc_label_smoke bpc_label_solver

bpc_label_smoke: $(SMOKE_OBJ)
	$(CXX) $(CXXFLAGS) $(SMOKE_OBJ) -o $@

bpc_label_solver: $(SOLVER_OBJ)
	$(CXX) $(CXXFLAGS) $(SOLVER_OBJ) $(GUROBI_LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

%.solver.o: %.cpp
	$(CXX) $(CPPFLAGS) $(GUROBI_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEP)

clean:
	rm -f $(SMOKE_OBJ) $(SOLVER_OBJ) $(DEP) bpc_label_smoke bpc_label_solver
