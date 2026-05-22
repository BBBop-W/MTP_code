cmake_minimum_required(VERSION 3.20)
project(bpc_label_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED GUROBI_HOME)
  if(DEFINED ENV{GUROBI_HOME})
    set(GUROBI_HOME "$ENV{GUROBI_HOME}")
  elseif(WIN32)
    file(GLOB _GUROBI_CANDIDATES "C:/gurobi*/win64")
    if(_GUROBI_CANDIDATES)
      list(SORT _GUROBI_CANDIDATES)
      list(REVERSE _GUROBI_CANDIDATES)
      list(GET _GUROBI_CANDIDATES 0 GUROBI_HOME)
    endif()
  elseif(APPLE)
    file(GLOB _GUROBI_CANDIDATES "/Library/gurobi*/macos_universal2")
    if(_GUROBI_CANDIDATES)
      list(SORT _GUROBI_CANDIDATES)
      list(REVERSE _GUROBI_CANDIDATES)
      list(GET _GUROBI_CANDIDATES 0 GUROBI_HOME)
    endif()
  endif()
endif()

if(NOT GUROBI_HOME)
  message(FATAL_ERROR "GUROBI_HOME is not set. Pass -DGUROBI_HOME=... or set the GUROBI_HOME environment variable.")
endif()

set(GUROBI_INCLUDE_DIR "${GUROBI_HOME}/include")
set(GUROBI_LIB_DIR "${GUROBI_HOME}/lib")

find_library(
  GUROBI_C_LIBRARY
  NAMES gurobi130 gurobi120 gurobi110 gurobi100
  PATHS "${GUROBI_LIB_DIR}"
  NO_DEFAULT_PATH
)
find_library(
  GUROBI_CXX_LIBRARY
  NAMES gurobi_c++md2022 gurobi_c++md2019 gurobi_c++md2017 gurobi_c++ gurobi_c++mdd2022 gurobi_c++mdd2019 gurobi_c++mdd2017
  PATHS "${GUROBI_LIB_DIR}"
  NO_DEFAULT_PATH
)

if(NOT EXISTS "${GUROBI_INCLUDE_DIR}/gurobi_c++.h")
  message(FATAL_ERROR "Could not find gurobi_c++.h under ${GUROBI_INCLUDE_DIR}")
endif()
if(NOT GUROBI_C_LIBRARY)
  message(FATAL_ERROR "Could not find the Gurobi C library under ${GUROBI_LIB_DIR}")
endif()
if(NOT GUROBI_CXX_LIBRARY)
  message(FATAL_ERROR "Could not find the Gurobi C++ library under ${GUROBI_LIB_DIR}")
endif()

set(PRICING_SRC
  src/DynamicSegmentation.cpp
  src/LabelPricing.cpp
  src/WagonPricing.cpp
)

add_executable(bpc_label_smoke
  ${PRICING_SRC}
  src/main.cpp
)
target_include_directories(bpc_label_smoke PRIVATE include)

add_executable(bpc_label_solver
  ${PRICING_SRC}
  src/InstanceIO.cpp
  src/Warmstart.cpp
  src/SolverPricing.cpp
  src/BpcSolver.cpp
  src/solver_main.cpp
)
target_include_directories(bpc_label_solver PRIVATE include "${GUROBI_INCLUDE_DIR}")
target_link_libraries(bpc_label_solver PRIVATE "${GUROBI_CXX_LIBRARY}" "${GUROBI_C_LIBRARY}")

if(MSVC)
  target_compile_options(bpc_label_smoke PRIVATE /O2 /W4)
  target_compile_options(bpc_label_solver PRIVATE /O2 /W4)
else()
  target_compile_options(bpc_label_smoke PRIVATE -O3 -Wall -Wextra -pedantic)
  target_compile_options(bpc_label_solver PRIVATE -O3 -Wall -Wextra -pedantic)
endif()
