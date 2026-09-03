cmake_minimum_required(VERSION 3.18)
project(qmt_cpp LANGUAGES CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Point CMake at the active conda environment ─────────────────────────────
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# ── pybind11 ─────────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)

# ── Eigen ────────────────────────────────────────────────────────────────────
find_package(Eigen3 CONFIG REQUIRED)

# ── NLopt ─────────────────────────────────────────────────────────────────────
find_package(NLopt CONFIG REQUIRED)

# ── Extension target ─────────────────────────────────────────────────────────
pybind11_add_module(opt_JSPqMT
    src/opt_JSPqMT.cpp
)

install(TARGETS opt_JSPqMT
        LIBRARY DESTINATION qmt)

target_link_libraries(opt_JSPqMT PRIVATE
    Eigen3::Eigen
    NLopt::nlopt
)

# Release-mode optimisations
target_compile_options(opt_JSPqMT PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)
