cmake_minimum_required(VERSION 3.18)
project(qmt_cpp LANGUAGES CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ─────────────────────────────────────────────────────────────────
# Prefer a version installed via pip (scikit-build-core / setup.py will have
# already placed it on the path); fall back to a system install.
find_package(pybind11 CONFIG REQUIRED)

# ── Eigen ────────────────────────────────────────────────────────────────────
# 1. Try an Eigen3 CMake config (system install or conda).
# 2. Fall back to the header-only copy bundled under extern/eigen if present.
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 3.4 QUIET MODULE)
endif()
if(NOT Eigen3_FOUND)
    set(_EIGEN_FALLBACK "${CMAKE_SOURCE_DIR}/extern/eigen")
    if(EXISTS "${_EIGEN_FALLBACK}/Eigen/Dense")
        message(STATUS "Eigen3 not found via CMake — using bundled copy at ${_EIGEN_FALLBACK}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE "${_EIGEN_FALLBACK}")
    else()
        message(FATAL_ERROR
            "Eigen3 >= 3.4 not found.\n"
            "Install it with 'conda install -c conda-forge eigen' (preferred).")
    endif()
endif()

# ── NLopt ─────────────────────────────────────────────────────────────────────
find_library(NLOPT_LIBRARY
    NAMES nlopt nlopt_cxx
    PATHS ${CONDA_PREFIX}/lib
    PATH_SUFFIXES lib
)
find_path(NLOPT_INCLUDE_DIR
    NAMES nlopt.h nlopt.hpp
    PATHS ${CONDA_PREFIX}/include
)

# Fallback: let CMake search default system paths too
if(NOT NLOPT_LIBRARY)
    find_library(NLOPT_LIBRARY NAMES nlopt nlopt_cxx)
endif()
if(NOT NLOPT_INCLUDE_DIR)
    find_path(NLOPT_INCLUDE_DIR NAMES nlopt.h nlopt.hpp)
endif()

if(NOT NLOPT_LIBRARY OR NOT NLOPT_INCLUDE_DIR)
    message(FATAL_ERROR
        "NLopt not found.\n"
        "Install it with: conda install -c conda-forge nlopt")
endif()

message(STATUS "NLopt library: ${NLOPT_LIBRARY}")
message(STATUS "NLopt include: ${NLOPT_INCLUDE_DIR}")

# ── Extension target ─────────────────────────────────────────────────────────
pybind11_add_module(opt_JSPqMT
    src/opt_JSPqMT.cpp
)

install(TARGETS opt_JSPqMT
        LIBRARY DESTINATION qmt)

target_link_libraries(opt_JSPqMT PRIVATE
    Eigen3::Eigen
    ${NLOPT_LIBRARY}
)

# Release-mode optimisations
target_compile_options(opt_JSPqMT PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)
