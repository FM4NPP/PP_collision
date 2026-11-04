# acts_material_propagator: a pybind11 module linked against an installed ACTS (find_package) and built with
# the same pybind11 ACTS's Python bindings were built with (PYBIND11_SOURCE_DIR), so it accepts the geometry
# and field objects those bindings create. Built in the worker image, see acts-material-worker.def.
cmake_minimum_required(VERSION 3.20)
project(acts_material_propagator CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(PYBIND11_SOURCE_DIR "" CACHE PATH "pybind11 source tree ACTS's Python bindings were built with")
set(ACTS_LIBRARY_DIR "" CACHE PATH "the ACTS libraries the acts Python package loads (put first in RPATH)")

find_package(Acts CONFIG REQUIRED COMPONENTS Core)
set(PYBIND11_FINDPYTHON ON)
add_subdirectory(${PYBIND11_SOURCE_DIR} pybind11)

pybind11_add_module(acts_material_propagator material_propagator.cpp)
target_link_libraries(acts_material_propagator PRIVATE Acts::Core)
if(ACTS_LIBRARY_DIR)
  set_target_properties(acts_material_propagator PROPERTIES INSTALL_RPATH "${ACTS_LIBRARY_DIR}")
endif()
install(TARGETS acts_material_propagator LIBRARY DESTINATION .)
