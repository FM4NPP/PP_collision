# acts_perigee_propagator: a pybind11 module linked against an installed ACTS (find_package) and built with the
# same pybind11 ACTS's Python bindings were built with (PYBIND11_SOURCE_DIR), so it accepts the field and geometry
# objects those bindings create. Built inside the ACTS image by build_images.sh.
cmake_minimum_required(VERSION 3.20)
project(acts_perigee_propagator CXX)

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

pybind11_add_module(acts_perigee_propagator perigee_propagator.cpp)
target_link_libraries(acts_perigee_propagator PRIVATE Acts::Core)
if(ACTS_LIBRARY_DIR)
  set_target_properties(acts_perigee_propagator PROPERTIES INSTALL_RPATH "${ACTS_LIBRARY_DIR}")
endif()
install(TARGETS acts_perigee_propagator LIBRARY DESTINATION .)
