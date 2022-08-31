cmake_minimum_required(VERSION 3.20.0)
project(reduction VERSION 1.0 LANGUAGES CXX)

find_package(OpenMPTarget REQUIRED NVPTX)


message(STATUS "ARCH:${OpenMPTarget_NVPTX_ARCH}")
message(STATUS "ARCH:${OpenMPTarget_NVPTX_FLAGS}")
message(STATUS "ARCH:${OpenMPTarget_NVPTX_FLAGS}")

add_subdirectory(openmp)
