cmake_minimum_required(VERSION 3.20.0)
project(OpenMPReduction LANGUAGES CXX) #CUDA)

option(WITH_OPENMP "Enable OpenMP Reduction tests" ON)
option(WITH_NEW_OPENMP "Enable OpenMP Reduction tests" OFF)
option(WITH_CUB "Enable OpenMP Reduction tests" OFF)
#find_package(CUDA REQUIRED)

if (WITH_OPENMP) 
  message(STATUS "Adding default OpenMP compilation")
  add_subdirectory(openmp)
endif()

if (WITH_NEW_OPENMP)
  #TODO at some point openmp  new would be integraded into compiler, so this option will be disabled
  message(STATUS "Adding default New-OpenMP compilation")
  add_subdirectory(openmpNew)
endif()


if (WITH_CUB)
  enable_language(CUDA)
  message(STATUS "Adding CUB compilation")
  add_subdirectory(cub)
endif()

add_subdirectory(scripts)
