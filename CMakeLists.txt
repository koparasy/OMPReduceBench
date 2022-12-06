cmake_minimum_required(VERSION 3.20.0)
project(OpenMPReduction LANGUAGES CXX) #CUDA)

option(WITH_OPENMP "Enable OpenMP Reduction tests" OFF)
option(WITH_NEW_OPENMP "Enable OpenMP Reduction tests" OFF)
option(WITH_CUB "Enable OpenMP Reduction tests" OFF)
option(WITH_RAJA "Enable RAJA Reduction tests" OFF)
option(WITH_KOKKOS "Enable KOKKOS Reduction tests" OFF)
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
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  message(STATUS "Adding CUB compilation")
  add_subdirectory(cub)
endif()

if (WITH_RAJA)
  set(ENABLE_CUDA On)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(raja REQUIRED)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict --extended-lambda --expt-relaxed-constexpr -DTHRUST_IGNORE_CUB_VERSION_CHECK")
  find_package(Threads REQUIRED)
  add_subdirectory(raja)
endif()

if (WITH_KOKKOS)
  find_package(Kokkos REQUIRED)
  add_subdirectory(kokkos)
endif()

add_subdirectory(scripts)
