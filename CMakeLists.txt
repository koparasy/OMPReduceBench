project(OpenMPReduction LANGUAGES CXX) #CUDA)
cmake_minimum_required(VERSION 3.20.0)
#find_package(CUDA REQUIRED)

#find_package(OpenMPTarget REQUIRED NVPTX)

#add_subdirectory(openmp)
add_subdirectory(openmp_new)
#add_subdirectory(cub)
#add_subdirectory(thrust)
