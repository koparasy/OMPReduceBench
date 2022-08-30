
set(SRC_FILES driver.cpp)


set(data_types "double;long;int;float")
set(omp_reduction "reduction(*:out);reduction(+:out);reduction(min:out);reduction(max:out)")
set(oper_util "mult_OP;add_OP;min_OP;max_OP")

foreach(dt IN LISTS data_types)
  foreach(ompR op IN ZIP_LISTS omp_reduction oper_util)
    message(STATUS "DT=${dt} Reduction Type=${ompR} operation=${op}")
    set(REDUCTION_TYPE ${dt})
    set(OMP_REDUCTION ${ompR})
    set(OPER_UTIL ${op})
  endforeach()
endforeach()

target_include_directories(driver PRIVATE "${CMAKE_SOURCE_DIR}/include")
