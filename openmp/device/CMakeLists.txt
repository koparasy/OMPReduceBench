
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
    set(DEST_SRC_FILE "${CMAKE_CURRENT_BINARY_DIR}/driver_${dt}_${op}.cpp")
    set(DEST_BIN_FILE "driver_${dt}_${op}.exe")
    configure_file(driver.cpp ${DEST_SRC_FILE})
    add_executable(${DEST_BIN_FILE} ${DEST_SRC_FILE})
    target_compile_options(${DEST_BIN_FILE} PRIVATE ${OpenMPTarget_NVPTX_FLAGS})
    target_link_libraries(${DEST_BIN_FILE} PRIVATE ${OpenMPTarget_NVPTX_FLAGS})
    target_include_directories(${DEST_BIN_FILE} PRIVATE "${CMAKE_SOURCE_DIR}/include")
  endforeach()
endforeach()

