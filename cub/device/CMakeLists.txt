

set(data_types "double;long;int;float")#;dim4;dim32")
set(omp_reduction "reduction(*:out);reduction(+:out);reduction(min:out);reduction(max:out)")
set(oper_util "MUL;ADD;MIN;MAX")

foreach(dt IN LISTS data_types)
  foreach(ompR op IN ZIP_LISTS omp_reduction oper_util)
    set(REDUCTION_TYPE ${dt})
    set(OMP_REDUCTION ${ompR})
    set(OPER_UTIL ${op})
    set(DEST_SRC_FILE "${CMAKE_CURRENT_BINARY_DIR}/cub_${dt}_${op}.cu")
    set(DEST_BIN_FILE "cub_${dt}_${op}.exe")
    message(STATUS "DT=${dt} Reduction Type=${ompR} operation=${op} ${DEST_SRC_FILE}")
    configure_file(cub.cu ${DEST_SRC_FILE})
    add_executable(${DEST_BIN_FILE} ${DEST_SRC_FILE})
    set_target_properties(${DEST_BIN_FILE} PROPERTIES FORCE_LANGUAGE CUDA)
    set_property(TARGET ${DEST_BIN_FILE} PROPERTY CUDA_ARCHITECTURES 70)
    target_compile_options(${DEST_BIN_FILE} PRIVATE)
    target_link_libraries(${DEST_BIN_FILE} PRIVATE)
    target_include_directories(${DEST_BIN_FILE} PRIVATE "${CMAKE_SOURCE_DIR}/include")
  endforeach()
endforeach()

