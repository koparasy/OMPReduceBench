#pragma once
#include <omp.h>

enum __llvm_omp_reduction_level : unsigned char {
  _WARP = 1 << 0,
  _TEAM = 1 << 1,
  _LEAGUE = 1 << 2,
};

enum __llvm_omp_reduction_element_type : int8_t {
  _INT8,
  _INT16,
  _INT32,
  _INT64,
  _FLOAT,
  _DOUBLE,
  _CUSTOM_TYPE,
};

enum __llvm_omp_reduction_initial_value_kind : unsigned char {
  _VALUE_ONE,
  _VALUE_MIN,
  _VALUE_MAX,
};

enum __llvm_omp_reduction_operation : unsigned char {
  /// Uses 0 initializer
  _ADD,
  _SUB,
  _BIT_OR,
  _BIT_XOR,
  _LOGIC_OR,

  /// Uses ~0 initializer
  _BIT_AND,

  /// Uses 1 initializer
  _MUL,
  _LOGIC_AND,

  /// Usesmin/max value initializer
  _MAX,
  _MIN,

  /// Uses custom initializer function.
  _CUSTOM_OP,
};

enum __llvm_omp_default_reduction_choices : int32_t {
  /// By default we will reduce a batch of elements completely before we move on
  /// to the next batch. If the _REDUCE_WARP_FIRST bit is set we will instead
  /// first reduce all warps and then move on to reduce warp results further.
  _REDUCE_WARP_FIRST = 1 << 0,

  _REDUCE_ATOMICALLY_AFTER_WARP = 1 << 1,
  _REDUCE_ATOMICALLY_AFTER_TEAM = 1 << 2,

  _REDUCE_LEAGUE_VIA_ATOMICS_WITH_OFFSET = 1 << 3,
  _REDUCE_LEAGUE_VIA_LARGE_BUFFER = 1 << 4,
  _REDUCE_LEAGUE_VIA_SYNCHRONIZED_SMALL_BUFFER = 1 << 5,
  _REDUCE_LEAGUE_VIA_PROCESSOR_IDX = 1 << 6,
  _REDUCE_LEAGUE_VIA_PROCESSOR_IDX_BATCHED = 1 << 7,

  _PRIVATE_BUFFER_IS_SHARED = 1 << 7,
};

/// TODO
enum __llvm_omp_reduction_allocation_configuration : unsigned char {
  _PREALLOCATED_IN_PLACE = 1 << 0,
  _PRE_INITIALIZED = 1 << 1,
};

/// TODO
typedef __attribute__((alloc_size(1))) void *(
    __llvm_omp_reduction_allocator_fn_ty)(size_t);

/// TODO
typedef void(__llvm_omp_reduction_initializer_fn_ty)(void *);

struct __llvm_omp_default_reduction_configuration_ty {

  __llvm_omp_reduction_level __level;

  __llvm_omp_reduction_allocation_configuration __alloc_config;

  __llvm_omp_reduction_operation __op;

  __llvm_omp_reduction_element_type __element_type;

  __llvm_omp_default_reduction_choices __choices;

  int32_t __item_size;
  int32_t __num_items;

  int32_t __batch_size;

  int32_t __num_participants;

  void *__buffer;

  // Counters need to be initialized prior to the reduction to 0.
  uint32_t *__counter1_ptr;
  uint32_t *__counter2_ptr;

  __llvm_omp_reduction_allocator_fn_ty *__allocator_fn;
  __llvm_omp_reduction_initializer_fn_ty *__initializer_fn;
};

struct __llvm_omp_default_reduction {
  __llvm_omp_default_reduction_configuration_ty *__config;
  void *__private_default_data;
};
