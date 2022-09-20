#include <iostream>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <chrono>
using namespace cub;
using namespace std;

#include "utilities.hpp"
#include "data_types.hpp"

#define DTYPE @REDUCTION_TYPE@ 

using util = @OPER_UTIL@<DTYPE>;

template<typename T>
void init(T* ptr, long size){
  for ( long i = 0 ; i < size; i++){
    ptr[i] = util::init(i, size);
  }
}

struct CUBOP{
  __device__ __forceinline__
    DTYPE operator()(const DTYPE &a, const DTYPE &b){
      return util::OP(a, b);
    }
};

CachingDeviceAllocator  g_allocator(true);

int main(int argc, char** argv)
{
  long elements = atol(argv[1]) *1024 * 1024 / sizeof(DTYPE);
  CUBOP oper;
  DTYPE out = util::init();
  DTYPE op_init = util::init();
  DTYPE *d_in = NULL;
  DTYPE *h_in = new DTYPE[elements];
  init(h_in, elements);

  cudaSetDevice(0);

  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(DTYPE) * elements));
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(DTYPE) * elements, cudaMemcpyHostToDevice));

  // Allocate device output array
  DTYPE *d_out = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(DTYPE) * 1));

  auto start = chrono::high_resolution_clock::now();
  // Request and allocate temporary storage
  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;

  CubDebugExit(DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, elements, oper, op_init));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run
  CubDebugExit(DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, elements, oper, op_init));
  CubDebugExit(cudaMemcpy(&out, d_out, sizeof(DTYPE) , cudaMemcpyDeviceToHost));
  auto end = chrono::high_resolution_clock::now();
  cout << "ELAPSED TIME: "
    << chrono::duration<double>(end - start).count() <<  endl;

  std::cout<< "SIZE OF ELEMENT:" << sizeof(DTYPE) << "\n"; 
  std::cout<< "REDUCTION TYPE:" << util::info() << "\n";

  if ( !util::validate(out, elements) ){
    std::cout << "FAIL\n";
    return -1;
  }

  std::cout << "PASS \n";


  // Cleanup
  //  if (h_in) delete[] h_in;
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  return 0;
}

