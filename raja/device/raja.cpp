#include <iostream>
#include <chrono>
#include <limits>

#include <RAJA/RAJA.hpp>

#include "utilities.hpp"
#include "data_types.hpp"

using namespace std;

#define DTYPE @REDUCTION_TYPE@

void* gpu_allocate(long size){
  void *ptr;
  cudaMalloc(&ptr,size);
  return ptr;
}

void gpu_free(void *ptr){
  cudaFree(ptr);
}

int main(int argc, char** argv)
{
  using util = @OPER_UTIL@<DTYPE>;

  if (argc != 2){
    std::cout << "Please provide the size of the reduction in MB" << "\n";
    return 10;
  }

  long elements = atol(argv[1]) *1024 * 1024 / sizeof(DTYPE);
  DTYPE *data = (DTYPE*) gpu_allocate(sizeof(DTYPE)*elements);
  util::reduction_type out(util::init());

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, elements),
      [=] RAJA_DEVICE (int i) {
      data[i] = util::init(i, elements);
      });

  auto start = chrono::high_resolution_clock::now();
  RAJA::forall< RAJA::cuda_exec<256, false /*async*/> >(
      RAJA::RangeSegment(0, elements), [=] __device__ (long i) {
        util::OP(out, data[i]);
      });

  auto end = chrono::high_resolution_clock::now();
  cout << "ELAPSED TIME: "
    << chrono::duration<double>(end - start).count() <<  endl;

  std::cout<< "SIZE OF ELEMENT:" << sizeof(DTYPE) << "\n";
  std::cout<< "REDUCTION TYPE:" << util::info() << "\n";

  if ( !util::validate(out.get(), elements) ){
    std::cout << "Value is " << out.get() << "\n";
    std::cout << "FAIL\n";
    return -1;
  }

  std::cout << "PASS \n";
  gpu_free(data);

  return 0;
}

