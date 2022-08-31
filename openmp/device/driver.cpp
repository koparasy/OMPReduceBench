#include <iostream>
#include <chrono>
#include <limits>

#include "utilities.hpp"
#include "data_types.hpp"

using namespace std;

#define DTYPE @REDUCTION_TYPE@ 

int main(int argc, char** argv)
{
  using util = @OPER_UTIL@<DTYPE>;
  
  if (argc != 2){
    std::cout << "Please provide the size of the reduction in MB" << "\n";
    return -1;
  }

  long elements = atol(argv[1]) *1024 * 1024 / sizeof(DTYPE);
  DTYPE *data = nullptr;
  DTYPE out = util::init();

#pragma omp target data map (alloc:data[0:elements])
  {
#pragma omp target teams distribute parallel for
    for ( long i = 0 ; i < elements; i++){
      data[i] = util::init(i);
    }

    auto start = chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for @OMP_REDUCTION@ 
    for ( long i = 0 ; i < elements; i++){
      out = util::OP(out, data[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Elapsed time in milliseconds: "
      << chrono::duration<double>(end - start).count() << " s" << endl;
  }

  std::cout<< "SIZE_OF_ELEMENT:" << sizeof(DTYPE) << "\n"; 

  if ( util::validate(out, elements) )
    std::cout << "PASS \n";
  else
    std::cout << "FAIL\n";


  free(data);

  return 0;
}

