#include <iostream>
#include <chrono>
#include <limits>
#include <algorithm>

#include "utilities.hpp"
#include "omp_new_reduce.hpp"
#include "data_types.hpp"

using namespace std;

#define DTYPE @REDUCTION_TYPE@
#define __MAX_TEAMS__ 3200
#define __SIZE__  (__MAX_TEAMS__ * sizeof(DTYPE))

#pragma omp begin declare target device_type(nohost)
static uint32_t LeagueCounter1 = 0;
static uint32_t LeagueCounter2 = 0;
static char LeagueBuffer[__SIZE__];
extern "C" void __llvm_omp_default_reduction_init(
    __llvm_omp_default_reduction *__restrict__ __private_copy,
    __llvm_omp_default_reduction *const __restrict__ __original_copy);
extern "C" void __llvm_omp_default_reduction_combine(
    __llvm_omp_default_reduction *__restrict__ __shared_out_copy,
    __llvm_omp_default_reduction *__restrict__ __private_copy);
#pragma omp end declare target


#pragma omp begin declare target device_type(nohost)
[[clang::loader_uninitialized]] static __llvm_omp_default_reduction_configuration_ty reduction_type;

template<class T>
const __llvm_omp_reduction_element_type getLLVMConstType(T *ptr){
  return __llvm_omp_reduction_element_type::_CUSTOM_TYPE;
}

template<>
const __llvm_omp_reduction_element_type getLLVMConstType(int *ptr){
  return __llvm_omp_reduction_element_type::_INT32;
}

template<>
const __llvm_omp_reduction_element_type getLLVMConstType(long *ptr){
  return __llvm_omp_reduction_element_type::_INT64;
}

template<>
const __llvm_omp_reduction_element_type getLLVMConstType(float *ptr){
  return __llvm_omp_reduction_element_type::_FLOAT;
}

template<>
const __llvm_omp_reduction_element_type getLLVMConstType(double *ptr){
  return __llvm_omp_reduction_element_type::_DOUBLE;
}

#pragma omp end declare target

using util = @OPER_UTIL@<DTYPE>;
template<class T>
void initGlobalData(long NumElements, void *buffer, uint32_t *counter1, uint32_t *counter2){
#pragma omp target
  {
    DTYPE *ptr = (DTYPE*) LeagueBuffer;
    for (int i = 0 ; i < __SIZE__ / sizeof(DTYPE); i++){
      ptr[i] = util::init();
    }

    reduction_type = {
      _LEAGUE,
      (__llvm_omp_reduction_allocation_configuration) (_PRE_INITIALIZED | _PREALLOCATED_IN_PLACE),
      _@OPER_UTIL@,
      getLLVMConstType((DTYPE*) nullptr),
      __llvm_omp_default_reduction_choices(0 | @REDUCE_POLICY@ ), // I will need to define RC as well
      sizeof(T),
      1,
      1, // Batch size lets figure out a way later to configure this
      0,
      &LeagueBuffer,
      &LeagueCounter1,
      &LeagueCounter2,
      nullptr,
      nullptr
    };
  }
}

void reduce_custom(const int numThreads, int numTeams, long elements, DTYPE *data, DTYPE *out){
#pragma omp target teams num_teams(numTeams) thread_limit(numThreads) firstprivate(elements)
    {
#pragma omp parallel num_threads(numThreads) firstprivate(elements) 
      {
        DTYPE temp = util::init();
        int teamId= omp_get_team_num();
        int numTeams = omp_get_num_teams();
        int threadId = omp_get_thread_num();
        int teamSize = omp_get_num_threads();
        int gridSize = numTeams * teamSize;
        __llvm_omp_default_reduction sout { nullptr, out };
        __llvm_omp_default_reduction lodr { &reduction_type , (void*)&temp };
        __llvm_omp_default_reduction_init(&lodr, nullptr);
       for (long i = teamSize*teamId + threadId; i < elements; i+=gridSize){
          temp = util::OP(temp, data[i]);
        }
        __llvm_omp_default_reduction_combine(&sout, &lodr) ;
      }
    }
}

int main(int argc, char** argv)
{
  if (argc != 2){
    std::cout << "Please provide the size of the reduction in MB" << "\n";
    return 10;
  }

  long elements = atol(argv[1])  * 1024L * 1024L / sizeof(DTYPE);
  initGlobalData<DTYPE>(elements, nullptr, nullptr, nullptr); //&LeagueBuffer, &LeagueCounter1, &LeagueCounter2);
  DTYPE *data = nullptr;
  DTYPE out = util::init();

  const int numThreads = 256; // This is the default value used when using cub. Lets see how this works out
  int numTeams =  (elements + numThreads -1) / numThreads ;
  numTeams = (numTeams < __MAX_TEAMS__) ? numTeams : __MAX_TEAMS__; // Again in my runs I checked that this is the max value for cub

  //TODO: allocate global device memory as johaness said 

#pragma omp target data map (alloc:data[0:elements])
  {
    // Initialize vectors
#pragma omp target teams distribute parallel for
    for ( long i = 0 ; i < elements; i++){
      data[i] = util::init(i, elements);// + util::init();
    }

    // Start reduction 
    auto start = chrono::high_resolution_clock::now();
    reduce_custom(numThreads, numTeams, elements, data, &out); 
    auto end = chrono::high_resolution_clock::now();
    cout << "ELAPSED TIME: "
      << chrono::duration<double>(end - start).count() <<  endl;
  }

  std::cout<< "SIZE OF ELEMENT:" << sizeof(DTYPE) << "\n"; 
  std::cout<< "REDUCTION TYPE:" << util::info() << "\n";

  if ( !util::validate(out, elements) ){
    std::cout << "Value is " << out << "\n";
    std::cout << "Value should had been " << util::correct(elements) << "\n";
    std::cout << "FAIL\n";
    return -1;
  }

  std::cout << "PASS \n";

  return 0;
}

