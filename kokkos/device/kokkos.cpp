#include <iostream>
#include <chrono>
#include <limits>

#include <Kokkos_Core.hpp>

#include "utilities.hpp"
#include "data_types.hpp"

using namespace std;

#define DTYPE @REDUCTION_TYPE@

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  using util = @OPER_UTIL@<DTYPE>;

  using ExecSpace = Kokkos::CudaSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  if (argc != 2){
    std::cout << "Please provide the size of the reduction in MB" << "\n";
    return 10;
  }

  long elements = atol(argv[1]) *1024 * 1024 / sizeof(DTYPE);
  DTYPE out = util::init();

  {
    typedef Kokkos::View<DTYPE*, Kokkos::LayoutLeft, Kokkos::CudaSpace> ViewVectorType;
    ViewVectorType data("data", elements);
    Kokkos::parallel_for("initialize",
        range_policy(0, elements),
        KOKKOS_LAMBDA(long i){
        data(i) = util::init(i, elements);
        });

    auto start = chrono::high_resolution_clock::now();
    Kokkos::parallel_reduce("reduce",
        range_policy(0, elements),
        KOKKOS_LAMBDA( int i, DTYPE &out){
          out = util::OP(out, data[i]);
        }, util::reduction_type(out));
    auto end = chrono::high_resolution_clock::now();
    cout << "ELAPSED TIME: "
      << chrono::duration<double>(end - start).count() <<  endl;
  }

  std::cout<< "SIZE OF ELEMENT:" << sizeof(DTYPE) << "\n";
  std::cout<< "REDUCTION TYPE:" << util::info() << "\n";
  Kokkos::finalize();

  if ( !util::validate(out, elements) ){
    std::cout << "Value is " << out << "\n";
    std::cout << "FAIL\n";
    return -1;
  }

  std::cout << "PASS \n";

  return 0;
}

