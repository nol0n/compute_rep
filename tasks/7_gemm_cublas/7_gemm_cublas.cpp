#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "gemm_cublas.h"

int main() {
  size_t n = 1536;
  std::vector<float> a = rng::float_vector(n * n, 10.0f, 30.0f);
  std::vector<float> b = rng::float_vector(n * n, 10.0f, 30.0f);
  
  GemmCUBLAS(a, b, n);
  auto start = std::chrono::high_resolution_clock::now();
  GemmCUBLAS(a, b, n);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by BlockGemmOMP: " << elapsed.count() << " seconds\n";

  // std::vector<float> ref_res = R_GemmCUBLAS(a, b, n);
  // std::vector<float> my_res = GemmCUBLAS(a, b, n);

  // for (int i = 0; i < n * n; i++) {
  //   if (fabs(ref_res[i] - my_res[i]) > 0.1f) {
  //     std::cout << "error\n"; std::exit(0);
  //   } 
  // }

  // std::cout << "ok\n";

  return 0;
}
