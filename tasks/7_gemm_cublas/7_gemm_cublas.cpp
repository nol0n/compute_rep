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

  return 0;
}
