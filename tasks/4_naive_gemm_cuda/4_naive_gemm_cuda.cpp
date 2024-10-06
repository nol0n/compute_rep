#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "naive_gemm_cuda.h"

int main() {
  size_t n = 1400;
  std::vector<float> a = rng::float_vector(n * n, 10.0f, 30.0f);
  std::vector<float> b = rng::float_vector(n * n, 10.0f, 30.0f);
  
  NaiveGemmCUDA(a, b, n);
  auto start = std::chrono::high_resolution_clock::now();
  NaiveGemmCUDA(a, b, n);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by NaiveGemmCUDA: " << elapsed.count() << " seconds\n";
  
  return 0;
}
