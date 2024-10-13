#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "gelu_ocl.h"

int main() {
  size_t n = 100'000'000;

  std::vector<float> a = rng::float_vector(n, 0.0f, 1.0f);
  
  GeluOCL(a);
  auto start = std::chrono::high_resolution_clock::now();
  GeluOCL(a);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by GeluOCL: " << elapsed.count() << " seconds\n";

  return 0;
}
