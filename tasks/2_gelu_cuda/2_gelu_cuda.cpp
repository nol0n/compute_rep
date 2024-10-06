#include <iomanip>
#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "gelu_cuda.h"

int main() {
  std::vector<float> sample = rng::float_vector(134'217'728, 0.0f, 1.0f);
  GeluCUDA(sample);
  auto start = std::chrono::high_resolution_clock::now();
  GeluCUDA(sample);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << std::setprecision(10);
  std::cout << "Time taken by GeluCUDA: " << elapsed.count() << " seconds\n";
  return 0;
}