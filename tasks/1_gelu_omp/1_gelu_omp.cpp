#include <iomanip>
#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "gelu_omp.h"

int main(int argc, char* argv[]) {
  std::vector<float> sample = rng::float_vector(134'217'728, 1.0f, 50.0f);
  auto start = std::chrono::high_resolution_clock::now();
  GeluOMP(sample);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << std::setprecision(10);
  std::cout << "Time taken by GeluOMP: " << elapsed.count() << " seconds\n";
  return 0;
}
