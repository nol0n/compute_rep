#include <vector>
#include <iostream>
#include <chrono>
#include "gelu_omp.h"

int main() {
  std::vector<float> sample(1'000'000'000, 34.0f);
  auto start = std::chrono::high_resolution_clock::now();
  GeluOMP(sample);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by GeluOMP: " << elapsed.count() << " seconds\n";
  return 0;
}
