#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "fft_cufft.h"

int main() {
  size_t n = 531072;
  size_t batch = 5;

  std::vector<float> a = rng::float_vector(2 * n * batch, 0.0f, 1.0f);

  FftCUFFT(a, batch);
  auto start = std::chrono::high_resolution_clock::now();
  FftCUFFT(a, batch);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by FftCUFFT: " << elapsed.count() << " seconds\n";

  return 0;
}
