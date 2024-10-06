#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <rng.hpp>

#include "block_gemm_omp.h"

int main() {
  size_t n = 1024;
  std::vector<float> a = rng::float_vector(n * n, 10.0f, 30.0f);
  std::vector<float> b = rng::float_vector(n * n, 10.0f, 30.0f);
  
  auto start = std::chrono::high_resolution_clock::now();
  BlockGemmOMP(a, b, n);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by Me-e-e: " << elapsed.count() << " seconds\n";

  start = std::chrono::high_resolution_clock::now();
  ArtemBlockGemmOMP(a, b, n);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Time taken by Artem: " << elapsed.count() << " seconds\n";

  start = std::chrono::high_resolution_clock::now();
  DamirBlockGemmOMP(a, b, n);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Time taken by Damir: " << elapsed.count() << " seconds\n";

  start = std::chrono::high_resolution_clock::now();
  OptimizedBlockGemmOMP(a, b, n);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Time taken by AI: " << elapsed.count() << " seconds\n";

  // size_t n = 512;
  // std::vector<float> a = rng::float_vector(n * n, 10.0f, 30.0f);
  // std::vector<float> b = rng::float_vector(n * n, 10.0f, 30.0f);

  // std::vector<float> my_res = BlockGemmOMP(a, b, n);
  // std::vector<float> dam_res = DamirBlockGemmOMP(a, b, n);
  // std::vector<float> art_res = ArtemBlockGemmOMP(a, b, n);
  // std::vector<float> ai_res = OptimizedBlockGemmOMP(a, b, n);

  // for (int i = 0; i < n * n; i++) {
  //   if (fabs(ai_res[i] - my_res[i]) > 0.000001f) {
  //     std::cout << "error\n"; std::exit(0); 
  //   }
  //   if (fabs(ai_res[i] - my_res[i]) > 0.000001f) {
  //     std::cout << "error\n"; std::exit(0); 
  //   }
  //   if (fabs(ai_res[i] - my_res[i]) > 0.000001f) {
  //     std::cout << "error\n"; std::exit(0); 
  //   }
  //   if (fabs(dam_res[i] - my_res[i]) > 0.000001f) {
  //     std::cout << "error\n"; std::exit(0); 
  //   }
  // }
  // std::cout << "ok\n";

  return 0;
}
