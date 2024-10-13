#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <rng.hpp>

#include <gemm_cublas.h>

TEST(GemmCUBLASTests, GemmCUBLAS) {
  
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> seconds;
  std::cout << std::setprecision(10);
  size_t n = 1536;

  for (int i = 0; i < 5; i++) {
    std::vector<float> a = rng::float_vector(n * n, 0.0f, 1.0f);
    std::vector<float> b = rng::float_vector(n * n, 0.0f, 1.0f);
    GemmCUBLAS(a, b, n);
    start = std::chrono::high_resolution_clock::now();
    GemmCUBLAS(a, b, n);
    end = std::chrono::high_resolution_clock::now();

    seconds = end - start;
    std::cout << " - " << seconds.count() << " s.\n";
  }
  
  EXPECT_EQ(1, 1);
}
