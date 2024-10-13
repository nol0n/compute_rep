#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <rng.hpp>

#include <gelu_ocl.h>

TEST(GeluOCLTests, GeluOCL) {
  
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> seconds;
  std::cout << std::setprecision(10);
  size_t n = 100'000'000;

  for (int i = 0; i < 5; i++) {
    std::vector<float> a = rng::float_vector(n, 0.0f, 1.0f);
    GeluOCL(a);
    start = std::chrono::high_resolution_clock::now();
    GeluOCL(a);
    end = std::chrono::high_resolution_clock::now();

    seconds = end - start;
    std::cout << " - " << seconds.count() << " s.\n";
  }
  
  EXPECT_EQ(1, 1);
}
