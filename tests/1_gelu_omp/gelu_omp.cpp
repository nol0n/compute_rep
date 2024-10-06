#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <rng.hpp>

#include <gelu_omp.h>

TEST(GeluTests, GeluOMP) {
  
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> seconds;
  std::cout << std::setprecision(10);

  for (int i = 0; i < 5; i++) {
    std::vector<float> data = rng::float_vector(134'217'728, 0.0f, 1.0f);
    start = std::chrono::high_resolution_clock::now();
    GeluOMP(data);
    end = std::chrono::high_resolution_clock::now();

    seconds = end - start;
    std::cout << " - " << seconds.count() << " s.\n";
  }
  
  EXPECT_EQ(1, 1);
}
