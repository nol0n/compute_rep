#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <rng.hpp>

#include <fft_cufft.h>

TEST(fftCUFFTTests, fftCUFFT) {
  
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> seconds;
  std::cout << std::setprecision(10);
  size_t n = 53107200;
  size_t batch = 5;

  for (int i = 0; i < 5; i++) {
    std::vector<float> a = rng::float_vector(2 * n * batch, 0.0f, 1.0f);
    FftCUFFT(a, batch);
    start = std::chrono::high_resolution_clock::now();
    FftCUFFT(a, batch);
    end = std::chrono::high_resolution_clock::now();

    seconds = end - start;
    std::cout << " - " << seconds.count() << " s.\n";
  }
  
  EXPECT_EQ(1, 1);
}
