#define _USE_MATH_DEFINES

#include "naive_gemm_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  size_t elements_count = input.size();
  std::vector<float> result(elements_count, 0.0f);
  const float* data_ptr = input.data();

#pragma omp parallel for firstprivate(elements_count)
  for (int i = 0; i < elements_count; ++i) {
    float x = data_ptr[i];
    result[i] =
        0.5 * x *
        (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * powf(x, 3))));
  }

  return result;
}
