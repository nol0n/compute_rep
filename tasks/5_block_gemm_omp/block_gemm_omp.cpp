#include <vector>
#include <immintrin.h>

#include "block_gemm_omp.h"

void transpose_matrix(std::vector<float>& a, int n) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      std::swap(a[i * n + j], a[j * n + i]);
    }
  }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  std::vector<float> b_T(b);
  transpose_matrix(b_T, n);
  constexpr int block_size = 32;

  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for collapse(2) firstprivate(n)
  for (int block_i = 0; block_i < n; block_i += block_size) {
    for (int block_j = 0; block_j < n; block_j += block_size) {
      for (int block_k = 0; block_k < n; block_k += block_size) {
        float a_block[block_size][block_size];
        float b_block[block_size][block_size];

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            a_block[i][j] = a[(block_i + i) * n + block_k + j];
            b_block[i][j] = b_T[(block_j + i) * n + block_k + j];
          }
        }

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            for (int k = 0; k < block_size; ++k) {
              result[(block_i + i) * n + block_j + j] += a_block[i][k] * b_block[j][k];
            }
          }
        }
      }
    }
  }

  return result;
}

std::vector<float> ArtemBlockGemmOMP(const std::vector<float>& a,
                                     const std::vector<float>& b, int size) {
  auto countElem = size * size;
  if (a.size() != countElem || b.size() != countElem)
    return {};

  std::vector<float> c(countElem, 0.0f);
  constexpr auto blockSize = 8;
  auto numBlocks = size / blockSize;

#pragma omp parallel for shared(a, b, c)
  for (int i = 0; i < numBlocks; ++i)
    for (int j = 0; j < numBlocks; ++j)
      for (int block = 0; block < numBlocks; ++block)
        for (int m = i * blockSize; m < (i + 1) * blockSize; ++m)
          for (int n = j * blockSize; n < (j + 1) * blockSize; ++n)
            for (int k = block * blockSize; k < (block + 1) * blockSize; ++k)
              c[m * size + n] += a[m * size + k] * b[k * size + n];

  return c;
}

std::vector<float> DamirBlockGemmOMP(const std::vector<float>& a,
                                     const std::vector<float>& b, int n) {
  std::vector<float> c(n * n);
  constexpr int block_sz = 16;
  if (n % block_sz == 0) {
#pragma omp parallel for
    for (int ii = 0; ii < n; ii += block_sz) {
      for (int kk = 0; kk < n; kk += block_sz) {
        for (int jj = 0; jj < n; jj += block_sz) {
          for (int i = 0; i < block_sz; i++) {
            for (int k = 0; k < block_sz; k++) {
              for (int j = 0; j < block_sz; j++) {
                c[(ii + i) * n + (jj + j)] +=
                    a[(ii + i) * n + (kk + k)] * b[(kk + k) * n + (jj + j)];
              }
            }
          }
        }
      }
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
          c[i * n + j] += a[i * n + k] * b[k * n + j];
        }
      }
    }
  }
  return c;
}

std::vector<float> OptimizedBlockGemmOMP(const std::vector<float>& a,
                                         const std::vector<float>& b, int n) {
  std::vector<float> b_T(b);
  transpose_matrix(b_T, n);

  constexpr int block_size = 32;
  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for collapse(2)
  for (int block_i = 0; block_i < n; block_i += block_size) {
    for (int block_j = 0; block_j < n; block_j += block_size) {
      for (int block_k = 0; block_k < n; block_k += block_size) {

        float a_block[block_size][block_size];
        float b_block[block_size][block_size];

        for (int i = 0; i < block_size; ++i) {
          for (int k = 0; k < block_size; ++k) {
            a_block[i][k] = a[(block_i + i) * n + block_k + k];
            b_block[i][k] = b_T[(block_j + i) * n + block_k + k];
          }
        }

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            for (int k = 0; k < block_size; ++k) {
              result[(block_i + i) * n + block_j + j] += a_block[i][k] * b_block[j][k];
            }
          }
        }

      }
    }
  }

  return result;
}
