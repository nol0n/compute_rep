#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n);

std::vector<float> ArtemBlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int size);

std::vector<float> DamirBlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n);

std::vector<float> OptimizedBlockGemmOMP(const std::vector<float>& a,
                                         const std::vector<float>& b, int n);

#endif  // __BLOCK_GEMM_OMP_H