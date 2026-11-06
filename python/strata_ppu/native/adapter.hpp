#pragma once
#include <cstdint>
#include <vector>

namespace strata_tensor {
// Private C++ linkage between this JIT module's Tensor binding and device TU.
// This is not an exported operator ABI or an installed header requirement.
std::vector<int64_t> layout(int n, int k, int experts);
void pack(const std::vector<uintptr_t>& p, int n, int k, int experts, uintptr_t stream);
void multiply(const std::vector<uintptr_t>& p, int n, int k, int experts, int rows,
              bool f16, int64_t ldx, int64_t ldy, uintptr_t stream);
}
