#pragma once
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace strata::ppu {
bool enabled(const char* name, bool default_value = true);
namespace python {
enum class Dtype { u8, i32, f16, bf16, f32 };
// A borrowed view. The engine owns storage through execution/graph lifetime.
// Only DLPack metadata is transferred; its deleter never frees engine memory.
struct Tensor {
    const void* data;
    Dtype dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides{};
    bool cpu = false;
};
bool loaded();
// device/stream/capture are supplied by the caller, without a second runtime
// query from the embedded interpreter. No LibTorch C++ ABI crosses this seam.
std::vector<int64_t> call(const char* operation, const std::vector<Tensor>& tensors,
    const std::vector<int64_t>& integers, int device, void* stream, bool capturing = false);
} // namespace python
} // namespace strata::ppu
