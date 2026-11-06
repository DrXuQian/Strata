#pragma once
#include <cstddef>
#include <cstdint>
#include "strata/ppu/quactlize/quactlize_ppu_pack.h"
#include "strata/ppu/quactlize/kpack_execution.h"

namespace strata::ppu::quactlize {
// ABI copied from the user's llama.cpp; these functions load existing DSOs.
struct Library {
    decltype(&quactlize_ppu_kpack_canonical_arrangement_v1) arrangement = nullptr;
    decltype(&quactlize_ppu_kpack_sizes_for_arrangement_v1) sizes = nullptr;
    decltype(&quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2) pack = nullptr;
    decltype(&quactlize_kpack_gemv_query_v1) query = nullptr;
    decltype(&quactlize_kpack_gemv_run_v1) run = nullptr;
};
bool enabled();
bool loaded();
const Library& library(); // Explicit opt-in + broken/missing library throws.
struct Shape {
    quactlize_ppu_placed_arrangement_v2 arrangement{};
    quactlize_ppu_kpack_sizes_v1 sizes{};
};
// Host-only admission by BOTH producer and consumer. No invented layout IDs.
bool describe(const qkg_call_v1& call, Shape& shape);
const qkg_config_v1& recipe();

enum class Input { f16, f32, q8_1 };
// Immutable dense/head weights only. Original GGUF bytes remain valid for every
// original reader. Additional packed storage is bounded and counted at startup.
size_t preload(const void* raw, int type, int n, int k);
void release(const void* raw) noexcept;

// False means no work submitted: the original path can run. Admitted launch
// errors throw. Mutable expert/staging pointers are NEVER cached by address.
bool dense(const void* raw, int type, int n, int k, const void* x, Input input,
           int rows, float* out, void* stream, int64_t ldx = 0,
           int64_t ldy = 0, float beta = 0, bool interleaved_gu = false);

// Device-driven expert routing; no ID readback. Eight weight slices at a time
// bound scratch regardless of the number of groups. Gate/up blob rows are
// [gate; up]; output is split into gate/up, or scattered through ent_dst (down).
bool grouped(const unsigned long long* ptr, const int32_t* start, const int32_t* count,
             const int32_t* ent_tok, const int32_t* ent_dst, int cap_groups, int cap_entries,
             const void* x_q8_1, size_t weight_offset, int type, int n, int k,
             float* out, float* up, void* stream);
} // namespace strata::ppu::quactlize
