#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstdint>

namespace strata::ppu {

// Reserve recurrence scratch before graph capture; no kernels or state updates.
bool prepare_gdn(int max_tokens, int HV, int S, void* stream);

// Strata state [K,HV,V]; q/k/v have independent token strides, in floats.
// final_state == nullptr leaves the input state unchanged (speculative verification).
// A false result leaves state untouched; callers must overwrite the entire output.
// allow_capture requires prepared scratch and replay on the original stream.
// Keep it false for session graphs captured on temporary streams.
bool gdn(const float* state, float* final_state, const float* q, const float* k, const float* v,
         int64_t q_stride, int64_t k_stride, int64_t v_stride,
         const float* gate, const float* beta, float* out, int T, int H, int HV, int S,
         float scale, bool allow_chunked, void* stream, bool allow_capture = false);

void gdn_norm(float* output, const float* z, const float* gamma, float eps, int HV, void* stream);

// Prefill only: exact selected lengths are already known on the host. Each query
// gathers its own selected cells; the FA call is full attention over that selection.
// All selected pages must be resident: use the full pool or the prefill staging pool.
bool qsa_prefill(const float* q, const kernels::QsaAttnPools& pools, const int32_t* ids,
                 const int32_t* host_steps, int64_t cap, const kernels::QsaShapes& shape,
                 float* out, int64_t T, void* stream);

// One expert's contiguous row segment through the real NoPad ABI (n_experts=1).
// FP16 -> BF16 operands and BF16 -> FP32 output are explicit rounding boundaries.
bool expert_f16(const uint16_t* x, const uint16_t* weight, float* out,
                int64_t rows, int64_t N, int64_t K, void* stream);

}  // namespace strata::ppu
