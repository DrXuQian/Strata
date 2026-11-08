#pragma once

#include <cstddef>
#include "strata/ppu/ncp-fa-lib.h"
#include "strata/ppu/ncp-moe-lib.h"
#include "strata/ppu/ncp-gdn-lib.h"

namespace strata::ppu {

// ABI copied from llama.cpp 2f288debe, ggml/src/ggml-cuda/ncp-*-lib.h.
// Keep the exported symbol names and argument types in sync with ncp_flash_lib.
struct Ncp {
    decltype(&ncp_flash_attn_fwd_v3) fa = nullptr;
    decltype(&ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad) moe = nullptr;
    decltype(&ncp_gdn_recurrent) recurrent = nullptr;
    decltype(&ncp_gdn_chunked_bf16) chunked = nullptr;
    decltype(&ncp_gdn_chunked_bf16_workspace_size) workspace_size = nullptr;
};

// Process-lifetime handles: captured graphs may reference code in these libraries.
const Ncp& ncp();
bool enabled(const char* name, bool default_value = true);

}  // namespace strata::ppu
