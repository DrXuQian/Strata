#include "strata/ppu/ncp.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <dlfcn.h>

namespace strata::ppu {

bool enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    return std::strcmp(value, "0") && strcasecmp(value, "false") && strcasecmp(value, "off");
}

namespace {
void* open_library(const char* name) {
    void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (handle) std::fprintf(stderr, "[strata-ppu] loaded %s\n", name);
    else std::fprintf(stderr, "[strata-ppu] %s unavailable: %s; using Strata kernels\n", name, dlerror());
    return handle;
}

template<class Function>
Function symbol(void* handle, const char* name) {
    if (!handle) return nullptr;
    dlerror();
    void* address = dlsym(handle, name);
    if (const char* error = dlerror()) {
        std::fprintf(stderr, "[strata-ppu] %s unavailable: %s\n", name, error);
        return nullptr;
    }
    return reinterpret_cast<Function>(address);
}
}

const Ncp& ncp() {
    static const Ncp api = [] {
        Ncp result;
#ifdef STRATA_NCP_FA
        if (enabled("STRATA_NCP_FA"))
            result.fa = symbol<decltype(result.fa)>(open_library("libncp_fa.so"), "ncp_flash_attn_fwd_v3");
#endif
#ifdef STRATA_NCP_MOE
        if (enabled("STRATA_NCP_MOE"))
            result.moe = symbol<decltype(result.moe)>(open_library("libncp_moe.so"),
                "ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad");
#endif
#ifdef STRATA_NCP_GDN
        if (enabled("STRATA_NCP_GDN")) {
            void* handle = open_library("libncp_gdn.so");
            result.recurrent = symbol<decltype(result.recurrent)>(handle, "ncp_gdn_recurrent");
            result.chunked = symbol<decltype(result.chunked)>(handle, "ncp_gdn_chunked_bf16");
            result.workspace_size = symbol<decltype(result.workspace_size)>(handle,
                "ncp_gdn_chunked_bf16_workspace_size");
            if (!result.workspace_size) result.chunked = nullptr;
        }
#endif
        return result;
    }();
    return api;
}

}  // namespace strata::ppu
