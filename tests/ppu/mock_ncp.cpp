// ABI fixtures only. These libraries never execute model math.
#include "strata/ppu/ncp.hpp"

#ifdef MOCK_FA
extern "C" int ncp_flash_attn_fwd_v3(
    const void*, const void*, const void*, void*, int b, int q, int k, int h, int hk, int d, int dv,
    long long qb, long long qh, long long qr, long long kb, long long kh, long long kr,
    long long vb, long long vh, long long vr, long long ob, long long oh, long long orow,
    float scale, float cap, int causal, int wl, int wr, int dtype, void*) {
    return b == 1 && q == 2 && k == 3 && h == 24 && hk == 2 && d == 256 && dv == 256 &&
        qb == 11 && qh == 12 && qr == 13 && kb == 21 && kh == 22 && kr == 23 &&
        vb == 31 && vh == 32 && vr == 33 && ob == 41 && oh == 42 && orow == 43 &&
        scale == 0.0625f && cap == 0 && causal == 0 && wl == -1 && wr == -1 && dtype == 0 ? 101 : -99;
}
#elif defined(MOCK_OLD_FA)
extern "C" int ncp_flash_attn_fwd_v2() { return -99; }
#endif

#ifdef MOCK_MOE
extern "C" int ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
    const void*, const void*, void*, const int*, const int*, int rows, int n, int k, int e, int expected, void*) {
    return rows == 7 && n == 1280 && k == 2560 && e == 1 && expected == 7 ? 102 : -99;
}
#endif

#ifdef MOCK_MOE_OLD
extern "C" int ncp_m_grouped_gemm_bf16_bf16_bf16_nt_contiguous(
    const void*, const void*, void*, const int*, int, int, int, int, int, void*) { return -99; }
#endif

#ifdef MOCK_GDN
extern "C" int ncp_gdn_recurrent(const float*, const float*, const float*, const float*, const float*,
    const float*, float*, float*, int b, int t, int h, int hv, int s, float scale, void*) {
    return b == 1 && t == 8 && h == 16 && hv == 48 && s == 128 && scale == 0.25f ? 103 : -99;
}
extern "C" int ncp_gdn_chunked_bf16(const void*, const void*, const void*, const float*, const float*,
    const float*, void*, float*, int b, int t, int h, int hv, int s, float scale, void*, size_t ws, void*) {
    return b == 1 && t == 128 && h == 16 && hv == 48 && s == 128 && scale == 0.25f && ws == 4096 ? 104 : -99;
}
#ifndef MOCK_NO_WORKSPACE
extern "C" size_t ncp_gdn_chunked_bf16_workspace_size(int b, int t, int h, int hv, int s) {
    return b == 1 && t == 128 && h == 16 && hv == 48 && s == 128 ? 4096 : 0;
}
#endif
#endif
