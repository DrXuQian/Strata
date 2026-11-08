#include "strata/ppu/kernels.hpp"
#include "strata/ppu/ncp.hpp"
#include "strata/core/dispatch_stats.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strata::ppu {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("strata PPU: ") + cudaGetErrorString(error));
}
bool accepted(int rc, const char* name, bool any_nonzero_declines = false) {
    if (rc == 0) return true;
    if (rc == -1 || any_nonzero_declines) {
        check(cudaGetLastError());
        strata::dispatch::note("ppu: NCP declined, using Strata kernels");
        return false;
    }
    throw std::runtime_error(std::string("strata PPU: ") + name + " failed, status " + std::to_string(rc));
}

size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

struct Scratch {
    std::mutex mutex;
    void* data = nullptr;
    size_t bytes = 0;
    std::vector<void*> retired;
    bool reserve(size_t need, cudaStream_t stream) {
        if (need <= bytes) return true;
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) return false;
        void* next = nullptr;
        const cudaError_t rc = cudaMalloc(&next, need);
        if (rc == cudaErrorMemoryAllocation) { cudaGetLastError(); return false; }
        check(rc);
        // Old graphs still own these addresses. Never free or resize their storage.
        if (data) retired.push_back(data);
        data = next;
        bytes = need;
        return true;
    }
};

Scratch& scratch_for(cudaStream_t stream, int kind) {
    int device;
    check(cudaGetDevice(&device));
    using Key = std::tuple<int, uintptr_t, int>;
    // CUDA contexts and captured graphs can outlive host thread-local objects.
    static auto* cache = new std::map<Key, std::unique_ptr<Scratch>>;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    auto& item = (*cache)[Key{device, reinterpret_cast<uintptr_t>(stream), kind}];
    if (!item) item = std::make_unique<Scratch>();
    return *item;
}

struct Arena {
    char* p;
    template<class T> T* take(size_t count) {
        T* result = reinterpret_cast<T*>(p);
        p += aligned(count * sizeof(T));
        return result;
    }
};

template<class T> __device__ T convert(float x);
template<> __device__ float convert<float>(float x) { return x; }
template<> __device__ __nv_bfloat16 convert<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

template<class T>
__global__ void pack_gdn(const float* q, const float* k, const float* v, T* qo, T* ko, T* vo,
                         const float* gate, const float* beta, float* gp, float* bp,
                         int64_t qs, int64_t ks, int64_t vs, int H, int HV, int S, int Tn) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(Tn) * HV * S) return;
    const int d = i % S, h = (i / S) % HV, t = i / (HV * S);
    if (h < H) {
        const int64_t qki = (int64_t(t) * H + h) * S + d;
        qo[qki] = convert<T>(q[int64_t(t) * qs + h * S + d]);
        ko[qki] = convert<T>(k[int64_t(t) * ks + h * S + d]);
    }
    // FLA pairs h / (HV/H), Strata pairs hv % H. Permute value heads so the
    // shared library can use its original (H,HV,S) specialization unchanged.
    const int hv = (h % (HV / H)) * H + h / (HV / H);
    vo[i] = convert<T>(v[int64_t(t) * vs + hv * S + d]);
    if (d == 0) {
        gp[t * HV + h] = gate[t * HV + hv];
        bp[t * HV + h] = beta[t * HV + hv];
    }
}

template<bool Pack>
__global__ void state_layout(const float* in, float* out, int H, int HV, int S, bool chunked) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(HV) * S * S) return;
    const int v = i % S, h = (i / S) % HV, k = i / (S * HV);
    const int he = (h % H) * (HV / H) + h / H;
    const int64_t external = (int64_t(he) * S + (chunked ? k : v)) * S + (chunked ? v : k);
    if (Pack) out[external] = in[i];
    else out[i] = in[external];
}

template<class T>
__global__ void unpack_gdn(const T* in, float* out, int64_t n, int H, int HV, int S) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = i % S, h = (i / S) % HV, t = i / (HV * S);
    const int he = (h % H) * (HV / H) + h / H;
    out[i] = float(in[(int64_t(t) * HV + he) * S + d]);
}

template<class T>
__global__ void widen(const T* in, float* out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = float(in[i]);
}

__global__ void f16_to_bf16(const uint16_t* in, __nv_bfloat16* out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(__half2float(__ushort_as_half(in[i])));
}

__global__ void one_expert(int* ids, int* counts, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = 0;
    if (i == 0) counts[0] = n;
}

__global__ void norm_gdn(float* y, const float* z, const float* gamma, float eps) {
    const int d = threadIdx.x, i = blockIdx.x * 128 + d;
    const float o = y[i];
    float ss = o * o;
    for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    __shared__ float sums[4];
    if ((d & 31) == 0) sums[d / 32] = ss;
    __syncthreads();
    const float sum = sums[0] + sums[1] + sums[2] + sums[3];
    y[i] = o * rsqrtf(sum / 128.0f + eps) * gamma[d] * (1.0f / (1.0f + __expf(-z[i])));
}

__global__ void pack_query(const float* q, __half* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(q[i]);
}

__device__ __half kv_value(const kernels::QsaAttnPools& p, bool value, int64_t row, int d, int D) {
    if (p.k_pool) return __ushort_as_half((value ? p.v_pool : p.k_pool)[row * D + d]);
    const int8_t* codes = value ? p.v_q : p.k_q;
    const uint16_t* scales = value ? p.v_scale : p.k_scale;
    return __float2half_rn(float(codes[row * D + d]) *
                          __half2float(__ushort_as_half(scales[row * (D / 64) + d / 64])));
}

__global__ void gather_qsa(kernels::QsaAttnPools p, const int32_t* ids, __half* k, __half* v,
                           int width, int H, int D, int page_size) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(width) * H * D) return;
    const int d = i % D, h = (i / D) % H, cell = ids[i / (H * D)];
    const int page = p.page_table[cell / page_size];
    const int64_t row = (int64_t(page) * H + h) * page_size + cell % page_size;
    k[i] = kv_value(p, false, row, d, D);
    v[i] = kv_value(p, true, row, d, D);
}
}  // namespace

bool prepare_gdn(int max_tokens, int HV, int S, void* stream) {
    if (!ncp().recurrent || max_tokens <= 0 || HV <= 0 || HV > 65535 || S != 128) return false;
    const size_t count = size_t(std::min(max_tokens, 256)) * HV * S;
    const size_t states = size_t(HV) * S * S;
    const auto cs = static_cast<cudaStream_t>(stream);
    Scratch& scratch = scratch_for(cs, 0);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    return scratch.reserve(2 * aligned(states * sizeof(float)) + 4 * aligned(count * sizeof(float)) +
                           2 * aligned(count / S * sizeof(float)), cs);
}

bool gdn(const float* state, float* final_state, const float* q, const float* k, const float* v,
         int64_t qs, int64_t ks, int64_t vs, const float* gate, const float* beta, float* out,
         int T, int H, int HV, int S, float scale, bool allow_chunked, void* stream, bool allow_capture) {
    const Ncp& api = ncp();
    const bool chunked = allow_chunked && T >= 128 && T % 64 == 0 && api.chunked &&
                         enabled("STRATA_NCP_GDN_CHUNKED");
    if ((!chunked && !api.recurrent) || !state || !q || !k || !v || !gate || !beta || !out ||
        T <= 0 || H <= 0 || HV <= 0 || HV > 65535 || HV % H || S != 128 ||
        qs < int64_t(H) * S || ks < int64_t(H) * S || vs < int64_t(HV) * S) return false;
    const auto cs = static_cast<cudaStream_t>(stream);
    if (!allow_capture) {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(cs, &capture));
        if (capture != cudaStreamCaptureStatusNone) return false;
    }
    const int block_t = std::min(T, 256);
    const size_t count = size_t(block_t) * HV * S, states = size_t(HV) * S * S;
    size_t ws_bytes = 0;
    if (chunked) {
        for (int t = 128; t <= 256; t += 64)
            ws_bytes = std::max(ws_bytes, api.workspace_size(1, t, H, HV, S));
        if (!ws_bytes) return false;
    }
    const size_t element = chunked ? sizeof(__nv_bfloat16) : sizeof(float);
    const size_t bytes = 2 * aligned(states * sizeof(float)) + 4 * aligned(count * element) +
                         2 * aligned(count / S * sizeof(float)) + aligned(ws_bytes);
    Scratch& scratch = scratch_for(cs, 0);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    if (!scratch.reserve(bytes, cs)) return false;
    Arena arena{static_cast<char*>(scratch.data)};
    float* h0 = arena.take<float>(states);
    float* ht = arena.take<float>(states);
    void* qp = arena.take<char>(count * element);
    void* kp = arena.take<char>(count * element);
    void* vp = arena.take<char>(count * element);
    void* op = arena.take<char>(count * element);
    float* gp = arena.take<float>(count / S);
    float* bp = arena.take<float>(count / S);
    void* ws = arena.take<char>(ws_bytes);
    state_layout<true><<<(states + 255) / 256, 256, 0, cs>>>(state, h0, H, HV, S, chunked);
    for (int first = 0; first < T;) {
        int nt = std::min(T - first, 256);
        if (chunked && T - first > nt && T - first - nt < 128) nt -= 64;
        const int64_t n = int64_t(nt) * HV * S;
        int rc;
        if (chunked) {
            auto* qb = static_cast<__nv_bfloat16*>(qp);
            auto* kb = static_cast<__nv_bfloat16*>(kp);
            auto* vb = static_cast<__nv_bfloat16*>(vp);
            auto* ob = static_cast<__nv_bfloat16*>(op);
            pack_gdn<<<(n + 255) / 256, 256, 0, cs>>>(q + first * qs, k + first * ks, v + first * vs,
                qb, kb, vb, gate + size_t(first) * HV, beta + size_t(first) * HV, gp, bp,
                qs, ks, vs, H, HV, S, nt);
            check(cudaGetLastError());
            rc = api.chunked(qb, kb, vb, gp, bp, h0, ob, ht, 1, nt, H, HV, S, scale, ws, ws_bytes, stream);
            if (!accepted(rc, "ncp_gdn_chunked_bf16")) return false;
            unpack_gdn<<<(n + 255) / 256, 256, 0, cs>>>(ob, out + size_t(first) * HV * S, n, H, HV, S);
        } else {
            auto* qf = static_cast<float*>(qp);
            auto* kf = static_cast<float*>(kp);
            auto* vf = static_cast<float*>(vp);
            auto* of = static_cast<float*>(op);
            pack_gdn<<<(n + 255) / 256, 256, 0, cs>>>(q + first * qs, k + first * ks, v + first * vs,
                qf, kf, vf, gate + size_t(first) * HV, beta + size_t(first) * HV, gp, bp,
                qs, ks, vs, H, HV, S, nt);
            check(cudaGetLastError());
            rc = api.recurrent(qf, kf, vf, gp, bp, h0, of, ht, 1, nt, H, HV, S, scale, stream);
            if (!accepted(rc, "ncp_gdn_recurrent")) return false;
            unpack_gdn<<<(n + 255) / 256, 256, 0, cs>>>(of, out + size_t(first) * HV * S, n, H, HV, S);
        }
        std::swap(h0, ht);
        first += nt;
    }
    if (final_state) state_layout<false><<<(states + 255) / 256, 256, 0, cs>>>(h0, final_state, H, HV, S, chunked);
    check(cudaGetLastError());
    strata::dispatch::note(chunked ? "ppu: NCP GDN chunked" : "ppu: NCP GDN recurrent");
    return true;
}

void gdn_norm(float* output, const float* z, const float* gamma, float eps, int HV, void* stream) {
    norm_gdn<<<HV, 128, 0, static_cast<cudaStream_t>(stream)>>>(output, z, gamma, eps);
    check(cudaGetLastError());
}

bool expert_f16(const uint16_t* x, const uint16_t* weight, float* out,
                int64_t rows, int64_t N, int64_t K, void* stream) {
    const auto fn = ncp().moe;
    if (!fn || !x || !weight || !out || rows <= 0 || rows > INT_MAX ||
        N <= 0 || N > INT_MAX || K <= 0 || K > INT_MAX || N % 8 || K % 8) return false;
    const auto cs = static_cast<cudaStream_t>(stream);
    const int block_t = int(std::min<int64_t>(rows, 256));
    const size_t wn = size_t(N) * K, xn = size_t(block_t) * K, yn = size_t(block_t) * N;
    const size_t bytes = aligned(wn * 2) + aligned(xn * 2) + aligned(yn * 2) +
                         aligned(block_t * sizeof(int)) + 256;
    Scratch& scratch = scratch_for(cs, 1);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    if (!scratch.reserve(bytes, cs)) return false;
    Arena arena{static_cast<char*>(scratch.data)};
    auto* w = arena.take<__nv_bfloat16>(wn);
    auto* a = arena.take<__nv_bfloat16>(xn);
    auto* y = arena.take<__nv_bfloat16>(yn);
    auto* ids = arena.take<int>(block_t);
    auto* counts = arena.take<int>(1);
    f16_to_bf16<<<(wn + 255) / 256, 256, 0, cs>>>(weight, w, wn);
    for (int64_t first = 0; first < rows; first += block_t) {
        const int nt = int(std::min<int64_t>(rows - first, block_t));
        f16_to_bf16<<<(nt * K + 255) / 256, 256, 0, cs>>>(x + first * K, a, nt * K);
        one_expert<<<(nt + 255) / 256, 256, 0, cs>>>(ids, counts, nt);
        check(cudaGetLastError());
        if (!accepted(fn(a, w, y, ids, counts, nt, int(N), int(K), 1, nt, stream), "ncp_moe_nopad", true)) return false;
        widen<<<(nt * N + 255) / 256, 256, 0, cs>>>(y, out + first * N, nt * N);
    }
    check(cudaGetLastError());
    strata::dispatch::note("ppu: NCP NoPad expert GEMM");
    return true;
}

bool qsa_prefill(const float* q, const kernels::QsaAttnPools& pools, const int32_t* ids,
                 const int32_t* host_steps, int64_t cap, const kernels::QsaShapes& s,
                 float* out, int64_t T, void* stream) {
    const auto fn = ncp().fa;
    if (!fn || !q || !ids || !host_steps || !out || !pools.page_table || T <= 0 || T > INT_MAX ||
        cap <= 0 || cap > INT_MAX || s.head_dim != 256 || s.n_head != 24 || s.n_head_kv != 2 ||
        s.page_size <= 0 || s.page_size > INT_MAX || pools.k_q4 || pools.v_q4) return false;
    if (pools.k_pool ? !pools.v_pool : (!pools.k_q || !pools.v_q || !pools.k_scale || !pools.v_scale)) return false;
    int width_max = 0;
    for (int64_t t = 0; t < T; ++t) {
        const int width = host_steps[t * kernels::kStepCount + kernels::kStepWidth];
        if (width <= 0 || width > cap) return false;
        width_max = std::max(width_max, width);
    }
    const auto cs = static_cast<cudaStream_t>(stream);
    // Host lengths cannot vary on graph replay. Decode keeps its device-length kernel.
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(cs, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;
    const int nq = int(s.n_head * s.head_dim), kv_row = int(s.n_head_kv * s.head_dim);
    const size_t kv_count = size_t(width_max) * kv_row;
    Scratch& scratch = scratch_for(cs, 2);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    if (!scratch.reserve(2 * aligned(nq * 2) + 2 * aligned(kv_count * 2), cs)) return false;
    Arena arena{static_cast<char*>(scratch.data)};
    auto* qh = arena.take<__half>(nq);
    auto* oh = arena.take<__half>(nq);
    auto* kh = arena.take<__half>(kv_count);
    auto* vh = arena.take<__half>(kv_count);
    for (int64_t t = 0; t < T; ++t) {
        const int width = host_steps[t * kernels::kStepCount + kernels::kStepWidth];
        const int64_t n = int64_t(width) * kv_row;
        pack_query<<<(nq + 255) / 256, 256, 0, cs>>>(q + t * nq, qh, nq);
        gather_qsa<<<(n + 255) / 256, 256, 0, cs>>>(pools, ids + t * cap, kh, vh,
            width, int(s.n_head_kv), int(s.head_dim), int(s.page_size));
        check(cudaGetLastError());
        const int rc = fn(qh, kh, vh, oh, 1, 1, width, int(s.n_head), int(s.n_head_kv), 256, 256,
            nq, 256, nq, n, 256, kv_row, n, 256, kv_row, nq, 256, nq,
            1.0f / std::sqrt(256.0f), 0.0f, 0, -1, -1, 0, stream);
        if (!accepted(rc, "ncp_flash_attn_fwd_v3", true)) return false;
        widen<<<(nq + 255) / 256, 256, 0, cs>>>(oh, out + t * nq, nq);
    }
    check(cudaGetLastError());
    strata::dispatch::note("ppu: NCP selected-cell FlashAttention");
    return true;
}

}  // namespace strata::ppu
