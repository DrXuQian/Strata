#include "strata/ppu/kernels.hpp"
#include "strata/ppu/python.hpp"
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
int device() { int d;check(cudaGetDevice(&d));return d; }
bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(stream,&status));
    return status!=cudaStreamCaptureStatusNone;
}
using python::Dtype;
using python::Tensor;

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
template<> __device__ __nv_bfloat16 convert<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

template<class T>
__global__ void pack_gdn(const float* q, const float* k, const float* v, T* qo, T* ko, T* vo,
                         const float* gate, const float* beta, float* gp, __nv_bfloat16* bp,
                         int64_t qs, int64_t ks, int64_t vs, int H, int HV, int S, int Tn, const int32_t* keep, int first) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(Tn) * HV * S) return;
    const int d = i % S, h = (i / S) % HV, t = i / (HV * S);
    const bool valid=!keep || t+first < max(*keep,0);
    if (h < H) {
        const int64_t qki = (int64_t(t) * H + h) * S + d;
        qo[qki] = convert<T>(valid ? q[int64_t(t) * qs + h * S + d] : 0.f);
        ko[qki] = convert<T>(valid ? k[int64_t(t) * ks + h * S + d] : 0.f);
    }
    // FLA pairs h / (HV/H), Strata pairs hv % H. Permute value heads so the
    // shared library can use its original (H,HV,S) specialization unchanged.
    const int hv = (h % (HV / H)) * H + h / (HV / H);
    vo[i] = convert<T>(valid ? v[int64_t(t) * vs + hv * S + d] : 0.f);
    if (d == 0) {
        gp[t * HV + h] = valid ? gate[t * HV + hv] : 0.f;
        bp[t * HV + h] = __float2bfloat16_rn(valid ? beta[t * HV + hv] : 0.f);
    }
}

template<bool Pack>
__global__ void state_layout(const float* in, float* out, int H, int HV, int S, const int32_t* keep = nullptr) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(HV) * S * S || (!Pack && keep && *keep<=0)) return;
    const int v = i % S, h = (i / S) % HV, k = i / (S * HV);
    const int he = (h % H) * (HV / H) + h / H;
    const int64_t external = (int64_t(he) * S + k) * S + v;
    if (Pack) out[external] = in[i];
    else out[i] = in[external];
}

template<class T>
__global__ void unpack_gdn(const T* in, float* out, int64_t n, int H, int HV, int S, const int32_t* keep = nullptr, int first = 0) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = i % S, h = (i / S) % HV, t = i / (HV * S);
    if (keep && t+first>=max(*keep,0)) return;
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

namespace {
constexpr int kGdnBlock=256;
size_t gdn_bytes(int HV,int S) {
    const size_t count=size_t(kGdnBlock)*HV*S,states=size_t(HV)*S*S;
    return 2*aligned(states*4)+4*aligned(count*2)+aligned(count/S*4)+aligned(count/S*2);
}
struct GdnBuffers {
    float *h0,*ht,*g;
    __nv_bfloat16 *q,*k,*v,*o,*b;
    explicit GdnBuffers(void* p,int HV,int S) {
        Arena a{static_cast<char*>(p)};
        const size_t count=size_t(kGdnBlock)*HV*S,states=size_t(HV)*S*S;
        h0=a.take<float>(states);ht=a.take<float>(states);
        q=a.take<__nv_bfloat16>(count);k=a.take<__nv_bfloat16>(count);
        v=a.take<__nv_bfloat16>(count);o=a.take<__nv_bfloat16>(count);
        g=a.take<float>(count/S);b=a.take<__nv_bfloat16>(count/S);
    }
    std::vector<Tensor> tensors(int T,int H,int HV,int S) {
        return {{q,Dtype::bf16,{1,T,H,S}},{k,Dtype::bf16,{1,T,H,S}},
            {v,Dtype::bf16,{1,T,HV,S}},{g,Dtype::f32,{1,T,HV}},{b,Dtype::bf16,{1,T,HV}},
            {h0,Dtype::f32,{1,HV,S,S}},{o,Dtype::bf16,{1,T,HV,S}},{ht,Dtype::f32,{1,HV,S,S}}};
    }
};
}
bool prepare_gdn(int max_tokens,int H,int HV,int S,void* stream) {
#ifndef STRATA_PPU_GDN
    return false;
#else
    if(!enabled("STRATA_PPU_GDN")||max_tokens<=0||H<=0||HV<=0||HV>65535||HV%H||S!=128)return false;
    const auto cs=static_cast<cudaStream_t>(stream);
    if(capturing(cs))throw std::runtime_error("prepare_gdn must precede graph capture");
    auto& scratch=scratch_for(cs,0);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    if(!scratch.reserve(gdn_bytes(HV,S),cs))throw std::runtime_error("actlizeLA: insufficient prepared GDN scratch");
    check(cudaMemsetAsync(scratch.data,0,scratch.bytes,cs));
    GdnBuffers b(scratch.data,HV,S);
    python::call("gdn_prepare",b.tensors(kGdnBlock,H,HV,S),{std::min(max_tokens,kGdnBlock)},device(),stream);
    return true;
#endif
}

bool gdn(const float* state,float* final_state,const float* q,const float* k,const float* v,
         int64_t qs,int64_t ks,int64_t vs,const float* gate,const float* beta,float* out,
         int T,int H,int HV,int S,float scale,bool allow_chunked,void* stream,bool allow_capture,
         const int32_t* keep,int out_begin) {
#ifndef STRATA_PPU_GDN
    return false;
#else
    (void)allow_chunked; // The installed core owns full-chunk versus tail selection.
    if(!enabled("STRATA_PPU_GDN")||!state||!q||!k||!v||!gate||!beta||(!out&&!final_state)||
        T<=0||H<=0||HV<=0||HV>65535||HV%H||S!=128||
        qs<int64_t(H)*S||ks<int64_t(H)*S||vs<int64_t(HV)*S)return false;
    if(std::abs(scale-1.f/std::sqrt(float(S)))>1.e-7f)
        throw std::invalid_argument("actlizeLA applies exactly one 1/sqrt(128) query scale");
    const auto cs=static_cast<cudaStream_t>(stream);
    const bool capture=capturing(cs);
    if(capture&&!allow_capture)return false;
    auto& scratch=scratch_for(cs,0);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    if(!scratch.reserve(gdn_bytes(HV,S),cs))throw std::runtime_error("actlizeLA: insufficient prepared GDN scratch");
    GdnBuffers b(scratch.data,HV,S);
    const size_t states=size_t(HV)*S*S;
    state_layout<true><<<(states+255)/256,256,0,cs>>>(state,b.h0,H,HV,S);
    for(int first=0;first<T;first+=kGdnBlock) {
        const int nt=std::min(T-first,kGdnBlock);
        const int64_t n=int64_t(nt)*HV*S;
        pack_gdn<<<(n+255)/256,256,0,cs>>>(q+first*qs,k+first*ks,v+first*vs,
            b.q,b.k,b.v,gate+size_t(first)*HV,beta+size_t(first)*HV,b.g,b.b,
            qs,ks,vs,H,HV,S,nt,keep,first);
        check(cudaGetLastError());
        python::call("gdn",b.tensors(nt,H,HV,S),{},device(),stream,capture);
        const int skip=std::max(0,out_begin-first);
        if(out&&skip<nt) {
            const int64_t remaining=int64_t(nt-skip)*HV*S;
            unpack_gdn<<<(remaining+255)/256,256,0,cs>>>(b.o+size_t(skip)*HV*S,
                out+size_t(first+skip)*HV*S,remaining,H,HV,S,keep,first+skip);
        }
        std::swap(b.h0,b.ht);
    }
    if(final_state)state_layout<false><<<(states+255)/256,256,0,cs>>>(b.h0,final_state,H,HV,S,keep);
    check(cudaGetLastError());
    strata::dispatch::note("ppu: actlizeLA Tensor GDN");
    return true;
#endif
}

void gdn_norm(float* output, const float* z, const float* gamma, float eps, int HV, void* stream) {
    norm_gdn<<<HV, 128, 0, static_cast<cudaStream_t>(stream)>>>(output, z, gamma, eps);
    check(cudaGetLastError());
}

bool expert_f16(const uint16_t* x, const uint16_t* weight, float* out,
                int64_t rows, int64_t N, int64_t K, void* stream) {
    if (!enabled("STRATA_PPU_MOE") || !x || !weight || !out || rows <= 0 || rows > INT_MAX ||
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
        python::call("moe",{{a,Dtype::bf16,{nt,K}},{w,Dtype::bf16,{1,N,K}},
            {y,Dtype::bf16,{nt,N}},{ids,Dtype::i32,{nt}},{counts,Dtype::i32,{1}}},
            {},device(),stream,capturing(cs));
        widen<<<(nt * N + 255) / 256, 256, 0, cs>>>(y, out + first * N, nt * N);
    }
    check(cudaGetLastError());
    strata::dispatch::note("ppu: DeepGEMM Tensor expert GEMM");
    return true;
}

bool qsa_prefill(const float* q, const kernels::QsaAttnPools& pools, const int32_t* ids,
                 const int32_t* host_steps, int64_t cap, const kernels::QsaShapes& s,
                 float* out, int64_t T, void* stream) {
    if (!enabled("STRATA_PPU_FA") || !q || !ids || !host_steps || !out || !pools.page_table || T <= 0 || T > INT_MAX ||
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
        python::call("fa",{{qh,Dtype::f16,{1,1,s.n_head,256}},
            {kh,Dtype::f16,{1,width,s.n_head_kv,256}},{vh,Dtype::f16,{1,width,s.n_head_kv,256}},
            {oh,Dtype::f16,{1,1,s.n_head,256}}},{},device(),stream);
        widen<<<(nq + 255) / 256, 256, 0, cs>>>(oh, out + t * nq, nq);
    }
    check(cudaGetLastError());
    strata::dispatch::note("ppu: FlashAttention Tensor selected cells");
    return true;
}

}  // namespace strata::ppu
