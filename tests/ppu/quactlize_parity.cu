// Physical-device, multi-kernel integration test. Not a simulator invocation.
#include "strata/ppu/quactlize.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ggml.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace qz = strata::ppu::quactlize;
namespace {
void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { put(v); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) const {
        std::vector<T> r(n); ck(cudaMemcpy(r.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return r;
    }
};
float value(size_t i, int seed = 0) { return float(int((i * 17 + i / 13 + seed * 37) % 127) - 63) / 256; }
struct Weight {
    std::vector<uint8_t> raw;
    std::vector<float> f;
    Weight(int q, int n, int k, int seed = 0) : raw(ggml_row_size((ggml_type) q, k) * n), f(size_t(n) * k) {
        std::vector<float> src(f.size());
        for (size_t i = 0; i < src.size(); ++i) src[i] = value(i, seed) * (1 + (i / 256) % 5);
        switch (q) {
#define QCASE(T, NAME) case T: quantize_row_##NAME##_ref(src.data(), reinterpret_cast<block_##NAME*>(raw.data()), src.size()); \
    dequantize_row_##NAME(reinterpret_cast<const block_##NAME*>(raw.data()), f.data(), f.size()); break
            QCASE(8, q8_0); QCASE(10, q2_K); QCASE(11, q3_K); QCASE(12, q4_K); QCASE(13, q5_K); QCASE(14, q6_K);
#undef QCASE
            default: throw std::runtime_error("test qtype");
        }
    }
};
std::vector<float> product(const std::vector<float>& x, const Weight& w, int rows, int n, int k, int ldx) {
    std::vector<float> y(size_t(rows) * n);
    for (int r = 0; r < rows; ++r) for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int col = 0; col < k; ++col)
            s += double(__half2float(__float2half_rn(x[size_t(r) * ldx + col]))) * w.f[size_t(j) * k + col];
        y[size_t(r) * n + j] = float(s);
    }
    return y;
}
void near(const std::vector<float>& got, const std::vector<float>& want, const char* name) {
    require(got.size() == want.size(), "length mismatch");
    double error = 0, norm = 0, peak = 0, worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]), "nonfinite output");
        const double d = double(got[i]) - want[i];
        error += d*d; norm += double(want[i])*want[i];
        worst = std::max(worst, std::abs(d)); peak = std::max(peak, std::abs(double(want[i])));
    }
    const double rel = std::sqrt(error / std::max(norm, 1e-20));
    std::printf("%s rel_l2=%.6g max_abs=%.6g peak=%.6g\n", name, rel, worst, peak);
    // Independent raw-GGUF F32 oracle: the library's FP16 affine arithmetic
    // is not bit-identical to ggml dequantization. Fixed bounds, never fitted
    // from the output; the box run must admit these numerically.
    require(rel < 0.02 && worst < 0.002 + 0.03 * peak, "numerical parity failed");
}
struct Q81 { __half d, sum; int8_t q[32]; };
static_assert(sizeof(Q81) == 36, "Q8_1 ABI");
std::vector<Q81> activations(int rows, int k, std::vector<float>& x, int seed = 0) {
    std::vector<Q81> a(size_t(rows) * k / 32); x.resize(size_t(rows) * k);
    for (size_t b = 0; b < a.size(); ++b) {
        a[b].d = __float2half_rn(float(b % 3 + 1) / 512);
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            a[b].q[j] = int((b * 13 + j * 7 + seed * 3) % 127) - 63;
            x[b * 32 + j] = __half2float(a[b].d) * a[b].q[j]; sum += a[b].q[j];
        }
        a[b].sum = __float2half_rn(__half2float(a[b].d) * sum);
    }
    return a;
}
void dense_cases(cudaStream_t stream) {
    constexpr int n = 256, k = 512, m = 3, ldx = k + 32, ldy = n + 16;
    std::vector<float> x(m * ldx);
    for (size_t i = 0; i < x.size(); ++i) x[i] = value(i);
    Device<float> dx(x), y(m * ldy);
    for (int q : {8,10,11,12,13,14}) {
        Weight w(q,n,k,q); Device<uint8_t> dw(w.raw);
        const auto ref = product(x,w,m,n,k,ldx);
        require(qz::preload(dw.p,q,n,k) == w.raw.size(), "load-time shuffle declined");
        std::vector<float> old(m * ldy, 0.125f); y.put(old);
        require(qz::dense(dw.p,q,n,k,dx.p,qz::Input::f32,m,y.p,stream,ldx,ldy,0.5f), "resident dense declined");
        ck(cudaStreamSynchronize(stream));
        auto got = y.get(old.size());
        std::vector<float> want = old;
        for (int r = 0; r < m; ++r) for (int j = 0; j < n; ++j) want[r*ldy+j] = ref[r*n+j] + 0.0625f;
        near(got,want,"resident strided F32 + beta");
        require(dw.get(w.raw.size()) == w.raw, "load shuffle altered original GGUF");
        qz::release(dw.p);
        // Reuse the address with different contents: a stale resident artifact
        // would pass a shape-only test but fail this replacement check.
        Weight replacement(q,n,k,q+1); dw.put(replacement.raw);
        std::vector<float> qx; auto xq = activations(m,k,qx); Device<Q81> dq(xq);
        require(qz::dense(dw.p,q,n,k,dq.p,qz::Input::q8_1,m,y.p,stream,0,n), "Q8_1 dense declined");
        ck(cudaStreamSynchronize(stream));
        near(y.get(m*n),product(qx,replacement,m,n,k,k),"uncached Q8_1 / reused address");
    }
    // Actual merged Flash-Next gate/up geometry: each half has 640 rows,
    // while the contiguous merged matrix satisfies the N=1280 contract.
    constexpr int gn = 1280, gk = 2560;
    Weight w(12,gn,gk); Device<uint8_t> dw(w.raw); Device<float> gy(m*gn);
    std::vector<float> xf(m*gk); std::vector<__half> xh(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = __half2float(xh[i] = __float2half_rn(value(i)));
    Device<__half> dh(xh);
    require(qz::dense(dw.p,12,gn,gk,dh.p,qz::Input::f16,m,gy.p,stream,0,0,0,true), "merged gate/up declined");
    ck(cudaStreamSynchronize(stream));
    auto ref = product(xf,w,m,gn,gk,gk); std::vector<float> want(ref.size());
    for (int r = 0; r < m; ++r) for (int j = 0; j < gn; ++j)
        want[r*gn+2*(j%(gn/2))+j/(gn/2)] = ref[r*gn+j];
    near(gy.get(want.size()),want,"interleaved merged gate/up");
}

void grouped_case(cudaStream_t stream, bool graph, bool down) {
    const int n = down ? 256 : 1280, k = down ? 512 : 2560, q = down ? 14 : 12;
    constexpr int cap = 11, rows = 17; // two batches, padded entries, empty groups
    Weight w0(q,n,k,0), w1(q,n,k,1);
    const size_t offset = 256;
    std::vector<uint8_t> blob0(offset+w0.raw.size()), blob1(offset+w1.raw.size());
    std::copy(w0.raw.begin(),w0.raw.end(),blob0.begin()+offset);
    std::copy(w1.raw.begin(),w1.raw.end(),blob1.begin()+offset);
    Device<uint8_t> d0(blob0), d1(blob1);
    std::vector<unsigned long long> ptr(cap);
    for (int i = 0; i < cap; ++i) ptr[i] = (unsigned long long) (i%2 ? d1.p : d0.p);
    ptr[1] = 0; // Empty groups must not dereference their weight pointer.
    std::vector<int32_t> start{0,2,2,3,5,6,8,9,10,11,13,13}, count{10}, tok(rows), dst(rows);
    for (int i = 0; i < rows; ++i) { tok[i] = (i*3)%rows; dst[i] = rows-1-i; }
    std::vector<float> x; auto xq = activations(rows,k,x);
    Device<Q81> dx(xq); Device<unsigned long long> dp(ptr);
    Device<int32_t> ds(start), dc(count), dt(tok), dd(dst);
    Device<float> out(size_t(rows)*n), up(size_t(rows)*n/2);
    auto submit = [&] {
        require(qz::grouped(dp.p,ds.p,dc.p,down ? nullptr : dt.p,down ? dd.p : nullptr,cap,rows,
            dx.p,offset,q,n,k,out.p,down ? nullptr : up.p,stream), "grouped adapter declined");
    };
    submit(); ck(cudaStreamSynchronize(stream));
    cudaGraph_t g = nullptr; cudaGraphExec_t exec = nullptr;
    cudaStream_t replay; ck(cudaStreamCreateWithFlags(&replay,cudaStreamNonBlocking));
    if (graph) {
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); submit();
        ck(cudaStreamEndCapture(stream,&g)); ck(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
    }
    for (int trial = 0; trial < 3; ++trial) {
        if (trial == 1) {
            // Same pointer, new expert contents and routing on graph replay.
            w0 = Weight(q,n,k,7);
            std::copy(w0.raw.begin(),w0.raw.end(),blob0.begin()+offset); d0.put(blob0);
            std::swap(ptr[0],ptr[9]); dp.put(ptr);
            count[0] = 9; dc.put(count);
            xq = activations(rows,k,x,5); dx.put(xq);
        } else if (trial == 2) { count[0] = 0; dc.put(count); }
        std::vector<float> sentinel(size_t(rows)*n,-17); out.put(sentinel);
        if (graph) { ck(cudaGraphLaunch(exec,replay)); ck(cudaStreamSynchronize(replay)); }
        else { submit(); ck(cudaStreamSynchronize(stream)); }
        std::vector<float> want(down ? size_t(rows)*n : size_t(rows)*n/2,down ? -17 : 0), wu(size_t(rows)*n/2);
        for (int group = 0; group < count[0]; ++group) {
            const Weight& w = ptr[group] == (unsigned long long) d0.p ? w0 : w1;
            for (int e = start[group]; e < start[group+1]; ++e) {
                const int source = down ? e : tok[e];
                std::vector<float> row(x.begin()+size_t(source)*k,x.begin()+size_t(source+1)*k);
                auto y = product(row,w,1,n,k,k);
                if (down) std::copy(y.begin(),y.end(),want.begin()+size_t(dst[e])*n);
                else for (int j = 0; j < n/2; ++j) { want[e*(n/2)+j]=y[j]; wu[e*(n/2)+j]=y[n/2+j]; }
            }
        }
        auto got = out.get(want.size());
        if (down) {
            std::vector<float> active_got, active_want;
            for (size_t i = 0; i < want.size(); ++i) {
                if (want[i] == -17) require(got[i] == -17,"scatter overwrote an unrouted row");
                else { active_got.push_back(got[i]); active_want.push_back(want[i]); }
            }
            near(active_got,active_want,"grouped down scatter");
        } else near(got,want,"grouped gate");
        if (!down) near(up.get(wu.size()),wu,"grouped up");
    }
    if (exec) ck(cudaGraphExecDestroy(exec)); if (g) ck(cudaGraphDestroy(g));
    ck(cudaStreamDestroy(replay));
}

void resident_graph(cudaStream_t stream) {
    constexpr int n = 256, k = 512, m = 2;
    Weight w(12,n,k); Device<uint8_t> dw(w.raw); Device<float> dx(m*k), dy(m*n);
    require(qz::preload(dw.p,12,n,k) == w.raw.size(),"resident graph preload declined");
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    require(qz::dense(dw.p,12,n,k,dx.p,qz::Input::f32,m,dy.p,stream),"resident capture declined");
    ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    cudaStream_t replay; ck(cudaStreamCreateWithFlags(&replay,cudaStreamNonBlocking));
    for (int trial = 0; trial < 2; ++trial) {
        std::vector<float> x(m*k);
        for (size_t i = 0; i < x.size(); ++i) x[i] = value(i,trial);
        dx.put(x); ck(cudaGraphLaunch(exec,replay)); ck(cudaStreamSynchronize(replay));
        near(dy.get(m*n),product(x,w,m,n,k,k),"resident ready-event graph replay");
    }
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(replay));
    qz::release(dw.p);
}

void fallback_case(cudaStream_t stream, bool budget) {
    Weight w(12,256,512); Device<uint8_t> dw(w.raw);
    Device<float> dx(512), dy(std::vector<float>(256,17));
    if (!budget) {
        require(!qz::loaded(),"cold-capture fixture already loaded a DSO");
        cudaGraph_t g;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        require(!qz::dense(dw.p,12,256,512,dx.p,qz::Input::f32,1,dy.p,stream),"cold capture did not decline");
        require(!qz::loaded(),"initialized a shared library during capture");
        ck(cudaStreamEndCapture(stream,&g)); ck(cudaGraphDestroy(g));
        qz::library();
    }
    if (budget) require(qz::preload(dw.p,12,256,512) == 0,"zero cache budget ignored");
    require(!qz::dense(dw.p,budget ? 12 : 21,256,512,dx.p,qz::Input::f32,1,dy.p,stream),"decline expected");
    if (!budget) {
        require(!qz::dense(dw.p,12,256,640,dx.p,qz::Input::f32,1,dy.p,stream),"unsupported K accepted");
        cudaGraph_t g;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        require(!qz::dense(dw.p,12,256,512,dx.p,qz::Input::q8_1,1,dy.p,stream),"allocated scratch during capture");
        ck(cudaStreamEndCapture(stream,&g)); ck(cudaGraphDestroy(g));
    }
    ck(cudaStreamSynchronize(stream));
    require(dy.get(256) == std::vector<float>(256,17),"declined call touched output");
}
}
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr,"usage: quactlize_parity dense|grouped|graph|fallback|budget\n"); return 2; }
    const std::string mode = argv[1];
    setenv("STRATA_QUACTLIZE","1",1);
    if (mode == "budget") { setenv("STRATA_QUACTLIZE_CACHE_MIB","0",1); setenv("STRATA_QUACTLIZE_SCRATCH_MIB","0",1); }
    try {
        int ndev = 0; auto rc = cudaGetDeviceCount(&ndev);
        if (rc != cudaSuccess || !ndev) { std::fprintf(stderr,"SKIP: no usable CUDA/PPU device\n"); return 77; }
        if (mode != "fallback") qz::library(); // The fallback fixture first checks a cold capture.
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        if (mode == "dense") dense_cases(stream);
        else if (mode == "grouped" || mode == "graph") {
            grouped_case(stream,mode=="graph",false); grouped_case(stream,mode=="graph",true);
            if (mode == "graph") resident_graph(stream);
        } else if (mode == "fallback" || mode == "budget") fallback_case(stream,mode=="budget");
        else throw std::runtime_error("unknown case");
        ck(cudaStreamSynchronize(stream)); ck(cudaStreamDestroy(stream));
        std::printf("PASS quactlize %s\n",mode.c_str()); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
