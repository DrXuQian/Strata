// Physical-device integration check. package plus adapter packing launches several
// kernels; this executable is not a single-launch simulator harness.
#include "strata/ppu/kernels.hpp"
#include "strata/ppu/python.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    std::vector<T> read(size_t n) const {
        std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
void near(const std::vector<float>& got, const std::vector<float>& want, float atol, float rtol) {
    if (got.size() != want.size()) throw std::runtime_error("size mismatch");
    for (size_t i = 0; i < got.size(); ++i)
        if (!std::isfinite(got[i]) || std::abs(got[i] - want[i]) > atol + rtol * std::abs(want[i]))
            throw std::runtime_error("numerical mismatch at " + std::to_string(i) + ": " +
                std::to_string(got[i]) + " vs " + std::to_string(want[i]));
}
float value(size_t i) { return float(int((i * 17 + i / 7) % 101) - 50) / 256.0f; }
float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

int gdn_case(bool commit, bool chunked, bool graph, cudaStream_t stream, bool prefix=false, bool partial=false) {
    constexpr int H = 16, HV = 48, S = 128, C = (2 * H + HV) * S;
    const int T = chunked ? 320 : 3; // Exercises the 256 + 64 chunk boundary.
    std::vector<float> h(size_t(T) * C), gate(T * HV), beta(T * HV), state(S * HV * S);
    for (size_t i = 0; i < state.size(); ++i) state[i] = value(i) * 0.03f;
    for (size_t i = 0; i < h.size(); ++i) h[i] = value(i + i / S);
    for (int t = 0; t < T; ++t)
        for (int head = 0; head < 2 * H; ++head) {
            double norm = 0;
            for (int k = 0; k < S; ++k) norm += double(h[t * C + head * S + k]) * h[t * C + head * S + k];
            for (int k = 0; k < S; ++k) h[t * C + head * S + k] /= float(std::sqrt(norm));
        }
    for (int i = 0; i < T * HV; ++i) { gate[i] = -0.03f - 0.01f * (i % 7); beta[i] = 0.1f + 0.01f * (i % 23); }
    std::vector<double> ref(state.begin(), state.end());
    std::vector<float> expected(size_t(T) * HV * S);
    const float scale = 1.0f / std::sqrt(float(S));
    std::vector<std::vector<float>> states;
    if(prefix)states.push_back(state);
    for (int t = 0; t < T; ++t) {
      for (int head = 0; head < HV; ++head) for (int v = 0; v < S; ++v) {
        const float* ht = h.data() + t * C;
        const int qh = head % H;
        const double decay = std::exp(double(gate[t * HV + head]));
        double dot = 0;
        for (int k = 0; k < S; ++k) {
            const float key = ht[H * S + qh * S + k];
            dot += ref[(k * HV + head) * S + v] * bf(key);
        }
        const float val = ht[2 * H * S + head * S + v];
        const double delta = (bf(val) - decay * dot) * bf(beta[t * HV + head]);
        double out = 0;
        for (int k = 0; k < S; ++k) {
            const float key = ht[H * S + qh * S + k], query = ht[qh * S + k];
            double& cell = ref[(k * HV + head) * S + v];
            cell = decay * cell + bf(key) * delta;
            out += cell * bf(query);
        }
        expected[(t * HV + head) * S + v] = float(out * scale);
      }
      if(prefix)states.emplace_back(ref.begin(),ref.end());
    }
    Device<float> ds(state), dh(h), dg(gate), db(beta), out(expected.size());
    Device<int32_t> accepted(std::vector<int32_t>{T});
    const std::vector<float> untouched(expected.size(),17.f);
    ck(cudaMemcpy(out.p,untouched.data(),untouched.size()*4,cudaMemcpyHostToDevice));
    if (graph) {
        if (!strata::ppu::prepare_gdn(T, H, HV, S, stream)) throw std::runtime_error("GDN graph scratch unavailable");
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        if (strata::ppu::gdn(ds.p, nullptr, dh.p, dh.p + H * S, dh.p + 2 * H * S,
                C, C, C, dg.p, db.p, out.p, T, H, HV, S, scale, false, stream))
            throw std::runtime_error("GDN captured scratch without explicit replay-stream ownership");
    }
    const bool used = strata::ppu::gdn(ds.p, commit ? ds.p : nullptr, dh.p, dh.p + H * S, dh.p + 2 * H * S,
        C, C, C, dg.p, db.p, out.p, T, H, HV, S, scale, chunked, stream, graph,
        prefix?accepted.p:nullptr,partial?1:0);
    if (graph) {
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        ck(cudaStreamEndCapture(stream, &captured));
        ck(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        if(prefix) {
            // Same captured graph, changing accepted lengths on the device.
            for(int kept : {0,1,T}) {
                ck(cudaMemcpy(accepted.p,&kept,4,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(ds.p,state.data(),state.size()*4,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(out.p,untouched.data(),untouched.size()*4,cudaMemcpyHostToDevice));
                ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));
                const auto got_state=ds.read(state.size());
                if(!kept && got_state!=state)throw std::runtime_error("zero commit modified state");
                near(got_state,states[kept],2e-3f,3e-2f);
                const auto y=out.read(expected.size());
                const size_t end=size_t(kept)*HV*S;
                near(std::vector<float>(y.begin(),y.begin()+end),
                     std::vector<float>(expected.begin(),expected.begin()+end),2e-3f,3e-2f);
                for(size_t i=end;i<y.size();++i)if(y[i]!=17.f)
                    throw std::runtime_error("commit wrote rejected output rows");
            }
        } else {
            ck(cudaGraphLaunch(executable, stream));
            ck(cudaStreamSynchronize(stream));
        }
        ck(cudaGraphExecDestroy(executable));
        ck(cudaGraphDestroy(captured));
    }
    if (!used) throw std::runtime_error("package library loaded but did not handle the GDN fixture");
    ck(cudaStreamSynchronize(stream));
    if(partial)std::fill(expected.begin(),expected.begin()+HV*S,17.f);
    near(out.read(expected.size()), expected, 2e-3f, 3e-2f);
    if (commit) near(ds.read(state.size()), std::vector<float>(ref.begin(), ref.end()), 2e-3f,
                     3e-2f);
    else if (ds.read(state.size()) != state) throw std::runtime_error("verification modified carried state");
    return 0;
}

int moe_case(cudaStream_t stream) {
    constexpr int T = 7, N = 1280, K = 2560;
    std::vector<uint16_t> x(T * K), w(N * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = __half_as_ushort(__float2half_rn(value(i)));
    for (size_t i = 0; i < w.size(); ++i) w[i] = __half_as_ushort(__float2half_rn(value(i + 59)));
    std::vector<float> ref(T * N);
    for (int t = 0; t < T; ++t) for (int n = 0; n < N; ++n) {
        double sum = 0;
        for (int k = 0; k < K; ++k)
            sum += double(bf(__half2float(__ushort_as_half(x[t * K + k])))) * bf(__half2float(__ushort_as_half(w[n * K + k])));
        ref[t * N + n] = bf(float(sum));
    }
    Device<uint16_t> dx(x), dw(w); Device<float> out(T * N);
    if (!strata::ppu::expert_f16(dx.p, dw.p, out.p, T, N, K, stream))
        throw std::runtime_error("package library loaded but did not handle the expert fixture");
    ck(cudaStreamSynchronize(stream));
    near(out.read(T * N), ref, 0.02f, 0.01f);
    return 0;
}

int fa_case(cudaStream_t stream, bool quantized) {
    constexpr int T = 2, H = 24, HK = 2, D = 256, P = 4, CAP = 5;
    auto shape = strata::kernels::qsa_real_shapes(); shape.page_size = P;
    std::vector<int32_t> table{2, 0, 1}, ids{0, 5, 8, 0, 0, 1, 3, 6, 8, 9};
    std::vector<int32_t> steps(T * strata::kernels::kStepCount);
    steps[strata::kernels::kStepWidth] = 3;
    steps[strata::kernels::kStepCount + strata::kernels::kStepWidth] = 5;
    std::vector<float> q(T * H * D);
    std::vector<uint16_t> k(3 * HK * P * D), v(k.size());
    std::vector<int8_t> kq(k.size()), vq(k.size());
    std::vector<uint16_t> ks(k.size() / 64), vs(ks.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = value(i + 17);
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __half_as_ushort(__float2half_rn(value(i + 31)));
        v[i] = __half_as_ushort(__float2half_rn(value(i + 73)));
    }
    if (quantized) {
        for (size_t i = 0; i < ks.size(); ++i) {
            ks[i] = __half_as_ushort(__float2half_rn(0.001f * (1 + i % 7)));
            vs[i] = __half_as_ushort(__float2half_rn(0.002f * (1 + i % 5)));
        }
        for (size_t i = 0; i < k.size(); ++i) {
            kq[i] = int8_t(int(i * 11 % 127) - 63); vq[i] = int8_t(int(i * 17 % 127) - 63);
            k[i] = __half_as_ushort(__float2half_rn(kq[i] * __half2float(__ushort_as_half(ks[i / 64]))));
            v[i] = __half_as_ushort(__float2half_rn(vq[i] * __half2float(__ushort_as_half(vs[i / 64]))));
        }
    }
    std::vector<float> ref(q.size());
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        const int width = steps[t * strata::kernels::kStepCount + strata::kernels::kStepWidth];
        double scores[CAP], denom = 0;
        for (int j = 0; j < width; ++j) {
            const int cell = ids[t * CAP + j], row = (table[cell / P] * HK + h / 12) * P + cell % P;
            double sum = 0;
            for (int d = 0; d < D; ++d) sum += double(__half2float(__float2half_rn(q[(t * H + h) * D + d]))) *
                __half2float(__ushort_as_half(k[row * D + d]));
            scores[j] = std::exp(sum / 16.0); denom += scores[j];
        }
        for (int d = 0; d < D; ++d) {
            double sum = 0;
            for (int j = 0; j < width; ++j) {
                const int cell = ids[t * CAP + j], row = (table[cell / P] * HK + h / 12) * P + cell % P;
                sum += scores[j] * __half2float(__ushort_as_half(v[row * D + d]));
            }
            ref[(t * H + h) * D + d] = float(sum / denom);
        }
    }
    Device<float> dq(q), out(q.size()); Device<uint16_t> dk(k), dv(v); Device<int32_t> dt(table), di(ids);
    Device<int8_t> dkq(kq), dvq(vq); Device<uint16_t> dks(ks), dvs(vs);
    strata::kernels::QsaAttnPools pools; pools.k_pool = dk.p; pools.v_pool = dv.p; pools.page_table = dt.p;
    if (quantized) {
        pools.k_pool = pools.v_pool = nullptr;
        pools.k_q = dkq.p; pools.v_q = dvq.p; pools.k_scale = dks.p; pools.v_scale = dvs.p;
    }
    if (!strata::ppu::qsa_prefill(dq.p, pools, di.p, steps.data(), CAP, shape, out.p, T, stream))
        throw std::runtime_error("package library loaded but did not handle the QSA fixture");
    ck(cudaStreamSynchronize(stream)); near(out.read(q.size()), ref, 5e-4f, 5e-3f);
    return 0;
}
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: ppu_package_parity gdn-verify|gdn-commit|gdn-graph|gdn-prefix|gdn-partial|gdn-chunked|moe|fa|fa-q8\n"); return 2; }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::puts("SKIP: no CUDA-compatible device"); return 77; }
    try {
        cudaStream_t stream; ck(cudaStreamCreate(&stream));
        const std::string mode = argv[1];
        int rc = 2;
        if (mode == "gdn-verify") rc = gdn_case(false, false, false, stream);
        else if (mode == "gdn-commit") rc = gdn_case(true, false, false, stream);
        else if (mode == "gdn-graph") rc = gdn_case(false, false, true, stream);
        else if (mode == "gdn-prefix") rc = gdn_case(true,false,true,stream,true);
        else if (mode == "gdn-partial") rc = gdn_case(false,false,true,stream,false,true);
        else if (mode == "gdn-chunked") rc = gdn_case(true, true, false, stream);
        else if (mode == "moe") rc = moe_case(stream);
        else if (mode == "fa" || mode == "fa-q8") rc = fa_case(stream, mode == "fa-q8");
        ck(cudaStreamDestroy(stream));
        std::puts(rc == 0 ? "PASS" : rc == 77 ? "SKIP: no device" : "unknown case");
        return rc;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
