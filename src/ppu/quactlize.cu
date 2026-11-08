#include "strata/ppu/quactlize.hpp"
#include "strata/ppu/quactlize_graph.hpp"
#include "strata/ppu/offline.hpp"
#include "strata/ppu/python.hpp"
#include "strata/core/dispatch_stats.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strata::ppu::quactlize {
namespace {
void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string("quactlize: ") + cudaGetErrorString(rc));
}
size_t align(size_t n) {
    if (n > SIZE_MAX - 255) throw std::overflow_error("quactlize: allocation overflow");
    return (n + 255) & ~size_t(255);
}
size_t budget(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return size_t(256) << 20;
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(v, &end, 10);
    if (*v == '-' || !end || *end || mb > (SIZE_MAX >> 24))
        throw std::runtime_error(std::string("quactlize: invalid MiB budget ") + name);
    return size_t(mb) << 20;
}
int device() { int d; check(cudaGetDevice(&d)); return d; }
bool capturing(void* stream) {
    cudaStreamCaptureStatus state;
    check(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &state));
    return state != cudaStreamCaptureStatusNone;
}
bool may_load(void* stream) {
    if (loaded()) return true;
    cudaStreamCaptureStatus state;
    check(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &state));
    return state == cudaStreamCaptureStatusNone; // No first-time DSO/runtime initialization during capture.
}
Call call(int type, int n, int k, int rows, int experts = 1) {
    Call c{};
    c.qtype = type; c.n = n; c.k = k; c.experts = experts; c.rows = rows;
    c.input_type = F32;
    c.a_row_stride = k; c.out_row_stride = n;
    return c;
}
void planes(Call& c, void* p, const Shape& s) {
    c.low = static_cast<uint8_t*>(p);
    c.high = s.sizes.high_bytes ? c.low + s.sizes.low_bytes : nullptr;
    c.units = c.low + s.sizes.low_bytes + s.sizes.high_bytes;
}
void pack(const void* raw, const Call& c, const Shape& s) {
    if (offline::enabled()) throw std::runtime_error("quactlize: runtime shuffle is forbidden in offline mode");
    package_pack(raw,c,s,device());
}
struct Artifact {
    int type, n, k;
    Shape shape;
    void* data = nullptr;
    cudaEvent_t ready = nullptr;
    ~Artifact() {
        if (ready) { cudaEventSynchronize(ready); cudaEventDestroy(ready); }
        if (data) cudaFree(data);
    }
};
using WeightKey = std::pair<int, uintptr_t>;
struct Resident {
    std::mutex mutex;
    std::map<WeightKey, std::shared_ptr<Artifact>> weights;
    std::map<int, size_t> bytes;
};
Resident& resident() { static auto* r = new Resident; return *r; }
std::shared_ptr<Artifact> find(const void* p) {
    auto& r = resident();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto it = r.weights.find({device(), reinterpret_cast<uintptr_t>(p)});
    return it == r.weights.end() ? nullptr : it->second;
}
struct Scratch {
    std::mutex mutex;
    void* data = nullptr;
    size_t bytes = 0;
    std::vector<void*> retired;
};
struct Pools {
    std::mutex mutex;
    std::map<std::tuple<int, uintptr_t, int>, std::unique_ptr<Scratch>> pools;
    std::map<int, size_t> bytes;
};
Pools& pools() { static auto* p = new Pools; return *p; }
Scratch& scratch(float* output, int kind) {
    auto& p = pools();
    std::lock_guard<std::mutex> lock(p.mutex);
    auto& s = p.pools[{device(), reinterpret_cast<uintptr_t>(output), kind}];
    if (!s) s = std::make_unique<Scratch>();
    return *s;
}
bool reserve(Scratch& s, size_t need, cudaStream_t stream) {
    if (need <= s.bytes) return true;
    cudaStreamCaptureStatus state;
    check(cudaStreamIsCapturing(stream, &state));
    if (state != cudaStreamCaptureStatusNone) {
        strata::dispatch::note("quactlize: capture needs unprepared scratch; original kernels");
        return false;
    }
    auto& p = pools();
    std::lock_guard<std::mutex> lock(p.mutex);
    const int dev = device();
    static const size_t limit = budget("STRATA_QUACTLIZE_SCRATCH_MIB");
    if (need > limit - std::min(limit, p.bytes[dev])) {
        strata::dispatch::note("quactlize: scratch budget exhausted; original kernels");
        return false;
    }
    void* next = nullptr;
    cudaError_t rc = cudaMalloc(&next, need);
    if (rc == cudaErrorMemoryAllocation) { cudaGetLastError(); return false; }
    check(rc);
    // Captured graphs keep old addresses. Storage belongs to the caller's
    // output, not a recyclable temporary stream handle. Concurrent use of the
    // same output is already excluded by the Strata caller's ordering contract.
    if (s.data) s.retired.push_back(s.data);
    s.data = next; s.bytes = need; p.bytes[dev] += need;
    return true;
}
struct Arena {
    uint8_t* p;
    template<class T> T* take(size_t n) {
        T* r = reinterpret_cast<T*>(p); p += align(n * sizeof(T)); return r;
    }
};
struct Q81 { __half d, sum; int8_t qs[32]; };
static_assert(sizeof(Q81) == 36, "Strata Q8_1 ABI");
__global__ void expand_q81(const Q81* x, float* y, int k, int rows) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < int64_t(k) * rows) y[i] = __half2float(x[i / 32].d) * float(x[i / 32].qs[i % 32]);
}
__global__ void finish(const float* x, float* y, int n, int rows, int64_t ldy, float beta, bool interleaved) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(n) * rows) return;
    const int j = i % n;
    const int dst = interleaved ? 2 * (j % (n / 2)) + j / (n / 2) : j;
    float* at = y + (i / n) * ldy + dst;
    *at = beta == 0 ? x[i] : x[i] + beta * *at;
}
// Packed prompt experts write into different slices of the same prompt buffer.
// Reorder each row in place so those slices do not each retain a temporary.
__global__ void interleave_rows(float* y, int n, int64_t ldy) {
    extern __shared__ float row[];
    float* dst = y + int64_t(blockIdx.x) * ldy;
    for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] = dst[j];
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        dst[2 * (j % (n / 2)) + j / (n / 2)] = row[j];
}
constexpr int kGroupBatch = 8;
__global__ void gather_packed(const unsigned long long* ptr,const int32_t* start,const int32_t* count,
    uint8_t* dst,size_t low,size_t high,size_t units,size_t off,int batch,int cap) {
    const size_t bytes=low+high+units,i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=bytes*kGroupBatch)return;
    const int e=i/bytes,g=batch+e;const size_t j=i%bytes;
    const size_t at=j<low ? size_t(e)*low+j : j<low+high ? low*kGroupBatch+size_t(e)*high+j-low
        : (low+high)*kGroupBatch+size_t(e)*units+j-low-high;
    dst[at]=g<cap&&g<*count&&start[g]<start[g+1]
        ? reinterpret_cast<const uint8_t*>(ptr[g])[off+j] : 0;
}
__global__ void gather_weights(const unsigned long long* ptr, const int32_t* start, const int32_t* count, uint8_t* raw,
                               size_t bytes, size_t off, int batch, int cap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= bytes * kGroupBatch) return;
    const int g = batch + i / bytes;
    raw[i] = g < cap && g < *count && start[g] < start[g + 1]
        ? reinterpret_cast<const uint8_t*>(ptr[g])[off + i % bytes] : 0;
}
__global__ void entry_groups(const int32_t* start, const int32_t* count, int32_t* group, int rows, int cap) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= rows) return;
    const int ng = min(max(*count, 0), cap);
    if (!ng || e >= start[ng]) { group[e] = -1; return; }
    int lo = 0, hi = ng;
    while (lo + 1 < hi) { int m = (lo + hi) / 2; if (start[m] <= e) lo = m; else hi = m; }
    group[e] = lo;
}
__global__ void gather_x(const Q81* x, const int32_t* tok, const int32_t* group,
                         float* a, int32_t* ids, int rows, int k, int batch) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(rows) * k) return;
    const int e = i / k, col = i % k, g = group[e] - batch;
    const bool valid = g >= 0 && g < kGroupBatch;
    if (col == 0) ids[e] = valid ? g : 0;
    if (!valid) { a[i] = 0; return; }
    const int64_t j = int64_t(tok ? tok[e] : e) * (k / 32) + col / 32;
    a[i] = __half2float(x[j].d) * float(x[j].qs[col % 32]);
}
__global__ void scatter_group(const float* y, const int32_t* group, const int32_t* dst,
                              float* out, float* up, int rows, int n, int batch) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(rows) * n) return;
    const int e = i / n, col = i % n, g = group[e] - batch;
    if (g < 0 || g >= kGroupBatch) return;
    if (up) (col < n / 2 ? out : up)[int64_t(e) * (n / 2) + col % (n / 2)] = y[i];
    else out[int64_t(dst[e]) * n + col] = y[i];
}
unsigned blocks(size_t elems) { return static_cast<unsigned>((elems + 255) / 256); }
// One grouped call's scratch: eight experts' planes (and their raw source), activations, outputs and ids.
bool grouped_scratch(const Shape& shape, int cap_entries, int n, int k, bool packed_input, size_t& need) {
    const size_t nx = size_t(cap_entries) * k, ny = size_t(cap_entries) * n;
    static const size_t limit = budget("STRATA_QUACTLIZE_SCRATCH_MIB");
    if (shape.sizes.raw_bytes > limit || nx > limit / 4 || ny > limit / 4) return false;
    const size_t capacity = (size_t(cap_entries) + 255) / 256 * 256;
    need = (packed_input ? 1 : 2) * align(shape.sizes.raw_bytes) + align(capacity * k * 4) + align(capacity * n * 4) +
        2 * align(capacity * 4);
    return true;
}
}

size_t preload(const void* raw, int type, int n, int k, const char* name) {
    if (!enabled() || !raw) return 0;
    Call c = call(type, n, k, 1);
    Shape shape;
    if (!describe(c, shape)) return 0;
    // The verifier can capture without an eager grouped call. Prepare every
    // supported format before that point, including formats only in experts.
    // Python retains a per-device once flag; no repeated allocation or JIT.
    python::call("prepare_quact",{},{},device(),nullptr);
    const offline::Record* disk=nullptr;
    if (offline::enabled()) {
        disk=name?offline::store().dense(name):nullptr;
        if(!disk)return 0;
        if(disk->type!=type||disk->n!=n||disk->k!=k||std::memcmp(&shape,&disk->shape,sizeof(shape)))
            throw std::runtime_error("quactlize: offline dense geometry/layout mismatch");
    }
    auto& r = resident();
    std::lock_guard<std::mutex> lock(r.mutex);
    const int dev = device();
    WeightKey key{dev, reinterpret_cast<uintptr_t>(raw)};
    if (r.weights.count(key)) throw std::runtime_error("quactlize: duplicate immutable weight registration");
    static const size_t limit = budget("STRATA_QUACTLIZE_CACHE_MIB");
    const size_t bytes = shape.sizes.raw_bytes;
    if (bytes > limit - std::min(limit, r.bytes[dev])) return 0;
    auto a = std::make_shared<Artifact>();
    a->type = type; a->n = n; a->k = k; a->shape = shape;
    cudaError_t rc = cudaMalloc(&a->data, bytes);
    if (rc == cudaErrorMemoryAllocation) { cudaGetLastError(); return 0; }
    check(rc);
    check(cudaEventCreateWithFlags(&a->ready, cudaEventDisableTiming));
    planes(c, a->data, shape);
    if(disk)check(cudaMemcpy(a->data,offline::store().data(*disk),bytes,cudaMemcpyHostToDevice));
    else pack(raw, c, shape);
    check(cudaEventRecord(a->ready, nullptr));
    r.weights.emplace(key, a); r.bytes[dev] += bytes;
    std::fprintf(stderr, "quactlize: %s type=%d N=%d K=%d mapping=%llx +%zu bytes\n",
                 disk?"offline load":"load shuffle",type,n,k,(unsigned long long)shape.arrangement.mapping_id,bytes);
    return bytes;
}
void release(const void* raw) noexcept {
    if (!enabled() || !raw) return;
    auto& r = resident();
    std::lock_guard<std::mutex> lock(r.mutex);
    int current;
    if (cudaGetDevice(&current) != cudaSuccess) return;
    auto it = r.weights.find({current, reinterpret_cast<uintptr_t>(raw)});
    if (it == r.weights.end()) {
        it = std::find_if(r.weights.begin(), r.weights.end(), [raw](const auto& item) {
            return item.first.second == reinterpret_cast<uintptr_t>(raw);
        });
    }
    if (it == r.weights.end()) return;
    const int owner = it->first.first;
    if (owner != current && cudaSetDevice(owner) != cudaSuccess) return;
    r.bytes[owner] -= it->second->shape.sizes.raw_bytes;
    r.weights.erase(it);
    if (owner != current) cudaSetDevice(current);
}

bool dense(const void* raw, int type, int n, int k, const void* x, Input input,
           int rows, float* out, void* stream, int64_t ldx, int64_t ldy, float beta, bool interleaved, bool packed_input) {
    if (!enabled() || !raw || !x || !out || rows <= 0 || n <= 0 || k <= 0) return false;
    if (ldx <= 0) ldx = k;
    if (ldy <= 0) ldy = n;
    if (ldx < k || ldy < n || (input == Input::q8_1 && (ldx != k || k % 32)) || (interleaved && n % 2)) return false;
    if (!may_load(stream)) return false;
    Call c = call(type, n, k, rows);
    c.capturing=capturing(stream);
    c.input_type = input == Input::f16 ? F16 : F32;
    c.a_row_stride = ldx;
    const bool in_place = packed_input && interleaved && beta == 0 && n <= 4096;
    const bool finish_output = beta != 0 || (interleaved && !in_place);
    c.out_row_stride = finish_output ? n : ldy;
    Shape shape;
    if (!describe(c, shape)) return false;
    const auto a = packed_input ? std::shared_ptr<Artifact>() : find(raw);
    if(offline::enabled()&&!packed_input&&!a)return false;
    // A prefix/submatrix may share the raw base address with a larger resident
    // tensor. Its plane strides differ: leave that view to the raw reader.
    if (a && (a->type != type || a->n != n || a->k != k)) return false;
    const size_t nx = size_t(rows) * k, ny = size_t(rows) * n;
    static const size_t limit = budget("STRATA_QUACTLIZE_SCRATCH_MIB");
    if ((!a && !packed_input && shape.sizes.raw_bytes > limit) ||
        (input == Input::q8_1 && nx > limit / sizeof(float)) ||
        (finish_output && ny > limit / sizeof(float))) return false;
    const size_t need = (a || packed_input ? 0 : align(shape.sizes.raw_bytes)) +
        (input == Input::q8_1 ? align(nx * sizeof(float)) : 0) + (finish_output ? align(ny * sizeof(float)) : 0);
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    auto& s = scratch(out, 0);
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!reserve(s, need, cs)) return false;
    Arena arena{static_cast<uint8_t*>(s.data)};
    void* packed = packed_input ? const_cast<void*>(raw) : a ? a->data : arena.take<uint8_t>(shape.sizes.raw_bytes);
    float* xf = input == Input::q8_1 ? arena.take<float>(nx) : nullptr;
    c.output = finish_output ? arena.take<float>(ny) : out;
    c.a = xf ? xf : x; c.stream = stream;
    planes(c, packed, shape);
    if (a) {
        cudaStreamCaptureStatus state;
        check(cudaStreamIsCapturing(cs, &state));
        check(cudaStreamWaitEvent(cs, a->ready, state == cudaStreamCaptureStatusNone ? 0 : cudaEventWaitExternal));
    } else if(!packed_input)pack(raw, c, shape);
    if (xf) expand_q81<<<blocks(nx), 256, 0, cs>>>(static_cast<const Q81*>(x), xf, k, rows);
    package_run(c,shape,device());
    if (in_place) interleave_rows<<<rows, 256, n * sizeof(float), cs>>>(out, n, ldy);
    if (finish_output) finish<<<blocks(ny), 256, 0, cs>>>(c.output, out, n, rows, ldy, beta, interleaved);
    check(cudaPeekAtLastError());
    strata::dispatch::note(packed_input ? "quactlize: offline expert GEMV" : a ? "quactlize: resident K-pack GEMV" : "quactlize: online shuffle + GEMV");
    return true;
}

bool grouped(const unsigned long long* ptr, const int32_t* start, const int32_t* count,
             const int32_t* ent_tok, const int32_t* ent_dst, int cap_groups, int cap_entries,
             const void* x_q8_1, size_t weight_offset, int type, int n, int k,
             float* out, float* up, void* stream, bool packed_input) {
    if (!enabled() || cap_groups <= 0 || cap_entries <= 0 || !ptr || !start || !count || !x_q8_1 || !out ||
        (!up && !ent_dst) || (up && (!ent_tok || n % 2)) || k % 32) return false;
    if(offline::enabled()&&!packed_input)return false;
    if (!may_load(stream)) return false;
    auto c = call(type, n, k, cap_entries, kGroupBatch);
    c.capturing=capturing(stream);
    Shape shape;
    if (!describe(c, shape)) return false;
    const size_t nx = size_t(cap_entries) * k, ny = size_t(cap_entries) * n;
    const size_t raw_bytes = shape.sizes.raw_bytes;
    size_t need = 0;
    if (!grouped_scratch(shape, cap_entries, n, k, packed_input, need)) return false;
    auto& s = scratch(out, up ? 1 : 2);
    std::lock_guard<std::mutex> lock(s.mutex);
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (!reserve(s, need, cs)) return false;
    Arena a{static_cast<uint8_t*>(s.data)};
    auto* raw = packed_input ? nullptr : a.take<uint8_t>(raw_bytes);
    planes(c, a.take<uint8_t>(raw_bytes), shape);
    c.a = a.take<float>(nx); c.output = a.take<float>(ny);
    auto* group = a.take<int32_t>(cap_entries);
    auto* ids = a.take<int32_t>(cap_entries);
    c.ids = ids; c.stream = stream;
    entry_groups<<<blocks(cap_entries), 256, 0, cs>>>(start, count, group, cap_entries, cap_groups);
    if (up) { check(cudaMemsetAsync(out, 0, ny / 2 * sizeof(float), cs)); check(cudaMemsetAsync(up, 0, ny / 2 * sizeof(float), cs)); }
    for (int batch = 0; batch < cap_groups; batch += kGroupBatch) {
        if(packed_input)gather_packed<<<blocks(raw_bytes),256,0,cs>>>(ptr,start,count,const_cast<uint8_t*>(c.low),
            shape.sizes.low_bytes/kGroupBatch,shape.sizes.high_bytes/kGroupBatch,shape.sizes.units_bytes/kGroupBatch,
            weight_offset,batch,cap_groups);
        else {
            gather_weights<<<blocks(raw_bytes), 256, 0, cs>>>(ptr, start, count, raw, raw_bytes / kGroupBatch, weight_offset, batch, cap_groups);
            pack(raw, c, shape);
        }
        gather_x<<<blocks(nx), 256, 0, cs>>>(static_cast<const Q81*>(x_q8_1), ent_tok, group,
            const_cast<float*>(static_cast<const float*>(c.a)), ids, cap_entries, k, batch);
        package_run(c,shape,device());
        scatter_group<<<blocks(ny), 256, 0, cs>>>(c.output, group, ent_dst, out, up, cap_entries, n, batch);
    }
    check(cudaPeekAtLastError());
    strata::dispatch::note(packed_input ? "quactlize: offline grouped experts, no shuffle" :
        "quactlize: mutable grouped experts, online shuffle + indexed GEMV");
    return true;
}

bool prepare_grouped(float* out, bool gate_up, int type, int n, int k, int cap_entries, void* stream,
                     bool packed_input) {
    if (!enabled() || !out || cap_entries <= 0 || k % 32 || (gate_up && n % 2)) return false;
    if (capturing(stream)) throw std::runtime_error("quactlize: grouped scratch must be prepared before graph capture");
    auto c = call(type, n, k, cap_entries, kGroupBatch);
    Shape shape;
    size_t need = 0;
    if (!describe(c, shape) || !grouped_scratch(shape, cap_entries, n, k, packed_input, need)) return false;
    auto& s = scratch(out, gate_up ? 1 : 2);
    std::lock_guard<std::mutex> lock(s.mutex);
    return reserve(s, need, static_cast<cudaStream_t>(stream));
}
} // namespace strata::ppu::quactlize
