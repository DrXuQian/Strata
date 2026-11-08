#include "strata/ppu/quactlize.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::ppu::quactlize {
bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QUACTLIZE");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}
namespace {
std::atomic<bool> is_loaded{false};
std::string location(const char* override_name, const char* directory, const char* file) {
    const char* p = override_name ? std::getenv(override_name) : nullptr;
    if (p && *p) return p;
    const char* root = std::getenv(directory);
    return root && *root ? std::string(root) + "/" + file : file;
}
void* open(const std::string& path) {
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error("quactlize: " + path + ": " + dlerror());
    return h; // Function pointers/captured graphs retain the DSO for process life.
}
template<class T> T symbol(void* h, const char* name) {
    dlerror();
    void* p = dlsym(h, name);
    const char* error = dlerror();
    if (error || !p) throw std::runtime_error(std::string("quactlize: missing ABI symbol ") + name);
    return reinterpret_cast<T>(p);
}
void status(int rc, const char* operation) {
    if (rc) throw std::runtime_error(std::string("quactlize: ") + operation + " status " + std::to_string(rc));
}
}
bool loaded() { return is_loaded.load(std::memory_order_acquire); }
const Library& library() {
    static const Library lib = [] {
        Library l;
        void* producer = open(location("QUACTLIZE_PPU_PACK_LIBRARY", "QUACTLIZE_PPU_BUNDLE", "libquactlize_ppu_pack.so"));
        void* execution = open(location(nullptr, "QUACTLIZE_KPACK_EXECUTION", "libquactlize_ppu_execution.so"));
#define BIND(member, handle, name) l.member = symbol<decltype(l.member)>(handle, #name)
        BIND(arrangement, producer, quactlize_ppu_kpack_canonical_arrangement_v1);
        BIND(sizes, producer, quactlize_ppu_kpack_sizes_for_arrangement_v1);
        BIND(pack, producer, quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2);
        BIND(query, execution, quactlize_kpack_gemv_query_v1);
        BIND(run, execution, quactlize_kpack_gemv_run_v1);
#undef BIND
        return l;
    }();
    is_loaded.store(true, std::memory_order_release);
    return lib;
}
const qkg_config_v1& recipe() {
    // An explicit correctness baseline, NOT a measured performance policy. No
    // JIT, legacy fmt fallback, online tuning, or guessed TC selector is used.
    static const qkg_config_v1 value{1, sizeof(qkg_config_v1), 16, 4, 1};
    return value;
}
bool describe(const qkg_call_v1& c, Shape& s) {
    s = {};
    if (!enabled() || (c.qtype != 8 && (c.qtype < 10 || c.qtype > 14)) ||
        c.n <= 0 || c.k <= 0 || c.experts <= 0 || c.rows <= 0) return false;
    const auto& l = library();
    status(l.arrangement(c.qtype, &s.arrangement), "canonical arrangement");
    if (s.arrangement.version != 2 || s.arrangement.reserved != 0 || s.arrangement.mapping_id == 0)
        throw std::runtime_error("quactlize: invalid producer arrangement-v2");
    int rc = l.sizes(c.n, c.k, c.experts, c.qtype, &s.arrangement, &s.sizes);
    if (rc == QKG_SHAPE || rc == QKG_OVERFLOW) return false;
    status(rc, "producer size query");
    qkg_sizes_v1 sizes{};
    rc = l.query(&c, &recipe(), &s.arrangement, &sizes);
    if (rc == QKG_SHAPE || rc == QKG_FORMAT || rc == QKG_OVERFLOW) return false;
    status(rc, "consumer query");
    const auto& b = s.sizes;
    if (!b.raw_bytes || b.raw_bytes > std::numeric_limits<size_t>::max() ||
        b.low_bytes > b.raw_bytes || b.high_bytes > b.raw_bytes - b.low_bytes ||
        b.units_bytes != b.raw_bytes - b.low_bytes - b.high_bytes ||
        sizes.low_bytes != b.low_bytes || sizes.high_bytes != b.high_bytes || sizes.units_bytes != b.units_bytes ||
        (s.arrangement.high_bits == 0) != (b.high_bytes == 0) || sizes.workspace_bytes != 0)
        throw std::runtime_error("quactlize: producer/consumer byte layout mismatch");
    return true;
}
} // namespace strata::ppu::quactlize
