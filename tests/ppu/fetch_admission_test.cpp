// Exercise the real dispatch gate without a GPU or an installed operator package.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cstdio>
#include <stdexcept>

namespace {
bool offline_on = false;
struct Source : strata::core::ExpertSource {
    bool fetchable = true;
    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
    bool copy_blob(int64_t, int64_t, uint8_t*) override { return false; }
    bool copy_blobs(const int32_t*, const int32_t*, uint8_t* const*, size_t) override { return false; }
    bool pcie_layer(int64_t) const override { return fetchable; }
};
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}
namespace strata::ppu::offline {
// The container suite checks layer lookup; here layer 2 has packed gate/up.
bool layer(int l) { return offline_on && l == 2; }
}
namespace strata::kernels::cpu {
const ExpertLayout& expert_layout() {
    throw std::runtime_error("admission gate must not read model weights");
}
}

int main() {
    using namespace strata::core;
    Source src;
    ExpertDispatch d;
    GpuPlanSink plan;
    int32_t resident = 0;
    unsigned long long destination = 0;
    d.src = &src;
    d.plan = &plan;
    d.res_mut = &resident;
    d.usage.assign(1, 3.0f);
    d.pcie_num = 128;
    plan.pcie_mode = 2;
    plan.dst2 = &destination;
    for (int mode : {1, 2}) {
        d.fetch_admit = mode;
        offline_on = false;
        require(d.fetch_can(2), "raw cache should allow direct admission");
        offline_on = true;
        require(!d.fetch_can(2), "raw DMA must not overwrite packed cache slots");
        require(d.fetch_can(1), "unpacked layers must retain direct admission");
        src.fetchable = false;
        require(!d.fetch_can(1), "unpinned layer must not allow admission");
        src.fetchable = true;
        plan.pcie_mode = 1;
        require(!d.fetch_can(1), "direct admission requires DMA mode");
        plan.pcie_mode = 2;
        d.remote_count = 1;
        require(!d.fetch_can(1), "remote tier must retain its admission guard");
        d.remote_count = 0;
    }
    d.fetch_admit = 0;
    require(!d.fetch_can(1), "direct admission must remain opt-in");
    std::puts("PASS raw/packed fetch admission in modes 0, 1 and 2");
}
