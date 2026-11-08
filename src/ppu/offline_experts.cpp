#include "strata/ppu/offline_experts.hpp"
#include "strata/ppu/offline.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cstring>
#include <stdexcept>
namespace strata::ppu::offline {
const uint8_t* gpu_blob(core::ExpertSource* raw,int64_t l,int64_t e) {
    if(layer(int(l))) {
        const auto& layout=kernels::cpu::expert_layout();
        if(!layout.native||l<0||size_t(l)>=layout.fmt.size())throw std::runtime_error("quactlize: offline expert layer absent");
        const auto& f=layout.fmt[size_t(l)];
        const auto* record = store().expert(int(l), int(e));
        if (!record || record->experts != layout.n_expert || f.up_off != record->shape.sizes.raw_bytes / 2 ||
            f.down_off != record->shape.sizes.raw_bytes)
            throw std::runtime_error("quactlize: offline expert count/offset mismatch");
        if(f.tail_off || f.d_type>=100 || f.d_type==40)throw std::runtime_error("quactlize: unsupported offline expert down layout");
        return expert(int(l),int(e),f.gu_type,f.d_type,int(f.n_embd),int(f.n_ff),layout.blob_bytes(l));
    }
    return raw?raw->blob_stable(l,e):nullptr;
}
namespace {
class GpuSource final : public core::ExpertSource {
    core::ExpertSource* raw_;
public:
    explicit GpuSource(core::ExpertSource* raw):raw_(raw) { inputs_=raw->model_inputs(); }
    const uint8_t* blob(int64_t l,int64_t e) override { return gpu_blob(raw_,l,e); }
    const uint8_t* blob_stable(int64_t l,int64_t e) override { return blob(l,e); }
    bool pinned(int64_t l,int64_t e) const override { return !layer(int(l))&&raw_->pinned(l,e); }
    bool transient(int64_t l,int64_t e) const override { return !layer(int(l))&&raw_->transient(l,e); }
    void note_async_read(const uint8_t* p,void* stream) override { raw_->note_async_read(p,stream); }
    bool copy_blob(int64_t l,int64_t e,uint8_t* dst) override {
        if(!layer(int(l)))return raw_->copy_blob(l,e,dst);
        const uint8_t* p=blob(l,e);
        std::memcpy(dst,p,kernels::cpu::expert_layout().blob_bytes(l));return true;
    }
    void begin_layer(int64_t l,const int32_t* ids,int64_t n) override { if(!layer(int(l)))raw_->begin_layer(l,ids,n); }
    void prefetch(int64_t l,int64_t e) override { if(!layer(int(l)))raw_->prefetch(l,e); }
};
}
std::unique_ptr<core::ExpertSource> gpu_source(core::ExpertSource* raw) {
    return enabled()&&raw?std::make_unique<GpuSource>(raw):nullptr;
}
}
