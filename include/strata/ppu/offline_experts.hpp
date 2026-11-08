#pragma once
#include <cstdint>
#include <memory>
namespace strata::core { class ExpertSource; }
namespace strata::ppu::offline {
// Only GPU clients use this view. CPU offload keeps the original ExpertSource.
const uint8_t* gpu_blob(core::ExpertSource* raw,int64_t layer,int64_t expert);
std::unique_ptr<core::ExpertSource> gpu_source(core::ExpertSource* raw);
}
