#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::ppu::quactlize {
bool enabled();
bool loaded();
// Serialized Strata container metadata. Values come from the installed
// package's canonical producer, not a copied operator ABI header.
struct Arrangement {
    int32_t version=0, layout=0, bits=0, high_bits=0, artifact_tile_k=0;
    int32_t transport_tile_k=0, group_size=0, reserved=0;
    uint64_t mapping_id=0;
};
struct Sizes { uint64_t raw_bytes=0, low_bytes=0, high_bytes=0, units_bytes=0; };
struct Shape { Arrangement arrangement{}; Sizes sizes{}; };
static_assert(sizeof(Shape)==72, "offline container descriptor size");
inline constexpr int F32=0, F16=1;
struct Call {
    int qtype=0,n=0,k=0,experts=1,rows=1,input_type=F32;
    int64_t a_row_stride=0,out_row_stride=0;
    const void* a=nullptr;
    const uint8_t *low=nullptr,*high=nullptr,*units=nullptr;
    const int32_t* ids=nullptr;
    float* output=nullptr;
    void* stream=nullptr;
    bool capturing=false;
};
bool describe(const Call& call, Shape& shape);
void package_pack(const void* raw,const Call& call,const Shape& shape,int device);
void package_run(const Call& call,const Shape& shape,int device);

enum class Input { f16, f32, q8_1 };
// Immutable dense/head weights only. Original GGUF bytes remain valid for every
// original reader. Additional packed storage is bounded and counted at startup.
size_t preload(const void* raw, int type, int n, int k, const char* name = nullptr);
void release(const void* raw) noexcept;

// False means no work submitted: the original path can run. Admitted launch
// errors throw. Mutable expert/staging pointers are NEVER cached by address.
bool dense(const void* raw, int type, int n, int k, const void* x, Input input,
           int rows, float* out, void* stream, int64_t ldx = 0,
           int64_t ldy = 0, float beta = 0, bool interleaved_gu = false, bool packed = false);

// Device-driven expert routing; no ID readback. Eight weight slices at a time
// bound scratch regardless of the number of groups. Gate/up blob rows are
// [gate; up]; output is split into gate/up, or scattered through ent_dst (down).
bool grouped(const unsigned long long* ptr, const int32_t* start, const int32_t* count,
             const int32_t* ent_tok, const int32_t* ent_dst, int cap_groups, int cap_entries,
             const void* x_q8_1, size_t weight_offset, int type, int n, int k,
             float* out, float* up, void* stream, bool packed = false);
} // namespace strata::ppu::quactlize
