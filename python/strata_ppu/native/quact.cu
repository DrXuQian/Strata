// Only Tensor plumbing lives here. Packing maps, metadata, rounding, and GEMV
// kernels below are compiled from the installed QUactlize package, unchanged.
#include "adapter.hpp"
#include "word_pack.hpp"
#include "ppu_placed_arrangement.hpp"
#include "gemv.cu"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata_tensor {
namespace {
#if QKG_QTYPE != 8
constexpr auto Type=gguf_scale::KType(QKG_QTYPE-10);
using Unit=gguf_scale::packed_unit::Unit<Type>;
using Raw=gguf_scale::unit_pack::Raw<Type>;
using Code=gguf_scale::CodeTraits<Type>;
#endif
struct Size {uint64_t raw,low,high,units;};
Size sizes(int n,int k,int e) {
    if(n<=0||k<=0||e<=0||n%256||k%256)
        throw std::invalid_argument("QUactlize requires positive N/K multiples of 256");
#if QKG_QTYPE == 8
    constexpr uint64_t block=32, raw=34, low=32, high=0, units=2;
#else
    if((k/256)%Unit::kSbPerUnit)throw std::invalid_argument("incomplete packed metadata unit");
    constexpr uint64_t block=256,raw=Raw::kBytes,low=Code::kLoBytes,high=Code::kHiBytes,units=Unit::kSbBytes;
#endif
    uint64_t count=uint64_t(n)*(k/block);
    if(count>uint64_t(INT64_MAX)/uint64_t(e)/raw)throw std::overflow_error("QUactlize span overflow");
    count*=e;return {count*raw,count*low,count*high,count*units};
}
unsigned grid(uint64_t n){return unsigned(std::min<uint64_t>((n+255)/256,65535));}
__global__ void pack_codes(const uint8_t* src,uint16_t* low,uint16_t* high,uint8_t* metadata,
                           int n,int k,Size sizes) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<sizes.low/2;i+=uint64_t(gridDim.x)*blockDim.x) {
#if QKG_QTYPE == 8
        low[i]=quactlize_pack::Q8::word(src,n,k,i);
#else
        low[i]=quactlize_pack::Plane<Type,false>::word(src,n,k,i);
#endif
    }
#if QKG_QTYPE != 8
#if QKG_QTYPE == 11 || QKG_QTYPE == 13 || QKG_QTYPE == 14
        for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<sizes.high/2;i+=uint64_t(gridDim.x)*blockDim.x)
            high[i]=quactlize_pack::Plane<Type,true>::word(src,n,k,i);
#endif
    constexpr int unit_bytes=Unit::kUnitTotal;
#else
    constexpr int unit_bytes=2;
#endif
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<sizes.units/unit_bytes;i+=uint64_t(gridDim.x)*blockDim.x) {
#if QKG_QTYPE == 8
        quactlize_pack::Q8::metadata(src,metadata,n,k,i);
#else
        quactlize_pack::metadata<Type>(src,metadata,n,k,i);
#endif
    }
}
}
std::vector<int64_t> layout(int n,int k,int experts) {
    const auto s=sizes(n,k,experts);
#if QKG_QTYPE == 8
    const auto a=q8_kpack2::arrangement();
#elif QKG_QTYPE == 12
    const auto a=ppu_arrangements::q4_kpack4_transpose_v1();
#else
    const auto a=ppu_arrangements::kquant_kpack_transpose_v1(QKG_QTYPE);
#endif
    return {a.version,a.layout,a.bits,a.high_bits,a.artifact_tile_k,a.transport_tile_k,
        a.group_size,a.reserved,int64_t(a.mapping_id),int64_t(s.raw),int64_t(s.low),int64_t(s.high),int64_t(s.units)};
}
void pack(const std::vector<uintptr_t>& p,int n,int k,int experts,uintptr_t stream) {
    const auto s=sizes(n,k,experts);
    pack_codes<<<grid(s.low/2),256,0,reinterpret_cast<hggcStream_t>(stream)>>>(
        reinterpret_cast<const uint8_t*>(p[0]),reinterpret_cast<uint16_t*>(p[1]),
        reinterpret_cast<uint16_t*>(p[2]),reinterpret_cast<uint8_t*>(p[3]),n,k,s);
    if(hggcGetLastError()!=hggcSuccess)throw std::runtime_error("QUactlize pack launch failed");
}
void multiply(const std::vector<uintptr_t>& p,int n,int k,int experts,int rows,
              bool f16,int64_t ldx,int64_t ldy,uintptr_t stream) {
    (void)sizes(n,k,experts);
    if(rows<=0||uint64_t(rows)*(n/16)>uint64_t(INT32_MAX)||ldx<k||ldy<n)
        throw std::invalid_argument("invalid QUactlize grid/strides");
    // The installed kernel's private call type stays within this one module.
    qkg_call_v1 c{};c.version=1;c.size=sizeof(c);c.qtype=QKG_QTYPE;
    c.n=n;c.k=k;c.experts=experts;c.rows=rows;c.mode=p[4]?QKG_INDEXED:QKG_DENSE;
    c.input_type=f16?QKG_F16:QKG_F32;c.channels=c.topk=1;
    c.a_row_stride=c.a_token_stride=ldx;c.ids_stride=1;c.out_row_stride=ldy;
    c.a=reinterpret_cast<const void*>(p[0]);c.low=reinterpret_cast<const uint8_t*>(p[1]);
    c.high=reinterpret_cast<const uint8_t*>(p[2]);c.units=reinterpret_cast<const uint8_t*>(p[3]);
    c.ids=reinterpret_cast<const int32_t*>(p[4]);c.output=reinterpret_cast<float*>(p[5]);
    c.stream=reinterpret_cast<void*>(stream);
    // Explicit untuned baseline matching the previous scalar rounding contract.
#if QKG_QTYPE == 8
    const int rc=QKG_CONCAT(kpack_q,QKG_QTYPE)::launch<16,4,true>(c,1);
#else
    const int rc=QKG_CONCAT(kpack_q,QKG_QTYPE)::launch<16,4>(c,1);
#endif
    if(rc)throw std::runtime_error("QUactlize Tensor GEMV launch failed");
}
}
