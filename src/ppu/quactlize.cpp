#include "strata/ppu/quactlize.hpp"
#include "strata/ppu/python.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace strata::ppu::quactlize {
bool enabled() {
    static const bool on=[] {const char* v=std::getenv("STRATA_QUACTLIZE");return v&&std::strcmp(v,"1")==0;}();
    return on;
}
bool loaded(){return python::loaded();}
bool describe(const Call& c,Shape& s) {
    s={};
    if(!enabled()||(c.qtype!=8&&(c.qtype<10||c.qtype>14))||c.n<=0||c.k<=0||c.experts<=0||c.rows<=0||
       c.n%256||c.k%256||((c.qtype==11||c.qtype==14)&&c.k%512))return false;
    using Key=std::tuple<int,int,int,int>;
    static std::mutex mutex;
    static std::map<Key,Shape> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const Key key{c.qtype,c.n,c.k,c.experts};
    if(auto it=cache.find(key);it!=cache.end()){s=it->second;return true;}
    const auto values=python::call("quact_layout",{},{c.qtype,c.n,c.k,c.experts},-1,nullptr,c.capturing);
    if(values.size()!=13)throw std::runtime_error("quactlize: invalid package layout response");
    int32_t* fields[]={&s.arrangement.version,&s.arrangement.layout,&s.arrangement.bits,&s.arrangement.high_bits,
        &s.arrangement.artifact_tile_k,&s.arrangement.transport_tile_k,&s.arrangement.group_size,&s.arrangement.reserved};
    for(size_t i=0;i<8;++i){
        if(values[i]<0||values[i]>INT32_MAX)throw std::runtime_error("quactlize: descriptor exceeds int32");
        *fields[i]=int32_t(values[i]);
    }
    s.arrangement.mapping_id=values[8];
    for(size_t i=9;i<13;++i)if(values[i]<0)throw std::runtime_error("quactlize: negative plane size");
    s.sizes={uint64_t(values[9]),uint64_t(values[10]),uint64_t(values[11]),uint64_t(values[12])};
    const auto& b=s.sizes;
    if(s.arrangement.version!=2||s.arrangement.reserved||!s.arrangement.mapping_id||!b.raw_bytes||
       b.raw_bytes>SIZE_MAX||b.low_bytes>b.raw_bytes||b.high_bytes>b.raw_bytes-b.low_bytes||
       b.units_bytes!=b.raw_bytes-b.low_bytes-b.high_bytes||(s.arrangement.high_bits==0)!=(b.high_bytes==0))
        throw std::runtime_error("quactlize: invalid byte-neutral package layout");
    cache.emplace(key,s);return true;
}
namespace {
using python::Dtype;
using python::Tensor;
Tensor bytes(const void* p,uint64_t n){return {p,Dtype::u8,{int64_t(n)}};}
}
void package_pack(const void* raw,const Call& c,const Shape& s,int device) {
    python::call("quact_pack",{bytes(raw,s.sizes.raw_bytes),bytes(c.low,s.sizes.low_bytes),
        bytes(c.high,s.sizes.high_bytes),bytes(c.units,s.sizes.units_bytes)},
        {c.qtype,c.n,c.k,c.experts},device,c.stream,c.capturing);
}
void package_run(const Call& c,const Shape& s,int device) {
    python::call("quact_run",{{c.a,c.input_type==F16?Dtype::f16:Dtype::f32,{c.rows,c.k},{c.a_row_stride,1}},
        bytes(c.low,s.sizes.low_bytes),bytes(c.high,s.sizes.high_bytes),bytes(c.units,s.sizes.units_bytes),
        {c.ids,Dtype::i32,{c.ids?c.rows:0}},
        {c.output,Dtype::f32,{c.rows,c.n},{c.out_row_stride,1}}},
        {c.qtype,c.n,c.k,c.experts,c.rows},device,c.stream,c.capturing);
}
} // namespace strata::ppu::quactlize
