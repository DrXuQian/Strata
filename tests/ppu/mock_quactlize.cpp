#include "strata/ppu/quactlize.hpp"
namespace {
void sizes(int q, int n, int k, int e, uint64_t& low, uint64_t& high, uint64_t& units) {
    const uint64_t nk = uint64_t(n) * k * e;
    const int lo[] = {2,2,4,4,4}, hi[] = {0,1,0,1,2}, unit[] = {20,14,16,16,18};
    low = q == 8 ? nk : nk / 8 * lo[q-10];
    high = q == 8 ? 0 : nk / 8 * hi[q-10];
    units = q == 8 ? nk / 32 * 2 : nk / 256 * unit[q-10];
}
}
#ifdef MOCK_PRODUCER
extern "C" int quactlize_ppu_kpack_canonical_arrangement_v1(int q, quactlize_ppu_placed_arrangement_v2* a) {
    *a = {}; a->version = 2; a->mapping_id = 123; a->high_bits = q == 11 || q == 13 || q == 14 ? 1 : 0;
#ifdef MOCK_BAD_DESCRIPTOR
    a->version = 1;
#endif
    return 0;
}
extern "C" int quactlize_ppu_kpack_sizes_for_arrangement_v1(int n, int k, int e, int q,
        const quactlize_ppu_placed_arrangement_v2*, quactlize_ppu_kpack_sizes_v1* s) {
    if (n % 256 || k % ((q == 11 || q == 14) ? 512 : 256)) return 24;
    sizes(q,n,k,e,s->low_bytes,s->high_bytes,s->units_bytes);
    s->raw_bytes = s->low_bytes + s->high_bytes + s->units_bytes;
#ifdef MOCK_NONNEUTRAL
    ++s->raw_bytes;
#endif
    return 0;
}
#ifndef MOCK_NO_PACK
extern "C" int quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(const uint8_t*,uint8_t*,uint8_t*,uint8_t*,
        int,int,int,int,const quactlize_ppu_placed_arrangement_v2*,void*) { return 0; }
#endif
#endif
#ifdef MOCK_CONSUMER
extern "C" int quactlize_kpack_gemv_query_v1(const qkg_call_v1* c,const qkg_config_v1*,
        const quactlize_ppu_placed_arrangement_v2*,qkg_sizes_v1* s) {
    *s = {}; sizes(c->qtype,c->n,c->k,c->experts,s->low_bytes,s->high_bytes,s->units_bytes);
#ifdef MOCK_MISMATCH
    ++s->units_bytes;
#endif
#ifdef MOCK_REJECT_LAYOUT
    return QKG_ARRANGEMENT;
#endif
#ifdef MOCK_SHAPE
    return QKG_SHAPE;
#endif
#ifdef MOCK_WORKSPACE
    s->workspace_bytes = 4096;
#endif
    return 0;
}
#ifndef MOCK_NO_RUN
extern "C" int quactlize_kpack_gemv_run_v1(const qkg_call_v1*,const qkg_config_v1*,
        const quactlize_ppu_placed_arrangement_v2*) { return 0; }
#endif
#endif
