// Host-only container tests: no device, producer or consumer needed.
#include "strata/ppu/offline.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
// Container tests exercise parsing/mapping only, independently of packages.
namespace strata::ppu::quactlize {
bool enabled() { return false; }
bool describe(const Call&,Shape&) { throw std::runtime_error("unexpected package access"); }
}
namespace off = strata::ppu::offline;
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string mode = argv[1];
    try {
        off::Store s(argv[2]);
        if (mode != "unbound") s.bind({argv[3]});
        const auto* d = s.dense("dense.weight");
        if (!d || s.dense("missing") || !s.layer(2) || s.layer(1) || s.expert(2,2))
            throw std::runtime_error("lookup contract");
        const auto* e = s.expert(2,1);
        if (!e || e->experts != 2) throw std::runtime_error("expert count");
        const auto* p = s.data(*d);
        if (std::memcmp(p,"\x03\x03\x03\x03",4)) throw std::runtime_error("dense bytes");
        if (s.data(*e)[e->bytes-1] != 5) throw std::runtime_error("raw down bytes");
        if (s.data(*d) != p) throw std::runtime_error("mapping moved");
        std::puts("PASS offline store"); return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"rejected: %s\n",e.what()); return 1;
    }
}
