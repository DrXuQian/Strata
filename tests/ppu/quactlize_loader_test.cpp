#include "strata/ppu/quactlize.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    qkg_call_v1 c{};
    c.version = 1; c.size = sizeof(c); c.qtype = std::atoi(argv[2]);
    c.n = std::atoi(argv[3]); c.k = std::atoi(argv[4]); c.experts = c.rows = c.channels = c.topk = 1;
    c.a_row_stride = c.k; c.out_row_stride = c.n;
    int result;
    try {
        strata::ppu::quactlize::Shape s;
        result = strata::ppu::quactlize::describe(c, s) ? 1 : 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); result = 2; }
    return result == std::atoi(argv[1]) ? 0 : 1;
}
