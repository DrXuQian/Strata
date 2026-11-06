#include "strata/ppu/ncp.hpp"
#include <cstdlib>
#include <future>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::vector<std::future<const strata::ppu::Ncp*>> tasks;
    for (int i = 0; i < 16; ++i)
        tasks.push_back(std::async(std::launch::async, [] { return &strata::ppu::ncp(); }));
    const auto& api = strata::ppu::ncp();
    for (auto& task : tasks) if (task.get() != &api) return 3;
    const int mask = (api.fa ? 1 : 0) | (api.moe ? 2 : 0) | (api.recurrent ? 4 : 0) | (api.chunked ? 8 : 0);
    if (mask != std::atoi(argv[1])) { std::cerr << "availability mask " << mask << '\n'; return 4; }
    if (api.fa && api.fa(nullptr, nullptr, nullptr, nullptr, 1, 2, 3, 24, 2, 256, 256,
        11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 0.0625f, 0.0f, 0, -1, -1, 0, nullptr) != 101) return 5;
    if (api.moe && api.moe(nullptr, nullptr, nullptr, nullptr, nullptr, 7, 1280, 2560, 1, 7, nullptr) != 102) return 6;
    if (api.recurrent && api.recurrent(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        1, 8, 16, 48, 128, 0.25f, nullptr) != 103) return 7;
    if (api.chunked && (api.workspace_size(1, 128, 16, 48, 128) != 4096 ||
        api.chunked(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        1, 128, 16, 48, 128, 0.25f, nullptr, 4096, nullptr) != 104)) return 8;
    return 0;
}
