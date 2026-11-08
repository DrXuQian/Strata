#include "strata/ppu/python.hpp"
#include <array>
#include <future>
#include <stdexcept>
#include <string>
#include <iostream>
using namespace strata::ppu::python;
void trial(int seed) {
    std::array<float,10> x{},y{};
    for(int i=0;i<10;++i){x[i]=float(seed+i);y[i]=-99.f;}
    auto addresses=call("copy",{{x.data(),Dtype::f32,{2,3},{5,1},true},
        {y.data(),Dtype::f32,{2,3},{4,1},true}},{},-1,nullptr);
    if(addresses!=std::vector<int64_t>{int64_t(uintptr_t(x.data())),int64_t(uintptr_t(y.data())),5,1})
        throw std::runtime_error("DLPack was copied or strides changed");
    for(int r=0;r<2;++r)for(int c=0;c<3;++c)
        if(y[r*4+c]!=x[r*5+c])throw std::runtime_error("strided output mismatch");
    if(y[3]!=-99.f||y[7]!=-99.f||y[9]!=-99.f)throw std::runtime_error("wrote outside borrowed view");
    // Capsules and consumed Tensor owners may die; engine storage stays alive.
    x[0]=123.f;y[0]=234.f;
}
int main() {
    try {
        std::array<std::future<void>,8> jobs;
        for(int i=0;i<8;++i)jobs[i]=std::async(std::launch::async,[i]{trial(i*17);});
        for(auto& job:jobs)job.get();
        const uint16_t bf[]={0x3f80,0x4000,0xc040};uint16_t out[3]{};
        call("copy",{{bf,Dtype::bf16,{3},{},true},{out,Dtype::bf16,{3},{},true}},{},-1,nullptr);
        for(int i=0;i<3;++i)if(bf[i]!=out[i])throw std::runtime_error("BF16 bits changed");
        call("empty",{{nullptr,Dtype::u8,{0},{},true}},{},-1,nullptr);
        try { call("raise",{},{},-1,nullptr);throw std::logic_error("error was swallowed"); }
        catch(const std::runtime_error& e) {
            if(std::string(e.what()).find("intentional adapter failure")==std::string::npos)throw;
        }
        trial(100); // Error handling releases the GIL and leaves Python usable.
        std::cout<<"PASS real Torch DLPack borrowing, strides, BF16, empty Tensor, threads and exceptions\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
