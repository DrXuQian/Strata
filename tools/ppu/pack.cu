// Offline preparation on the box, using the existing shared GPU producer.
#include "strata/ppu/offline.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>

namespace qz = strata::ppu::quactlize;
namespace off = strata::ppu::offline;
namespace fs = std::filesystem;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* raw = nullptr; void* packed = nullptr; size_t capacity = 0;
    ~Device() { if(raw)cudaFree(raw);if(packed)cudaFree(packed); }
    void reserve(size_t n) {
        if (n <= capacity) return;
        if(raw)ck(cudaFree(raw));raw=nullptr;if(packed)ck(cudaFree(packed));packed=nullptr;
        ck(cudaMalloc(&raw,n));ck(cudaMalloc(&packed,n));capacity=n;
    }
    std::vector<uint8_t> pack(const void* src,int type,int n,int k,const qz::Shape& s) {
        reserve(s.sizes.raw_bytes);
        ck(cudaMemcpy(raw,src,s.sizes.raw_bytes,cudaMemcpyHostToDevice));
        auto* lo=static_cast<uint8_t*>(packed);
        auto* hi=s.sizes.high_bytes?lo+s.sizes.low_bytes:nullptr;
        auto* units=lo+s.sizes.low_bytes+s.sizes.high_bytes;
        int rc=qz::library().pack(static_cast<const uint8_t*>(raw),lo,hi,units,n,k,1,type,&s.arrangement,nullptr);
        if(rc)throw std::runtime_error("GPU producer status "+std::to_string(rc));
        std::vector<uint8_t> result(s.sizes.raw_bytes);
        ck(cudaMemcpy(result.data(),packed,result.size(),cudaMemcpyDeviceToHost));return result;
    }
};
bool describe(int type,int n,int k,qz::Shape& s) {
    qkg_call_v1 c{};c.version=1;c.size=sizeof(c);c.qtype=type;c.n=n;c.k=k;
    c.experts=c.rows=c.channels=c.topk=1;c.a_row_stride=k;c.out_row_stride=n;
    return qz::describe(c,s);
}
bool raw_down_supported(int type) {
    switch (type) {
        case 2: case 3: case 6: case 7: case 8: case 13: case 14: case 20: case 23: case 42: return true;
        default: return false;
    }
}
void sync(const fs::path& p,bool directory=false) {
    int fd=open(p.c_str(),O_RDONLY|(directory?O_DIRECTORY:0));
    if(fd<0)throw std::runtime_error("cannot open for fsync: "+p.string());
    int rc=fsync(fd);close(fd);if(rc)throw std::runtime_error("fsync failed: "+p.string());
}
}
int main(int argc,char** argv) {
    std::string input,output;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--help") { std::cout<<"strata-ppu-pack --gguf SHARD1 --out NEW_DIRECTORY\n";return 0; }
        if(i+1<argc && a=="--gguf")input=argv[++i];
        else if(i+1<argc && a=="--out")output=argv[++i];
        else { std::cerr<<"unknown or missing option: "<<a<<'\n';return 2; }
    }
    if(input.empty()||output.empty()){std::cerr<<"--gguf and --out are required\n";return 2;}
    fs::path partial;
    bool own_partial = false;
    try {
        setenv("STRATA_QUACTLIZE","1",1);
        int devices=0;ck(cudaGetDeviceCount(&devices));if(!devices)throw std::runtime_error("no PPU device");
        qz::library();
        const auto paths=strata::gguf_split_paths(input);
        strata::GgufModel model(paths);
        std::vector<off::Source> sources;
        for(const auto& p:paths)sources.push_back(off::identify(p));
        fs::path destination=fs::absolute(output);
        if(fs::exists(destination))throw std::runtime_error("output exists; choose a new directory");
        fs::create_directories(destination.parent_path());
        partial=destination.string()+".partial."+std::to_string(getpid());
        if(!fs::create_directory(partial))throw std::runtime_error("partial directory already exists");
        own_partial = true;
        std::ofstream data(partial/"weights.bin",std::ios::binary),index(partial/"index.txt");
        if(!data||!index)throw std::runtime_error("cannot create pack files");
        index<<"strata.quactlize.pack 1\n";
        for(const auto& s:sources)off::write_source(index,s);
        uint64_t offset=0;size_t records=0;Device device;
        auto emit=[&](off::Record& r,const std::vector<uint8_t>& bytes) {
            r.offset=offset;r.bytes=bytes.size();r.hash=off::checksum(bytes.data(),bytes.size());
            data.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            if(!data)throw std::runtime_error("writing weights.bin failed");
            off::write_record(index,r);offset+=bytes.size();++records;
        };
        for(size_t si=0;si<model.size();++si)for(const auto& t:model.shard(si).tensors()) {
            if(t.shape.size()!=2 || t.name=="token_embd.weight" || t.shape[0]>INT_MAX || t.shape[1]>INT_MAX)continue;
            off::Record r;r.name=t.name;r.type=t.type;r.n=int(t.shape[1]);r.k=int(t.shape[0]);
            if(!describe(r.type,r.n,r.k,r.shape))continue;
            if(!model.in_bounds(t,si)||strata::tensor_payload_bytes(t)!=r.shape.sizes.raw_bytes)
                throw std::runtime_error("invalid tensor payload "+t.name);
            emit(r,device.pack(model.shard(si).tensor_data(t),r.type,r.n,r.k,r.shape));
            std::cout<<"packed dense "<<t.name<<'\n';
        }
        for(size_t si=0;si<model.size();++si)for(const auto& gate:model.shard(si).tensors()) {
            const std::string suffix=".ffn_gate_exps.weight";
            if(!gate.name.ends_with(suffix)||!gate.name.starts_with("blk.")||gate.shape.size()!=3)continue;
            const std::string prefix=gate.name.substr(0,gate.name.size()-suffix.size());
            size_t us=0,ds=0;const auto* up=model.find(prefix+".ffn_up_exps.weight",&us);
            const auto* down=model.find(prefix+".ffn_down_exps.weight",&ds);
            if(!up||!down||up->type!=gate.type||up->shape!=gate.shape||down->shape.size()!=3||
                down->shape[0]!=gate.shape[1]||down->shape[1]!=gate.shape[0]||down->shape[2]!=gate.shape[2])
                throw std::runtime_error("expert shapes differ: "+prefix);
            if(!gate.shape[0]||!gate.shape[1]||!gate.shape[2]||gate.shape[0]>INT_MAX||gate.shape[1]>INT_MAX/2||gate.shape[2]>INT_MAX)
                throw std::runtime_error("expert extent exceeds ABI");
            off::Record r;r.layer=std::stoi(prefix.substr(4));r.type=gate.type;r.down_type=down->type;
            r.experts=int(gate.shape[2]);
            r.n=int(2*gate.shape[1]);r.k=int(gate.shape[0]);
            if(!raw_down_supported(r.down_type) || !describe(r.type,r.n,r.k,r.shape)) {
                std::cout<<"raw expert layer "<<r.layer<<'\n';continue;
            }
            if(!model.in_bounds(gate,si)||!model.in_bounds(*up,us)||!model.in_bounds(*down,ds))
                throw std::runtime_error("truncated expert tensor");
            const size_t experts=gate.shape[2],gu=strata::tensor_payload_bytes(gate)/experts,
                db=strata::tensor_payload_bytes(*down)/experts;
            if(2*gu!=r.shape.sizes.raw_bytes)throw std::runtime_error("gate/up span mismatch");
            std::vector<uint8_t> raw(2*gu);
            for(size_t e=0;e<experts;++e) {
                std::memcpy(raw.data(),model.shard(si).tensor_data(gate)+e*gu,gu);
                std::memcpy(raw.data()+gu,model.shard(us).tensor_data(*up)+e*gu,gu);
                auto packed=device.pack(raw.data(),r.type,r.n,r.k,r.shape);
                packed.insert(packed.end(),model.shard(ds).tensor_data(*down)+e*db,
                    model.shard(ds).tensor_data(*down)+(e+1)*db);
                r.expert=int(e);r.name=prefix+".expert."+std::to_string(e);emit(r,packed);
            }
            std::cout<<"packed expert layer "<<r.layer<<": "<<experts<<" byte-neutral GPU blobs\n";
        }
        if(!records)throw std::runtime_error("no admitted quactlize matrices");
        for(size_t i=0;i<paths.size();++i)if(!off::same(sources[i],off::identify(paths[i])))
            throw std::runtime_error("source changed during preparation");
        data.close();index.close();if(!data||!index)throw std::runtime_error("closing pack files failed");
        sync(partial/"weights.bin");sync(partial/"index.txt");sync(partial,true);
        if(syscall(SYS_renameat2,AT_FDCWD,partial.c_str(),AT_FDCWD,destination.c_str(),1)!=0)
            throw std::runtime_error("cannot publish pack (destination must not exist)");
        partial.clear();sync(destination.parent_path(),true);
        std::cout<<"published "<<records<<" records, "<<offset<<" bytes: "<<destination<<'\n';return 0;
    } catch(const std::exception& e) {
        if(own_partial && !partial.empty()){std::error_code ec;fs::remove_all(partial,ec);}
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
