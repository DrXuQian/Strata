#include "strata/ppu/offline.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace strata::ppu::offline {
namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("quactlize offline: " + s); }
}
Source identify(const std::string& path) {
    Source s; s.path = std::filesystem::canonical(path).string();
    struct stat st{};
    if (stat(s.path.c_str(), &st) || !S_ISREG(st.st_mode) || st.st_size <= 0) fail("invalid source " + path);
    s.device = st.st_dev; s.inode = st.st_ino; s.bytes = st.st_size;
    s.seconds = st.st_mtim.tv_sec; s.nanoseconds = st.st_mtim.tv_nsec;
    return s;
}
bool same(const Source& a, const Source& b) {
    return a.path == b.path && a.device == b.device && a.inode == b.inode && a.bytes == b.bytes &&
        a.seconds == b.seconds && a.nanoseconds == b.nanoseconds;
}
uint64_t checksum(const void* data, size_t bytes) {
    auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h; // Corruption check, not a cryptographic authentication mechanism.
}
void write_source(std::ostream& o, const Source& s) {
    o << "source " << std::quoted(s.path) << ' ' << s.device << ' ' << s.inode << ' ' << s.bytes << ' '
      << s.seconds << ' ' << s.nanoseconds << '\n';
}
void write_record(std::ostream& o, const Record& r) {
    const auto& a = r.shape.arrangement; const auto& b = r.shape.sizes;
    o << "record " << std::quoted(r.name) << ' ' << r.layer << ' ' << r.expert << ' ' << r.experts << ' ' << r.type << ' '
      << r.down_type << ' ' << r.n << ' ' << r.k << ' ' << r.offset << ' ' << r.bytes << ' ' << r.hash << ' '
      << a.version << ' ' << a.layout << ' ' << a.bits << ' ' << a.high_bits << ' ' << a.artifact_tile_k << ' '
      << a.transport_tile_k << ' ' << a.group_size << ' ' << a.reserved << ' ' << a.mapping_id << ' '
      << b.raw_bytes << ' ' << b.low_bytes << ' ' << b.high_bytes << ' ' << b.units_bytes << '\n';
}
Store::Store(const std::string& directory) {
    const auto root = std::filesystem::path(directory);
    std::ifstream index(root / "index.txt");
    std::string line;
    if (!std::getline(index, line) || line != "strata.quactlize.pack 1") fail("unknown or missing index version");
    uint64_t end = 0; size_t count = 0;
    std::set<std::string> paths;
    while (std::getline(index, line)) {
        if (line.size() > 16384 || ++count > 1000000) fail("index exceeds limits");
        std::istringstream in(line); std::string kind, tail;
        in >> kind;
        if (kind == "source") {
            Source s;
            if (!(in >> std::quoted(s.path) >> s.device >> s.inode >> s.bytes >> s.seconds >> s.nanoseconds) ||
                (in >> tail) || s.path.empty() || !s.bytes || !paths.insert(s.path).second) fail("invalid source record");
            sources_.push_back(s);
        } else if (kind == "record") {
            Record r; auto& a = r.shape.arrangement; auto& b = r.shape.sizes;
            if (!(in >> std::quoted(r.name) >> r.layer >> r.expert >> r.experts >> r.type >> r.down_type >> r.n >> r.k
                >> r.offset >> r.bytes >> r.hash >> a.version >> a.layout >> a.bits >> a.high_bits >> a.artifact_tile_k
                >> a.transport_tile_k >> a.group_size >> a.reserved >> a.mapping_id
                >> b.raw_bytes >> b.low_bytes >> b.high_bytes >> b.units_bytes) || (in >> tail)) fail("invalid tensor record");
            if (r.name.empty() || r.experts<=0 || r.n <= 0 || r.k <= 0 || r.n % 256 || r.k % 256 ||
                (r.type != 8 && (r.type < 10 || r.type > 14)) || a.version != 2 || a.reserved || !a.mapping_id ||
                !b.raw_bytes || b.low_bytes > b.raw_bytes || b.high_bytes > b.raw_bytes - b.low_bytes ||
                b.units_bytes != b.raw_bytes - b.low_bytes - b.high_bytes ||
                r.bytes < b.raw_bytes || r.offset != end || r.bytes > SIZE_MAX - end ||
                (a.high_bits == 0) != (b.high_bytes == 0)) fail("invalid geometry, layout, or span");
            end += r.bytes;
            if (r.layer == -1 && r.expert == -1) {
                if (r.experts!=1 || r.bytes != b.raw_bytes || !dense_.emplace(r.name,r).second) fail("duplicate or invalid dense record");
            } else {
                if (r.layer < 0 || r.expert < 0 || r.down_type < 0 ||
                    !experts_.emplace(std::make_pair(r.layer,r.expert),r).second) fail("duplicate or invalid expert record");
            }
        } else fail("unknown index record");
    }
    if (!index.eof() || sources_.empty() || !end) fail("incomplete index");
    // A layer must include every expert 0..E-1, all with the same geometry.
    int last_layer = -1, next_expert = 0; const Record* first = nullptr;
    for (const auto& [key,r] : experts_) {
        if (key.first != last_layer) {
            if(first && next_expert!=first->experts)fail("incomplete expert layer");
            last_layer = key.first; next_expert = 0; first = &r;
        }
        if (key.second != next_expert++ || r.n != first->n || r.k != first->k || r.type != first->type ||
            r.down_type != first->down_type || r.bytes != first->bytes || r.experts!=first->experts ||
            std::memcmp(&r.shape, &first->shape, sizeof(r.shape))) fail("inconsistent expert layer");
    }
    if(first && next_expert!=first->experts)fail("incomplete expert layer");
    fd_ = open((root / "weights.bin").c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st{};
    if (fd_ < 0 || fstat(fd_, &st) || uint64_t(st.st_size) != end) {
        if (fd_ >= 0) close(fd_);
        fd_ = -1;
        fail("weights.bin size mismatch");
    }
    bytes_ = size_t(end);
    void* p = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { close(fd_); fd_ = -1; fail("cannot map weights.bin"); }
    map_ = static_cast<const uint8_t*>(p);
}
Store::~Store() { if (map_) munmap(const_cast<uint8_t*>(map_),bytes_); if (fd_ >= 0) close(fd_); }
void Store::bind(const std::vector<std::string>& shards) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shards.size() != sources_.size()) fail("source shard count differs");
    for (size_t i = 0; i < shards.size(); ++i)
        if (!same(identify(shards[i]),sources_[i])) fail("source GGUF changed or moved: " + shards[i]);
    bound_ = true;
}
const Record* Store::dense(const std::string& name) const { auto i=dense_.find(name); return i==dense_.end()?nullptr:&i->second; }
const Record* Store::expert(int l,int e) const { auto i=experts_.find({l,e});return i==experts_.end()?nullptr:&i->second; }
bool Store::layer(int l) const { return expert(l,0) != nullptr; }
const uint8_t* Store::data(const Record& r) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!bound_) fail("source GGUF has not been validated");
    if (r.offset > bytes_ || r.bytes > bytes_ - r.offset) fail("payload outside mapping");
    if (!checked_.count(r.offset)) {
        if (checksum(map_+r.offset,r.bytes) != r.hash) fail("payload checksum mismatch: " + r.name);
        checked_[r.offset] = true;
    }
    return map_+r.offset;
}
bool enabled() {
    const char* p = std::getenv("STRATA_QUACTLIZE_PACK");
    return quactlize::enabled() && p && *p;
}
Store& store() {
    static Store* s = [] {
        const char* p=std::getenv("STRATA_QUACTLIZE_PACK");
        if (!p || !*p) fail("STRATA_QUACTLIZE_PACK is unset");
        return new Store(p);
    }();
    return *s;
}
void bind(const std::vector<std::string>& shards) { if (enabled()) store().bind(shards); }
bool layer(int l) { return enabled() && store().layer(l); }
const uint8_t* expert(int l,int e,int type,int down,int n,int ff,size_t bytes) {
    if (!layer(l)) return nullptr;
    const Record* r=store().expert(l,e);
    if (!r || r->type != type || r->down_type != down || r->k != n || r->n != 2*ff || r->bytes != bytes)
        fail("expert geometry differs from runtime layer " + std::to_string(l));
    quactlize::Shape s;
    quactlize::Call c{}; c.qtype=type;c.n=2*ff;c.k=n;
    c.experts=c.rows=1;c.a_row_stride=n;c.out_row_stride=2*ff;
    if (!quactlize::describe(c,s) || std::memcmp(&s,&r->shape,sizeof(s))) fail("expert producer/consumer layout mismatch");
    return store().data(*r);
}
} // namespace strata::ppu::offline
