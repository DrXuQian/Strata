#pragma once
#include "strata/ppu/quactlize.hpp"
#include <map>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ppu::offline {
struct Source {
    std::string path;
    uint64_t device = 0, inode = 0, bytes = 0;
    int64_t seconds = 0, nanoseconds = 0;
};
Source identify(const std::string& path);
bool same(const Source& a, const Source& b);
uint64_t checksum(const void* data, size_t bytes);
struct Record {
    std::string name;
    int layer = -1, expert = -1, experts = 1, type = -1, down_type = -1, n = 0, k = 0;
    uint64_t offset = 0, bytes = 0, hash = 0;
    quactlize::Shape shape{};
};
// Each expert region is [packed gate/up planes | original down bytes]. The
// byte-neutral gate/up region keeps Strata's original down offset and stride.
class Store {
public:
    explicit Store(const std::string& directory);
    ~Store();
    Store(const Store&) = delete;
    void bind(const std::vector<std::string>& shards) const;
    const Record* dense(const std::string& name) const;
    const Record* expert(int layer, int expert) const;
    bool layer(int layer) const;
    const uint8_t* data(const Record& record) const;
    const std::vector<Source>& sources() const { return sources_; }
private:
    int fd_ = -1;
    const uint8_t* map_ = nullptr;
    size_t bytes_ = 0;
    std::vector<Source> sources_;
    std::map<std::string, Record> dense_;
    std::map<std::pair<int,int>, Record> experts_;
    mutable std::mutex mutex_;
    mutable bool bound_ = false;
    mutable std::map<uint64_t, bool> checked_;
};
bool enabled();
Store& store();
void bind(const std::vector<std::string>& shards);
bool layer(int layer);
const uint8_t* expert(int layer, int expert, int gu_type, int down_type,
                      int n_embd, int n_ff, size_t bytes);
void write_source(std::ostream& out, const Source& source);
void write_record(std::ostream& out, const Record& record);
} // namespace strata::ppu::offline
