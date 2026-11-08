#include "strata/ppu/python.hpp"
#include <Python.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <strings.h>

#ifndef STRATA_PPU_PYTHON
#define STRATA_PPU_PYTHON "python3"
#endif
#ifndef STRATA_PPU_MODULE_DIR
#define STRATA_PPU_MODULE_DIR "."
#endif

namespace strata::ppu {
bool enabled(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    return !v || !*v ? fallback : std::strcmp(v,"0") && strcasecmp(v,"false") && strcasecmp(v,"off");
}
namespace python {
namespace {
// DLPack's stable, unversioned capsule protocol (not a Torch C++ interface).
struct Device { int32_t type, id; };
struct Type { uint8_t code, bits; uint16_t lanes; };
struct DLTensor { void* data; Device device; int32_t ndim; Type dtype;
    int64_t* shape; int64_t* strides; uint64_t byte_offset; };
struct Managed { DLTensor tensor; void* context; void (*deleter)(Managed*); };
struct Owner {
    Managed managed{};
    std::vector<int64_t> shape, strides;
};
void destroy(Managed* p) { delete static_cast<Owner*>(p->context); }
void capsule_destroy(PyObject* p) {
    if (PyCapsule_IsValid(p, "dltensor"))
        destroy(static_cast<Managed*>(PyCapsule_GetPointer(p, "dltensor")));
}
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* v = nullptr) : p(v) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
};
std::string error() {
    PyObject *type = nullptr, *value = nullptr, *trace = nullptr;
    PyErr_Fetch(&type, &value, &trace);
    PyErr_NormalizeException(&type, &value, &trace);
    Ref t(type), v(value), tb(trace), module(PyImport_ImportModule("traceback"));
    if (module.p && type) {
        Ref lines(PyObject_CallMethod(module.p,"format_exception","OOO",type,value?value:Py_None,trace?trace:Py_None));
        Ref separator(PyUnicode_FromString(""));
        Ref joined(lines.p ? PyUnicode_Join(separator.p, lines.p) : nullptr);
        if (joined.p) return PyUnicode_AsUTF8(joined.p);
    }
    PyErr_Clear();
    Ref str(value ? PyObject_Str(value) : nullptr);
    return str.p ? PyUnicode_AsUTF8(str.p) : "unknown Python error";
}
void require(PyObject* p) { if (!p) throw std::runtime_error("strata PPU packages: " + error()); }
std::atomic<bool> ready{false};
std::once_flag once;
PyObject* entry = nullptr; // Process lifetime: graphs retain package code and prepared owners.
void initialize() {
    std::call_once(once, [] {
        if (!Py_IsInitialized()) {
            PyConfig config;
            PyConfig_InitPythonConfig(&config);
            config.parse_argv = 0;
            config.install_signal_handlers = 0;
            PyStatus status = PyConfig_SetBytesString(&config,&config.program_name,STRATA_PPU_PYTHON);
            if (!PyStatus_Exception(status))
                status = PyConfig_SetBytesString(&config,&config.executable,STRATA_PPU_PYTHON);
            if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
            const std::string why = PyStatus_Exception(status) ? (status.err_msg ? status.err_msg : "initialization failed") : "";
            PyConfig_Clear(&config);
            if (!why.empty()) throw std::runtime_error("strata embedded Python: " + why);
            PyEval_SaveThread();
        }
        const auto gil = PyGILState_Ensure();
        try {
            Ref path(PyUnicode_FromString(STRATA_PPU_MODULE_DIR)); require(path.p);
            if (PyList_Insert(PySys_GetObject("path"),0,path.p)) require(nullptr);
            Ref module(PyImport_ImportModule("strata_ppu.adapter")); require(module.p);
            entry = PyObject_GetAttrString(module.p,"call"); require(entry);
            if (!PyCallable_Check(entry)) throw std::runtime_error("strata PPU adapter.call is not callable");
            ready.store(true, std::memory_order_release);
        } catch (...) { PyGILState_Release(gil); throw; }
        PyGILState_Release(gil);
    });
}
PyObject* capsule(const Tensor& t, int device) {
    if (t.shape.empty() || (!t.strides.empty() && t.strides.size() != t.shape.size()))
        throw std::invalid_argument("strata PPU: invalid Tensor rank/strides");
    int64_t elements = 1;
    for (int64_t s : t.shape) {
        if (s < 0 || (s && elements > INT64_MAX / s)) throw std::invalid_argument("strata PPU: Tensor size overflow");
        elements *= s;
    }
    if (elements && !t.data) throw std::invalid_argument("strata PPU: null Tensor storage");
    for (auto s : t.strides) if (s < 0) throw std::invalid_argument("strata PPU: negative Tensor stride");
    auto o = std::make_unique<Owner>();
    o->shape = t.shape; o->strides = t.strides;
    Type type{};
    switch(t.dtype) {
        case Dtype::u8: type={1,8,1}; break;
        case Dtype::i32: type={0,32,1}; break;
        case Dtype::f16: type={2,16,1}; break;
        case Dtype::bf16: type={4,16,1}; break;
        case Dtype::f32: type={2,32,1}; break;
    }
    o->managed.tensor={const_cast<void*>(t.data),{t.cpu?1:2,t.cpu?0:device},int32_t(o->shape.size()),type,
        o->shape.data(),o->strides.empty()?nullptr:o->strides.data(),0};
    o->managed.context=o.get(); o->managed.deleter=destroy;
    PyObject* p=PyCapsule_New(&o->managed,"dltensor",capsule_destroy);
    if(p)o.release();
    return p;
}
}
bool loaded() { return ready.load(std::memory_order_acquire); }
std::vector<int64_t> call(const char* op,const std::vector<Tensor>& tensors,
    const std::vector<int64_t>& integers,int device,void* stream,bool capturing) {
    if (capturing && !loaded()) throw std::runtime_error("strata PPU: initialize installed packages before graph capture");
    initialize();
    const auto gil=PyGILState_Ensure();
    try {
        std::vector<int64_t> result;
        { // Destroy all Python references before releasing the GIL.
            Ref ts(PyTuple_New(tensors.size())), ns(PyTuple_New(integers.size()));
            require(ts.p); require(ns.p);
            for(size_t i=0;i<tensors.size();++i) {
                auto* p=capsule(tensors[i],device); require(p); PyTuple_SET_ITEM(ts.p,i,p);
            }
            for(size_t i=0;i<integers.size();++i) {
                auto* p=PyLong_FromLongLong(integers[i]); require(p); PyTuple_SET_ITEM(ns.p,i,p);
            }
            Ref args(Py_BuildValue("sOOiKO",op,ts.p,ns.p,device,
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),capturing?Py_True:Py_False));
            require(args.p);
            Ref ret(PyObject_CallObject(entry,args.p)); require(ret.p);
            if (ret.p != Py_None) {
                Ref seq(PySequence_Fast(ret.p,"adapter result must be integers")); require(seq.p);
                for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq.p);++i) {
                    auto n=PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq.p,i));
                    if(PyErr_Occurred())require(nullptr);
                    result.push_back(n);
                }
            }
        }
        PyGILState_Release(gil);
        return result;
    } catch(...) { PyGILState_Release(gil); throw; }
}
} // namespace python
} // namespace strata::ppu
