#include <Python.h>
#include "adapter.hpp"
#include <stdexcept>

namespace {
std::vector<uintptr_t> pointers(PyObject* tensors, Py_ssize_t expected) {
    if (!PyTuple_Check(tensors) || PyTuple_GET_SIZE(tensors) != expected)
        throw std::invalid_argument("wrong Tensor operand count");
    std::vector<uintptr_t> p;
    for (Py_ssize_t i=0;i<expected;++i) {
        auto* item=PyTuple_GET_ITEM(tensors,i);
        if (item==Py_None) { p.push_back(0); continue; }
        PyObject* ptr=PyObject_CallMethod(item,"data_ptr",nullptr);
        if(!ptr)throw std::invalid_argument("operand is not a Tensor");
        const auto value=PyLong_AsUnsignedLongLong(ptr);Py_DECREF(ptr);
        if(PyErr_Occurred())throw std::invalid_argument("invalid Tensor pointer");
        p.push_back(static_cast<uintptr_t>(value));
    }
    return p;
}
PyObject* failure(const std::exception& e) {
    if(!PyErr_Occurred())PyErr_SetString(PyExc_RuntimeError,e.what());
    return nullptr;
}
PyObject* layout(PyObject*,PyObject* args) {
    int n,k,e;
    if(!PyArg_ParseTuple(args,"iii",&n,&k,&e))return nullptr;
    try {
        const auto v=strata_tensor::layout(n,k,e);
        auto* result=PyTuple_New(v.size());if(!result)return nullptr;
        for(size_t i=0;i<v.size();++i) {
            auto* value=PyLong_FromLongLong(v[i]);
            if(!value){Py_DECREF(result);return nullptr;}
            PyTuple_SET_ITEM(result,i,value);
        }
        return result;
    }catch(const std::exception& e){return failure(e);}
}
PyObject* pack(PyObject*,PyObject* args) {
    PyObject* t;int n,k,e;unsigned long long stream;
    if(!PyArg_ParseTuple(args,"OiiiK",&t,&n,&k,&e,&stream))return nullptr;
    try {strata_tensor::pack(pointers(t,4),n,k,e,stream);Py_RETURN_NONE;}
    catch(const std::exception& e){return failure(e);}
}
PyObject* multiply(PyObject*,PyObject* args) {
    PyObject* t;int n,k,e,m,f16;long long ldx,ldy;unsigned long long stream;
    if(!PyArg_ParseTuple(args,"OiiiiiLLK",&t,&n,&k,&e,&m,&f16,&ldx,&ldy,&stream))return nullptr;
    try {strata_tensor::multiply(pointers(t,6),n,k,e,m,f16,ldx,ldy,stream);Py_RETURN_NONE;}
    catch(const std::exception& e){return failure(e);}
}
PyMethodDef methods[]={{"layout",layout,METH_VARARGS,nullptr},{"pack",pack,METH_VARARGS,nullptr},
    {"multiply",multiply,METH_VARARGS,nullptr},{nullptr,nullptr,0,nullptr}};
#define STRATA_STRING_(X) #X
#define STRATA_STRING(X) STRATA_STRING_(X)
#define STRATA_JOIN_(X,Y) X##Y
#define STRATA_JOIN(X,Y) STRATA_JOIN_(X,Y)
PyModuleDef module={PyModuleDef_HEAD_INIT,STRATA_STRING(STRATA_MODULE),nullptr,-1,methods};
}
PyMODINIT_FUNC STRATA_JOIN(PyInit_,STRATA_MODULE)(){return PyModule_Create(&module);}
