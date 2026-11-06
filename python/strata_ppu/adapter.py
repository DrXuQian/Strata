"""Tensor/stream adapters. All operator packages remain unmodified.

The embedded interpreter and prepared owners live as long as Strata. A package
upgrade takes effect on the next process start; native images are never hot
unloaded while a captured graph can reference them.
"""
import importlib
import math
import os

import torch

_modules = {}
_plans = {}
_work = {}
_warmed = set()
_devices = set()
_streams = {}
_layouts = {}


def _module(name, capture=False):
    if name in _modules:
        return _modules[name]
    if capture:
        raise RuntimeError(f"warm up {name} before graph capture; first-use import/JIT is forbidden")
    if name == "fa":
        # The installed FlashAttention binding already accepts caller-owned out.
        result = importlib.import_module("flash_attn_2_cuda")
        if not callable(getattr(result, "fwd", None)):
            raise RuntimeError("flash-attention-for-sail must export flash_attn_2_cuda.fwd")
    elif name == "moe":
        package = importlib.import_module("deep_gemm")
        result = getattr(package, "m_grouped_bf16_gemm_nt_nopad", None)
        if result is None:
            result = getattr(package, "m_grouped_gemm_bf16_bf16_bf16_nt_nopad", None)
        if not callable(result):
            raise RuntimeError("installed DeepGEMM-for-sail lacks the BF16 NoPad Tensor entry")
    elif name == "gdn":
        from .jit import gdn_library
        target = os.environ.get("STRATA_PPU_TARGET", "ppu10")
        if target != "ppu10":
            raise RuntimeError("the processed-input actlizeLA PreparedGDNCore currently requires ppu10")
        result = gdn_library(target)
    elif name.startswith("quact:"):
        from .jit import quactlize
        result = quactlize(int(name.split(":")[1]))
    else:
        raise ValueError(f"unknown operator {name}")
    _modules[name] = result
    return result


def _device(index, capturing):
    if index in _devices:
        return
    if capturing:
        raise RuntimeError("prepare the PPU device before capture")
    from actlize_la.serving.device import validate_device_target
    target = os.environ.get("STRATA_PPU_TARGET", "ppu10")
    validate_device_target(target, torch.cuda.get_device_name(index))
    _devices.add(index)


def _tensor(t, shape, dtype, *, contiguous=True):
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_cuda:
        raise ValueError(f"expected device {dtype} Tensor {shape}, got {t.shape}/{t.dtype}/{t.device}")
    if contiguous and not t.is_contiguous():
        raise ValueError("operator requires contiguous Tensor")


def _gdn_plan(tensors, capture):
    from actlize_la.serving import PreparedGDNCore, gdn_core_workspace_specs
    q, k, v, g, beta, initial, output, final = tensors
    b, t, hk, _ = q.shape
    hv = v.shape[2]
    key = (str(q.device), torch.cuda.current_stream().cuda_stream,
           tuple((x.data_ptr(), tuple(x.shape)) for x in tensors))
    if key in _plans:
        return _plans[key]
    if capture:
        raise RuntimeError(f"GDN T={t} was not prepared on this stream before capture")
    library = _module("gdn")
    # Every prepared length borrows slices of one maximum workspace. Calls on
    # the same stream are ordered; other streams have independent workspaces.
    work_key = (str(q.device), key[1], b, hk, hv)
    if work_key not in _work:
        specs = gdn_core_workspace_specs(b, 256, hk, hv)
        _work[work_key] = {n: torch.empty(shape, dtype=dtype, device=q.device)
                           for n, (shape, dtype) in specs.items()}
    specs = gdn_core_workspace_specs(b, t, hk, hv)
    work = {n: _work[work_key][n].view(-1)[:math.prod(shape)].view(shape)
            for n, (shape, dtype) in specs.items()}
    plan = PreparedGDNCore(library, q=q, k=k, v=v, g=g, beta=beta,
                          initial_state=initial, output=output, final_state=final, workspace=work)
    _plans[key] = plan
    return plan


def _quact_layout(qtype, n, k, experts, capture):
    key = (qtype, n, k, experts)
    if key not in _layouts:
        # Once loaded, layout() is a host-only integer query. It launches no
        # kernels, JIT, allocations, device queries, or filesystem operations.
        values = tuple(_module(f"quact:{qtype}", capture).layout(n, k, experts))
        if len(values) != 13 or values[0] != 2 or values[7] or not values[8]:
            raise RuntimeError("QUactlize returned an unsupported layout descriptor")
        raw, low, high, units = values[9:]
        if min(raw, low, units) <= 0 or high < 0 or raw != low + high + units:
            raise RuntimeError("QUactlize packed planes must remain byte-neutral")
        _layouts[key] = values
    return _layouts[key]


def _quact(op, tensors, integers, stream, capture):
    qtype, n, k, experts = integers[:4]
    layout = _quact_layout(qtype, n, k, experts, capture)
    module = _module(f"quact:{qtype}", capture)
    low_bytes, high_bytes, units_bytes = layout[10:]
    if op == "quact_pack":
        raw, low, high, units = tensors
        _tensor(raw, (layout[9],), torch.uint8)
    else:
        a, low, high, units, ids, out = tensors
        rows = integers[4]
        if a.dtype not in (torch.float16, torch.float32):
            raise ValueError("QUactlize activation must be FP16/FP32")
        _tensor(a, (rows, k), a.dtype, contiguous=False)
        _tensor(out, (rows, n), torch.float32, contiguous=False)
        if a.stride(1) != 1 or out.stride(1) != 1 or a.stride(0) < k or out.stride(0) < n:
            raise ValueError("invalid QUactlize row stride")
        if ids.numel():
            _tensor(ids, (rows,), torch.int32)
        elif experts != 1:
            raise ValueError("grouped QUactlize requires device expert IDs")
    for t, size in ((low, low_bytes), (high, high_bytes), (units, units_bytes)):
        _tensor(t, (size,), torch.uint8)
    if op == "quact_pack":
        module.pack((raw, low, high if high.numel() else None, units), n, k, experts, stream)
    else:
        module.multiply((a, low, high if high.numel() else None, units,
                         ids if ids.numel() else None, out),
                        n, k, experts, rows, int(a.dtype == torch.float16), a.stride(0), out.stride(0), stream)


def _execute(op, t, integers, stream, capture):
    if op == "prepare_quact":
        if capture:
            raise RuntimeError("QUactlize preparation is forbidden in capture")
        device = torch.cuda.current_device()
        key = ("quact", device)
        if key not in _warmed:
            for qtype in (8, 10, 11, 12, 13, 14):
                layout = _quact_layout(qtype, 256, 512, 1, False)
                raw = torch.zeros(layout[9], dtype=torch.uint8, device=device)
                planes = tuple(torch.zeros(n, dtype=torch.uint8, device=device) for n in layout[10:])
                a = torch.zeros((1, 512), dtype=torch.float32, device=device)
                out = torch.empty((1, 256), dtype=torch.float32, device=device)
                ids = torch.empty(0, dtype=torch.int32, device=device)
                if not os.environ.get("STRATA_QUACTLIZE_PACK"):
                    _quact("quact_pack", (raw, *planes), (qtype, 256, 512, 1), stream, False)
                _quact("quact_run", (a, *planes, ids, out), (qtype, 256, 512, 1, 1), stream, False)
            torch.cuda.current_stream().synchronize()
            _warmed.add(key)
    elif op == "fa":
        q, k, v, out = t
        result = _module("fa", capture).fwd(q, k, v, out, None, 0.0,
            q.shape[-1] ** -0.5, False, -1, -1, 0.0, False, None)
        if result[0].data_ptr() != out.data_ptr():
            raise RuntimeError("FlashAttention ignored caller-owned output")
    elif op == "moe":
        a, weight, out, ids, rows = t
        fn = _module("moe", capture)
        key = ("moe", str(a.device), tuple(a.shape), tuple(weight.shape))
        if capture and key not in _warmed:
            raise RuntimeError("DeepGEMM shape needs an eager JIT/warm-up call before capture")
        fn(a, weight, out, ids, m_rows=rows)
        _warmed.add(key)
    elif op == "gdn_prepare":
        if capture:
            raise RuntimeError("GDN preparation is forbidden in capture")
        for nt in range(1, integers[0] + 1):
            values = tuple(x[:, :nt] if i in (0, 1, 2, 3, 4, 6) else x for i, x in enumerate(t))
            _gdn_plan(values, False)()
        torch.cuda.current_stream().synchronize()
    elif op == "gdn":
        _gdn_plan(t, capture)()
    elif op in ("quact_pack", "quact_run"):
        _quact(op, t, integers, stream, capture)
    else:
        raise ValueError(f"unknown Strata operation {op}")


def call(operation, capsules, integers, device, stream, capturing):
    if operation == "quact_layout":
        return _quact_layout(*integers, capturing)
    with torch.cuda.device(device), torch.inference_mode():
        _device(device, capturing)
        key = (device, stream)
        if key not in _streams:
            # ExternalStream wraps the already existing handle. It neither
            # creates a stream nor synchronizes; pass device explicitly.
            _streams[key] = torch.cuda.ExternalStream(stream, device=device)
        with torch.cuda.stream(_streams[key]):
            # A legacy capsule has no producer-side implicit stream wait. The
            # engine has already enqueued its packing/events on this stream.
            tensors = tuple(torch.utils.dlpack.from_dlpack(x) for x in capsules)
            if any(t.device.index != device or not t.is_cuda for t in tensors):
                raise ValueError("Tensor/device mismatch at Strata PPU boundary")
            return _execute(operation, tensors, integers, stream, capturing)
