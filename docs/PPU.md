# PPU: Qwen3.8-Flash-Next

This is an opt-in Linux source build. It uses the PPU SDK's CUDA-compatible
runtime and ACBLAS, and the existing `ncp_flash_lib` libraries. The normal CUDA,
HIP and SYCL builds do not enable this backend.

## Build

Use CMake 3.24 or newer. Select a CUDA architecture that your **PPU SDK** maps to
the intended device. The architecture name alone does not prove a PPU generation:
the local SDK 2.1.1 wrapper also maps `sm_90a` to an older PPU target. Do not define
`__CUDA_ARCH__` or `ACOMPUTE_VERSION` to override the compiler's target.

Use the SDK's runtime environment as well as its compiler. The CUDA compatibility
libraries load `libhggcrt` and ACBLAS at runtime; a headers/compiler-only SDK can
build the program but cannot execute it.

For an SDK exposing the CUDA-compatible `80` target:

```sh
cmake -S . -B build-ppu \
  -DSTRATA_ENABLE_PPU=ON \
  -DCMAKE_CUDA_COMPILER="$PPU_SDK/CUDA_SDK/bin/nvcc" \
  -DCUDAToolkit_ROOT="$PPU_SDK/CUDA_SDK" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-ppu --target strata ncp_parity -j 3
```

`STRATA_GGML_DIR` supplies ggml's CPU expert implementation. Omitting it fetches
Strata's pinned llama.cpp. This does not link llama.cpp's CUDA backend or its
private kernel implementations into Strata.

Place these files beside `build-ppu/strata` (and beside `ncp_parity` for tests),
or provide their directory through `LD_LIBRARY_PATH`:

| Library | Required entry |
| --- | --- |
| `libncp_fa.so` | `ncp_flash_attn_fwd_v3` |
| `libncp_moe.so` | `ncp_m_grouped_gemm_bf16_bf16_bf16_nt_nopad` |
| `libncp_gdn.so` | `ncp_gdn_recurrent` |
| Same GDN library, optional chunked path | `ncp_gdn_chunked_bf16`, `ncp_gdn_chunked_bf16_workspace_size` |

The executables carry an `$ORIGIN` runpath. Library handles remain open for the
process lifetime because captured graphs may still reference their code.
No Torch, Triton or CUTLASS headers are required by this build.

The declarations in `include/strata/ppu/ncp-*-lib.h` are copied from the local
llama.cpp revision `2f288debe17e4acd261dcb86961f3a93ce8a26f2`. The GDN library must
include **`(H, HV, S) = (16, 48, 128)`** for this model. Strata pairs value heads
with Q/K by `hv % 16`, while FLA groups three value heads per Q/K head. The adapter
permutes V, gate, beta and state to FLA order and restores output/state to Strata
order. Q/K retain their 16 heads. No additional `(48,48,128)` specialization or
library ABI change is needed.

## Execution paths

| Operation | PPU path |
| --- | --- |
| Dense prompt projections | Existing dequantization and FP32-accumulating ACBLAS GEMMs |
| Prompt expert gate/up and down | Existing expert order and staging, then NCP NoPad with one expert per call; BF16 operands/output |
| GDN prompt processing | BF16 chunked NCP for lengths divisible by 64 and at least 128; otherwise FP32 recurrent NCP |
| Eager GDN single-token decode | FP32 recurrent NCP, state converted back to Strata's layout |
| Speculative GDN verification | FP32 recurrent NCP with the final state discarded; the existing commit replays the accepted prefix |
| QSA prompt attention | Gather the exact selected cells through the page table, then FP16 NCP FA, one query per call |
| QSA decode / batched slots | Existing kernels with device-resident lengths |

GDN scratch is bounded to 256 tokens of Q/K/V. The state conversion is between
Strata's `[K,HV,V]`, recurrent NCP's `[HV,V,K]`, and chunked NCP's `[HV,K,V]`.
Raw gates and sigmoid beta are passed unchanged. Output normalization retains
Flash-Next's sigmoid gate.

QSA supports FP16 and Q8 KV pools in the NCP prompt adapter. It uses the fully
resident pool or prefill's complete staging pool; ring caches retain Strata's path.
Selection already enforces causality, so the gathered cells use full attention,
without another causal mask. Q4/mixed/rotated pools retain Strata's path. The FA ABI takes sequence
lengths on the host; graph-replayed decode cannot bake in its changing lengths.

NVIDIA MMQ, programmatic dependent launch and CTA cluster kernels are disabled
for PPU. The existing SIMT decode kernels remain available. The PPU branch of
the native MMVQ address helper uses C++ unsigned address arithmetic to avoid the
SDK 2.1.1 `mad.wide` pointer-lowering compiler crash.

This does not implement persistent BF16 expert repacking, grouped multi-expert
NCP decode, NCP state snapshots, or NCP FA decode with device-side lengths.
DeepSeek and the separate `strata-split` engine are rejected by this build mode.

## Controls and diagnostics

`STRATA_NCP_FA`, `STRATA_NCP_MOE` and `STRATA_NCP_GDN` are CMake options, defaulting
to ON only in a PPU build. The same environment variable names accept `0`,
`false` or `off` to disable individual libraries at process startup.
`STRATA_NCP_GDN_CHUNKED=0` selects recurrent GDN for prompt processing too.

When using `setup.py`, `STRATA_SKIP_DRIVER_CHECK=1` skips its minimum NVIDIA
driver-version check for the PPU compatibility runtime. The default check stays
enabled. Select the PPU compiler and backend with the CMake options above.

Missing libraries, old FA symbols and unsupported shapes keep the existing
Strata computation. GDN module-load and workspace errors are reported as errors.
The adapter does not write carried state until the whole NCP operation succeeds;
a declined call's output is overwritten by the fallback.

Set `STRATA_DISPATCH_REPORT=/path/to/dispatch.json` to distinguish a library
being loaded from an NCP path actually being selected. The report includes
`ppu: NCP GDN recurrent`, `ppu: NCP GDN chunked`, `ppu: NCP NoPad expert GEMM`
and `ppu: NCP selected-cell FlashAttention` counters.

Scratch is separate per device, stream and operation. The verifier reserves its
recurrence scratch at initialization, before its first capture. Allocation growth occurs
outside capture; old addresses are retained for existing graphs. Capturing before
sufficient scratch has been prepared uses the existing kernels. Only the verifier
opts into captured NCP recurrence and replays on its original stream. The session
graphs captured on temporary streams retain Strata's kernels, even if a reused
stream address happens to have cached scratch.

NCP BF16/FP16 conversion and accumulation may differ from the default kernels.
These paths are not claimed to be bit-identical or faster without a device and
model comparison. The existing graph/capture and expert residency policies remain
the basis of execution.

## Validation

On the PPU box, this command builds the engine and runs all seven device checks
against the shared libraries. It does not build or replace those libraries. Any
missing device/library or failed fixture stops the command:

```sh
PPU_SDK=/path/to/PPU_SDK \
NCP_LIB_DIR=/path/to/shared/ncp-libs \
LLAMA_CPP_DIR=/path/to/llama.cpp \
PPU_CUDA_ARCH=80 \
bash tools/ppu/check.sh
```

Choose the architecture for the box's SDK. To run the same prepared
Qwen3.8-Flash-Next model with NCP disabled and enabled after those checks, append
the engine arguments. Use a real pretokenized prompt of at least 128 tokens to
exercise chunked prefill, and the GGUF shard that contains the PLE table:

```sh
PPU_SDK=/path/to/PPU_SDK \
NCP_LIB_DIR=/path/to/shared/ncp-libs \
LLAMA_CPP_DIR=/path/to/llama.cpp \
PPU_CUDA_ARCH=80 \
bash tools/ppu/check.sh -- \
  --pack /path/to/strata-pack \
  --native /path/to/model-00001-of-00002.gguf \
  --ple-gguf /path/to/model-00002-of-00002.gguf \
  --tokens-file /path/to/prompt.ids \
  --prefill 128 --max-context 4096 --max-new 32 --kv int8 --greedy --stats
```

The shard names are examples; use this pack's actual files. Pass the same MTP,
expert-cache and other model arguments to both runs through this command.
Logs, binary/library hashes and dispatch reports go into a dated directory under
`build-ppu`. Model output equivalence and performance still need to be assessed
from the saved runs; a zero process exit code does not establish either result.

Loader tests require only a C++17 compiler and Python:

```sh
python tests/ppu/test_loader.py --build-dir /workspace/strata-ppu-loader-tests
```

They use ABI fixture libraries, not simulated model computations, to check
concurrent initialization, all exported signatures, `$ORIGIN` lookup, missing
libraries, stale FA symbols, missing NoPad symbols, incomplete GDN libraries and
runtime disable flags.

Run numerical tests on a PPU with matching NCP libraries:

```sh
build-ppu/ncp_parity gdn-verify
build-ppu/ncp_parity gdn-commit
build-ppu/ncp_parity gdn-graph
build-ppu/ncp_parity gdn-chunked
build-ppu/ncp_parity moe
build-ppu/ncp_parity fa
build-ppu/ncp_parity fa-q8
```

References run on the CPU. Fixtures cover nonzero GDN state, modulo head mapping,
verification without state writes, first-capture graph execution and the capture
ownership guard, a 320-token chunk boundary, non-tile-aligned
expert rows, sparse selection, permuted KV pages and different selected lengths.
Exit 77 means a required device or library is absent; an available library that
declines a required fixture fails the test. These are physical-device tests with
packing and output kernels, not single-launch simulator inputs.

Then compare the same Qwen3.8-Flash-Next pack, prompts, context, sampling and MTP
settings with each NCP switch on and off. Check output quality, accepted draft
tokens, memory use, prefill latency and decode throughput separately. No physical
PPU or full-model performance result is supplied by the source build alone.
