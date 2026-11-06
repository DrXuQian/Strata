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
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-ppu --target strata ncp_parity -j 3
```

`STRATA_GGML_DIR` supplies ggml's CPU expert implementation. Omitting it fetches
Strata's pinned llama.cpp. This does not link llama.cpp's CUDA backend or its
private kernel implementations into Strata.

The box script's `LLAMA_CPP_DIR` is also optional. Leave it unset to fetch that
pinned source; set it to an existing checkout for an offline build or to reuse
local CPU ggml sources. The engine links ggml statically and needs no llama.cpp
checkout or executable at runtime. NCP and quactlize use their shared libraries
directly.

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

## Quactlize

Build with `-DSTRATA_QUACTLIZE=ON` (the PPU build's default), then opt in at
process startup with `STRATA_QUACTLIZE=1`. The runtime default is off. This uses
the same additive producer/execution C ABI as the local llama.cpp, with no
changes to the shared libraries:

```sh
export QUACTLIZE_PPU_BUNDLE=/path/to/quactlize/producer-directory
export QUACTLIZE_KPACK_EXECUTION=/path/to/quactlize/execution-directory
export STRATA_QUACTLIZE=1
```

The first directory contains `libquactlize_ppu_pack.so`; the second contains
`libquactlize_ppu_execution.so`. `QUACTLIZE_PPU_PACK_LIBRARY=/absolute/file.so`
overrides the producer path, as in llama.cpp. An explicitly enabled integration
with a missing/incompatible library fails with its path or missing symbol.

| Path | Shuffle and computation |
| --- | --- |
| Immutable native dense/head weights | GPU shuffle at load into a bounded resident K-pack cache; ready event waited by readers, including graph capture |
| Dense decode and native prompt GEMMs | Arrangement-aware K-pack GEMV, with F16/F32 inputs; Q8_1-only callers expand their existing activation values first |
| Prompt expert gate/up | Pack the current contiguous `[gate; up]` blob; convert the result to Strata's interleaved output order |
| Grouped expert decode | Gather the current device-selected pointers in batches of eight, shuffle, run indexed GEMV, and restore entry/destination order |
| Expert down | Quactlize when its type/geometry is admitted; otherwise the original down kernel, or NCP/ACBLAS prompt path |

The GPU producer supplies the canonical layout descriptor. Both producer and
consumer must accept it and agree on every plane size. Supported formats are
Q8_0 and Q2_K through Q6_K, subject to the loaded library's geometry checks.
The current consumer requires N and K multiples of 256; Q3_K/Q6_K need K a
multiple of 512. Flash-Next's merged gate/up is N=1280, K=2560. Its individual
640-row gate/up and K=640 down are not admitted separately. IQ and Q2_0 packs
retain their original expert kernels; enabling quactlize does not change their
quantization format.

Without `STRATA_QUACTLIZE_PACK`, dynamic experts are shuffled again after
selection. Their raw pointers can change content during
adaptive swaps, PCIe staging and prompt loans. No packed expert is cached by
pointer, and graph replay repacks the current contents without reading IDs on
the host. This preserves CPU offload, prefetch and cache policies. It adds GPU
gather/shuffle work and is not the byte-neutral persistent-buffer integration
used in llama.cpp.

The original GPU GGUF bytes also remain for immutable matrices so existing
fused, sliced and fallback readers stay valid. `STRATA_QUACTLIZE_CACHE_MIB`
bounds the additional resident copies per device (default 256 MiB, 0 disables
those copies). `STRATA_QUACTLIZE_SCRATCH_MIB` bounds all retained temporary
allocations per device (default 256 MiB). For raw inputs, allocation failure or budget exhaustion
uses the original reader before any work is submitted. These are allocation
budgets, not pre-reserved memory. The resident copies are charged in the native
dense allocation report and logged individually at load.

Scratch belongs to each output allocation/operation. It can grow outside
capture; older storage is retained for live graphs and counts against the
budget. A raw-input capture that needs unprepared scratch retains the original kernels.
Prepared graph calls read fresh activation, routing and expert bytes on replay.
Prompt ring slots are released only after all raw-weight readers finish.

This first binding uses the shared SIMT reader with the explicit
`columns=16, warps=4, split=1` recipe. It does not import llama.cpp's measured
GEMV policy, TC dispatcher, JIT or automatic prefill selector. The recipe is a correctness baseline, not a measured speed choice.
FP16 weight/activation arithmetic can differ from native integer-dot kernels;
numerical and model A/B checks on the box are required before evaluating speed.

Dispatch reports distinguish `quactlize: resident K-pack GEMV`,
`quactlize: online shuffle + GEMV` and
`quactlize: mutable grouped experts, online shuffle + indexed GEMV` from
scratch/capture fallback. A loaded library alone does not show that a model
used it.

## Prepare quactlize weights offline

There are two different prepared directories:

| Directory | Contents and use |
| --- | --- |
| `--pack /path/to/strata-pack` | Strata's model index, dense tensors and tokenizer, prepared by `tools/iq_pack.py`. This is also required by the original Strata engine. |
| `STRATA_QUACTLIZE_PACK=/path/to/quactlize-pack` | Canonical K-pack weight planes, prepared once by `strata-ppu-pack`. Only the optional quactlize integration reads it. |

Prepare both on the box. Reuse an existing Strata pack if it was prepared from
these same GGUF shards. The quactlize producer runs on the PPU during preparation;
the shared libraries are reused without changes. No llama.cpp path is required.
Use the SDK's normal runtime environment and set the two quactlize library paths
shown above before running the producer.

```sh
# Original Strata model preparation, once (omit if this pack already exists).
python tools/iq_pack.py --gguf /path/to/model-00001-of-00002.gguf \
  --out /path/to/strata-pack --compat-bf16

# Add this target to the PPU build configured above.
cmake --build build-ppu --target strata strata-ppu-pack quactlize_parity -j 3
build-ppu/strata-ppu-pack \
  --gguf /path/to/model-00001-of-00002.gguf \
  --out /path/to/quactlize-pack

# Check numerical parity on the box, then use the artifact for inference.
build-ppu/quactlize_parity offline
STRATA_QUACTLIZE=1 STRATA_QUACTLIZE_PACK=/path/to/quactlize-pack \
  build-ppu/strata generate --gpu 0 \
  --pack /path/to/strata-pack \
  --native /path/to/model-00001-of-00002.gguf \
  --tokens-file /path/to/prompt.ids \
  --prefill 128 --max-context 4096 --max-new 32 --kv int8 --greedy --stats
```

`--gguf` discovers all shards. The output must be a new directory. Preparation
writes a temporary sibling directory and publishes it only after the files are
complete. It packs one tensor/expert at a time. `index.txt` records the descriptor,
plane sizes, source identity and payload checksums; `weights.bin` holds the data.
Keep the original GGUFs for CPU offload, unsupported matrices and token tables.
The offline directory adds disk storage, roughly the size of its supported
matrices and their raw expert down projections.

Each converted GPU expert blob contains packed merged gate/up followed by the
original down bytes. The blob size and down offset are unchanged, so residency
slots keep their existing sizes. Startup fill, adaptive swaps, prompt loans,
refill and Foresight prediction all load these bytes. CPU decode and CPU prompt
sharing still use the original source. Eviction into a CPU RAM slot reloads
raw bytes from that source rather than copying the packed GPU slot back.
Ordinary PCIe miss staging stays raw and uses the original expert kernels.

Offline prompt gate/up reads the packed planes and interleaves its output in
place. Grouped decode gathers the already packed planes for the shared indexed
reader; it does **no runtime shuffle**. This still adds gather and activation
conversion work. Down remains on the original kernel or NCP/ACBLAS prompt path.
Immutable dense/head copies load from disk within `STRATA_QUACTLIZE_CACHE_MIB`;
a missing or over-budget dense record keeps the raw reader. The original dense
GPU allocations remain for sliced and fused readers.

The runtime adapter forbids invoking the producer in offline mode. A converted
expert cannot fall back to a raw reader: insufficient scratch, unsupported graph
capture or mismatched layout stops with an error. Increase
`STRATA_QUACTLIZE_SCRATCH_MIB` if needed. Warm the same grouped operation before
capture so its temporary storage exists. Producer/consumer descriptor and plane
sizes must agree with the artifact. Both existing DSOs are still needed at
runtime for admission; the producer's shuffle entry is used only by preparation.

This mode currently requires `--native` and one visible GPU (`--gpu 0`); peer,
remote and layer-split caches are rejected. Q8_0/Q2_K–Q6_K merged gate/up must
satisfy the consumer geometry. IQ, Q2_0, scaled-down layouts and NVFP4 down remain
raw. No quantization format is changed. The file identity check binds the artifact
to the canonical GGUF paths, sizes, inode/device and modification timestamps.
Moving/replacing a shard requires preparing again. Keep model and artifact files
immutable during use. Per-record FNV-1a detects accidental corruption; it is not
a cryptographic authentication scheme.

For the three model A/B runs below, add
`STRATA_QUACTLIZE_PACK=/path/to/quactlize-pack` to the environment. The two runs
with quactlize disabled ignore it. Dispatch reports include
`quactlize: offline expert GEMV` and
`quactlize: offline grouped experts, no shuffle`. Confirm these counters before
interpreting timings. Source compilation alone establishes no speedup.

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
python tests/ppu/test_quactlize_loader.py --build-dir /workspace/strata-quactlize-loader-tests
python tests/ppu/test_offline_store.py --build-dir /workspace/strata-offline-store-tests
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

To include quactlize, use the existing libraries on the box:

```sh
PPU_SDK=/path/to/PPU_SDK \
NCP_LIB_DIR=/path/to/shared/ncp-libs \
PPU_CUDA_ARCH=80 \
WITH_QUACTLIZE=1 \
QUACTLIZE_PPU_BUNDLE=/path/to/quactlize/producer-directory \
QUACTLIZE_KPACK_EXECUTION=/path/to/quactlize/execution-directory \
bash tools/ppu/check.sh -- \
  --pack /path/to/strata-pack \
  --native /path/to/model-00001-of-00002.gguf \
  --ple-gguf /path/to/model-00002-of-00002.gguf \
  --tokens-file /path/to/prompt.ids \
  --prefill 128 --max-context 4096 --max-new 32 --kv int8 --greedy --stats
```

This builds `quactlize_parity` and `strata-ppu-pack` and adds six strict checks:
dense, grouped, graph, fallback, budget and offline. The CPU oracle quantizes and dequantizes raw GGUF
blocks through ggml, independently of the GPU producer. Tests include all six
formats, strided inputs/outputs, beta, F16/F32/Q8_1 input, merged gate/up order,
mutable expert replacement at the same address, two batches of routed groups,
empty groups, destination scatter, cross-stream graph replay, the ready event,
disk-loaded weights, prepacked expert replacement on graph replay, in-place
prefill interleaving, and calls that must decline without touching output.
The offline container tests cover corrupt/truncated files, missing experts,
source identity and incompatible metadata without a GPU. Missing hardware returns
77 and stops the box script; it is not a numerical pass.

With model arguments, logs contain three runs: native (`NCP=0, quactlize=0`),
NCP (`1,0`) and NCP plus quactlize (`1,1`). Compare the last two to isolate this
change. Confirm a quactlize computation counter in `dispatch-quactlize-1.json`;
unsupported quantization or insufficient memory can leave a model entirely on
the original path. Quality/performance admission is still a separate review of
the saved results. Local source compilation cannot establish a PPU box result.
