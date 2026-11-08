# PPU: installed operator packages

This opt-in backend targets Qwen3.8-Flash-Next on PPU1.0 / ZW810. The engine uses
the PPU SDK's CUDA-compatible runtime and ACBLAS. Strata contains the adaptation
layer; operator packages are installed normally and remain unchanged.

| Operation | Installed provider | Strata integration |
| --- | --- | --- |
| Selected-cell prompt attention | `flash-attention-for-sail` (`flash_attn_2_cuda.fwd`) | FP16 Tensor views and caller-owned output |
| Prompt expert GEMMs | `DeepGEMM-for-sail` (`deep_gemm.m_grouped_bf16_gemm_nt_nopad`) | BF16 Tensor views; package owns its JIT |
| GDN prefill, verification, accepted-prefix commit | `actlizeLA` (`actlize_la.serving.PreparedGDNCore`) | BF16 Q/K/V/beta, FP32 log gate and KV state; prepared workspaces |
| Quantized dense and routed expert matrices | `QUactlize` | Strata Tensor binding JIT-compiles the installed canonical producer and GEMV kernels |

There is no `libncp_*.so` loader, copied operator C ABI header, or requirement to
build an adapter in an operator repository. Strata embeds the selected Python
interpreter and passes borrowed DLPack views. It does not link LibTorch's C++ ABI.
The original Strata raw-weight CPU offload, predictive fills, cache swaps and
prefill loans remain in use.

## Install and build

Use the PPU platform's Torch in one Python environment. Install the four operator
packages in that same environment. Existing FlashAttention and DeepGEMM host
extensions are used through their Tensor APIs. DeepGEMM prepares its own kernels.

A complete actlizeLA serving wheel is used directly. With a host/source install,
Strata invokes the installed package's existing serving CMake target in its JIT
cache, then uses the same prepared Python interface. QUactlize uses the K-pack/JIT interfaces from its `develop` branch (including
`quactlize/runtime/compiler.py` and `quactlize/packing/word_pack.hpp`). It needs its installed
JIT source/header closure, including the pinned actlize dependency. A minimal
host wheel that omits those files cannot supply them. Editable pip installs of
complete source trees work without changing any operator files:

```sh
# Only needed if these packages are not already installed with their sources.
python -m pip install --no-build-isolation -e /path/to/actlizeLA
python -m pip install --no-build-isolation -e /path/to/QUactlize
```

Paths above are installation inputs. The engine discovers packages through
Python imports; it takes no llama.cpp or operator-source command-line path.
Missing sources, a missing selected entry, or a corrupt packaged library produce
an explicit error. No incompatible binary or CPU computation substitutes for a
requested provider.

Activate that environment and the normal SDK environment, then build:

```sh
export PPU_SDK=/path/to/complete/PPU_SDK
export PPU_CUDA_ARCH=80
export STRATA_PPU_TARGET=ppu10
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/lib64:$PPU_SDK/lib:$PPU_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build-ppu-python \
  -DSTRATA_ENABLE_PPU=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER="$PPU_SDK/CUDA_SDK/bin/nvcc" \
  -DCUDAToolkit_ROOT="$PPU_SDK/CUDA_SDK" \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-ppu-python \
  --target strata strata-ppu-pack ppu_package_parity quactlize_parity -j 3
```

CMake requires Python 3.10+ with embedding headers/library. It copies the Strata
adapter into the build directory. Keep that directory and Python environment
with the executable. Changing Python's major/minor version needs a rebuild;
ordinary compatible operator package updates do not.

`STRATA_PPU_FA`, `STRATA_PPU_MOE`, `STRATA_PPU_GDN`, and `STRATA_QUACTLIZE` are
CMake options enabled by `STRATA_ENABLE_PPU`. The first three also have runtime
switches of the same name, default on. Runtime quantized execution requires
`STRATA_QUACTLIZE=1`. All features remain off in a normal non-PPU build. NVIDIA
MMQ, clusters and PDL remain disabled in the PPU build. ACBLAS handles prompt
shapes outside the installed operators' admitted paths.

## Updating operators and preparing JIT

Use `python -m pip install -U ...` in the selected environment, then restart
Strata. For editable installs, update the installed source checkout and restart.
Strata keeps native images and graph owners alive for the process lifetime.

`STRATA_PPU_JIT_CACHE` defaults to `~/.cache/strata/ppu`.
`STRATA_PPU_JIT_JOBS` defaults to 2 for actlizeLA's build; `CMAKE` selects its CMake
executable. Cache identities include operator sources/headers, adapter sources,
compiler/SDK identity, flags and Python ABI. Entries are locked during build and
have image checksums. Kernel-only updates invalidate the affected JIT entry.
QUactlize compiler flags and runtime libraries come from the installed package.

```sh
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" python -m strata_ppu --prepare
```

This discovers the packages and prepares native images. It does not establish
numerical parity. Preparation/import/JIT happens before capture; replay launches
the captured device work without Python, compilation, or host routing readback.

## Offline weights and offload

Two prepared directories have distinct purposes:

| Setting | Contents |
| --- | --- |
| `--pack DIR` | Original Strata index, dense weights and tokenizer |
| `STRATA_QUACTLIZE_PACK=DIR` | Canonical K-pack planes and GPU expert blobs |

```sh
python tools/iq_pack.py --gguf "$MODEL" --out "$STRATA_PACK" --compat-bf16
build-ppu-python/strata-ppu-pack --gguf "$MODEL" --out "$KPACK"

STRATA_QUACTLIZE=1 STRATA_QUACTLIZE_PACK="$KPACK" \
  build-ppu-python/strata generate --gpu 0 --pack "$STRATA_PACK" --native "$MODEL" \
  --tokens-file "$PROMPT_IDS" --prefill 128 --spec 4 \
  --max-context 4096 --max-new 32 --kv int8 --greedy --stats
```

`MODEL` is the first GGUF shard; both preparers discover the remaining shards.
`--compat-bf16` is needed for the supplied UD-Q4_K_XL pack. Strata's GGUF Python
reader may use an installed compatible `gguf` package; `STRATA_GGUF_PY` remains
an optional setup-time override. Runtime has no llama.cpp dependency. CMake
uses Strata's pinned ggml dependency unless `STRATA_GGML_DIR` is explicitly set.

The K-pack output must be a new directory. It is published atomically after
preparation. The version-1 container and canonical layouts remain compatible
with previously prepared packs: startup compares the actual installed producer's
descriptor and plane sizes before using them. A changed layout is rejected and
requires repacking. Source identity, per-record corruption checks, complete
expert sets, and geometry checks remain enforced.

GPU expert blobs contain packed merged gate/up followed by original down bytes,
with unchanged size/down offset. Initial fill, predictive prefetch, swaps,
refill and prompt loans use the converted source. CPU execution and CPU
complement eviction use the original raw GGUF source. Packed grouped execution
reads fresh device routing and expert pointers on replay. Offline execution
never shuffles model weights at runtime. Raw PCIe misses retain their raw reader.
`STRATA_ADAPT_FETCH` does not admit those raw misses directly into packed cache
slots. Packed layers keep the converted adaptive refill path, including with
`STRATA_ADAPT_FETCH=2`; unpacked layers retain upstream direct admission.

Supported K-pack types are Q8_0 and Q2_K through Q6_K, with N/K divisible by 256
and Q3_K/Q6_K K divisible by 512. Other shapes/types retain their original raw
reader. Converted experts fail closed if their packed reader cannot execute.
The current Tensor binding selects the package's scalar SIMT GEMV recipe
(columns 16, warps 4, split 1), including multirow prompt calls. This is an
untuned correctness baseline; it makes no speed claim and does not import a
measured tensor-core prefill policy.

Additional immutable dense copies are bounded by `STRATA_QUACTLIZE_CACHE_MIB`
(default 256); retained QUactlize scratch is bounded by
`STRATA_QUACTLIZE_SCRATCH_MIB` (default 256). Raw-input calls can decline before
submission if scratch is unavailable. Graph storage remains alive until process
exit. actlizeLA's shared per-stream workspace and Python/Torch runtime memory
are additional allocations.

GDN converts Strata's head order and `[K,Hv,V]` state to the package's
`[1,Hv,K,V]` order. Q scale is applied once by actlizeLA. Verification leaves the
carried state unchanged. Commit masks the rejected suffix using the device
acceptance count and uses the same BF16 operands/core; zero acceptance preserves
the state exactly. Partial replay preserves earlier output rows. These are
BF16 numerical boundaries, not bit-identical FP32 Strata recurrence. QSA decode
keeps Strata's device-length paged attention; FlashAttention handles prompt
selections. Session graphs on temporary streams retain their existing GDN path;
the prepared native verifier uses the package on its owned stream.

## Run on the box

The end-to-end command uses the model and MTP paths supplied for this task:

```sh
PPU_SDK=/path/to/complete/PPU_SDK PYTHON=python3 bash tools/ppu/run_box.sh
```

It builds/checks the adapters, prepares missing model packs, creates a prompt
with the pack's tokenizer, records a routing trace, builds a cache profile, then
runs three model configurations with the same prompt/MTP/profile/cache: original
providers, installed FA/DeepGEMM/actlizeLA, and those providers plus offline
QUactlize. The cache defaults to 256 expert slots so the run can exercise GPU
expert residency. `EXPERT_PROFILE` and `EXPERT_CACHE` override those choices.
The trace run is profile preparation, not a benchmark.

Defaults are:

- `MODEL_DIR=/sim/eec/shared/AI_workspace/llm-models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL`
- `MTP_RT=/sim/eec/shared/AI_workspace/llm-models/MTP-GGUF-modelzoo/Qwen3.8-Flash-Next-MTP/rt`
- `BUILD_DIR=build-ppu-python`, `PPU_CUDA_ARCH=80`, one visible device.

Run `bash tools/ppu/run_box.sh --help` for directory/command overrides. Existing
Strata and K-pack directories are reusable with `STRATA_MODEL_PACK` and
`STRATA_QUACTLIZE_PACK`. MTP takes the runtime directory containing `dense.txt`,
`dense.bin`, and `experts.bin`; use `WITH_MTP=0` to omit it. Native packs still
require `--spec >= 2`. Native loading discovers the PLE shard automatically.

For custom engine arguments use `bash tools/ppu/check.sh -- <generate args>`.
`STRATA_CHECK_MODEL_ONLY=1` reuses a previously checked build for the model runs.
Logs include package locations/versions, source/build identities, individual
operator results, model output, and dispatch reports. Check that packed expert
and provider dispatch counts are nonzero. Process exit alone does not establish
model quality, token equivalence, or performance.

Host checks are `tests/ppu/test_bridge.py` (real CPU Torch, embedded Python,
DLPack ownership/strides/BF16/threads/errors), `test_packages.py` (8 host interface/cache checks),
`test_offline_store.py` (19 source/container checks), and `test_fetch_admission.py`
(raw/packed cache admission in fetch modes 0, 1 and 2). Device tests cover FA with FP16/Q8 KV, BF16 MoE,
GDN verification/commit/chunks/graph/accepted-prefix/partial output, six quant
formats, dynamic routing/weight replacement, graph replay, budgets and offline
artifacts. A no-device exit 77 is a failed box validation, not a pass.
