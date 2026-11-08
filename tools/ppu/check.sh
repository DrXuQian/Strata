#!/usr/bin/env bash
# Build Strata against the PPU SDK and the existing, shared NCP libraries.
# Optional arguments after -- are passed unchanged to `strata generate` for A/B runs.
set -euo pipefail

: "${PPU_SDK:?Set PPU_SDK to the complete PPU SDK root}"
: "${NCP_LIB_DIR:?Set NCP_LIB_DIR to the existing libncp_fa/moe/gdn.so directory}"
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to the llama.cpp checkout for CPU ggml}"
: "${PPU_CUDA_ARCH:?Set PPU_CUDA_ARCH to the explicit CUDA target supported by your PPU SDK}"

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${BUILD_DIR:-"$repo/build-ppu"}
out=${OUT_DIR:-"$build/check-$(date -u +%Y%m%dT%H%M%SZ)"}
cmake_bin=${CMAKE:-cmake}
with_quactlize=${WITH_QUACTLIZE:-0}
if [[ "$with_quactlize" != 0 && "$with_quactlize" != 1 ]]; then
    printf 'WITH_QUACTLIZE must be 0 or 1\n' >&2
    exit 1
fi
targets=(strata ncp_parity)
if [[ "$with_quactlize" == 1 ]]; then
    : "${QUACTLIZE_KPACK_EXECUTION:?Set the existing quactlize execution library directory}"
    if [[ -z ${QUACTLIZE_PPU_PACK_LIBRARY:-} ]]; then
        : "${QUACTLIZE_PPU_BUNDLE:?Set the existing quactlize producer library directory}"
        export QUACTLIZE_PPU_PACK_LIBRARY="$QUACTLIZE_PPU_BUNDLE/libquactlize_ppu_pack.so"
    fi
    for lib in "$QUACTLIZE_PPU_PACK_LIBRARY" "$QUACTLIZE_KPACK_EXECUTION/libquactlize_ppu_execution.so"; do
        if [[ ! -r "$lib" ]]; then printf 'Missing shared quactlize library: %s\n' "$lib" >&2; exit 1; fi
    done
    targets+=(quactlize_parity)
fi
if [[ ${1:-} == -- ]]; then shift; fi

for lib in fa moe gdn; do
    if [[ ! -r "$NCP_LIB_DIR/libncp_${lib}.so" ]]; then
        printf 'Missing shared NCP library: %s\n' "$NCP_LIB_DIR/libncp_${lib}.so" >&2
        exit 1
    fi
done
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
export LD_LIBRARY_PATH="$NCP_LIB_DIR:$PPU_SDK/CUDA_SDK/lib64:$PPU_SDK/lib:$PPU_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$with_quactlize" == 1 ]]; then
    export LD_LIBRARY_PATH="$QUACTLIZE_KPACK_EXECUTION:$(dirname -- "$QUACTLIZE_PPU_PACK_LIBRARY"):$LD_LIBRARY_PATH"
fi

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short
    git -C "$repo" diff --binary HEAD | sha256sum
    sha256sum "$repo/cmake/ppu_backend.cmake" "$repo"/src/ppu/* "$repo"/include/strata/ppu/*.h \
        "$repo"/include/strata/ppu/*.hpp "$repo"/include/strata/ppu/quactlize/*.h \
        "$repo/tests/ppu/ncp_parity.cu" "$repo/tools/ppu/check.sh"
    if [[ "$with_quactlize" == 1 ]]; then
        sha256sum "$QUACTLIZE_PPU_PACK_LIBRARY" "$QUACTLIZE_KPACK_EXECUTION/libquactlize_ppu_execution.so" \
            "$repo/tests/ppu/quactlize_parity.cu"
        printf 'quactlize_cache_mib=%s scratch_mib=%s\n' "${STRATA_QUACTLIZE_CACHE_MIB:-256}" "${STRATA_QUACTLIZE_SCRATCH_MIB:-256}"
    fi
    printf 'model_args:'
    printf ' %q' "$@"
    printf '\n'
    "$PPU_SDK/CUDA_SDK/bin/nvcc" --version
    sha256sum "$NCP_LIB_DIR/libncp_fa.so" "$NCP_LIB_DIR/libncp_moe.so" "$NCP_LIB_DIR/libncp_gdn.so"
} > "$out/provenance.txt"

"$cmake_bin" -S "$repo" -B "$build" \
    -DSTRATA_ENABLE_PPU=ON \
    -DSTRATA_NCP_FA=ON -DSTRATA_NCP_MOE=ON -DSTRATA_NCP_GDN=ON -DSTRATA_QUACTLIZE=ON \
    -DCMAKE_CUDA_COMPILER="$PPU_SDK/CUDA_SDK/bin/nvcc" \
    -DCUDAToolkit_ROOT="$PPU_SDK/CUDA_SDK" \
    -DCMAKE_CUDA_ARCHITECTURES="$PPU_CUDA_ARCH" \
    -DSTRATA_GGML_DIR="$LLAMA_CPP_DIR" \
    -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$out/configure.log"
"$cmake_bin" --build "$build" --target "${targets[@]}" -j "${JOBS:-3}" 2>&1 | tee "$out/build.log"
sha256sum "$build/strata" "$build/ncp_parity" >> "$out/provenance.txt"

for case_name in gdn-verify gdn-commit gdn-graph gdn-chunked moe fa fa-q8; do
    # Exit 77 (no device/library) is a failure of this box validation command.
    # The executable itself distinguishes that skip from a numerical failure.
    env STRATA_NCP_FA=1 STRATA_NCP_MOE=1 STRATA_NCP_GDN=1 STRATA_NCP_GDN_CHUNKED=1 \
        "$build/ncp_parity" "$case_name" 2>&1 | tee "$out/$case_name.log"
done
if [[ "$with_quactlize" == 1 ]]; then
    sha256sum "$build/quactlize_parity" >> "$out/provenance.txt"
    for case_name in dense grouped graph fallback budget; do
        "$build/quactlize_parity" "$case_name" 2>&1 | tee "$out/quactlize-$case_name.log"
    done
fi

if (( $# )); then
    for enabled in 0 1; do
        printf 'Running model with NCP=%s\n' "$enabled"
        env STRATA_NCP_FA="$enabled" STRATA_NCP_MOE="$enabled" STRATA_NCP_GDN="$enabled" \
            STRATA_NCP_GDN_CHUNKED="$enabled" STRATA_QUACTLIZE=0 STRATA_DISPATCH_STATS=1 \
            STRATA_DISPATCH_REPORT="$out/dispatch-ncp-$enabled.json" \
            "$build/strata" generate "$@" 2>&1 | tee "$out/model-ncp-$enabled.log"
    done
    if [[ "$with_quactlize" == 1 ]]; then
        env STRATA_NCP_FA=1 STRATA_NCP_MOE=1 STRATA_NCP_GDN=1 STRATA_NCP_GDN_CHUNKED=1 \
            STRATA_QUACTLIZE=1 STRATA_DISPATCH_STATS=1 \
            STRATA_DISPATCH_REPORT="$out/dispatch-quactlize-1.json" \
            "$build/strata" generate "$@" 2>&1 | tee "$out/model-quactlize-1.log"
    fi
fi
printf 'Checks completed. Logs: %s\n' "$out"
if (( $# )); then
    printf 'Compare model output, dispatch counters (including actual quactlize selection) and timings across the model logs.\n'
fi
