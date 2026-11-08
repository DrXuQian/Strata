#!/usr/bin/env bash
# Build and check the installed-package PPU backend. Arguments after -- are
# passed to all three model A/B runs without changing model/cache settings.
set -euo pipefail
: "${PPU_SDK:?Set PPU_SDK to the complete installed SDK}"
export PPU_CUDA_ARCH=${PPU_CUDA_ARCH:-80}
export STRATA_PPU_TARGET=${STRATA_PPU_TARGET:-ppu10}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${BUILD_DIR:-"$repo/build-ppu-python"}
out=${OUT_DIR:-"$build/check-$(date -u +%Y%m%dT%H%M%SZ)"}
cmake_bin=${CMAKE:-cmake}
python_bin=${PYTHON:-python3}
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/lib64:$PPU_SDK/lib:$PPU_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == -- ]]; then shift; fi
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
provenance="$out/provenance.txt"
if [[ ${STRATA_CHECK_MODEL_ONLY:-0} == 1 ]]; then provenance="$out/provenance-model.txt"; fi
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short
    git -C "$repo" diff --binary HEAD | sha256sum
    printf 'python=%s target=%s cuda_arch=%s\n' "$python_bin" "$STRATA_PPU_TARGET" "$PPU_CUDA_ARCH"
    printf 'model_args:'; printf ' %q' "$@"; printf '\n'
    "$PPU_SDK/CUDA_SDK/bin/nvcc" --version
} > "$provenance"

if [[ ${STRATA_CHECK_MODEL_ONLY:-0} != 1 ]]; then
# Pure host checks, followed by real package discovery/JIT. A no-device skip
# is a failure of this box command, not a successful numerical measurement.
"$python_bin" "$repo/tests/ppu/test_bridge.py" --build-dir "$out/bridge" 2>&1 | tee "$out/bridge.log"
"$python_bin" "$repo/tests/ppu/test_offline_store.py" --build-dir "$out/offline-store" 2>&1 | tee "$out/offline-store.log"
"$python_bin" "$repo/tests/ppu/test_fetch_admission.py" --build-dir "$out/fetch-admission" \
    --cuda-include "$PPU_SDK/CUDA_SDK/include" 2>&1 | tee "$out/fetch-admission.log"
"$python_bin" "$repo/tests/ppu/test_packages.py" -v 2>&1 | tee "$out/package-contracts.log"
"$python_bin" -m strata_ppu --prepare 2>&1 | tee "$out/packages.log"

"$cmake_bin" -S "$repo" -B "$build" \
    -DSTRATA_ENABLE_PPU=ON -DSTRATA_PPU_FA=ON -DSTRATA_PPU_MOE=ON -DSTRATA_PPU_GDN=ON -DSTRATA_QUACTLIZE=ON \
    -DCMAKE_CUDA_COMPILER="$PPU_SDK/CUDA_SDK/bin/nvcc" \
    -DCUDAToolkit_ROOT="$PPU_SDK/CUDA_SDK" -DCMAKE_CUDA_ARCHITECTURES="$PPU_CUDA_ARCH" \
    -DPython3_EXECUTABLE="$python_bin" -DSTRATA_GGML_DIR="${STRATA_GGML_DIR:-}" \
    -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$out/configure.log"
"$cmake_bin" --build "$build" --target strata strata-ppu-pack ppu_package_parity quactlize_parity \
    -j "${JOBS:-3}" 2>&1 | tee "$out/build.log"
sha256sum "$build/strata" "$build/strata-ppu-pack" "$build/ppu_package_parity" "$build/quactlize_parity" >> "$provenance"
for test_case in gdn-verify gdn-commit gdn-graph gdn-prefix gdn-partial gdn-chunked moe fa fa-q8; do
    env STRATA_PPU_FA=1 STRATA_PPU_MOE=1 STRATA_PPU_GDN=1 \
        "$build/ppu_package_parity" "$test_case" 2>&1 | tee "$out/$test_case.log"
done
for test_case in dense grouped graph fallback budget offline; do
    "$build/quactlize_parity" "$test_case" 2>&1 | tee "$out/quactlize-$test_case.log"
done
fi
if (( $# )); then
    for providers in 0 1; do
        env STRATA_PPU_FA="$providers" STRATA_PPU_MOE="$providers" STRATA_PPU_GDN="$providers" \
            STRATA_QUACTLIZE=0 STRATA_DISPATCH_STATS=1 STRATA_DISPATCH_REPORT="$out/dispatch-packages-$providers.json" \
            "$build/strata" "$@" 2>&1 | tee "$out/model-packages-$providers.log"
    done
    env STRATA_PPU_FA=1 STRATA_PPU_MOE=1 STRATA_PPU_GDN=1 STRATA_QUACTLIZE=1 STRATA_DISPATCH_STATS=1 \
        STRATA_DISPATCH_REPORT="$out/dispatch-quactlize.json" \
        "$build/strata" "$@" 2>&1 | tee "$out/model-quactlize.log"
fi
printf 'Completed command checks. Logs: %s\n' "$out"
printf 'Assess model output and dispatch counts before comparing performance.\n'
