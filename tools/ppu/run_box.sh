#!/usr/bin/env bash
# End-to-end Qwen3.8-Flash-Next check on the user's box, using installed packages.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
PPU_SDK=/path/to/sdk bash tools/ppu/run_box.sh
Uses the supplied /sim/eec/shared/AI_workspace model and MTP locations.
Overrides: MODEL_DIR, MTP_RT, STRATA_MODEL_PACK, STRATA_QUACTLIZE_PACK,
PROMPT_IDS, EXPERT_PROFILE, EXPERT_CACHE (default 256), WITH_MTP (default 1),
BUILD_DIR, OUT_DIR, PYTHON, CMAKE, JOBS, PPU_CUDA_ARCH (default 80).
The four operator packages and platform Torch must already be installed in PYTHON.
Creates new pack directories if absent. Existing prepared packs are reused.
HELP
    exit 0
fi
: "${PPU_SDK:?Set PPU_SDK to the complete installed PPU SDK}"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export BUILD_DIR=${BUILD_DIR:-"$repo/build-ppu-python"}
export OUT_DIR=${OUT_DIR:-"$BUILD_DIR/box-$(date -u +%Y%m%dT%H%M%SZ)"}
export PPU_CUDA_ARCH=${PPU_CUDA_ARCH:-80}
export STRATA_PPU_TARGET=${STRATA_PPU_TARGET:-ppu10}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON=${PYTHON:-python3}
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/lib64:$PPU_SDK/lib:$PPU_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
model_dir=${MODEL_DIR:-/sim/eec/shared/AI_workspace/llm-models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL}
mtp=${MTP_RT:-/sim/eec/shared/AI_workspace/llm-models/MTP-GGUF-modelzoo/Qwen3.8-Flash-Next-MTP/rt}
pack=${STRATA_MODEL_PACK:-"$repo/pack/qwen38-ppu-bf16"}
export STRATA_QUACTLIZE_PACK=${STRATA_QUACTLIZE_PACK:-"$repo/pack/qwen38-ppu-kpack"}
mapfile -t shards < <("$PYTHON" - "$model_dir" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
files=sorted(p.glob('*-00001-of-*.gguf'))
if not files:files=sorted(p.glob('*.gguf'))
for f in files:print(f)
PY
)
if (( ${#shards[@]} != 1 )); then
    printf 'Expected one first GGUF shard in %s; set MODEL_DIR to the model directory.\n' "$model_dir" >&2
    exit 2
fi
model=${shards[0]}
if [[ ${WITH_MTP:-1} == 1 ]]; then
    for file in dense.txt dense.bin experts.bin; do
        [[ -s "$mtp/$file" ]] || { printf 'Missing MTP runtime file: %s\n' "$mtp/$file" >&2; exit 2; }
    done
fi
bash "$repo/tools/ppu/check.sh"
if [[ ! -s "$pack/dense.txt" ]]; then
    "$PYTHON" "$repo/tools/iq_pack.py" --gguf "$model" --out "$pack" --compat-bf16
fi
if [[ ! -s "$STRATA_QUACTLIZE_PACK/index.txt" ]]; then
    "$BUILD_DIR/strata-ppu-pack" --gguf "$model" --out "$STRATA_QUACTLIZE_PACK"
fi
prompt=${PROMPT_IDS:-"$OUT_DIR/prompt.ids"}
if [[ -z ${PROMPT_IDS:-} ]]; then
    "$PYTHON" - "$repo" "$pack" "$prompt" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
import check_lookup_window_serve as fixture
fixture.PROMPT = ('<|im_start|>user\nRepeat the following text exactly:\n\n' + fixture.TEXT*4 +
                  '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
ids=fixture.prompt_ids(Path(sys.argv[2])/'tokenizer')
if not 128<=len(ids)<=2048:raise SystemExit(f'fixture length outside test range: {len(ids)}')
Path(sys.argv[3]).write_text(' '.join(map(str,ids))+'\n')
print(f'Prepared {len(ids)} prompt tokens using the pack tokenizer')
PY
fi
args=(--gpu 0 --pack "$pack" --native "$model" --tokens-file "$prompt" --prefill 128 \
      --spec 4 --max-context 4096 --max-new "${MAX_NEW:-32}" --kv int8 --greedy --stats)
if [[ ${WITH_MTP:-1} == 1 ]]; then args+=(--mtp "$mtp"); fi
profile=${EXPERT_PROFILE:-"$OUT_DIR/expert-profile.bin"}
if [[ -z ${EXPERT_PROFILE:-} ]]; then
    # A trace from this prompt ranks the cache. A/B runs then use the same
    # profile and cache size; the trace run is not a performance measurement.
    env STRATA_PPU_FA=0 STRATA_PPU_MOE=0 STRATA_PPU_GDN=0 STRATA_QUACTLIZE=0 \
        "$BUILD_DIR/strata" generate "${args[@]}" --expert-cache 0 \
        --dump-routing "$OUT_DIR/profile.trace" 2>&1 | tee "$OUT_DIR/profile-run.log"
    [[ -s "$OUT_DIR/profile.trace" ]] || { printf 'Routing trace is empty; cannot rank the expert cache.\n' >&2; exit 1; }
    "$PYTHON" "$repo/tools/make_profile.py" "$OUT_DIR/profile.trace" --no-base --out "$profile"
fi
STRATA_CHECK_MODEL_ONLY=1 bash "$repo/tools/ppu/check.sh" -- "${args[@]}" \
    --expert-profile "$profile" --expert-cache "${EXPERT_CACHE:-256}"
printf 'Box run complete; inspect model output and dispatch reports in %s\n' "$OUT_DIR"
