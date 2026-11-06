"""Inspect installed providers and optionally prepare their native images."""
import argparse
import importlib
import importlib.metadata
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="prepare imports/JIT on the current PPU")
    args = parser.parse_args()
    import torch
    from . import adapter
    providers = {}
    for name in ("flash_attn", "deep_gemm", "actlize_la", "quactlize"):
        module = importlib.import_module(name)
        distributions = importlib.metadata.packages_distributions().get(name, [])
        providers[name] = dict(path=module.__file__, distributions={
            d: importlib.metadata.version(d) for d in distributions})
    print(json.dumps(dict(python=sys.executable, torch=torch.__version__,
                         target=os.environ.get("STRATA_PPU_TARGET", "ppu10"), packages=providers), indent=2))
    if args.prepare:
        if not torch.cuda.is_available():
            print("SKIP: no usable PPU; package imports are not device validation", file=sys.stderr)
            return 77
        device = torch.cuda.current_device()
        adapter._device(device, False)
        adapter._module("fa")
        adapter._module("moe")
        adapter._module("gdn")
        with torch.inference_mode():
            adapter._execute("prepare_quact", (), (), torch.cuda.current_stream().cuda_stream, False)
        print("Prepared installed providers; numerical validation is in ppu_package_parity/quactlize_parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
