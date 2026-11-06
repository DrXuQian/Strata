#!/usr/bin/env python3
"""Build/run the actual embedded bridge with this interpreter's CPU Torch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import textwrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    out = parser.parse_args().build_dir.resolve()
    import torch  # Fail early if the selected interpreter does not have Torch.
    out.mkdir(parents=True, exist_ok=True)
    package = out / "strata_ppu"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "adapter.py").write_text(textwrap.dedent('''
        import torch
        def call(op, capsules, integers, device, stream, capturing):
            tensors = tuple(torch.utils.dlpack.from_dlpack(x) for x in capsules)
            if op == "copy":
                a,b = tensors
                b.copy_(a)
                return (a.data_ptr(),b.data_ptr(),*a.stride())
            if op == "empty":
                assert tensors[0].numel() == 0
                return
            raise ValueError("intentional adapter failure")
    '''))
    root = Path(__file__).resolve().parents[2]
    libdir = sysconfig.get_config_var("LIBDIR")
    name = sysconfig.get_config_var("LDLIBRARY").removeprefix("lib").split(".so")[0]
    exe = out / "bridge-test"
    command = [os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-pthread",
               f"-I{root / 'include'}", f"-I{sysconfig.get_path('include')}",
               f'-DSTRATA_PPU_PYTHON="{sys.executable}"', f'-DSTRATA_PPU_MODULE_DIR="{out}"',
               root / "src/ppu/python.cpp", root / "tests/ppu/bridge_test.cpp",
               f"-L{libdir}", f"-Wl,-rpath,{libdir}", f"-l{name}", "-o", exe]
    subprocess.run(list(map(str, command)), check=True)
    subprocess.run([str(exe)], check=True, timeout=90)


if __name__ == "__main__":
    main()
