#!/usr/bin/env python3
"""Build the real host dispatch gate; discard unused GPU paths at link time."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cuda-include", type=Path, required=True)
    args = parser.parse_args()
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    exe = out / "fetch-admission-test"
    subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++20", "-O1", "-pthread",
        "-ffunction-sections", "-fdata-sections", "-DSTRATA_QUACTLIZE=1",
        "-I", str(repo / "include"), "-I", str(args.cuda_include.resolve()),
        str(repo / "src/core/expert_source.cpp"),
        str(repo / "tests/ppu/fetch_admission_test.cpp"),
        "-Wl,--gc-sections", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
