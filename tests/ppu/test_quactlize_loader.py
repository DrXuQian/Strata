#!/usr/bin/env python3
"""Host-only DSO/ABI negative controls; no CUDA SDK or device required."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    build = parser.parse_args().build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[2]
    cc = [os.environ.get("CXX", "c++"), "-std=c++17", "-I", str(src / "include")]
    driver = build / "quactlize_loader"
    subprocess.run(cc + [str(src / "tests/ppu/quactlize_loader_test.cpp"), str(src / "src/ppu/quactlize.cpp"),
                        "-ldl", "-o", str(driver)], check=True)
    cases = [("complete", [], 1), ("missing", None, 2), ("no_pack", ["NO_PACK"], 2),
             ("no_run", ["NO_RUN"], 2), ("descriptor", ["BAD_DESCRIPTOR"], 2),
             ("bytes", ["MISMATCH"], 2), ("nonneutral", ["NONNEUTRAL"], 2),
             ("reader_layout", ["REJECT_LAYOUT"], 2), ("workspace", ["WORKSPACE"], 2),
             ("reader_declines", ["SHAPE"], 0)]
    count = 0
    for name, flags, expected in cases:
        directory = build / name
        directory.mkdir(exist_ok=True)
        for role, lib in (("PRODUCER", "pack"), ("CONSUMER", "execution")):
            path = directory / f"libquactlize_ppu_{lib}.so"
            path.unlink(missing_ok=True)
            if flags is not None:
                subprocess.run(cc + ["-shared", "-fPIC", f"-DMOCK_{role}"] + [f"-DMOCK_{f}" for f in flags] +
                               [str(src / "tests/ppu/mock_quactlize.cpp"), "-o", str(path)], check=True)
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env.update(STRATA_QUACTLIZE="1", QUACTLIZE_PPU_PACK_LIBRARY=str(directory / "libquactlize_ppu_pack.so"),
                   QUACTLIZE_KPACK_EXECUTION=str(directory))
        checks = [(expected, 12, 256, 512)]
        if name == "complete":
            checks += [(1, q, 256, 512) for q in (8, 10, 11, 13, 14)]
            checks += [(0, 21, 1280, 2560), (0, 42, 2560, 640), (0, 12, 640, 2560),
                       (0, 11, 256, 256), (0, 8, 2560, 640), (1, 12, 1280, 2560)]
        for expect, q, n, k in checks:
            subprocess.run([str(driver), str(expect), str(q), str(n), str(k)], env=env, check=True)
            count += 1
        if name == "missing":
            env["STRATA_QUACTLIZE"] = "0"
            subprocess.run([str(driver), "0", "12", "256", "512"], env=env, check=True)
            count += 1
        print(f"PASS {name}", flush=True)
    print(f"PASS {count} host-only loader/admission cases")


if __name__ == "__main__":
    main()
