#!/usr/bin/env python3
"""Compile and exercise the NCP loader without CUDA, models, or external libraries."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX", "c++")
    common = [compiler, "-std=c++17", "-I", str(source / "include")]

    def run(command, **kwargs):
        subprocess.run([str(x) for x in command], check=True, **kwargs)

    driver = build / "loader_test"
    run(common + ["-DSTRATA_NCP_FA", "-DSTRATA_NCP_MOE", "-DSTRATA_NCP_GDN",
                  source / "tests/ppu/loader_test.cpp", source / "src/ppu/ncp.cpp",
                  "-ldl", "-pthread", "-Wl,-rpath,$ORIGIN", "-o", driver])
    cases = [
        ("missing", [], 0, {}),
        ("complete", ["FA", "MOE", "GDN"], 15, {}),
        ("old_fa", ["OLD_FA", "MOE", "GDN"], 14, {}),
        ("no_nopad", ["FA", "MOE_OLD", "GDN"], 13, {}),
        ("no_workspace", ["FA", "MOE", "GDN", "NO_WORKSPACE"], 7, {}),
        ("disabled", ["FA", "MOE", "GDN"], 0,
         {"STRATA_NCP_FA": "0", "STRATA_NCP_MOE": "OFF", "STRATA_NCP_GDN": "false"}),
    ]
    for name, flags, expected, overrides in cases:
        directory = build / name
        directory.mkdir(exist_ok=True)
        shutil.copy2(driver, directory / driver.name)
        for lib in ("fa", "moe", "gdn"):
            path = directory / f"libncp_{lib}.so"
            path.unlink(missing_ok=True)
            if not flags:
                continue
            lib_flags = [lib.upper()] if lib.upper() in flags else []
            if lib == "fa" and "OLD_FA" in flags:
                lib_flags.append("OLD_FA")
            if lib == "moe" and "MOE_OLD" in flags:
                lib_flags.append("MOE_OLD")
            if lib == "gdn" and "NO_WORKSPACE" in flags:
                lib_flags.append("NO_WORKSPACE")
            run(common + [f"-DMOCK_{flag}" for flag in lib_flags] + ["-shared", "-fPIC",
                source / "tests/ppu/mock_ncp.cpp", "-o", path])
        env = dict(os.environ)
        # Only fixture libraries may satisfy the test, including the missing case.
        env.pop("LD_PRELOAD", None)
        env["LD_LIBRARY_PATH"] = str(directory)
        for key in ("STRATA_NCP_FA", "STRATA_NCP_MOE", "STRATA_NCP_GDN"):
            env.pop(key, None)
        env.update(overrides)
        run([directory / driver.name, expected], env=env)
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
