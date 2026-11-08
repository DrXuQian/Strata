#!/usr/bin/env python3
"""Exercise source binding, expert completeness and corrupt offline artifacts."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def checksum(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    build = parser.parse_args().build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    driver = build / "offline_store_test"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-I", str(repo / "include"),
                    str(repo / "tests/ppu/offline_store_test.cpp"), str(repo / "src/ppu/offline.cpp"),
                    str(repo / "src/ppu/quactlize.cpp"), "-ldl", "-pthread", "-o", str(driver)], check=True)
    source = build / "source.gguf"
    source.write_bytes(b"source identity fixture")
    stat = source.stat()
    # std::quoted expects double quotes and escapes, independently of the writer.
    quoted = '"' + str(source).replace('\\', '\\\\').replace('"', '\\"') + '"'
    src = f"source {quoted} {stat.st_dev} {stat.st_ino} {stat.st_size} {stat.st_mtime_ns // 10**9} {stat.st_mtime_ns % 10**9}"
    raw = 69632
    payloads = [bytes([3])*raw, bytes([4])*(raw+128), bytes([5])*(raw+128)]
    records, offset = [], 0
    for i, data in enumerate(payloads):
        # Q8_0 fixture: byte-neutral three-plane shape, expert payload keeps raw down.
        fields = ['record', '"dense.weight"' if i == 0 else f'"expert.{i-1}"',
                  -1 if i == 0 else 2, -1 if i == 0 else i-1, 1 if i == 0 else 2,
                  8, -1 if i == 0 else 8, 256, 256, offset, len(data), checksum(data),
                  2, 1, 8, 0, 256, 256, 32, 0, 123, raw, 65536, 0, 4096]
        records.append(' '.join(map(str, fields)))
        offset += len(data)
    good = ['strata.quactlize.pack 1', src, *records]
    data = b''.join(payloads)
    tests = [("valid", good, data, "bound", source, 0),
             ("unbound", good, data, "unbound", source, 1),
             ("version", [good[0]+'0', *good[1:]], data, "bound", source, 1),
             ("truncated", good, data[:-1], "bound", source, 1),
             ("corrupt_dense", good, b'X'+data[1:], "bound", source, 1),
             ("corrupt_down", good, data[:-1]+b'X', "bound", source, 1),
             ("missing_expert", good[:-1], b''.join(payloads[:-1]), "bound", source, 1),
             ("no_sources", [good[0], *records], data, "bound", source, 1),
             ("duplicate_source", [good[0], src, src, *records], data, "bound", source, 1)]
    other = build / "other.gguf"
    other.write_bytes(source.read_bytes())
    tests.append(("wrong_source", good, data, "bound", other, 1))
    # Mutate fields without using the implementation's serialization helpers.
    for name, row, col, val in [("overlap", 3, 9, "0"), ("layout_version", 2, 12, "1"),
                                ("nonneutral", 2, 24, "4095"), ("expert_gap", 3, 3, "2"),
                                ("expert_mismatch", 4, 7, "512"), ("wrong_count", 3, 4, "3"),
                                ("bad_geometry", 2, 8, "640"), ("bad_type", 2, 5, "21")]:
        rows = good.copy()
        f = shlex.split(rows[row]); f[col] = val
        rows[row] = ' '.join(f)
        tests.append((name, rows, data, "bound", source, 1))
    for name, lines, blob, mode, model, expect in tests:
        root = build / name
        root.mkdir(exist_ok=True)
        (root / "index.txt").write_text('\n'.join(lines)+'\n')
        (root / "weights.bin").write_bytes(blob)
        result = subprocess.run([str(driver), mode, str(root), str(model)], capture_output=True, text=True)
        if result.returncode != expect:
            raise RuntimeError(f"{name}: wanted {expect}, got {result.returncode}: {result.stdout}{result.stderr}")
        print(f"PASS {name}")
    # Same path, different metadata must also invalidate a prepared pack.
    source.write_bytes(b"changed source")
    result = subprocess.run([str(driver), "bound", str(build / "valid"), str(source)], capture_output=True)
    assert result.returncode == 1, "changed source was accepted"
    print(f"PASS {len(tests)+1} host-only offline container cases")


if __name__ == "__main__":
    main()
