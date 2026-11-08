"""Prepare package sources in a content-addressed cache, never in capture.

Source installs may be editable pip installs. No operator checkout is patched,
and no generated file is written into site-packages or an operator checkout.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def source_root(package, required):
    module = importlib.import_module(package)
    root = Path(module.__file__).resolve().parent.parent
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise RuntimeError(
            f"installed {package} lacks its JIT sources: {', '.join(missing)}. "
            "Install the full host/source distribution, or use "
            f"'{sys.executable} -m pip install --no-build-isolation -e /path/to/{package}'. "
            "Strata does not modify or download operator source at run time.")
    return root


def sdk_root():
    value = os.environ.get("PPU_SDK") or os.environ.get("PPU_SDK_ROOT")
    if not value:
        raise RuntimeError("set PPU_SDK to the installed PPU SDK for first-use JIT")
    root = Path(value).resolve(strict=True)
    if not (root / "bin/hgcc").is_file():
        raise RuntimeError(f"{root} has no bin/hgcc")
    return root


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(root, folders):
    entries = []
    suffixes = {".h", ".hpp", ".cuh", ".cu", ".cpp", ".inc", ".inl", ".cmake", ".py", ".json", ".txt"}
    for folder in folders:
        for p in sorted((root / folder).rglob("*")):
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
                entries.append((str(p.relative_to(root)), file_hash(p)))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def identity(kind, root, folders, sdk, options):
    runtime = {}
    for name in ("VERSION.txt", "bin/ppu_clang", "bin/ppu-llc", "lib/libhggcrt1.so",
                 "lib/libhggc_wrapper.so", "lib/libhggc.so", "lib/libhg_wrapper.so"):
        path = sdk / name
        if path.is_file():
            runtime[name] = file_hash(path)
    return dict(schema=1, kind=kind, source=tree_hash(root, folders),
                adapter=tree_hash(Path(__file__).parent, ["native"]),
                builder=file_hash(__file__), python=sysconfig.get_config_var("SOABI"),
                sdk=str(sdk), compiler=file_hash(sdk / "bin/hgcc"),
                sdk_headers=tree_hash(sdk, ["include"]), runtime=runtime,
                options=options)


@contextmanager
def cache_entry(record):
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get("STRATA_PPU_JIT_CACHE", Path.home() / ".cache/strata/ppu"))
    entry = root / key
    entry.mkdir(parents=True, exist_ok=True)
    with (entry / "build.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield entry


def run(command, directory, label):
    log = directory / (label + ".log")
    print(f"[strata-ppu] JIT {label}; log={log}", file=sys.stderr, flush=True)
    with log.open("w") as f:
        p = subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"PPU JIT {label} failed (exit {p.returncode}); see {log}")


def valid(entry, image, record):
    receipt = entry / "receipt.json"
    if not receipt.is_file() or not image.is_file():
        return False
    try:
        saved = json.loads(receipt.read_text())
        return saved == dict(identity=record, image=file_hash(image))
    except (OSError, ValueError):
        return False


def publish(entry, image, record):
    temporary = entry / "receipt.tmp"
    temporary.write_text(json.dumps(dict(identity=record, image=file_hash(image)), sort_keys=True))
    temporary.replace(entry / "receipt.json")


def quactlize(qtype):
    # These are installed QUactlize's kernels and producer. Strata supplies only
    # a Tensor binding, launch selection and a cache, not a second quantizer.
    root = source_root("quactlize", ["quactlize/runtime/compiler.py", "quactlize/packing/word_pack.hpp",
        "quactlize/execution/gemv.cu", "third_party/actlize/include/cutlass/cutlass.h"])
    from quactlize.runtime.compiler import FLAGS, LIBRARIES
    sdk = sdk_root()
    module = f"_strata_quact_{qtype}"
    native = Path(__file__).parent / "native"
    flags = [*FLAGS, f"-DQKG_QTYPE={qtype}"]
    includes = [root / "quactlize/include", root / "quactlize/execution", root / "quactlize/packing",
                root / "third_party/actlize/include", root / "third_party/actlize/tools/util/include", native]
    folders = ["quactlize", "third_party/actlize/include", "third_party/actlize/tools/util/include"]
    host = os.environ.get("CXX", "c++")
    host_version = subprocess.check_output([host, "--version"], text=True)
    record = identity("quactlize-tensor", root, folders, sdk, [qtype, flags, host_version])
    with cache_entry(record) as entry:
        image = entry / (module + sysconfig.get_config_var("EXT_SUFFIX"))
        if not valid(entry, image, record):
            run([sdk / "bin/hgcc", *flags, *[f"-I{p}" for p in includes],
                 "-c", native / "quact.cu", "-o", entry / "device.o"], entry, "quact-device")
            run([host, "-std=c++17", "-O2", "-fPIC", "-fvisibility=hidden",
                 f"-I{sysconfig.get_path('include')}", f"-DSTRATA_MODULE={module}",
                 "-c", native / "tensor_module.cpp", "-o", entry / "binding.o"], entry, "quact-binding")
            run([host, "-shared", entry / "binding.o", entry / "device.o",
                 f"-L{sdk / 'lib'}", f"-Wl,-rpath,{sdk / 'lib'}",
                 *[f"-l{name}" for name in LIBRARIES], "-o", image], entry, "quact-link")
            if identity("quactlize-tensor", root, folders, sdk, [qtype, flags, host_version]) != record:
                raise RuntimeError("installed QUactlize/SDK changed during JIT; restart")
            publish(entry, image, record)
    spec = importlib.util.spec_from_file_location(module, image)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def gdn_library(target):
    from actlize_la.serving import NativeLibrary
    import actlize_la
    # A selected complete wheel must validate itself; a corrupt wheel is an
    # error, not permission to silently substitute another implementation.
    package = Path(actlize_la.__file__).resolve().parent
    if any((package / name).exists() for name in ("_native/wheel.json", "_native/manifest.json")):
        return NativeLibrary(target=target)
    # Host install: use the package's own CMake target and Python API.
    root = source_root("actlize_la", ["csrc/serving/CMakeLists.txt", "cmake/GdnHgccArch.cmake",
                                    "third_party/actlize/CMakeLists.txt"])
    sdk = sdk_root()
    cmake = os.environ.get("CMAKE", "cmake")
    jobs = int(os.environ.get("STRATA_PPU_JIT_JOBS", "2"))
    if jobs < 1:
        raise ValueError("STRATA_PPU_JIT_JOBS must be positive")
    folders = ["csrc", "include", "cmake", "third_party/actlize"]
    options = [target, subprocess.check_output([cmake, "--version"], text=True)]
    record = identity("actlize-la-serving", root, folders, sdk, options)
    with cache_entry(record) as entry:
        image = entry / "install/lib/libactlize_la_serving.so"
        if not valid(entry, image, record):
            run([cmake, "-S", root / "csrc/serving", "-B", entry / "build",
                 f"-DALA_TARGET={target}", f"-DPPU_SDK_ROOT={sdk}",
                 "-DALA_BUILD_SERVING_EXAMPLE=OFF", "-DCMAKE_BUILD_TYPE=Release",
                 f"-DCMAKE_INSTALL_PREFIX={entry / 'install'}"], entry, "actlize-configure")
            run([cmake, "--build", entry / "build", "--parallel", jobs], entry, "actlize-build")
            run([cmake, "--install", entry / "build"], entry, "actlize-install")
            if identity("actlize-la-serving", root, folders, sdk, options) != record:
                raise RuntimeError("installed actlizeLA/SDK changed during JIT; restart")
            publish(entry, image, record)
    return NativeLibrary(image, target=target)
