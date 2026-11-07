#!/usr/bin/env python3
"""Compile kernels/cuda/src/*.cu to PTX with NVRTC, without nvcc or a GPU.

    python scripts/nvrtc_ptx.py --all             # what the committed PTX is
    python scripts/nvrtc_ptx.py name [name ...]   # e.g. gated_delta

This is how the committed PTX is built, on Linux or Windows: NVRTC compiles
CUDA C++ to PTX in-process and needs neither a host compiler nor a GPU.
The committed PTX is built with NVRTC 13.0, whose PTX ISA 9.0 loads on any
driver for CUDA 13.0 or newer (a 13.1 nvcc would emit ISA 9.1, which a 13.0
driver rejects). Install it with
`pip install "nvidia-cuda-nvrtc==13.0.*" "nvidia-cuda-runtime==13.0.*"
"nvidia-cuda-cccl==13.0.*"`. Options mirror scripts/build_ptx.sh (the nvcc
route): --std=c++17, fast math, compute_80 (LIGHT_PTX_ARCH overrides).

NVRTC has no <cuda_runtime.h> or <limits.h>; sources that include them get
small stubs (the runtime API is host-side; device code needs only INFINITY,
NAN and the integer limits). cuda_fp16.h and cuda_bf16.h come from the
nvidia-cuda-runtime and nvidia-cuda-cccl wheels (same version as NVRTC).
"""
import ctypes, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "kernels", "cuda", "src")
OUT = os.path.join(ROOT, "kernels", "cuda", "ptx")
ARCH = os.environ.get("LIGHT_PTX_ARCH", "compute_80")


def find_nvrtc():
    import nvidia
    for base in nvidia.__path__:
        for pat in ("cu13/bin/x86_64/nvrtc64_*.dll", "cu13/lib/libnvrtc.so*", "cuda_nvrtc/lib/libnvrtc.so*",
                    "cuda_nvrtc/bin/nvrtc64_*.dll"):
            hits = sorted(glob.glob(os.path.join(base, pat)))
            # Skip the builtins library and the .alt variant (whose own
            # builtins DLL the wheel does not ship).
            hits = [h for h in hits if "builtins" not in os.path.basename(h) and ".alt." not in os.path.basename(h)]
            if hits:
                inc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))), "include")
                if not os.path.isdir(inc):
                    inc = os.path.join(os.path.dirname(os.path.dirname(hits[0])), "include")
                return hits[0], inc
    sys.exit("NVRTC not found: pip install nvidia-cuda-nvrtc==13.0.*")


lib_path, include_dir = find_nvrtc()
if os.name == "nt":
    os.add_dll_directory(os.path.dirname(lib_path))
    os.environ["PATH"] = os.path.dirname(lib_path) + os.pathsep + os.environ.get("PATH", "")  # nvrtc loads its builtins DLL by name
nvrtc = ctypes.CDLL(lib_path)
prog_t = ctypes.c_void_p


def check(rc, what, prog=None):
    if rc == 0:
        return
    msg = ctypes.c_char_p(nvrtc.nvrtcGetErrorString(rc)).value.decode() if hasattr(nvrtc, "nvrtcGetErrorString") else str(rc)
    log = ""
    if prog is not None:
        n = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, buf)
        log = buf.value.decode(errors="replace")
    sys.exit(f"{what}: {msg}\n{log}")


nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p


def compile_one(path):
    name = os.path.splitext(os.path.basename(path))[0]
    src = open(path, "rb").read()
    # `long` is 64-bit on Linux (what the Go launchers are written against)
    # but 32-bit on Windows, and NVRTC follows the host: a kernel parameter
    # of type `unsigned long` would become .u32 here and .u64 there, silently
    # mismatching the launcher. Compile with every lone `long` widened to
    # `long long`, which is what `long` means on LP64. (size_t is 64-bit on
    # both.) The kernels are extern "C", so their names do not change.
    src = re.sub(rb"\blong(\s+long)?\b", lambda m: m.group(0) if m.group(1) else b"long long", src)
    # Header stubs: NVRTC lacks cuda_runtime.h (which under nvcc also pulls
    # in <math.h>, the source of INFINITY / NAN) and <limits.h>.
    headers = [b"#ifndef INFINITY\n#define INFINITY __int_as_float(0x7f800000)\n#endif\n"
               b"#ifndef NAN\n#define NAN __int_as_float(0x7fffffff)\n#endif\n",
               b"#pragma once\n#define CHAR_BIT 8\n#define INT_MAX 2147483647\n#define INT_MIN (-INT_MAX - 1)\n"
               b"#define UINT_MAX 4294967295U\n#define LLONG_MAX 9223372036854775807LL\n"
               b"#define LLONG_MIN (-LLONG_MAX - 1LL)\n#define ULLONG_MAX 18446744073709551615ULL\n"]
    header_names = [b"cuda_runtime.h", b"limits.h"]
    prog = prog_t()
    hs = (ctypes.c_char_p * len(headers))(*headers)
    hn = (ctypes.c_char_p * len(header_names))(*header_names)
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, path.encode(), len(headers), hs, hn), f"{name}: create")
    opts = [f"--gpu-architecture={ARCH}", "--std=c++17", "--use_fast_math", f"-I{SRC}", f"-I{include_dir}"]
    if os.path.isdir(os.path.join(include_dir, "cccl")):  # nv/target etc. (nvidia-cuda-cccl wheel)
        opts.append(f"-I{os.path.join(include_dir, 'cccl')}")
    copts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), copts)
    check(rc, f"{name}: compile", prog)
    n = ctypes.c_size_t()
    check(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n)), f"{name}: ptx size")
    buf = ctypes.create_string_buffer(n.value)
    check(nvrtc.nvrtcGetPTX(prog, buf), f"{name}: ptx")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    out = os.path.join(OUT, name + ".ptx")
    with open(out, "wb") as f:
        f.write(buf.value)
    print(f"nvrtc -> {os.path.relpath(out, ROOT)} ({len(buf.value)} bytes)")


names = sys.argv[1:]
if not names:
    sys.exit("name the .cu files to compile, or --all")
if names == ["--all"]:
    paths = sorted(glob.glob(os.path.join(SRC, "*.cu")))
else:
    paths = [os.path.join(SRC, n if n.endswith(".cu") else n + ".cu") for n in names]
for p in paths:
    compile_one(p)
