#!/usr/bin/env python3
"""Compile kernels/cuda/src/*.cu to PTX with NVRTC, without nvcc or a GPU.

    python scripts/nvrtc_ptx.py name [name ...]   # e.g. gated_delta

The counterpart of scripts/build_ptx.sh for machines without the CUDA
toolkit (e.g. Windows without Visual Studio): NVRTC compiles CUDA C++ to
PTX in-process and needs neither a host compiler nor a GPU. Install it with
`pip install "nvidia-cuda-nvrtc==13.1.*" "nvidia-cuda-runtime==13.1.*"
"nvidia-cuda-cccl==13.1.*"` (13.1 matches the toolkit the
committed PTX was built with). Options mirror build_ptx.sh:
--std=c++17, fast math, compute_80 (LIGHT_PTX_ARCH overrides).

NVRTC has no <cuda_runtime.h>; sources that include it get an empty stub
(the runtime API is host-side, device code does not need it). cuda_fp16.h
and cuda_bf16.h come from the nvidia-cuda-runtime and nvidia-cuda-cccl
wheels (same 13.1 version), installed next to NVRTC.
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
            hits = [h for h in hits if "builtins" not in os.path.basename(h)]
            if hits:
                inc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))), "include")
                if not os.path.isdir(inc):
                    inc = os.path.join(os.path.dirname(os.path.dirname(hits[0])), "include")
                return hits[0], inc
    sys.exit("NVRTC not found: pip install nvidia-cuda-nvrtc==13.1.*")


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
    # `long` is 64-bit on Linux (the nvcc builds of the committed PTX) but
    # 32-bit on Windows, and NVRTC follows the host: a kernel parameter of
    # type `unsigned long` becomes .u32 here and .u64 there, silently
    # mismatching the Go launcher. Only fixed-width types are accepted.
    code = re.sub(rb"//[^\n]*|/\*.*?\*/", b"", src, flags=re.S)
    if re.search(rb"\bsize_t\b", code) or re.search(rb"\blong\b", re.sub(rb"\blong\s+long\b", b"", code)):
        sys.exit(f"{name}: uses long / size_t, whose width differs between Windows and Linux; "
                 "use int or long long (or build this file with scripts/build_ptx.sh on Linux)")
    # Header stubs: NVRTC lacks cuda_runtime.h.
    headers = [b""]
    header_names = [b"cuda_runtime.h"]
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
    sys.exit("name the .cu files to compile; the existing PTX is built with nvcc (scripts/build_ptx.sh)")
paths = [os.path.join(SRC, n if n.endswith(".cu") else n + ".cu") for n in names]
for p in paths:
    compile_one(p)
