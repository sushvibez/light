#!/usr/bin/env python3
"""Entry point for `colab run` on a CPU VM: clone the repo, install Go, run
Go tests (with -heavy: LIGHT_HEAVY_TESTS=1, real checkpoints), and print everything to stdout (the VM
and its files go away when this returns).

    colab run --high-mem scripts/colab/run_cpu_tests.py [-heavy] [-run REGEX] [-timeout 90m] [pkg ...]

    # real-weight checks of the newest models
    colab run --high-mem scripts/colab/run_cpu_tests.py -heavy -run 'LFM3|Spark' ./models
    # the whole models package, tiny fixtures only (too heavy for the laptop)
    colab run --high-mem scripts/colab/run_cpu_tests.py ./models

The default packages are ./... . --high-mem (Colab Pro) gives ~50 GB RAM;
without it the VM has ~12 GB, enough for the tiny tests and ~2B models.
WebGPU tests skip (no wgpu-native on the VM). Checkpoints download from the
Hub inside the VM.
"""
import os, subprocess, sys, urllib.request

REPO = os.environ.get("REPO_URL", "https://github.com/sushvibez/light")
BRANCH = os.environ.get("BRANCH", "main")
WORK = "/content/light"


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def stream(args, env):
    # A Jupyter kernel does not show a child's own stdout: pipe and re-print.
    p = subprocess.Popen(args, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


args = sys.argv[1:]
run, timeout, heavy, pkgs = "", "90m", False, []
i = 0
while i < len(args):
    if args[i] == "-heavy":
        heavy, i = True, i + 1
    elif args[i] == "-run":
        run, i = args[i + 1], i + 2
    elif args[i] == "-timeout":
        timeout, i = args[i + 1], i + 2
    else:
        pkgs.append(args[i])
        i += 1
pkgs = pkgs or ["./..."]

want = urllib.request.urlopen("https://go.dev/VERSION?m=text").read().decode().splitlines()[0]
if not os.path.exists("/usr/local/go/bin/go") or want not in subprocess.run(
        ["/usr/local/go/bin/go", "version"], capture_output=True, text=True).stdout:
    sh(f"curl -fsSL https://go.dev/dl/{want}.linux-amd64.tar.gz -o /tmp/go.tgz && "
       "rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz")
if os.path.isdir(f"{WORK}/.git"):
    sh(f"git -C {WORK} fetch -q --depth 1 origin {BRANCH} && git -C {WORK} reset -q --hard FETCH_HEAD")
else:
    sh(f"git clone -q --depth 1 --branch {BRANCH} {REPO} {WORK}")
sh(f"git -C {WORK} log -1 --format='%h %s'")
sh("nproc; free -g | head -2")

env = {**os.environ, "PATH": "/usr/local/go/bin:" + os.environ["PATH"], "CGO_ENABLED": "0"}
if heavy:
    env["LIGHT_HEAVY_TESTS"] = "1"
cmd = ["go", "test", "-count=1", f"-timeout={timeout}", "-v"]
if run:
    cmd += ["-run", run]
cmd += pkgs
print("$", " ".join(cmd), flush=True)
sys.exit(stream(cmd, env))
