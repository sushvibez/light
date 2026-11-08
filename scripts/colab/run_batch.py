#!/usr/bin/env python3
"""Entry point for `colab run`: clone the repo and run gpu_batch.sh on the VM.

    colab run --gpu L4 scripts/colab/run_batch.py [group ...]

`colab run` releases the VM when this returns, taking /content/logs with it,
so everything worth keeping goes to stdout: the batch's own streaming
output, its summary, and the tail of every failing group's log.
"""
import glob, os, subprocess, sys

REPO = os.environ.get("REPO_URL", "https://github.com/sushvibez/light")
BRANCH = os.environ.get("BRANCH", "main")
WORK, LOGS = "/content/light", "/content/logs"

if not os.path.isdir(f"{WORK}/.git"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO, WORK], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# Pipe and re-print: a Jupyter kernel does not capture a child's own
# stdout, so without this `colab run` shows nothing of the batch.
p = subprocess.Popen(["bash", f"{WORK}/scripts/colab/gpu_batch.sh", *sys.argv[1:]],
                     env={**os.environ, "WORK": WORK, "LOGS": LOGS, "BRANCH": BRANCH},
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()

for log in sorted(glob.glob(f"{LOGS}/*.log")):
    text = open(log, errors="replace").read()
    if "--- FAIL" in text or "\nFAIL" in text or "panic:" in text:
        print(f"\n===== failure log tail: {os.path.basename(log)} =====")
        print("\n".join(text.splitlines()[-80:]))
sys.exit(rc)
