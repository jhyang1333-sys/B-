import os
import pathlib
import subprocess
import sys

CWD = pathlib.Path(r"C:\Users\lenovo\Desktop\科研招聘项目\B新")
CMD = [
    str(CWD / ".conda" / "python.exe"),
    "scripts/run_energy_pipeline.py",
    "--refresh-cache",
    "--no-cache",
    "--cache-key",
    "timed_test",
]

print("Running run_energy_pipeline.py with timeout=120s ...", flush=True)
env = os.environ.copy()
env["PYTHONPATH"] = str(CWD / "src")

try:
    subprocess.run(CMD, cwd=str(CWD), check=True, timeout=120, env=env)
except subprocess.TimeoutExpired:
    print("Process timed out after 120 seconds.", file=sys.stderr, flush=True)
    sys.exit(124)
except subprocess.CalledProcessError as exc:
    print(
        f"Process exited with code {exc.returncode}.", file=sys.stderr, flush=True)
    sys.exit(exc.returncode)
