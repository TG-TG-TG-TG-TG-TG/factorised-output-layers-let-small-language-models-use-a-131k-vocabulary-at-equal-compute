"""Wait until no process command line matches <tag>.out (e.g. a chain started with '> runs/round11/<tag>.out').
The pattern is built here, so this process's own command line (which only holds <tag>) never matches.
python round11_wait.py chainR1"""
import subprocess
import sys
import time

pat = sys.argv[1] + r"\.out"
while subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
    time.sleep(15)
print(f"[wait] nothing matches {pat}; going on", flush=True)
