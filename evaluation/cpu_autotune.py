import sys
import time
import argparse
import os
import multiprocessing
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = f"{os.path.abspath('.')}/tvm_cputest/python:{env.get('PYTHONPATH', '')}"

parser = argparse.ArgumentParser()
parser.add_argument("--kick-the-tires", action="store_true", help="Run CPU autotune with single workload for AE kick-the-tires.")
args = parser.parse_args()

cmd = ["python3", "cpu_autotune_submodule.py"]
if args.kick_the_tires:
    cmd.append("--kick-the-tires")
subprocess.run(cmd, env=env)
