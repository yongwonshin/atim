import sys
import time
import argparse
import os
import multiprocessing
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = f"{os.path.abspath('.')}/tvm_cputest/python:{env['PYTHONPATH']}"

subprocess.run(["python3", "cpu_autotune_submodule.py"], env=env)
