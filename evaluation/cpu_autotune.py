import sys
import time
import argparse
import os
import multiprocessing
import subprocess

env = os.environ.copy()
tvm_dir = os.path.abspath("..") + "/python"
new_tvm_dir = tvm_dir[:-7] + "/evaluation/tvm_cputest/python"
python_path = env["PYTHONPATH"]
python_path = python_path.replace(tvm_dir, new_tvm_dir)
env["PYTHONPATH"] = python_path

print(env["PYTHONPATH"])

subprocess.run(["python3", "cpu_autotune_submodule.py"], env=env)
