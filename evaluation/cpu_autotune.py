import sys
import os
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = f"{os.path.abspath('.')}/tvm_cputest/python:{env.get('PYTHONPATH', '')}"
sys.path.insert(0, "tvm_cputest/python")

cmd = ["python3", "cpu_autotune_submodule.py", *sys.argv[1:]]
print(cmd)
subprocess.run(cmd, env=env)