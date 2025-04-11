import os
import subprocess
import re
import pandas as pd
import json


def extract_va_times(output):
    print(output)
    try:
        map_kernel = float(re.search(r'map function kernel execution time\s*:\s*([0-9.]+)', output).group(1))
        dpu_cpu = float(re.search(r'DPU-CPU Time \(ms\):\s*([0-9.]+)', output).group(1))
        return map_kernel, dpu_cpu
    except Exception as e:
        print("[VA] Failed to extract:", e)
        return 0.0, 0.0

def extract_red_times(output):
    try:
        kernel = float(re.search(r'reduction function kernel execution time\s*:\s*([0-9.]+)', output).group(1))
        host = float(re.search(r'host reduction execution time\s*:\s*([0-9.]+)', output).group(1))
        return kernel, host
    except Exception as e:
        print("[RED] Failed to extract:", e)
        return 0.0, 0.0

def run_make_and_execute(workload, L, dpus):
    folder = f"./baseline/simplepim/benchmarks/{workload}"
    env = os.environ.copy()
    env["NR_DPUS"] = str(dpus)
    env["NR_ELEMENTS"] = str(L)

    subprocess.run(["rm", "-rf", "bin"], cwd=folder)
    subprocess.run(["make"], cwd=folder, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        result = subprocess.run(["./bin/host"], cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return result.stdout.decode()
    except Exception as e:
        print(f"[{folder}] Execution failed for {dpus} DPUs:", e)
        return ""

def main():
    tasks = [
        ("va", 1048576, 0),
        ("red", 524288, 1),
        ("va", 16777216, 7),
        ("red", 8388608, 8),
        ("va", 67108864, 14),
        ("red", 34554432, 15),
        ("red", 67108864, 22),
    ]

    jtasks = {}
    with open("./reproduced/simplepim_parameters.json", "r") as f:
        arr = json.load(f)
        for j in arr:
            key, value = list(j.items())[0]
            jtasks[key] = value


    for workload, L, row in tasks:
        key = f"{workload}_{L}"
        if key not in jtasks.keys():
            print(f"Skipping {key} (no parameters found)")
            continue
        print(f"\n=== Running {workload.upper()} with L={L} ===")

        best_dpus = jtasks[key]
        extractor = extract_va_times if workload == "va" else extract_red_times
        print(workload, L, best_dpus)
        output = run_make_and_execute(workload, L, best_dpus)
        times = extractor(output)

        df = pd.read_csv("./reproduced/result_poly.csv")
        df.iloc[row, 18] = times[0]
        df.iloc[row, 19] = times[1]
        df.to_csv("./reproduced/result_poly.csv", index=False)

        print(f"  Extracted times: {times}")




if __name__ == "__main__":
    main()