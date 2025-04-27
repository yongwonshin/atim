import os
import subprocess
import re
import pandas as pd
import json
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--kick-the-tires", action="store_true", help="Run CPU autotune with single workload for AE kick-the-tires.")
    parser.add_argument("--workload", type=str, help="Specify a single workload in [va, red, mtv, mmtv, ttv, gemv, geva]")
    parser.add_argument("--m", type=int, default=1, help="M dimension")
    parser.add_argument("--n", type=int, default=1, help="N dimension")
    parser.add_argument("--k", type=int, default=1, help="K dimension")
    parser.add_argument("--jsonfile", type=str, default="./reproduced/simplepim_parameters.json")
    args = parser.parse_args()

    tasks = [
        ("va", 1048576, 0),
        ("red", 524288, 1),
        ("va", 16777216, 7),
        ("red", 8388608, 8),
        ("va", 67108864, 14),
        ("red", 34554432, 15),
        ("red", 67108864, 22),
    ]

    if args.workload:
        if args.kick_the_tires:
            raise ValueError("Cannot specify --workload with --kick-the-tires.")
        if args.workload not in ["va", "red"]:
            raise ValueError(f"Invalid workload: {args.workload}. Must be one of [va, red] in SimplePIM.")
        if args.m <= 0:
            raise ValueError("M, N, and K must be positive integers.")
        if (args.workload, args.m) not in tasks:
            print(f"Warning: {args.workload}, {args.m} is not in the list of tasks.")
        tasks = [(args.workload, args.m)]
    if args.kick_the_tires:
        if args.m != 1:
            print("Warning: Kick-the-tires is set, ignore M.")
        tasks = [("red", 8388608)]
    if args.n != 1 or args.k != 1:
        print("Warning: For workloads supported by SimplePIM, N and K values are always 1. Ignoring.")

    jtasks = {}
    with open(args.jsonfile, "r") as f:
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

        # hardcoded for now
        df = pd.read_csv("./reproduced/result_poly.csv")
        df.iloc[row, 18] = times[0]
        df.iloc[row, 19] = times[1]
        df.to_csv("./reproduced/result_poly.csv", index=False)

        print(f"  Extracted times: {times}")




if __name__ == "__main__":
    main()