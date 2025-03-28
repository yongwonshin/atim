import os
import subprocess
import re
import pandas as pd

DPUS = [512, 1024, 1536, 2048]

def extract_va_times(output):
    try:
        map_kernel = float(re.search(r'map function kernel execution time\s*:\s*([0-9.]+)', output).group(1))
        dpu_cpu = float(re.search(r'DPU-CPU Time \(ms\):\s*([0-9.]+)', output).group(1))
        return map_kernel, dpu_cpu
    except Exception as e:
        print("[VA] Failed to extract:", e)
        return 0, 0

def extract_red_times(output):
    try:
        kernel = float(re.search(r'reduction function kernel execution time\s*:\s*([0-9.]+)', output).group(1))
        host = float(re.search(r'host reduction execution time\s*:\s*([0-9.]+)', output).group(1))
        return kernel, host
    except Exception as e:
        print("[RED] Failed to extract:", e)
        return 0, 0

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

def search(workload, L):
    best_dpus = None
    best_times = None
    best_sum = float("inf")

    extractor = extract_va_times if workload == "va" else extract_red_times

    for dpus in DPUS:
        print(f"Testing {workload} with L={L}, DPUs={dpus}...")
        output = run_make_and_execute(workload, L, dpus)
        times = extractor(output)
        if times:
            total = sum(times)
            print(f"  Extracted times: {times}, total: {total}")
            if total < best_sum:
                best_sum = total
                best_times = times
                best_dpus = dpus

    return best_dpus, best_times

def main():
    tasks = [
        ("va", 1048576),
        ("red", 524288),
        ("va", 16777216),
        ("red", 8388608),
        ("va", 67108864),
        ("red", 34554432),
        ("red", 67108864),
    ]

    results = []

    for workload, L in tasks:
        print(f"\n=== Running {workload.upper()} with L={L} ===")
        best_dpus, best_times = search(workload, L)
        if best_times:
            results.append({
                "type": workload,
                "L": L,
                "dpus": best_dpus,
                "times": best_times
            })
        else:
            print(f"Failed to extract results for {workload} with L={L}")

    print("\n===== Summary =====")
    for r in results:
        print(f"[{r['type'].upper()}] L={r['L']} → Best DPUs={r['dpus']}, Times={r['times']}, Total={sum(r['times'])}")

    df = pd.read_csv("./graph/result_poly.csv")
    row_sequence = [0, 1, 7, 8, 14, 15, 22]

    for i, r in enumerate(results):
        df.iloc[row_sequence[i], 18] = r["times"][0]
        df.iloc[row_sequence[i], 19] = r["times"][1]
    df.to_csv("./graph/result_poly.csv", index=False)


if __name__ == "__main__":
    main()