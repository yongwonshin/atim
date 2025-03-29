import os
import subprocess
from simplepim_eval import extract_va_times, extract_red_times, run_make_and_execute
import json


DPUS = [512, 1024, 1536, 2048]

# def run_make_and_execute(workload, L, dpus):
#     folder = f"./baseline/simplepim/benchmarks/{workload}"
#     env = os.environ.copy()
#     env["NR_DPUS"] = str(dpus)
#     env["NR_ELEMENTS"] = str(L)

#     subprocess.run(["rm", "-rf", "bin"], cwd=folder)
#     subprocess.run(["make"], cwd=folder, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     try:
#         result = subprocess.run(["./bin/host"], cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
#         return result.stdout.decode()
#     except Exception as e:
#         print(f"[{folder}] Execution failed for {dpus} DPUs:", e)
#         return ""


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
                best_dpus = dpus
    return best_dpus

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
    best_dpus = search(workload, L)
    key = f"{workload}_{L}"
    results.append({key: best_dpus})

    output_path = "./reproduced/simplepim_parameters.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")