import os
import subprocess
from simplepim_eval import extract_va_times, extract_red_times, run_make_and_execute
import json
import argparse


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

parser = argparse.ArgumentParser()
parser.add_argument("--kick-the-tires", action="store_true", help="Run CPU autotune with single workload for AE kick-the-tires.")
parser.add_argument("--workload", type=str, help="Specify a single workload in [va, red, mtv, mmtv, ttv, gemv, geva]")
parser.add_argument("--m", type=int, default=1, help="M dimension")
parser.add_argument("--n", type=int, default=1, help="N dimension")
parser.add_argument("--k", type=int, default=1, help="K dimension")
parser.add_argument("--jsonfile", type=str, default="./reproduced/simplepim_parameters.json")
args = parser.parse_args()

DPUS = [512, 1024, 1536, 2048]
tasks = [
    ("va", 1048576),
    ("red", 524288),
    ("va", 16777216),
    ("red", 8388608),
    ("va", 67108864),
    ("red", 34554432),
    ("red", 67108864),
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

results = []

for workload, L in tasks:
    print(f"\n=== Running {workload.upper()} with L={L} ===")
    best_dpus = search(workload, L)
    key = f"{workload}_{L}"
    results.append({key: best_dpus})

    os.makedirs(os.path.dirname(args.jsonpath), exist_ok=True)

    with open(args.jsonpath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.jsonpath}")