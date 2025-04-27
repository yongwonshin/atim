import argparse
from tasks import get_tasks

def get_tune_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kick-the-tires", action="store_true", help="Run CPU autotune with single workload for AE kick-the-tires.")
    parser.add_argument("--workload", type=str, help="Specify a single workload in [va, red, mtv, mmtv, ttv, gemv, geva]")
    parser.add_argument("--m", type=int, default=1, help="M dimension, default=1")
    parser.add_argument("--n", type=int, default=1, help="N dimension, default=1")
    parser.add_argument("--k", type=int, default=1, help="K dimension, default=1")
    return parser

def args_to_tasks(args):
    all_tasks = get_tasks()
    if args.workload:
        if args.kick_the_tires:
            raise ValueError("Cannot specify --workload with --kick-the-tires.")
        if args.workload not in ["va", "red", "mtv", "mmtv", "ttv", "gemv", "geva"]:
            raise ValueError(f"Invalid workload: {args.workload}. Must be one of [va, red, mtv, mmtv, ttv, gemv, geva].")
        if args.m <= 0 or args.n <= 0 or args.k <= 0:
            raise ValueError("M, N, and K must be positive integers.")
        _tasks = [(args.workload, args.m, args.n, args.k)]
        if _tasks[0] not in all_tasks:
            print(f"Warning: {args.workload}, {args.m}, {args.n}, {args.k} is not in the list of tasks.")
        return _tasks
    else:
        if args.m != 1 or args.n != 1 or args.k != 1:
            print("Warning: Either M, N, or K are set but ignored when not using --workload to evaluate a single workload.")
        return get_tasks(args.kick_the_tires)