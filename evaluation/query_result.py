import os
import argparse
import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.meta_schedule.database import JSONDatabase
import flush


parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
parser.add_argument("--only_show", action="store_true")
parser.add_argument("--only_run", action="store_true")
parser.add_argument("--num_cores", default=os.cpu_count(), type=int)
args = parser.parse_args()

target = Target(f"llvm --num-cores={args.num_cores}")

def ndarray(size, dtype="int32"):
    arr = np.random.randint(0, 50, size=size, dtype=dtype)
    ret = tvm.nd.array(arr)
    flush.flush_cache_clflush(arr)
    return ret


def ndzeros(size, dtype="int32"):
    arr = np.zeros(size, dtype=dtype)
    ret = tvm.nd.array(arr)
    flush.flush_cache_clflush(arr)
    return ret


def query(workdir: str, only_show: False, only_run: False) -> None:
    if not workdir:
        print("")
        return
    parsed = workdir.split("/")[-1].split("_")
    comp = parsed[1]
    m, n, k = int(parsed[2]), int(parsed[3]), int(parsed[4])

    database = JSONDatabase(work_dir=workdir)
    all_records = database.get_all_tuning_records()
    top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
    assert len(top_record.run_secs) == 1
    mod = top_record.workload.mod
    schedule = ms.tir_integration.compile_tir(database, mod, target)
    if not only_run:
        schedule.trace.show()
    mod = schedule.mod
    if only_show:
        return
    f = tvm.build(schedule.mod, target=target)
    evaluator = f.time_evaluator(f.entry_name, tvm.cpu(0), number=1, repeat=100, flushing_cache=True)

    times = []
    print(workdir)

    if comp == "va":
        a = ndarray((m,))
        b = ndarray((m,))
        c = ndzeros((m,))
        t = evaluator(a, b, c)
        times.append(t.median * 1000)

    if comp == "mmtv":
        a = ndarray((m, n, k))
        b = ndarray((m, k))
        c = ndzeros((m, n))
        t = evaluator(a, b, c)
        times.append(t.median * 1000)

    if comp == "mtv":
        a = ndarray((m, k))
        b = ndarray((k,))
        c = ndzeros((m,))
        t = evaluator(a, b, c)
        times.append(t.median * 1000)

    if comp == "red":
        a = ndarray((m,), dtype=np.int64)
        c = ndzeros((1,), dtype=np.int64)
        t = evaluator(a, c)
        times.append(t.median * 1000)

    if comp == "ttv":
        a = ndarray((m, n, k))
        b = ndarray((k,))
        c = ndzeros((m, n))
        t = evaluator(a, b, c)
        times.append(t.median * 1000)

    if comp == "polyva":
        a = ndarray((m,))
        b = ndarray((m,))
        alpha = ndarray((1,))
        beta = ndarray((1,))
        c = ndzeros((m,))
        t = evaluator(a, b, c, alpha, beta)
        times.append(t.median * 1000)

    if comp == "polygemv1":
        a = ndarray((m, k))
        b = ndarray((k,))
        alpha = ndarray((1,))
        c = ndzeros((m,))
        t = evaluator(a, b, c, alpha)
        times.append(t.median * 1000)

    print(times)


if __name__ == "__main__":
    query(args.workdir, args.only_show, args.only_run)
