import tvm
import tvm.testing
import multiprocessing
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.meta_schedule.database import JSONDatabase
import argparse

from bench import *
from workloads import get_workload


def query(workdir: str, only_show=False, only_run=False) -> None:
    target = Target(f"upmem --num-cores={multiprocessing.cpu_count()}")
    parsed = workdir.split("/")[-1].split("_")
    dtype = "int32"
    if "red" in workdir or "dot" in workdir:
        dtype = "int64"

    database = JSONDatabase(work_dir=workdir)
    all_records = database.get_all_tuning_records()
    top_record = sorted(all_records, key=lambda rec: rec.run_secs[0])[0]
    assert len(top_record.run_secs) == 1
    mod = top_record.workload.mod
    schedule = lambda *args, **kwargs: ms.tir_integration.compile_tir(database, mod, target)
    if not only_run:
        schedule().trace.show()
        l = tvm.lower(schedule().mod)
        print(l)
    if only_show:
        return

    op_class = get_workload(
        parsed[1],
    )
    workload = op_class(
        repeat=100,
        warmup=10,
        compile_only=False,
        output_format="tab",
        verbose=1,
        use_time_evaluator=True
    )
    workload.test(
        schedule,
        M=int(parsed[2]),
        N=int(parsed[3]),
        K=int(parsed[4]),
        n_xb=-1,
        n_yb=-1,
        n_t=-1,
        n_rt=-1,
        n_cache=-1,
        dtype=dtype,
    )
    return workload.recent_time_tuple


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
    parser.add_argument("--only_show", action="store_true")
    parser.add_argument("--only_run", action="store_true")
    args = parser.parse_args()

    query(args.workdir, args.only_show, args.only_run)
