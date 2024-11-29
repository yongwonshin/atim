import logging
import tempfile

import numpy as np
import pytest
import tvm
import sys
import time
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from typing import Callable
import os
from tvm import te, runtime, topi, tir
from tvm.meta_schedule import Database
from tvm.meta_schedule.database import JSONDatabase

from bench import *
import argparse
from main_result import get_module

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default="bench_autotuner_result", type=str)
parser.add_argument("--only_show", action="store_true")
parser.add_argument("--only_run", action="store_true")
args = parser.parse_args()

target = Target("upmem --num-cores=96")


def query(workdir: str, only_show: False, only_run: False) -> None:
    parsed = workdir.split("_")
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
    if only_show:
        return

    op_class = get_module(
        parsed[1],
    )
    workload = op_class(
        repeat=100,
        warmup=10,
        verbose=False,
        compile_only=False,
        output_format="tab"
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


if __name__ == "__main__":
    query(args.workdir, args.only_show, args.only_run)
