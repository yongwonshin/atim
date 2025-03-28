import traceback
from typing import Any, List
import os
import re
import glob
import time
import subprocess
from abc import abstractmethod
from functools import lru_cache
import sys
import itertools
import signal

import tvm
from tvm.tir.transform import *
from tvm.target import Target
import numpy as np


class SymbolSpace:
    def __init__(self, symbols, output_symbol):
        self.symbols = {s: None for s in symbols}
        self.output_symbol = output_symbol

    def __getattr__(self, attr):
        if attr == "output":
            return self.symbols[self.output_symbol]
        if attr in self.symbols.keys():
            return self.symbols[attr]
        raise AttributeError(f"{attr} not in symbols")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "symbols" or attr == "output_symbol":
            self.__dict__[attr] = value
        elif attr == "output":
            self.symbols[self.output_symbol] = value
        elif attr in self.symbols.keys():
            self.symbols[attr] = value
        else:
            raise AttributeError(f"{attr} not in symbols")

    def __iter__(self):
        return iter(self.symbols.values())


def cleanup():
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./errors"):
        os.makedirs("./errors")
    for fname in os.listdir("./results"):
        os.remove("./results/" + fname)
    for fname in os.listdir("./errors"):
        os.remove("./errors/" + fname)
    files_to_remove = glob.glob("./temp*")
    for file in files_to_remove:
        os.remove(file)


@lru_cache(maxsize=4)
def host_array(dim, dtype, intdist=50):
    dimjoin = "_".join(map(str, dim))
    fname = f"../{dtype}_{dimjoin}.bin"
    if os.path.exists(fname) and intdist == 50:
        return np.fromfile(fname, dtype=dtype).reshape(dim)
    if dtype[:5] == "float":
        return np.random.rand(*dim).astype(dtype)
    else:
        return np.random.randint(0, intdist, dim).astype(dtype)


class UPMEMWorkload:
    def __init__(
        self,
        profile="empty",
        required={},
        repeat=3,
        warmup=1,
        bench=False,
        compile_only=False,
        verbose=0,
        symbols=[],
        output_symbol="",
        use_dummy=False,
        opt_level=-1,
        timeout=None,
        perform_h2d=True,
        perform_free=True,
        automatic_set_fname=True,
        record_schedule=True,
        record_lower=True,
        record_splitted_ir=True,
        record_upmem_c=True,
        record_host_llvm=True,
        max_correctness_indices=32,
        use_time_evaluator=True,
        output_format="all",
        ignore_wrong=False,
    ):
        self.profile = profile
        self.scheduler = None
        self.config = {}
        self.symbols = symbols
        self.output_symbol = output_symbol
        self.required = required
        self.opt_level = opt_level
        self.use_dummy = use_dummy
        self.repeat = repeat
        self.warmup = warmup
        self.compile_only = compile_only
        self.bench = bench
        self.timeout = timeout
        self.verbose = verbose
        self.perform_h2d = perform_h2d
        self.perform_free = perform_free
        self.automatic_set_fname = automatic_set_fname
        self.record_schedule = record_schedule
        self.record_lower = record_lower
        self.record_splitted_ir = record_splitted_ir
        self.record_upmem_c = record_upmem_c
        self.record_host_llvm = record_host_llvm
        self.max_correctness_indices = max_correctness_indices
        self.use_time_evaluator = use_time_evaluator
        self.output_format = output_format
        self.ignore_wrong = ignore_wrong
        # Fixed
        self.target = tvm.target.Target(target="upmem --num-cores=96", host="llvm")
        self.target_device = tvm.device("upmem", 0)

        # Internal
        self.fname = ""
        self.sch = None
        self.host = SymbolSpace(symbols, output_symbol)
        self.dev = SymbolSpace(symbols, output_symbol)
        self.benchmark_results = []
        self.results = []
        self.hand_tuned = []
        self.index = 0

        self.log_dir = f"./logs/results_{time.strftime('%m_%d_%H_%M_%S')}"
        os.makedirs(self.log_dir, exist_ok=True)

        recent_logs_symlink = "./recent_logs"
        if os.path.islink(recent_logs_symlink) or os.path.exists(recent_logs_symlink):
            os.remove(recent_logs_symlink)
        os.symlink(self.log_dir, recent_logs_symlink)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        if attr in self.config:
            return self.config[attr]
        if attr in self.required:
            return self.required[attr]
        raise AttributeError(f"{attr} not found")

    def extract_config(self, args):
        return {k: vars(args)[k] for k in self.required.keys()}

    def pre_kernel(self, sch):
        curr_pass_ctx = tvm.ir.transform.PassContext.current()
        curr_cfg = dict()
        for key, value in curr_pass_ctx.config.items():
            curr_cfg[key] = value
        tir_compiler_cfg = {
            # "tir.UnrollLoop": {"explicit_unroll": False},
            "tir.UpmemUseDummyKernel": self.use_dummy,
            "tir.UpmemKernelOptimize": self.opt_level,
        }
        # Merge two configs
        curr_cfg = {**curr_cfg, **tir_compiler_cfg}
        with tvm.transform.PassContext(config=curr_cfg):
            if self.record_schedule:
                with open(f"./{self.log_dir}/{self.fname}/schedule.py", "w") as f:
                    print("from tvm.script import ir as I", file=f)
                    print("from tvm.script import tir as T", file=f)
                    print(sch.mod, file=f)
                    print(sch.trace, file=f)

            if self.record_lower:
                with open(f"./{self.log_dir}/{self.fname}/lower.py", "w") as f:
                    l = tvm.lower(sch.mod)
                    print("from tvm.script import ir as I", file=f)
                    print("from tvm.script import tir as T", file=f)
                    print(l, file=f)

            if self.record_splitted_ir:
                with open(f"./{self.log_dir}/{self.fname}/split.py", "w") as f:
                    mp, _ = Target.canon_target_map_and_host({self.target: l}, "llvm")
                    m = mp[self.target]
                    m = BindTarget(self.target)(m)
                    m = VerifyMemory()(m)
                    m = AnnotateEntryFunc()(m)
                    m = ThreadSync("global")(m)
                    m = ThreadSync("shared")(m)
                    m = ThreadSync("shared.dyn")(m)
                    m = MergeDynamicSharedMemoryAllocations()(m)
                    m = ThreadSync("warp")(m)
                    m = InferFragment()(m)
                    m = LowerThreadAllreduce()(m)
                    m = AnnotateDeviceRegions()(m)
                    m = ExtractPimTransferSchedule()(m)
                    m = SplitHostDevice()(m)
                    m = SplitPimTransfer()(m)
                    m = Simplify()(m)
                    print("from tvm.script import ir as I", file=f)
                    print("from tvm.script import tir as T", file=f)
                    print(m, file=f)

            self.func = tvm.build(self.sch.mod, target=self.target, name="kernel")

            if self.record_upmem_c:
                with open(f"./{self.log_dir}/{self.fname}/upmem.c", "w") as f:
                    print(self.func.imported_modules[0].get_source(), file=f)

            if self.record_host_llvm:
                with open(f"./{self.log_dir}/{self.fname}/host.ll", "w") as f:
                    print("; LLVM Source\n", file=f)
                    print(self.func.get_source(), file=f)

    def post_kernel(self):
        if self.max_correctness_indices > 0:
            host_flatten = self.host.output.flatten()
            dev_flatten = self.dev.output.asnumpy().flatten()
            different_indices = np.where(host_flatten != dev_flatten)[0]
            if len(different_indices) > 0:
                with open(f"./{self.log_dir}/{self.fname}/wrong.txt", "w") as file:
                    print(f"Number of different indices: {different_indices.size}", file=file)
                    for idx in different_indices[: self.max_correctness_indices]:
                        print(
                            f"Index: {idx}, Host: {host_flatten[idx]}, Device: {dev_flatten[idx]}",
                            file=file,
                        )

    @abstractmethod
    def fetch_data(self):
        pass

    @abstractmethod
    def host_version(self):
        pass

    def benchmark_command(self):
        return ""

    def benchmark(self, **kwargs):
        config = {**self.required, **kwargs}
        cmd = self.benchmark_command(config)
        if not cmd or os.path.exists(f"baseline/prim/{self.profile}") is False:
            return 0, 0, 0
        try:
            process = subprocess.Popen(
                f"cd baseline/prim/{self.profile} && {cmd}",
                preexec_fn=os.setpgrp,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            result, error = process.communicate(timeout=self.timeout)
        except subprocess.CalledProcessError as e:
            result = result.decode("utf-8")
            if "iffer" in result:
                if not self.ignore_wrong: # Wrong
                    raise ValueError("Wrong")
            else:
                raise RuntimeError(f"Failed: {error.decode('utf-8')}")
        except subprocess.TimeoutExpired as e:
            for _ in range(5):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    raise TimeoutError("Benchmark command timed out and process was killed")
                time.sleep(1)
                if process.poll() is not None:
                    print("Process terminated successfully")
                    raise TimeoutError("Benchmark command timed out and process was killed")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()  # 종료될 때까지 대기
            except ProcessLookupError:
                raise TimeoutError("Benchmark command timed out and process was killed")
            raise TimeoutError("Benchmark command timed out and process was killed")

        try:
            result = result.decode("utf-8")
            if "iffer" in result:
                if not self.ignore_wrong: # Wrong
                    raise ValueError("Wrong")

            bench_before_kernel_time = re.search("CPU-DPU Time \(ms\): (\d+\.\d+)", result)
            bench_before_kernel_time = float(bench_before_kernel_time.group(1))
            bench_kernel_time = re.search("DPU Kernel Time \(ms\): (\d+\.\d+)", result)
            bench_kernel_time = float(bench_kernel_time.group(1))
            bench_after_kernel_time = re.search("DPU-CPU Time \(ms\): (\d+\.\d+)", result)
            bench_after_kernel_time = float(bench_after_kernel_time.group(1))
            time_tuple = (
                bench_before_kernel_time,
                bench_kernel_time,
                bench_after_kernel_time,
            )
        except AttributeError as e:
            print(traceback.format_exc(), file=sys.stderr)
            return [1000.0, 1000.0, 1000.0]
        self.benchmark_results.append(time_tuple)
        # print("\t".join(map(str, time_tuple)))
        return time_tuple


    def is_passed(self):
        if self.dtype[:5] == "float":
            return np.max(np.abs(self.dev.output.asnumpy() - self.host.output)) < 0.01
        else:
            return np.max(np.abs(self.dev.output.asnumpy() - self.host.output)) == 0


    def kernel(self, use_time_evaluator=False):
        if use_time_evaluator:
            evaluator = self.func.time_evaluator(
                self.func.entry_name,
                dev=self.target_device,
                number=10,
                repeat=100,
                bench=True,
            )
            repeated_costs: List[List[float]] = []
            profile_result = evaluator(*self.dev)
            repeated_costs.append(profile_result.results)
            costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
            return costs
        else:
            self.func(*self.dev)


    def h2d(self):
        for symbol in self.symbols:
            if symbol != self.output_symbol:
                self.dev.__setattr__(
                    symbol,
                    tvm.nd.array(
                        self.host.__getattr__(symbol),
                        self.target_device,
                        symbol=symbol,
                    ),
                )
        if self.output_symbol:
            self.dev.output = tvm.nd.empty(self.host.output.shape, self.dtype, self.target_device)


    def print_header(self):
        print("BK\tK\tAK\tD2H\tTOT\tPASS\tCONF")


    def dump_handtune_max(self):
        if len(self.hand_tuned) == 0:
            return None
        print("Best config")
        min_conf = min(self.hand_tuned, key=lambda x: x[1])[0]
        print(min_conf)
        self.hand_tuned = []
        return min_conf


    def evaluate_time_evaluator(self):
        costs = self.kernel(use_time_evaluator=True)
        before_kernel = []
        kernel = []
        after_kernel = []
        for i in range(100):
            before_kernel.append(costs[3 * i])
            kernel.append(costs[3 * i + 1])
            after_kernel.append(costs[3 * i + 2])
        mean_before_kernel = np.mean(before_kernel) * 1e3
        mean_kernel = np.mean(kernel) * 1e3
        mean_after_kernel = np.mean(after_kernel) * 1e3
        return (mean_before_kernel, mean_kernel, mean_after_kernel, 0, mean_before_kernel + mean_kernel + mean_after_kernel)


    def test(self, scheduler, **kwargs):
        ret = "ERROR"
        self.config = {**self.required, **kwargs}
        self.scheduler = scheduler
        self.sch = scheduler(**self.config)
        if self.automatic_set_fname or not self.fname:
            self.fname = f"{scheduler.__name__}_{self.index}"
        os.makedirs(self.log_dir + "/" + self.fname, exist_ok=True)
        self.index += 1

        self.fetch_data()
        self.host_version()

        try:
            # with open(f"./{self.log_dir}/{self.fname}.txt", "w") as f:
            self.pre_kernel(self.sch)
            if self.compile_only:
                return
            self.target_device.load_function(self.func)
            if self.perform_h2d:
                self.h2d()
            self.target_device.sync()
            time_tuple = self.evaluate_time_evaluator()
            self.recent_time_tuple = time_tuple
            flag = self.is_passed()
            if self.verbose >= 0:
                if self.output_format == "all":
                    print(
                        "\t".join([f"{x:.3f}" for x in time_tuple])
                        + f"\t{flag}\t{self.config.__repr__()}"
                    )
                elif self.output_format == "tab":
                    print(f"{time_tuple[0]}\t{time_tuple[1]}\t{time_tuple[2]}\t{time_tuple[4]}")
                elif self.output_format == "kernel":
                    print(time_tuple[1])
            if flag:
                self.hand_tuned.append([self.config, time_tuple[4]])
            self.post_kernel()
            ret = f"{time_tuple[1]}" if flag or self.ignore_wrong else "WRONG"
        except Exception as e:
            with open(f"./{self.log_dir}/{self.fname}/error.txt", "w") as f:
                print(traceback.format_exc(), file=f)
                ret = "ERROR"
        finally:
            if self.perform_free:
                self.target_device.free()
            return ret
