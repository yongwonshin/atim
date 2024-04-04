import traceback
from typing import Any
import os
import re
import glob
import time
import subprocess
from abc import abstractmethod
from functools import lru_cache

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
    ):
        self.profile = profile
        self.repeat = repeat
        self.warmup = warmup
        self.required = required
        self.compile_only = compile_only
        self.bench = bench
        self.verbose = verbose
        self.symbols = symbols
        self.output_symbol = output_symbol

        self.benchmark_results = []
        self.results = []

        self.index = 0
        self.sch = None
        self.fname = ""
        self.target = tvm.target.Target(target="upmem", host="llvm")
        self.target_device = tvm.device("upmem", 0)
        self.config = {}
        self.host = SymbolSpace(symbols, output_symbol)
        self.dev = SymbolSpace(symbols, output_symbol)

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

    def pre_kernel(self, file, sch):
        l = tvm.lower(sch.mod)

        print("[LOWER]", file=file)
        print(l, file=file)
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
        # m = MakePackedAPI()(m)
        # m = FP8StorageLegalize()(m)
        # m = BF16StorageLegalize()(m)
        # m = LowerDeviceKernelLaunch()(m)
        print("[TIR with PIM data copy]\n", m, file=file)

        print("\n\n[UPMEM source]\n", file=file)
        print(self.func.imported_modules[0].get_source(), file=file)

        print("\n\n[LLVM Source]\n", file=file)
        print(self.func.get_source(), file=file)

    def post_kernel(self, file):
        host_flatten = self.host.output.flatten()
        dev_flatten = self.dev.output.asnumpy().flatten()
        print("[Correctness Test]", file=file)
        print("Host: ", host_flatten[-32:], file=file)
        print("Device: ", dev_flatten[-32:], file=file)
        print(
            "Maximum Difference: ",
            np.max(np.abs(dev_flatten - host_flatten)),
            file=file,
        )

        different_indices = np.where(host_flatten != dev_flatten)[0]
        if different_indices.size > 100:
            different_indices = np.concatenate([different_indices[:50], different_indices[-50:]])
        for idx in different_indices:
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
        if not self.bench:
            return
        config = {**self.required, **kwargs}
        cmd = self.benchmark_command(config)
        if not cmd or os.path.exists(f"baseline/{self.profile}") is False:
            return 0, 0, 0
        result = subprocess.check_output(
            f"cd baseline/{self.profile} && {cmd}",
            shell=True,
            stderr=subprocess.DEVNULL,
        )
        result = result.decode("utf-8")

        partial_cpudpu = re.findall("Elapsed Time\(1\) \(ms\): (\d+\.*\d*)", result)
        partial_kernel = re.findall("Elapsed Time\(2\) \(ms\): (\d+\.*\d*)", result)
        partial_dpucpu = re.findall("Elapsed Time\(3\) \(ms\): (\d+\.*\d*)", result)
        if self.verbose >= 1:
            print("iter\tBK\tK\tAK")
            print("------------------------------")
            for j, (c, k, d) in enumerate(zip(partial_cpudpu, partial_kernel, partial_dpucpu)):
                time_tuple = (float(c), float(k), float(d))
                print(str(j) + "\t" + "\t".join([f"{float(x):.3f}" for x in time_tuple]))
            print("------------------------------")

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
        self.benchmark_results.append(time_tuple)
        print("\t".join(map(str, time_tuple)))

    def is_passed(self):
        if self.dtype[:5] == "float":
            return np.max(np.abs(self.dev.output.asnumpy() - self.host.output)) < 0.01
        else:
            return np.max(np.abs(self.dev.output.asnumpy() - self.host.output)) == 0

    def kernel(self):
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

    def file_suffix(self):
        return ""

    def test(self, scheduler, **kwargs):
        self.config = {**self.required, **kwargs}
        self.sch = scheduler(**self.config)
        suffix = self.file_suffix()
        if len(suffix) > 0:
            suffix = "_" + suffix
        self.fname = f"{self.profile}_{format(self.index, '02')}_{scheduler.__name__}{suffix}.txt"
        self.index += 1

        self.fetch_data()
        self.host_version()

        timestamp = tvm._ffi.get_global_func("device_api.upmem.timestamp")
        elapsed_time = tvm._ffi.get_global_func("device_api.upmem.elapsed_time")

        try:
            with open("./results/" + self.fname + ".txt", "w") as f:
                self.func = tvm.build(self.sch.mod, target=self.target, name="kernel")
                self.pre_kernel(f, self.sch)
                if self.compile_only:
                    return

                self.target_device.load_function(self.func)
                times = []

                if self.verbose >= 1:
                    print("iter\tBK\tK\tAK\tD2H\tTOT")
                    print("------------------------------")
                self.h2d()
                for j in range(self.repeat + self.warmup):
                    total_start = time.time()
                    timestamp("start")
                    self.kernel()
                    timestamp("end")
                    total_end = time.time()

                    if j >= self.warmup:
                        before_kernel_time = elapsed_time("before_kernel") / 1e6
                        kernel_time = elapsed_time("kernel") / 1e6
                        after_kernel_time = elapsed_time("after_kernel") / 1e6
                        d2h_time = elapsed_time("d2h") / 1e6
                        total_time = (total_end - total_start) * 1e3
                        time_tuple = (
                            before_kernel_time,
                            kernel_time,
                            after_kernel_time,
                            d2h_time,
                            total_time,
                        )
                        times.append(time_tuple)
                        if self.verbose >= 1:
                            print(str(j) + "\t" + "\t".join([f"{x:.3f}" for x in time_tuple]))
                if self.verbose >= 1:
                    print("------------------------------")
                time_tuple = np.mean(times, axis=0)
                flag = self.is_passed()
                print(
                    "\t".join([f"{x:.3f}" for x in time_tuple])
                    + f"\t{flag}\t{self.config.__repr__()}"
                )
                # print(
                #     f"{self.config['n_b']}\t{time_tuple[1]}\t{time_tuple[2]}\t{time_tuple[3]}\t{flag}"
                # )
                self.post_kernel(f)

        except Exception as e:
            with open("./errors/" + self.fname + ".txt", "w") as f:
                print(traceback.format_exc(), file=f)
        finally:
            self.target_device.free()
