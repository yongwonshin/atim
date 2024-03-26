import traceback
from typing import Any
import tvm
import os
from tvm.tir.transform import *
from tvm.target import Target
import numpy as np
from abc import abstractmethod
from functools import lru_cache


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
    def __init__(self, profile="empty", required={}, repeat=10, symbols=[], output_symbol=""):
        self.profile = profile
        self.repeat = repeat
        self.required = required
        self.symbols = symbols
        self.output_symbol = output_symbol
        self.index = 0
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

    def pre_kernel(self, file, sch):
        l = tvm.lower(sch.mod)

        print("[LOWER]", file=file)
        print(l, file=file)
        target = tvm.target.Target(target="upmem", host="llvm")
        mp, _ = Target.canon_target_map_and_host({target: l}, "llvm")
        m = mp[target]
        m = BindTarget(target)(m)
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
        m = MakePackedAPI()(m)
        m = FP8StorageLegalize()(m)
        m = BF16StorageLegalize()(m)
        m = LowerDeviceKernelLaunch()(m)
        print("[TIR with PIM data copy]\n", m, file=file)

        self.func = tvm.build(sch.mod, target=target, name="kernel")
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
                        symbol=self.func[f"copy_{symbol}"],
                    ),
                )
        if self.output_symbol:
            self.dev.output = tvm.nd.empty(self.host.output.shape, self.dtype, self.target_device)

    def file_suffix(self):
        return ""

    def test(self, scheduler, **kwargs):
        self.config = {**self.required, **kwargs}
        sch = scheduler(**self.config)
        suffix = self.file_suffix()
        if len(suffix) > 0:
            suffix = "_" + suffix
        fname = f"{self.profile}_{format(self.index, '02')}_{scheduler.__name__}{suffix}.txt"
        self.index += 1

        self.fetch_data()
        self.host_version()

        try:
            with open("./results/" + fname + ".txt", "w") as f:
                self.pre_kernel(f, sch)
                self.h2d()
                kernel_times = []
                after_kernel_times = []
                for _ in range(self.repeat):
                    self.kernel()
                    ktime = tvm._ffi.get_global_func("device_api.upmem.kernel_time")() / 1e6
                    aktime = tvm._ffi.get_global_func("device_api.upmem.after_kernel_time")() / 1e6
                    kernel_times.append(ktime)
                    after_kernel_times.append(aktime)
                tvm._ffi.get_global_func("device_api.upmem.release_resources")()
                kernel_time = np.mean(kernel_times)
                after_kernel_time = np.mean(after_kernel_times)
                total_time = kernel_time + after_kernel_time
                self.post_kernel(f)
                flag = self.is_passed()
                print(
                    "{:<30} {:<8.3f} {:<8.3f} {:<8.3f} {:<6} {:<10}".format(
                        fname,
                        kernel_time,
                        after_kernel_time,
                        total_time,
                        "PASS" if flag else "FAIL",
                        kwargs.__repr__(),
                    )
                )
        except Exception as e:
            with open("./errors/" + fname + ".txt", "w") as f:
                print(traceback.format_exc(), file=f)
