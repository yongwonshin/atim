import math
import argparse
import numpy as np
import tvm
import time
import math
from tvm.script import tir as T
import subprocess
from tvm.tir.transform import *

from base import UPMEMWorkload, cleanup
from tensor import host_array


def va_prim_schedule(L, dtype):
    @tvm.script.ir_module
    class VAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, (L,), dtype=dtype)
            B = T.match_buffer(b, (L,), dtype=dtype)
            C = T.match_buffer(c, (L,), dtype=dtype)
            for i in T.serial(0, L):
                with T.block("C"):
                    C[i] = A[i] + B[i]

    return VAModule


def vaTile(L, n_b, n_t, n_c, dtype):
    sch = tvm.tir.Schedule(va_prim_schedule(L, dtype))
    block_c = sch.get_block("C")
    (i,) = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, "A", "local")
    cb = sch.cache_read(block_c, "B", "local")
    cc = sch.cache_write(block_c, "C", "local")
    # ib, it, ii, ic = sch.split(i, factors=[n_b, n_t, None, n_c])
    ib, it = sch.split(i, factors=[None, math.ceil(L / n_b / 2) * 2])
    it, ii, ic = sch.split(it, factors=[n_t, None, n_c])
    sch.compute_at(ca, ii)
    sch.compute_at(cb, ii)
    sch.reverse_compute_at(cc, ii)
    sch.bind(ib, "blockIdx.x")
    sch.bind(it, "threadIdx.x")
    sch.annotate(sch.get_block("A_local"), "pragma_explicit_h2d", True)
    sch.annotate(sch.get_block("B_local"), "pragma_explicit_h2d", True)
    sch.parallel(ic)
    return sch


class VA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="va",
            required=dict(L=2500000, dtype="int32", n_b=4, n_t=16, n_c=256),
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array(self.L, self.dtype)
        self.host.B = host_array(self.L, self.dtype)

    def host_version(self):
        self.host.C = self.host.A + self.host.B

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_c"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_b']} NR_TASKLETS={config['n_t']} \
            TYPE={pbtype} BL={bl} make && \
            ./bin/host_code -i {config['L']} -w {self.warmup} -e {self.repeat}"

    def simplepim(self, L, n_b):
        cmd = 'cd "/root/dev/SimplePIM/benchmarks/va"\
                sed -i "s/dpu_number = [0-9]\+;/dpu_number = $i;/" Param.h\
                sed -i "s/nr_elements = [0-9]\+;/nr_elements = $i;/" Param.h\
                make > /dev/null 2> /dev/null\
                ./bin/host | grep -E "initial CPU-DPU input transfer|DPU Kernel Time|DPU-CPU Time" | awk \'{printf "%s\t", $NF}\'\
                echo ""'
        result = subprocess.check_output(
            cmd,
            shell=True,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--L", default=2500000, type=int)
    parser.add_argument("-b", "--n_b", default=4, type=int)
    parser.add_argument("-t", "--n_t", default=16, type=int)
    parser.add_argument("-c", "--n_c", default=64, type=int)
    parser.add_argument("-dtype", "--dtype", default="int32", type=str)

    parser.add_argument("-w", "--warmup", default=1, type=int)
    parser.add_argument("-e", "--repeat", default=3, type=int)
    parser.add_argument("-v", "--verbose", default=0, type=int)
    parser.add_argument("-bench", "--bench", default=False, action="store_true")
    parser.add_argument("-custom", "--custom", default=False, action="store_true")
    parser.add_argument("-compile_only", "--compile_only", default=False, action="store_true")

    args = parser.parse_args()

    cleanup()
    va = VA(
        repeat=args.repeat,
        warmup=args.warmup,
        bench=args.bench,
        verbose=args.verbose,
        compile_only=args.compile_only,
    )

    if not args.custom:
        config = va.extract_config(args)
        va.benchmark(**config)
        va.test(vaTile, **config)
    else:  # custom test config
        configs = [
            (1000000, 1, 16, 256, "int32"),
            (4000000, 4, 16, 256, "int32"),
            (16000000, 16, 16, 256, "int32"),
            (64000000, 64, 16, 256, "int32"),
            (2500000, 1, 16, 256, "int32"),
            (2500000, 4, 16, 256, "int32"),
            (2500000, 16, 16, 256, "int32"),
            (2500000, 64, 16, 256, "int32"),
            (160000000, 256, 16, 256, "int32"),
            (160000000, 512, 16, 256, "int32"),
            (160000000, 1024, 8, 256, "int32"),
            (160000000, 2048, 4, 256, "int32"),
        ]
        # for L, n_b, _, _, _ in configs:
        #     va.simplepim(L, n_b)
        for L, n_b, n_t, n_c, dtype in configs:
            va.benchmark(L=L, n_b=n_b, n_t=n_t, n_c=n_c, dtype=dtype)
        for L, n_b, n_t, n_c, dtype in configs:
            va.test(vaTile, L=L, n_b=n_b, n_c=n_c, n_t=n_t, dtype=dtype)
            time.sleep(0.1)
