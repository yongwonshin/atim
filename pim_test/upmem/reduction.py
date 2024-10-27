import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse
from tqdm import tqdm

def upmem_dot_factory(M, dtype):
    @tvm.script.ir_module
    class DOTModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (M,), dtype=dtype)
            C = T.match_buffer(c, (1,), dtype=dtype)
            for i in T.grid(M):
                with T.block("C"):
                    with T.init():
                        C[0] = 0
                    v_i = T.axis.remap("R", [i])
                    C[0] = C[0] + A[v_i] * B[v_i]

    return DOTModule


def red_prim_schedule(L, dtype):
    M = 1

    @tvm.script.ir_module
    class ReductionModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, [L, M], dtype=dtype)
            B = T.match_buffer(b, [M], dtype=dtype)
            for i, k in T.grid(L, M):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("RS", [i, k])
                    with T.init():
                        B[v_k] = 0
                    B[v_k] = B[v_k] + A[v_i, v_k]

    return ReductionModule


def crossReduction(L, n_b, n_t, n_c, dtype):
    sch = tvm.tir.Schedule(upmem_dot_factory(L, dtype))
    block = sch.get_block("C")
    i = sch.get_loops(block)[0]
    i, _ = sch.split(i, factors=[n_b, None])
    rf = sch.rfactor(i, factor_axis=0)  # C_rf

    _, k = sch.get_loops(rf)
    t, k = sch.split(k, [16, None])
    krf = sch.rfactor(t, factor_axis=0, mem_scope="shared")  # C_rf_rf
    i, t, k = sch.get_loops(krf)
    sch.reverse_compute_at(rf, t)
    sch.bind(i, "blockIdx.x")
    sch.bind(t, "threadIdx.x")
    return sch


class REDUCE(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="reduction",
            required=dict(L=8388608, dtype="int64", n_b=1024, n_t=16, n_c=64),
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = np.ones((L,), self.dtype)
        self.host.B = np.ones((L,), self.dtype)
        self.host.C = np.zeros((1,), self.dtype)

    def host_version(self):
        self.host.C = np.dot(self.host.A, self.host.B)

    def h2d(self):
        self.dev.A = tvm.nd.array(self.host.A, self.target_device, symbol="A")
        self.dev.B = tvm.nd.array(self.host.B, self.target_device, symbol="B")
        self.dev.C = tvm.nd.empty((1,), self.dtype, self.target_device)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_c"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_b']} \
            NR_TASKLETS={config['n_t']} TYPE={pbtype} BL={bl} VERSION=HANDSHAKE make >/dev/null 2>/dev/null && \
            ./bin/host_code -i {config['L']} -w {self.warmup} -e {self.repeat}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--schedule", default="gemvRTile", type=str)
    parser.add_argument("-i", "--L", default=8388608, type=int)
    parser.add_argument("-dtype", "--dtype", default="int64", type=str)
    parser.add_argument("-b", "--n_b", default=1024, type=int)
    parser.add_argument("-c", "--n_c", default=64, type=int)
    parser.add_argument("-t", "--n_t", default=16, type=int)

    parser.add_argument("-w", "--warmup", default=1, type=int)
    parser.add_argument("-e", "--repeat", default=3, type=int)
    parser.add_argument("-v", "--verbose", default=0, type=int)
    parser.add_argument("-bench", "--bench", default=False, action="store_true")
    parser.add_argument("-custom", "--custom", default=False, action="store_true")
    parser.add_argument("-compile_only", "--compile_only", default=False, action="store_true")

    args = parser.parse_args()

    cleanup()
    reduce = REDUCE(
        repeat=args.repeat,
        warmup=args.warmup,
        bench=args.bench,
        verbose=args.verbose,
        compile_only=args.compile_only,
    )

    if not args.custom:
        config = reduce.extract_config(args)
        reduce.benchmark(**config)
        reduce.test(crossReduction, **config)
    else:  # custom test config
        #
        dpus = [512, 1024, 2048]
        tasklets = [16, 20, 24]
        cache_size = [8, 16, 32, 64, 128, 256]
        configs = [(33554432, d, t, c, "int32") for d in dpus for t in tasklets for c in cache_size]
        # configs = [
        #     (33554432, 2048, 16, 64, "int32")
        #     # (65536, 1, 16, 128, "int64"),
        #     # (6553600, 1, 16, 128, "int64"),
        #     # (6553600 * 4, 4, 16, 128, "int64"),
        #     # (6553600 * 16, 16, 16, 128, "int64"),
        #     # (6553600 * 64, 64, 16, 128, "int64"),
        #     # (6500000, 1, 16, 128, "int64"),
        #     # (6500000, 4, 16, 128, "int64"),
        #     # (6500000, 16, 16, 128, "int64"),
        #     # (6500000, 64, 16, 128, "int64"),
        #     # (400000000, 256, 16, 128, "int64"),
        #     # (400000000, 512, 16, 128, "int64"),
        #     # (400000000, 1024, 16, 128, "int64"),
        #     # (400000000, 2048, 16, 128, "int64"),
        # ]

        max_time = 1e9
        with tqdm(total=len(configs), leave=True) as pbar:
            for config in configs:
                L, n_b, n_t, n_c, dtype = config
                try:
                    tuples = reduce.benchmark(L=L, n_b=n_b, n_t=n_t, n_c=n_c, dtype=dtype)
                    total_time = tuples[1] + tuples[2]
                    if total_time < max_time:
                        max_time = total_time
                        best_config = config
                except ValueError as e:
                    tuples = ("wrong", "", "")
                except RuntimeError as e:
                    tuples = ("fail", "", "")
                except TimeoutError as e:
                    tuples = ("timeout", "", "")
                except Exception as e:
                    tuples = ("exception", "", "")
                    print(e)
                tqdm.write("\t".join([str(x) for x in list(tuples) + list(config)]))
                pbar.update(1)
        # for L, n_b, n_t, n_c, dtype in configs:
        #     reduce.be(crossReduction, L=L, n_b=n_b, n_t=n_t, n_c=n_c, dtype=dtype)
