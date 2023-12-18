import tvm
from tvm import te
import numpy as np
from tvm.ir.module import IRModule
from tvm.script import tir as T

N_XB = 4
N_YB = 4
N_YT = 16
N_CACHE = 64
N_RTHREADS = 16

M = 2048
K = 2048

dtype = "float32"
target = tvm.target.Target(target="upmem", host="llvm")

@tvm.script.ir_module
class UPMEMModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        A = T.match_buffer(a, (M, K), dtype=dtype)
        B = T.match_buffer(b, (K,), dtype=dtype)
        C = T.match_buffer(c, (M,), dtype=dtype)
        for i, k in T.grid(M, K):
            with T.block("C"):
                v_i, v_k = T.axis.remap("SR", [i, k])
                with T.init():
                    C[v_i] = T.float16(0)
                C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]
ir_module = UPMEMModule

sch = tvm.tir.Schedule(ir_module)
block_c = sch.get_block("C")
_, k = sch.get_loops(block_c)

xb, xo = sch.split(k, factors = [N_XB, None])
block_crf = sch.rfactor(xb, factor_axis=0)

ca = sch.cache_read(block_crf, 0, "local")
cb = sch.cache_read(block_crf, 1, "local")
cc = sch.cache_write(block_crf, 0, "local")

i, xb, k = sch.get_loops(block_crf)
yb, yo, yi, yc = sch.split(i, factors=[N_YB, N_YT, M // N_YB // N_YT // 2, None])
xo, xi = sch.split(k, factors=[K // N_XB // N_CACHE, None])
sch.reorder(xb, yb, yo, yi, yc, xo, xi)

sch.compute_at(ca, xo)
sch.compute_at(cb, xo)
sch.reverse_compute_at(cc, yi)

sch.bind(xb, "blockIdx.x")
sch.bind(yb, "blockIdx.y")
sch.bind(yo, "threadIdx.x")

sch.annotate(xb, "bank", True)
sch.annotate(yb, "bank", True)

sch.decompose_reduction(block_crf, xo)

i, _ = sch.get_loops(block_c)
it, ii = sch.split(i, factors=[N_RTHREADS, None])
sch.parallel(it)

# print(sch.mod)

#new_mod = tvm.lower(sch.mod)
#print(new_mod)
func = tvm.build(sch.mod, target=target, name="gemm")
# evaluate_operation(func, log=log)

# if (
#     target.kind.name == "cuda"
#     or target.kind.name == "rocm"
#     or target.kind.name.startswith("opencl")
#     or target.kind.name.startswith("hbmpim")
# ):
#     dev_module = func.imported_modules[0]
#     print("-----GPU Code-----")
#     print(dev_module.get_source())
# else:
#     print("-----Code-----")
#     print(func.get_source())