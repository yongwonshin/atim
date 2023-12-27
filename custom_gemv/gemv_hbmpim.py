import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.tir.tensor_intrin.hbmpim import (
    HBMPIM_INPUT_INTRIN,
    HBMPIM_WEIGHT_INTRIN,
    HBMPIM_MAC_INTRIN,
    HBMPIM_PARTIAL_REDUCTION_INTRIN,
)

N_CHAN = 64
N_BANK = 16
N_PU = 8
N_GRF_A = 8
N_GRF_B = 8
N_ELEM_IN_GRF = 16


class BankOrdering:
    DENSE = 1
    EVEN = 2
    ODD = 3


N_OUT_TILE = N_CHAN * N_PU * N_GRF_B
N_IN_TILE = N_GRF_A * N_ELEM_IN_GRF

# The size of the matrix
# (M, K) x (K, N)
M = 4096 * 2
K = 1024
N = 1

log = []

# The default tensor data type in tvm
dtype = "float16"

target = tvm.target.Target(target="hbmpim", host="llvm")
dev = tvm.device(target.kind.name, 0)


# Random generated tensor for testing
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(
    np.random.rand(
        K,
    ).astype(dtype),
    dev,
)

answer = np.dot(a.numpy(), b.numpy())


@tvm.script.ir_module
class HBMPIMModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle, p: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # Create buffer from handles.
        A = T.match_buffer(a, (M, K), dtype=dtype)
        B = T.match_buffer(b, (K,), dtype=dtype)
        C = T.match_buffer(c, (M,), dtype=dtype)
        P = T.match_buffer(p, (M, N_ELEM_IN_GRF), dtype=dtype)
        # for i_0, i_1 in T.grid(M // N_OUT_TILE, N_CHAN):
        #     with T.block("R"):
        #         with T.init():
        #             pass
        #         for i_2, k, r in T.grid(N_OUT_TILE // N_CHAN, K // N_ELEM_IN_GRF, N_ELEM_IN_GRF):
        #             with T.block("P"):
        #                 v_i_0, v_i_1, v_i_2, v_k, v_r = T.axis.remap("SSSRR", [i_0, i_1, i_2, k, r])
        #                 v_i = v_i_0 * N_OUT_TILE + v_i_1 * N_OUT_TILE // N_CHAN + v_i_2
        #                 with T.init():
        #                     pass
        #                     # P[v_i, v_r] = T.float16(0)
        #                 P[v_i, v_r] = P[v_i, v_r] + A[v_i, v_k*N_ELEM_IN_GRF + v_r] * B[v_k*N_ELEM_IN_GRF + v_r]
        #         for i_2, r in T.grid(N_OUT_TILE // N_CHAN, N_ELEM_IN_GRF):
        #             with T.block("C"):
        #                 v_i_0, v_i_1, v_i_2, v_r = T.axis.remap("SSSR", [i_0, i_1, i_2, r])
        #                 v_i = v_i_0 * N_OUT_TILE + v_i_1 * N_OUT_TILE // N_CHAN + v_i_2
        #                 with T.init():
        #                     pass
        #                     # C[v_i] = T.float16(0)
        #                 C[v_i] = C[v_i] + P[v_i, v_r]
        # for b in T.grid(1):
        #     with T.block("R"):
        #         with T.init():
        #             pass
        for i, k, r in T.grid(M, K // N_ELEM_IN_GRF, N_ELEM_IN_GRF):
            with T.block("P"):
                v_i, v_k, v_r = T.axis.remap("SRS", [i, k, r])
                with T.init():
                    pass
                    # P[v_i, v_r] = 0
                P[v_i, v_r] = (
                    P[v_i, v_r] + A[v_i, v_k * N_ELEM_IN_GRF + v_r] * B[v_k * N_ELEM_IN_GRF + v_r]
                )
        for i, r in T.grid(M, N_ELEM_IN_GRF):
            with T.block("C"):
                v_i, v_r = T.axis.remap("SR", [i, r])
                with T.init():
                    pass
                    # C[v_i] = 0
                C[v_i] = C[v_i] + P[v_i, v_r]


ir_module = HBMPIMModule
print(type(ir_module))
print(ir_module.script())


def te_to_tir():
    # TVM Matrix Vector Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K,), name="B", dtype=dtype)
    C = te.compute((M,), lambda i: te.sum(A[i, k] * B[k], axis=k), name="C")

    func = te.create_prim_func([A, B, C])
    ir_module_from_te = IRModule({"main": func})
    print(ir_module_from_te.script())


# te_to_tir()


def evaluate_operation(func, log):
    c = tvm.nd.array(np.zeros((M,), dtype=dtype), dev)
    p = tvm.nd.array(np.zeros((M, N_ELEM_IN_GRF), dtype=dtype), dev)
    func(a, b, c, p)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-1, atol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c, p).mean
    log.append(("runtime", mean_time))


# schedule
sch = tvm.tir.Schedule(ir_module)
block_p = sch.get_block("P")
cache_read_a = sch.cache_read(block_p, 0, "local")
cache_read_b = sch.cache_read(block_p, 1, "local")
cache_write_p = sch.cache_write(block_p, 0, "local")
i, k, r = sch.get_loops(block_p)
i_0, i_1, i_2, i_3 = sch.split(i, factors=[None, N_CHAN, N_PU, N_GRF_B])
k_1, k_0, k_2 = sch.split(k, factors=[None, N_BANK // N_PU, N_GRF_A])
sch.reorder(i_1, i_0, i_2, k_0, k_1, i_3, k_2)
sch.compute_at(cache_read_a, loop=i_3)
sch.compute_at(cache_read_b, loop=k_1)
sch.reverse_compute_at(cache_write_p, i_2)
_, _, _, _, _, _, ax_a = sch.get_loops(cache_read_a)
sch.tensorize(ax_a, HBMPIM_WEIGHT_INTRIN)
_, _, _, _, _, ax_b = sch.get_loops(cache_read_b)
sch.tensorize(ax_b, HBMPIM_INPUT_INTRIN)
sch.blockize(k_2)
sch.tensorize(k_2, HBMPIM_MAC_INTRIN)
_, _, _, ax_p, _ = sch.get_loops(cache_write_p)
sch.tensorize(ax_p, HBMPIM_PARTIAL_REDUCTION_INTRIN)
# sch.vectorize(ax_b)
# sch.vectorize(r)
sch.bind(i_1, "blockIdx.x")
sch.bind(i_2, "puIdx.x")
sch.annotate(i_1, "bank", BankOrdering.DENSE)
sch.annotate(i_2, "bank", BankOrdering.DENSE)
sch.annotate(k_0, "bank", BankOrdering.DENSE)
sch.annotate(i_0, "pim", True)
sch.annotate(i_3, "barrier", True)

block_c = sch.get_block("C")
i, r = sch.get_loops(block_c)
# cache_read_p = sch.cache_read(block_c, 0, "local")
i_0, i_1, i_2, i_3 = sch.split(i, factors=[None, N_CHAN, N_PU, N_GRF_B])
sch.reorder(i_1, i_0, i_2, i_3)
# sch.compute_at(cache_read_p, i_0)
sch.bind(i_1, "blockIdx.x")
sch.bind(i_3, "threadIdx.x")
# sch.bind(i_2, "puIdx.x") # NOTE: single bank mode
sch.bind(i_2, "bankIdx.x")  # NOTE: single bank mode
sch.annotate(i_1, "bank", BankOrdering.DENSE)
sch.annotate(i_2, "bank", BankOrdering.ODD)
print(sch.mod.script())

# NOTE 1: 위에서 i_2가 pim_gemm.cl에서 n_out_tile에 해당하지만, i_2는 8개인 반면 n_out_tile은 1개이다. 이 차이가 발생하는 이유는 PIM block에 weight가 implicit하게 매핑되기 때문이다. 문제는 i_2가 blockIdx, threadIdx로 바인딩 될 수 없다는 것이다.
# NOTE 2: 생성되는 OpenCL 코드를 보면, (1) A 로컬 버퍼에서 데이터를 읽는다. (2) B 로컬 버퍼에서 데이터를 읽는다. (3) 필요하면, C 버퍼를 초기화한다. (4) A, B를 이용하여 연산을 수행하고 결과를 C 로컬 버퍼에 저장한다. 하지만 pim_gemm.cl에서는 (1), (2) 과정이 있지만, (3)은 (2) 과정에서 implicit하게 수행된다응 차이점이 있다.
# NOTE 3: Reduction 과정이 더 복잡하다.
# NOTE 4: threadIdx.x를 loop variable에 binding 할 수 없으며, 더 복잡한 방법을 요구한다.

from tvm.ir.transform import PassContext, Sequential

# Define the passes you want to apply
# In this case, let's use the 'simplify' pass
passes = [
    tvm.tir.transform.StorageFlatten(64),
    tvm.tir.transform.LowerInitBlock(),
    tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
    tvm.tir.transform.ConvertBlocksToOpaque(),
    tvm.tir.transform.CompactBufferAllocation(),
    tvm.tir.transform.Simplify(),
    #   tvm.tir.transform.LowerMatchBuffer(),
    #   tvm.tir.transform.UnifyThreadBinding(),
    #   tvm.tir.transform.LowerOpaqueBlock(),
    #   tvm.tir.transform.FlattenBuffer(),
    #   tvm.tir.transform.Simplify(),
]

# Apply the passes
# with PassContext():
#     lowered_module = Sequential(passes)(sch.mod)

# print(lowered_module.script())

func = tvm.build(sch.mod, target=target, name="gemm")
# evaluate_operation(func, log=log)

if (
    target.kind.name == "cuda"
    or target.kind.name == "rocm"
    or target.kind.name.startswith("opencl")
    or target.kind.name.startswith("hbmpim")
):
    dev_module = func.imported_modules[0]
    print("-----GPU Code-----")
    print(dev_module.get_source())
else:
    print("-----Code-----")
    print(func.get_source())
