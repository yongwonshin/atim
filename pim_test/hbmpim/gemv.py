import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.ir.module import IRModule
from tvm.ir.expr import GlobalVar
from tvm.tir.schedule import Schedule, BlockRV, LoopRV
from tvm.tir import PrimFunc
from tvm.script import tir as T
from tvm.script import from_source
from typing import Mapping
from tvm.tir.tensor_intrin.hbmpim import (
    HBMPIM_INPUT_INTRIN,
    HBMPIM_WEIGHT_INTRIN,
    HBMPIM_MAC_INTRIN,
    HBMPIM_PARTIAL_REDUCTION_INTRIN,
)

np.random.seed(1113)

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
M = 4096 * 1
K = 1024 * 1
N = 1

log = []

# The default tensor data type in tvm
dtype = "int16"

target = tvm.target.Target(target="hbmpim", host="llvm")
dev = tvm.device(target.kind.name, 0)


def convert_to_int16(arr: np.ndarray) -> np.ndarray:
    return np.frombuffer(
        arr.flatten().tobytes(), dtype=np.int16, count=len(arr.flatten())
    ).reshape(arr.shape)


def convert_to_fp16(arr: np.ndarray) -> np.ndarray:
    return np.frombuffer(
        arr.flatten().tobytes(), dtype=np.float16, count=len(arr.flatten())
    ).reshape(arr.shape)


def transform_block_(block: np.ndarray) -> np.ndarray:
    # 8 * 128 블록을 8 * 8 블록으로 변환한 후 transpose
    reshaped = block.reshape(8, 8, 16).transpose(1, 0, 2)
    # 다시 1차원으로 변환
    return reshaped.reshape(8, 128)


def transform_array(original_array: np.ndarray) -> np.ndarray:
    transformed_blocks = list[np.ndarray]()
    for i in range(0, original_array.shape[0], 8):
        for j in range(0, original_array.shape[1], 128):
            block = original_array[i : i + 8, j : j + 128]
            transformed_block = transform_block_(block)
            transformed_blocks.append(transformed_block)
    result_array = (
        np.concatenate(
            [block.reshape(1, 8, 128) for block in transformed_blocks], axis=0
        )
        .reshape(original_array.shape[0] // 8, original_array.shape[1] // 128, 8, 128)
        .transpose(0, 2, 1, 3)
    )
    result_array = result_array.reshape(original_array.shape)
    return result_array


data_A = np.random.standard_normal(size=(M, K)).astype("float16")
np.random.shuffle(data_A)
# data_A = np.full((M, K), 1, dtype="float16")
data_A = convert_to_int16(data_A)

data_B = np.random.standard_normal(size=(K,)).astype("float16")
np.random.shuffle(data_B)
# data_B = np.full((K), 1, dtype="float16")
data_B = convert_to_int16(data_B)

answer = np.matmul(
    convert_to_fp16(data_A), convert_to_fp16(data_B).reshape(K, 1)
).flatten()
# print(answer)

# Random generated tensor for testing
# a = tvm.nd.array(np.full((M, K), 15360, dtype=dtype), dev, mem_scope="internal")
# b = tvm.nd.array(np.full((K,), 15360, dtype=dtype), dev)
# c = tvm.nd.array(np.zeros((M,), dtype=dtype), dev)
a = tvm.nd.array(transform_array(data_A))
b = tvm.nd.array(data_B, dev)
c = tvm.nd.array(np.zeros((M,), dtype=dtype), dev)


@tvm.script.ir_module
class HBMPIMModule:
    @T.prim_func
    def main(
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # Create buffer from handles.
        A = T.match_buffer(a, (M, K), dtype=dtype)
        B = T.match_buffer(b, (K,), dtype=dtype)
        C = T.match_buffer(c, (M,), dtype=dtype)

        for i, k in T.grid(M, K):
            with T.block("C"):
                v_i, v_k = T.axis.remap("SR", [i, k])
                with T.init():
                    C[v_i] = 0
                C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]


ir_module = HBMPIMModule
# print(type(ir_module))
# print(ir_module.script())


def te_to_tir():
    # TVM Matrix Vector Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K,), name="B", dtype=dtype)
    C = te.compute((M,), lambda i: te.sum(A[i, k] * B[k], axis=k), name="C")

    func = te.create_prim_func([A, B, C])
    ir_module_from_te = IRModule({"main": func})
    # print(ir_module_from_te.script())


# te_to_tir()


def evaluate_operation(func, log):
    a_h2d = tvm.nd.array(a, dev, mem_scope="internal", symbol=func[f"copy_A"])

    for _ in range(1):
        func(a_h2d, b, c)

    # tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-1, atol=1e-3)

    # evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    # mean_time = evaluator(a, b, c).mean
    # log.append(("runtime", mean_time))


def schedule() -> {Schedule, LoopRV, BlockRV, BlockRV, BlockRV}:
    # schedule
    sch = Schedule(ir_module)
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    k, r = sch.split(k, [None, N_ELEM_IN_GRF])
    block_crf = sch.rfactor(r, factor_axis=1, mem_scope="internal")
    sch.reorder_block_iter_var(block_crf, [1, 2, 0])  # ?
    i, k, r = sch.get_loops(block_crf)
    i_0, i_1, i_2, i_3 = sch.split(i, factors=[None, N_CHAN, N_PU, N_GRF_B])
    k_0, k_1, k_2 = sch.split(k, factors=[None, N_BANK // N_PU, N_GRF_A])
    sch.reorder(i_1, i_0, i_2, k_1, k_0, i_3, k_2)
    cache_write_p = sch.cache_write(block_crf, 0, "local")
    sch.reverse_compute_at(cache_write_p, i_2)
    cache_read_a = sch.cache_read(block_crf, 0, "local")
    cache_read_b = sch.cache_read(block_crf, 1, "local")
    sch.compute_at(cache_read_a, loop=i_3)
    sch.compute_at(cache_read_b, loop=k_0)

    # binding
    sch.bind(i_1, "blockIdx.x")
    sch.bind(i_2, "puIdx.x")
    sch.annotate(i_1, "bank", BankOrdering.DENSE)
    sch.annotate(i_0, "pim", True)
    sch.annotate(i_0, "change_pim_mode", True)
    sch.annotate(i_2, "bank", BankOrdering.DENSE)
    sch.annotate(k_1, "bank", BankOrdering.DENSE)
    sch.annotate(i_3, "barrier", True)

    # reduction schedule
    block_c = sch.get_block("C")
    i, r = sch.get_loops(block_c)
    i_0, i_1, i_2, i_3 = sch.split(i, factors=[None, N_CHAN, N_PU, N_GRF_B])
    sch.reorder(i_1, i_0, i_2, i_3)

    # reduction binding
    sch.bind(i_1, "blockIdx.x")
    # sch.bind(i_2, "puIdx.x") # NOTE: single bank mode
    sch.bind(i_2, "bankIdx.x")  # NOTE: single bank mode
    sch.bind(i_3, "threadIdx.x")
    sch.annotate(i_1, "bank", BankOrdering.DENSE)
    sch.annotate(i_2, "bank", BankOrdering.ODD)
    # print(sch.mod.script())
    return sch, k_2, cache_write_p, cache_read_a, cache_read_b


def tensorize(
    sch: Schedule,
    k_2: LoopRV,
    cache_write_p: BlockRV,
    cache_read_a: BlockRV,
    cache_read_b: BlockRV,
) -> Schedule:
    sch.blockize(k_2)
    sch.tensorize(k_2, HBMPIM_MAC_INTRIN)
    _, _, _, ax_p, _ = sch.get_loops(cache_write_p)
    sch.blockize(ax_p)
    sch.tensorize(ax_p, HBMPIM_PARTIAL_REDUCTION_INTRIN)
    _, _, _, _, _, _, ax_a = sch.get_loops(cache_read_a)
    sch.blockize(ax_a)
    sch.tensorize(ax_a, HBMPIM_WEIGHT_INTRIN)
    _, _, _, _, _, ax_b = sch.get_loops(cache_read_b)
    sch.blockize(ax_b)
    sch.tensorize(ax_b, HBMPIM_INPUT_INTRIN)
    return sch


def data_copy_func(mod: IRModule) -> Mapping[GlobalVar, PrimFunc]:
    m = dict()
    for var in mod.get_global_vars():
        name = var.name_hint
        if name.startswith("copy_"):
            m[var] = mod[var]
    return m


sch, *args = schedule()
# mod = sch.mod
# mod = from_source(sch.mod.script())

# from tvm.ir.transform import *
from tvm.tir.transform import *
from tvm.target import Target

l = tvm.lower(sch.mod)

# print("[LOWER]\n", l)
target = tvm.target.Target(target="hbmpim", host="llvm")
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
# print("[TIR with PIM data copy]\n", m)

sch = tensorize(sch, *args)
tensorized_mod = sch.mod
# print("[TIR with tensorized module]\n", tensorized_mod)

data_copy_func_map = data_copy_func(m)
# print(data_copy_m)

# print("[TIR with tensorized module and copy code]\n", tensorized_mod)

func = tvm.build(
    sch.mod, target=target, name="kernel", data_copy_func_map=data_copy_func_map
)

if (
    target.kind.name == "cuda"
    or target.kind.name == "rocm"
    or target.kind.name.startswith("opencl")
    or target.kind.name.startswith("hbmpim")
):
    dev_module = func.imported_modules[0]
    print("-----GPU Code-----")
    # print(dev_module.get_source())
    print("-----Host Code-----")
    # print(func.get_source())
else:
    print("-----Code-----")
    # print(func.get_source())

evaluate_operation(func, log=log)

# very relaxed check
tvm.testing.assert_allclose(answer, convert_to_fp16(c.numpy()), rtol=1e-1, atol=1e-1)

