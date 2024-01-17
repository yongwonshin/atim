import os

import tvm
from tvm.target import Target
from tvm.tir.transform import *
import numpy as np
from tvm.script import tir as T

target = tvm.target.Target(target="upmem", host="llvm")

def upmem_gemv_factory(m_, k_, dtype_):
    @tvm.script.ir_module
    class UPMEMModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, (m_, k_), dtype=dtype_)
            B = T.match_buffer(b, (k_,), dtype=dtype_)
            C = T.match_buffer(c, (m_,), dtype=dtype_)
            for i, k in T.grid(m_, k_):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]
    return UPMEMModule

def gemvRCTile(m_, k_, n_xb_, n_yb_, dtype_):
    M, K, N_XB, N_YB, N_YT, N_CACHE, N_RTHREADS, dtype = m_, k_, n_xb_, n_yb_, 16, 64, 16, dtype_
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    xb, xo = sch.split(k, factors = [N_XB, None])
    block_crf = sch.rfactor(xb, factor_axis=0)
    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")
    i, xb, k = sch.get_loops(block_crf)
    yb, yo, yi, yc = sch.split(i, factors=[N_YB, N_YT, None, 2])
    xo, xi = sch.split(k, factors=[None, N_CACHE])
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
    return sch, (m_, k_, dtype_)

def gemvRTile(m_, k_, n_yb_, dtype_):
    M, K, N_YB, N_YT, N_CACHE, N_RTHREADS, dtype = m_, k_, n_yb_, 16, 64, 16, dtype_
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, xo = sch.get_loops(block_c)
    ca = sch.cache_read(block_c, 0, "local")
    cb = sch.cache_read(block_c, 1, "local")
    cc = sch.cache_write(block_c, 0, "local")
    i, k = sch.get_loops(block_c)
    yb, yo, yi, yc = sch.split(i, factors=[N_YB, N_YT, None, 2])
    xo, xi = sch.split(k, factors=[None, N_CACHE])
    sch.reorder(yb, yo, yi, yc, xo, xi)
    sch.compute_at(ca, xo)
    sch.compute_at(cb, xo)
    sch.reverse_compute_at(cc, yi)
    sch.bind(yb, "blockIdx.x")
    sch.bind(yo, "threadIdx.x")
    sch.annotate(yb, "bank", True)
    return sch, (m_, k_, dtype_)

def StridedBankTile(m_, k_, n_xb_, n_yb_, dtype_):
    M, K, N_XB, N_YB, N_YT, N_CACHE, N_RTHREADS, dtype = m_, k_, n_xb_, n_yb_, 16, 64, 16, dtype_
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)

    xb, x2, xo = sch.split(k, factors = [N_XB, 2, None])
    block_crf = sch.rfactor(x2, factor_axis=0)
    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")
    i, xb, x2, xo = sch.get_loops(block_crf)
    yb, yo, yi, yc = sch.split(i, factors=[N_YB, N_YT, M // N_YB // N_YT // 2, None])
    xo, xi = sch.split(xo, factors=[K // N_XB // N_CACHE, None])
    sch.reorder(x2, xb, yb, yo, yi, yc, xo, xi)
    sch.compute_at(ca, xo)
    sch.compute_at(cb, xo)
    sch.reverse_compute_at(cc, yo)
    sch.bind(x2, "blockIdx.x")
    sch.bind(yb, "blockIdx.y")
    sch.bind(yo, "threadIdx.x")
    # sch.decompose_reduction(block_crf, xo)
    i, _ = sch.get_loops(block_c)
    it, ii = sch.split(i, factors=[N_RTHREADS, None])
    sch.parallel(it)
    return sch, (M, K, dtype)


def HBMStyleTile(dtype_):
    M, K, N_CHAN, N_BANK, N_PU, N_GRF_A, N_GRF_B, dtype = 4096, 1024, 64, 16, 8, 8, 8, dtype_
    sch = tvm.tir.Schedule(upmem_gemv_factory(M, K, dtype))
    block_c = sch.get_block("C")
    _, k = sch.get_loops(block_c)
    k_1, k_0, k_2 = sch.split(k, factors=[None, N_BANK // N_PU, 128])
    block_crf = sch.rfactor(k_0, factor_axis=0)
    cache_read_a = sch.cache_read(block_crf, 0, "local")
    cache_read_b = sch.cache_read(block_crf, 1, "local")
    cache_write_p = sch.cache_write(block_crf, 0, "local")
    i, k_1, k_0, k_2 = sch.get_loops(block_crf)
    i_1, i_2, i_3 = sch.split(i, factors=[N_CHAN, N_PU, N_GRF_B])
    sch.reorder(i_1, i_2, k_0, k_1, i_3, k_2)
    sch.compute_at(cache_read_a, loop=i_3)
    sch.compute_at(cache_read_b, loop=k_1)
    sch.reverse_compute_at(cache_write_p, i_3)
    sch.blockize(k_2)
    sch.bind(i_1, "blockIdx.x")
    sch.bind(i_2, "blockIdx.y")
    sch.bind(k_0, "blockIdx.z")
    sch.bind(i_3, "threadIdx.x")
    i, r = sch.get_loops(block_c)
    i_0, i_1, i_2, i_3 = sch.split(i, factors=[None, N_CHAN, N_PU, N_GRF_B])
    sch.reorder(i_1, i_0, i_2, i_3)
    return sch, (M, K, dtype)

successes = []
failures = []

def func_test(fname, sch, M, K, dtype):
    try:
        flag = False
        with open("./results/" + fname + ".txt", "w") as f:
            l = tvm.lower(sch.mod)
            
            print("[LOWER]")
            print(l, file=f)
            mp, _ = Target.canon_target_map_and_host({ target: l }, "llvm")
            m = mp[target]
            m = BindTarget(target)(m)
            m = VerifyMemory()(m)
            m = AnnotateEntryFunc()(m)
            m = AnnotateDeviceRegions()(m)
            m = ExtractPimTransferSchedule()(m)
            m = SplitHostDevice()(m)
            print("[TIR with PIM data copy]\n", file=f)
            print(m["main"], file=f)
            
            func = tvm.build(sch.mod, target=target, name="gemm")
            print("\n\n[UPMEM source]\n", file=f)
            print(func.imported_modules[0].get_source(), file=f)
            device = tvm.device(target.kind.name, 0)
            
            if dtype[:5] == "float":
                na = np.random.rand(M, K).astype(dtype)
                nb = np.random.rand(K,).astype(dtype)
            else:
                na = np.random.randint(0, 100, (M, K)).astype(dtype)
                nb = np.random.randint(0, 100, (K,)).astype(dtype)
            nc = np.dot(na,nb)

            a = tvm.nd.array(na, device)
            b = tvm.nd.array(nb, device)
            c = tvm.nd.array(nc, device)

            func(a, b, c)
            print("\n\n[Correctness Test]\n", file=f)
            print("RESULT", file=f)
            print(c.asnumpy()[:32], file=f)
            print("", file=f)
            print("EXPECTED", file=f)
            print(nc[:32], file=f)
            res = np.max(c.asnumpy() - nc)
            print(res, file=f)
            if np.abs(res) < 0.01:
                flag = True
            else:
                flag = False
        if flag:
            successes.append(fname)
        else:
            failures.append(fname)
    except Exception as e:
        with open("./errors/" + fname + ".txt", "w") as f:
            f.write(str(e))
        failures.append(fname)

def gemv2048_4_4():
    sch, args = gemvRCTile(2048, 2048, 4, 4, "float32")
    func_test("gemv2048_4_4", sch, *args)
    
def gemv2048_4():
    sch, args = gemvRTile(2048, 2048, 4, "float32")
    func_test("gemv2048_4", sch, *args)
    
def gemv8192_4_4():
    sch, args = gemvRCTile(8192, 8192, 4, 4, "float32")
    func_test("gemv8192_4_4", sch, *args)
    
def gemv8192_8_8():
    sch, args = gemvRCTile(8192, 8192, 8, 8, "float32")
    func_test("gemv8192_8_8", sch, *args)
    
def gemv8192_16_16():
    sch, args = gemvRCTile(8192, 8192, 16, 16, "float32")
    func_test("gemv8192_16_16", sch, *args)
    
def gemv8192_16_16_int():
    sch, args = gemvRCTile(8192, 8192, 16, 16, "int32")
    func_test("gemv8192_16_16_int", sch, *args)
    
def gemv_8192_4096_16_16():
    sch, args = gemvRCTile(8192, 4096, 16, 16, "float32")
    func_test("gemv8192_4096_16_16", sch, *args)
    
def gemv_4096_8192_16_16():
    sch, args = gemvRCTile(4096, 8192, 16, 16, "float32")
    func_test("gemv8192_4096_16_16", sch, *args)
    
def gemv_32768_32768_32_32():
    sch, args = gemvRCTile(32768, 32768, 32, 32, "int32")
    func_test("gemv32768_32768_32_32", sch, *args)
    
def gemv_32768_32768_32_64():
    sch, args = gemvRCTile(32768, 32768, 32, 64, "float32")
    func_test("gemv32768_32768_32_64", sch, *args)
    
def strided_gemv_8192_8192_16_16():
    sch, args = StridedBankTile(8192, 8192, 16, 16, "int32")
    func_test("strided_gemv_8192_8192_16_16", sch, *args)

def HBMStyle():
    sch, args = HBMStyleTile("int32")
    func_test("hbmstyle_float32", sch, *args)
    
def cleanup():
    for fname in os.listdir("./results"):
        os.remove("./results/" + fname)
    for fname in os.listdir("./errors"):
        os.remove("./errors/" + fname)
    
def test_all():
    cleanup()
    gemv2048_4_4()
    gemv2048_4()
    gemv8192_4_4()
    gemv8192_8_8()
    gemv8192_16_16()
    gemv8192_16_16_int()
    gemv_8192_4096_16_16()
    gemv_4096_8192_16_16()
    gemv_32768_32768_32_32()
    gemv_32768_32768_32_64()
    strided_gemv_8192_8192_16_16()
    HBMStyle()

    for fname in successes:
        print(fname, "PASS")
    for fname in failures:
        print(fname, "FAIL")

test_all()