import sys
sys.path.append("/root/dev/tvm/python")

import os
import tvm
from tvm.target import Target
from tvm.tir.transform import *
import numpy as np
from tvm.script import tir as T

target = tvm.target.Target(target="upmem", host="llvm")
successes = []
failures = []

def upmem_gemv_factory(M, N, L, dtype):
    @tvm.script.ir_module
    class UPMEMModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            A = T.match_buffer(a, (M, N), dtype=dtype)
            B = T.match_buffer(b, (N, L), dtype=dtype)
            C = T.match_buffer(c, (M, L), dtype=dtype)
            for i, j, k in T.grid(M, L, N):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.int32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    return UPMEMModule

def gemmTile(m, n, l, mb, nb, lb, mc, nc, lc, dtype):
    sch = tvm.tir.Schedule(upmem_gemv_factory(m, n, l, dtype))
    block_c = sch.get_block("C")
    _, _, k = sch.get_loops(block_c)

    kb, ko = sch.split(k, factors=[nb, None])
    block_crf = sch.rfactor(kb, factor_axis=0)

    ca = sch.cache_read(block_crf, 0, "local")
    cb = sch.cache_read(block_crf, 1, "local")
    cc = sch.cache_write(block_crf, 0, "local")

    i, j, kb, ko = sch.get_loops(block_crf)
    ib, io, ii, ic = sch.split(i, factors=[mb, 16, None, mc])
    jb, jo, jc = sch.split(j, factors=[lb, None, lc])
    ko, ki = sch.split(ko, factors=[None, nc])
    sch.reorder(ib, jb, kb, io, jo, ii, ko, ic, jc,ki)

    sch.compute_at(ca, ko)
    sch.compute_at(cb, ko)
    sch.reverse_compute_at(cc, ii)

    sch.bind(ib, "blockIdx.x")
    sch.bind(jb, "blockIdx.y")
    sch.bind(kb, "blockIdx.z")
    sch.bind(io, "threadIdx.x")

    i, _, _ = sch.get_loops(block_c)
    it, ii = sch.split(i, factors=[16, None])
    sch.parallel(it)
    return sch, (m, n, l, dtype)

def func_test(fname, sch, M, N, L, dtype):
    try:
        flag = False
        with open("./results/" + fname + ".txt", "w") as f:
            l = tvm.lower(sch.mod)
            
            print("Testing", fname)
            print("[LOWER]", file=f)
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
                na = np.random.rand(M, N).astype(dtype)
                nb = np.random.rand(N, L).astype(dtype)
            else:
                na = np.random.randint(0, 100, (M, N)).astype(dtype)
                nb = np.random.randint(0, 100, (N, L)).astype(dtype)
            nc = np.zeros((M, L), dtype=dtype)

            a = tvm.nd.array(na, device)
            b = tvm.nd.array(nb, device)
            c = tvm.nd.array(nc, device)
            func(a, b, c)
            nc = np.dot(na, nb)
            
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
        
def gemmTest(m, n, l, mb, nb, lb, mc, nc, lc, dtype, name):
    sch, args = gemmTile(m, n, l, mb, nb, lb, mc, nc, lc, dtype)
    func_test(name, sch, *args)

def cleanup():
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./errors"):
        os.makedirs("./errors")
    for fname in os.listdir("./results"):
        os.remove("./results/" + fname)
    for fname in os.listdir("./errors"):
        os.remove("./errors/" + fname)
    
def test_all():
    cleanup()
    gemmTest(512, 512, 512, 4, 4, 4, 8, 16, 8, "int32", "gemm_A")
    gemmTest(512, 1024, 512, 4, 4, 4, 8, 16, 8, "int32", "gemm_B")
    gemmTest(512, 2048, 512, 4, 4, 4, 8, 16, 8, "int32", "gemm_C")
    gemmTest(512, 2048, 1024, 4, 4, 4, 8, 16, 8, "int32", "gemm_D")
    gemmTest(2048, 2048, 2048, 4, 4, 4, 8, 16, 8, "int32", "gemm_E")
    gemmTest(2048, 2048, 2048, 8, 8, 8, 8, 16, 8, "int32", "gemm_F")
    gemmTest(2048, 2048, 2048, 8, 8, 8, 8, 32, 8, "int32", "gemm_G")
    gemmTest(2048, 2048, 2048, 8, 8, 8, 8, 64, 4, "int32", "gemm_H")
    gemmTest(2048, 2048, 2048, 8, 8, 8, 8, 32, 8, "float32", "gemm_I")
    gemmTest(4096, 4096, 4096, 8, 8, 8, 8, 32, 8, "float32", "gemm_J")

    for fname in successes:
        print(fname, "PASS")
    for fname in failures:
        print(fname, "FAIL")

test_all()