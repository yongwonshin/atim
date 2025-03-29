import tvm
from tvm.script import ir as I
from tvm.script import tir as T


def get_base_module(op_type, M, N, K, dtype):
    if op_type == "mtv":
        return upmem_mtv_factory(M, K, dtype)
    elif op_type == "ttv":
        return upmem_ttv_factory(M, N, K, dtype)
    elif op_type == "gemv":
        return upmem_gemv_factory(M, K, dtype)
    elif op_type == "va":
        return upmem_va_factory(M, dtype)
    elif op_type == "ta":
        return upmem_ta_factory(M, N, K, dtype)
    elif op_type == "geva":
        return upmem_geva_factory(M, dtype)
    elif op_type == "polymixed":
        return upmem_poly_mixed_factory(M, N, dtype)
    elif op_type == "dot":
        dtype = "int64"
        return upmem_dot_factory(M, dtype)
    elif op_type == "red":
        dtype = "int64"
        return upmem_red_factory(M, dtype)
    elif op_type == "innerprod":
        dtype = "int64"
        return upmem_innerprod_factory(M, N, K, dtype)
    elif op_type == "mmtv":
        return upmem_mmtv_factory(M, N, K, dtype)
    else:
        raise Exception(f"Unknown operator type: {op_type}")
# basic kernels
def upmem_va_factory(M, dtype):
    @tvm.script.ir_module
    class VAModule:
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
            C = T.match_buffer(c, (M,), dtype=dtype)
            for i in T.grid(M):
                with T.block("C"):
                    v_i = T.axis.remap("S", [i])
                    C[v_i] = A[v_i] + B[v_i]

    return VAModule


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


def upmem_red_factory(M, dtype):
    @tvm.script.ir_module
    class REDModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A"],
                }
            )
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (1,), dtype=dtype)
            for i in T.grid(M):
                with T.block("C"):
                    with T.init():
                        B[0] = 0
                    v_i = T.axis.remap("R", [i])
                    B[0] = B[0] + A[v_i]

    return REDModule


def upmem_mtv_factory(M, K, dtype):
    @tvm.script.ir_module
    class MTVModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (M, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M,), dtype=dtype)
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]

    return MTVModule


# higher dimension kernels
def upmem_ta_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class TAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(b, (M, N, K), dtype=dtype)
            C = T.match_buffer(c, (M, N, K), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                    C[v_i, v_j, v_k] = A[v_i, v_j, v_k] + B[v_i, v_j, v_k]

    return TAModule


def upmem_innerprod_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class InnerprodModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(b, (M, N, K), dtype=dtype)
            C = T.match_buffer(c, (1,), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    with T.init():
                        C[0] = 0
                    v_i, v_j, v_k = T.axis.remap("RRR", [i, j, k])
                    C[0] = C[0] + A[v_i, v_j, v_k] * B[v_i, v_j, v_k]

    return InnerprodModule


def upmem_ttv_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class TTVModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M, N), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[v_i, v_j] = 0
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_j, v_k] * B[v_k]

    return TTVModule


# batched multi-head attention
def upmem_mmtv_factory(M, N, K, dtype):
    @tvm.script.ir_module
    class MTTVModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A"],
                }
            )
            A = T.match_buffer(a, (M, N, K), dtype=dtype)
            B = T.match_buffer(
                b,
                (
                    M,
                    K,
                ),
                dtype=dtype,
            )
            C = T.match_buffer(c, (M, N), dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[v_i, v_j] = 0
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_j, v_k] * B[v_i, v_k]

    return MTTVModule


# polybench kernels
def upmem_geva_factory(M, dtype):
    @tvm.script.ir_module
    class PolyVAModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, alpha: T.handle, beta: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["A", "B"],
                }
            )
            A = T.match_buffer(a, (M,), dtype=dtype)
            B = T.match_buffer(b, (M,), dtype=dtype)
            C = T.match_buffer(c, (M,), dtype=dtype)
            ALPHA = T.match_buffer(alpha, (1,), dtype=dtype)
            BETA = T.match_buffer(beta, (1,), dtype=dtype)
            alpha_val = ALPHA[0]
            beta_val = BETA[0]
            for i in T.grid(M):
                with T.block("C"):
                    v_i = T.axis.remap("S", [i])
                    C[v_i] = alpha_val * A[v_i] + beta_val * B[v_i]

    return PolyVAModule


def upmem_gemv_factory(M, K, dtype):
    @tvm.script.ir_module
    class PolyGEMV1Module:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle, alpha: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (M, K), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M), dtype=dtype)
            ALPHA = T.match_buffer(alpha, (1,), dtype=dtype)
            alpha_val = ALPHA[0]
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + alpha_val * A[v_i, v_k] * B[v_k]

    return PolyGEMV1Module


def upmem_poly_mixed_factory(M, N, dtype):
    @tvm.script.ir_module
    class PolyMixedModule:
        @T.prim_func
        def main(c: T.handle, u1: T.handle, v1: T.handle, u2: T.handle, v2: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                    "pragma_explicit_h2d": ["U1", "V1", "U2", "V2"],
                }
            )
            C = T.match_buffer(c, (M, N), dtype=dtype)
            U1 = T.match_buffer(u1, (M,), dtype=dtype)
            V1 = T.match_buffer(v1, (N,), dtype=dtype)
            U2 = T.match_buffer(u2, (M,), dtype=dtype)
            V2 = T.match_buffer(v2, (N,), dtype=dtype)
            for i, j in T.grid(M, N):
                with T.block("C"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    with T.init():
                        C[v_i, v_j] = 0
                    C[v_i, v_j] = C[v_i, v_j] + U1[v_i] * V1[v_j] + U2[v_i] * V2[v_j]

    return PolyMixedModule


def upmem_poly_gemv2_factory(M, K, dtype):
    @tvm.script.ir_module
    class PolyGEMV2Module:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": T.bool(True), "pragma_explicit_h2d": ["A"]}
            )
            A = T.match_buffer(a, (K, M), dtype=dtype)
            B = T.match_buffer(b, (K,), dtype=dtype)
            C = T.match_buffer(c, (M), dtype=dtype)
            for i, k in T.grid(M, K):
                with T.block("C"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    with T.init():
                        C[v_i] = 0
                    C[v_i] = C[v_i] + A[v_k, v_i] * B[v_k]

    return PolyGEMV2Module