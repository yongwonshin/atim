# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((2048, 2048), "float32"), B: T.Buffer((2048,), "float32"), C: T.Buffer((2048,), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        C_local = T.allocate([2], "float32", "local")
        A_local = T.allocate([64], "float32", "local")
        B_local = T.allocate([64], "float32", "local")
        for y_inner_outer in range(256):
            C_local_1 = T.Buffer((2,), data=C_local, scope="local", align=8)
            for y_c in range(2):
                C_local_1[y_c] = T.float32(0)
                for k_outer in range(32):
                    A_local_1 = T.Buffer((64,), data=A_local, scope="local")
                    for ax1 in range(64):
                        A_1 = T.Buffer((4194304,), data=A.data)
                        A_local_1[ax1] = A_1[threadIdx_x * 1048576 + y_inner_outer * 4096 + y_c * 2048 + k_outer * 64 + ax1]
                    B_local_1 = T.Buffer((64,), data=B_local, scope="local")
                    for ax0 in range(64):
                        B_local_1[ax0] = B[k_outer * 64 + ax0]
                    for k_inner in range(64):
                        C_local_1[y_c] = C_local_1[y_c] + A_local_1[k_inner] * B_local_1[k_inner]
            for y_inner_inner in range(2):
                C[threadIdx_x * 512 + y_inner_outer * 2 + y_inner_inner] = C_local_1[y_inner_inner]