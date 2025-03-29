from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_cpu_gemv_8192_1_8192:
    @T.prim_func
    def main(A: T.Buffer((8192, 8192), "int32"), B: T.Buffer((8192,), "int32"), C: T.Buffer((8192,), "int32"), ALPHA: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((8192,), "int32")
        alpha_val: T.int32 = ALPHA[0]
        for i_0_i_1_fused in T.parallel(8192, annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for i_2_init, i_3_init in T.grid(1, 1):
                with T.block("C_init"):
                    v_i = T.axis.spatial(8192, i_0_i_1_fused + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C_global[v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C_global[v_i] = 0
            for k_0, i_2, k_1, i_3 in T.grid(128, 1, 64, 1):
                with T.block("C_update"):
                    v_i = T.axis.spatial(8192, i_0_i_1_fused + i_2 + i_3)
                    v_k = T.axis.reduce(8192, k_0 * 64 + k_1)
                    T.reads(C_global[v_i], A[v_i, v_k], B[v_k])
                    T.writes(C_global[v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C_global[v_i] = C_global[v_i] + alpha_val * A[v_i, v_k] * B[v_k]
            for ax0 in range(1):
                with T.block("C_global"):
                    v0 = T.axis.spatial(8192, i_0_i_1_fused + ax0)
                    T.reads(C_global[v0])
                    T.writes(C[v0])
                    C[v0] = C_global[v0]
# from tvm import tir
def apply_trace_cpu_gemv_8192_1_8192(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
  l2, l3 = sch.get_loops(block=b0)
  v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64, decision=[1024, 8, 1, 1])
  l8, l9, l10, l11 = sch.split(loop=l2, factors=[v4, v5, v6, v7], preserve_unit_iters=True)
  v12, v13 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64, decision=[128, 64])
  l14, l15 = sch.split(loop=l3, factors=[v12, v13], preserve_unit_iters=True)
  sch.reorder(l8, l9, l14, l10, l15, l11)
  b16 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")
  sch.reverse_compute_at(block=b16, loop=l9, preserve_unit_loops=True, index=-1)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=1536)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
  v17 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v17)
  sch.enter_postproc()
  b18 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b18, ann_key="meta_schedule.parallel")
  sch.unannotate(block_or_loop=b18, ann_key="meta_schedule.vectorize")
  sch.unannotate(block_or_loop=b18, ann_key="meta_schedule.unroll_explicit")
  b19, b20 = sch.get_child_blocks(b18)
  l21, l22, l23, l24, l25, l26 = sch.get_loops(block=b19)
  l27 = sch.fuse(l21, l22, preserve_unit_iters=True)
  sch.parallel(loop=l27)
  sch.annotate(block_or_loop=l27, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l27, ann_key="pragma_unroll_explicit", ann_val=1)
  l28, l29 = sch.get_loops(block=b20)
  b30 = sch.get_block(name="C", func_name="main")
  l31, l32, l33, l34, l35 = sch.get_loops(block=b30)
  b36 = sch.decompose_reduction(block=b30, loop=l32)
