from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_cpu_mtv_4096_1_4096:
    @T.prim_func
    def main(A: T.Buffer((4096, 4096), "int32"), B: T.Buffer((4096,), "int32"), C: T.Buffer((4096,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0_i_1_fused in T.parallel(512, annotations={"pragma_auto_unroll_max_step": T.int64(16), "pragma_unroll_explicit": T.int64(1)}):
            for i_2_init, i_3_init in T.grid(8, 1):
                with T.block("C_init"):
                    v_i = T.axis.spatial(4096, i_0_i_1_fused * 8 + i_2_init + i_3_init)
                    T.reads()
                    T.writes(C[v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_i] = 0
            for k_0, i_2, k_1, i_3 in T.grid(4096, 8, 1, 1):
                with T.block("C_update"):
                    v_i = T.axis.spatial(4096, i_0_i_1_fused * 8 + i_2 + i_3)
                    v_k = T.axis.reduce(4096, k_0 + k_1)
                    T.reads(C[v_i], A[v_i, v_k], B[v_k])
                    T.writes(C[v_i])
                    T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    C[v_i] = C[v_i] + A[v_i, v_k] * B[v_k]
# from tvm import tir
def apply_trace_cpu_mtv_4096_1_4096(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
  l2, l3 = sch.get_loops(block=b0)
  v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64, decision=[8, 64, 8, 1])
  l8, l9, l10, l11 = sch.split(loop=l2, factors=[v4, v5, v6, v7], preserve_unit_iters=True)
  v12, v13 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64, decision=[4096, 1])
  l14, l15 = sch.split(loop=l3, factors=[v12, v13], preserve_unit_iters=True)
  sch.reorder(l8, l9, l14, l10, l15, l11)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=1536)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
  v16 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v16)
  sch.enter_postproc()
  b17 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b17, ann_key="meta_schedule.parallel")
  sch.unannotate(block_or_loop=b17, ann_key="meta_schedule.vectorize")
  sch.unannotate(block_or_loop=b17, ann_key="meta_schedule.unroll_explicit")
  b18, = sch.get_child_blocks(b17)
  l19, l20, l21, l22, l23, l24 = sch.get_loops(block=b18)
  l25 = sch.fuse(l19, l20, preserve_unit_iters=True)
  sch.parallel(loop=l25)
  sch.annotate(block_or_loop=l25, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l25, ann_key="pragma_unroll_explicit", ann_val=1)
  b26 = sch.get_block(name="C", func_name="main")
  l27, l28, l29, l30, l31 = sch.get_loops(block=b26)
  b32 = sch.decompose_reduction(block=b26, loop=l28)
