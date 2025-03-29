from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_cpu_mmtv_256_512_512:
    @T.prim_func
    def main(A: T.Buffer((256, 512, 512), "int32"), B: T.Buffer((256, 512), "int32"), C: T.Buffer((256, 512), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_global = T.alloc_buffer((256, 512), "int32")
        for i_0_j_0_fused_fused in T.parallel(32768):
            for i_1, j_1 in T.grid(1, 2):
                for i_2_init, j_2_init, i_3_init, j_3_init in T.grid(1, 2, 1, 1):
                    with T.block("C_init"):
                        v_i = T.axis.spatial(256, i_0_j_0_fused_fused // 128 + i_1 + i_2_init + i_3_init)
                        v_j = T.axis.spatial(512, i_0_j_0_fused_fused % 128 * 4 + j_1 * 2 + j_2_init + j_3_init)
                        T.reads()
                        T.writes(C_global[v_i, v_j])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_i, v_j] = 0
                for k_0, i_2, j_2, k_1, i_3, j_3 in T.grid(8, 1, 2, 64, 1, 1):
                    with T.block("C_update"):
                        v_i = T.axis.spatial(256, i_0_j_0_fused_fused // 128 + i_1 + i_2 + i_3)
                        v_j = T.axis.spatial(512, i_0_j_0_fused_fused % 128 * 4 + j_1 * 2 + j_2 + j_3)
                        v_k = T.axis.reduce(512, k_0 * 64 + k_1)
                        T.reads(C_global[v_i, v_j], A[v_i, v_j, v_k], B[v_i, v_k])
                        T.writes(C_global[v_i, v_j])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        C_global[v_i, v_j] = C_global[v_i, v_j] + A[v_i, v_j, v_k] * B[v_i, v_k]
            for ax0 in range(1):
                for ax1_fused in T.vectorized(4):
                    with T.block("C_global"):
                        v0 = T.axis.spatial(256, i_0_j_0_fused_fused // 128 + ax0)
                        v1 = T.axis.spatial(512, i_0_j_0_fused_fused % 128 * 4 + ax1_fused)
                        T.reads(C_global[v0, v1])
                        T.writes(C[v0, v1])
                        C[v0, v1] = C_global[v0, v1]
# from tvm import tir
def apply_trace_cpu_mmtv_256_512_512(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
  l2, l3, l4 = sch.get_loops(block=b0)
  v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64, decision=[256, 1, 1, 1])
  l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8], preserve_unit_iters=True)
  v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64, decision=[128, 2, 2, 1])
  l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16], preserve_unit_iters=True)
  v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64, decision=[8, 64])
  l23, l24 = sch.split(loop=l4, factors=[v21, v22], preserve_unit_iters=True)
  sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
  b25 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")
  sch.reverse_compute_at(block=b25, loop=l17, preserve_unit_loops=True, index=-1)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=1536)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
  v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)
  sch.enter_postproc()
  b27 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.parallel")
  sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.vectorize")
  sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.unroll_explicit")
  b28, b29 = sch.get_child_blocks(b27)
  l30, l31, l32, l33, l34, l35, l36, l37, l38, l39 = sch.get_loops(block=b28)
  l40 = sch.fuse(l30, l31, preserve_unit_iters=True)
  sch.parallel(loop=l40)
  l41, l42, l43 = sch.get_loops(block=b29)
  l44 = sch.fuse(l41, preserve_unit_iters=True)
  sch.parallel(loop=l44)
  l45 = sch.fuse(l43, preserve_unit_iters=True)
  sch.vectorize(loop=l45)
  b46 = sch.get_block(name="C", func_name="main")
  l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b46)
  b56 = sch.decompose_reduction(block=b46, loop=l50)
