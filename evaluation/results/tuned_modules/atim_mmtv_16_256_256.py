from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_atim_mmtv_16_256_256:
    @T.prim_func
    def main(A: T.Buffer((16, 256, 256), "int32"), B: T.Buffer((16, 256), "int32"), C: T.Buffer((16, 256), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.optimization_level": T.int64(4)})
            C_rf_global = T.alloc_buffer((16, 16, 256), "int32")
            C_rf_global_local = T.alloc_buffer((16, 16, 256), "int32", scope="local")
            A_local = T.alloc_buffer((16, 256, 256), "int32", scope="local")
            B_local = T.alloc_buffer((16, 256), "int32", scope="local")
            for k_0 in T.thread_binding(16, thread="blockIdx.x", annotations={"bank": T.int64(1), "pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(0)}):
                for i_0 in T.thread_binding(16, thread="blockIdx.y", annotations={"bank": T.int64(1)}):
                    for j_0 in T.thread_binding(1, thread="blockIdx.z", annotations={"bank": T.int64(1)}):
                        for j_1 in T.thread_binding(16, thread="threadIdx.x"):
                            for i_1, i_2, j_2 in T.grid(1, 1, 1):
                                for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(1, 8, 1, 2):
                                    with T.block("C_rf_init"):
                                        v_i = T.axis.spatial(16, i_0 + i_1 + i_2 + i_3_init + i_4_init)
                                        v_j = T.axis.spatial(256, j_0 * 256 + j_1 * 16 + j_2 * 16 + j_3_init * 2 + j_4_init)
                                        vk_0 = T.axis.spatial(16, k_0)
                                        T.reads()
                                        T.writes(C_rf_global_local[v_i, vk_0, v_j])
                                        T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                        C_rf_global_local[v_i, vk_0, v_j] = 0
                                for k_1_0 in range(1):
                                    for ax0, ax1, ax2 in T.grid(1, 16, 16):
                                        with T.block("A_local"):
                                            v0 = T.axis.spatial(16, i_0 + ax0)
                                            v1 = T.axis.spatial(256, j_1 * 16 + ax1)
                                            v2 = T.axis.spatial(256, k_0 * 16 + ax2)
                                            T.reads(A[v0, v1, v2])
                                            T.writes(A_local[v0, v1, v2])
                                            T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                            A_local[v0, v1, v2] = A[v0, v1, v2]
                                    for ax0, ax1 in T.grid(1, 16):
                                        with T.block("B_local"):
                                            v0 = T.axis.spatial(16, i_0 + ax0)
                                            v1 = T.axis.spatial(256, k_0 * 16 + ax1)
                                            T.reads(B[v0, v1])
                                            T.writes(B_local[v0, v1])
                                            T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                            B_local[v0, v1] = B[v0, v1]
                                    for i_3, j_3, k_1_1, i_4, j_4, k_1_2 in T.grid(1, 8, 2, 1, 2, 8):
                                        with T.block("C_rf_update"):
                                            v_i = T.axis.spatial(16, i_0 + i_1 + i_2 + i_3 + i_4)
                                            v_j = T.axis.spatial(256, j_0 * 256 + j_1 * 16 + j_2 * 16 + j_3 * 2 + j_4)
                                            vk_0 = T.axis.spatial(16, k_0)
                                            vk_1 = T.axis.reduce(16, k_1_0 * 16 + k_1_1 * 8 + k_1_2)
                                            T.reads(C_rf_global_local[v_i, vk_0, v_j], A_local[v_i, v_j, vk_0 * 16 + vk_1], B_local[v_i, vk_0 * 16 + vk_1])
                                            T.writes(C_rf_global_local[v_i, vk_0, v_j])
                                            T.block_attr({"meta_schedule.meta_schedule_rfactor_producer_block": T.int64(1), "meta_schedule.tiling_structure": "SSSRSRSR"})
                                            C_rf_global_local[v_i, vk_0, v_j] = C_rf_global_local[v_i, vk_0, v_j] + A_local[v_i, v_j, vk_0 * 16 + vk_1] * B_local[v_i, vk_0 * 16 + vk_1]
                                for ax0, ax1, ax2 in T.grid(1, 1, 16):
                                    with T.block("C_rf_global_local"):
                                        v0 = T.axis.spatial(16, i_0 + ax0)
                                        v1 = T.axis.spatial(16, k_0 + ax1)
                                        v2 = T.axis.spatial(256, j_1 * 16 + ax2)
                                        T.reads(C_rf_global_local[v0, v1, v2])
                                        T.writes(C_rf_global[v0, v1, v2])
                                        C_rf_global[v0, v1, v2] = C_rf_global_local[v0, v1, v2]
            for i in T.serial(16, annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(0)}):
                for j in range(256):
                    with T.block("C_init"):
                        v_i, v_j = T.axis.remap("SS", [i, j])
                        T.reads()
                        T.writes(C[v_i, v_j])
                        T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                        C[v_i, v_j] = 0
                    for k_0 in range(16):
                        with T.block("C_update"):
                            vk_0, v_i, v_j = T.axis.remap("RSS", [k_0, i, j])
                            T.reads(C[v_i, v_j], C_rf_global[v_i, vk_0, v_j])
                            T.writes(C[v_i, v_j])
                            T.block_attr({"meta_schedule.meta_schedule_rfactor_consumer_block": T.int64(1), "meta_schedule.random_compute_producer": T.int64(1)})
                            C[v_i, v_j] = C[v_i, v_j] + C_rf_global[v_i, vk_0, v_j]
# from tvm import tir
def apply_trace_atim_mmtv_16_256_256(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  l2, l3, l4 = sch.get_loops(block=b0)
  v5, v6 = sch.sample_perfect_tile2(loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[16, 16])
  l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
  b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
  sch.annotate(block_or_loop=b9, ann_key="meta_schedule.meta_schedule_rfactor_producer_block", ann_val=1)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.meta_schedule_rfactor_consumer_block", ann_val=1)
  b10 = sch.get_block(name="C_rf", func_name="main")
  sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
  sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
  l11, l12, l13, l14 = sch.get_loops(block=b10)
  v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l11, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[16, 1, 1, 1, 1])
  l20, l21, l22, l23, l24 = sch.split(loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True)
  v25, v26, v27, v28, v29 = sch.sample_perfect_tile(loop=l12, n=5, max_innermost_factor=256, min_innermost_factor=1, decision=[1, 16, 1, 8, 2])
  l30, l31, l32, l33, l34 = sch.split(loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True)
  v35, v36, v37 = sch.sample_perfect_tile(loop=l14, n=3, max_innermost_factor=256, min_innermost_factor=1, decision=[1, 2, 8])
  l38, l39, l40 = sch.split(loop=l14, factors=[v35, v36, v37], preserve_unit_iters=True)
  sch.reorder(l13, l20, l30, l21, l31, l22, l32, l38, l23, l33, l39, l24, l34, l40)
  sch.bind(loop=l13, thread_axis="blockIdx.x")
  sch.bind(loop=l20, thread_axis="blockIdx.y")
  sch.bind(loop=l30, thread_axis="blockIdx.z")
  sch.reorder(l31, l21)
  sch.bind(loop=l31, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=l13, ann_key="bank", ann_val=1)
  sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
  sch.annotate(block_or_loop=l30, ann_key="bank", ann_val=1)
  b41 = sch.cache_write(block=b10, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b41, loop=l32, preserve_unit_loops=True, index=-1)
  b42 = sch.cache_read(block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10])
  sch.compute_at(block=b42, loop=l38, preserve_unit_loops=True, index=-1)
  v43 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
  b44 = sch.cache_read(block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10])
  sch.compute_at(block=b44, loop=l38, preserve_unit_loops=True, index=-1)
  v45 = sch.sample_categorical(candidates=[1], probs=[1], decision=0)
  sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
  v46 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
  b47 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
  sch.enter_postproc()
  b48 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
  b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
  l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b49)
  l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b50)
  l75, l76, l77, l78, l79, l80, l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b51)
  sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=0)
  l89, l90, l91, l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b52)
  l99, l100, l101 = sch.get_loops(block=b53)
  sch.annotate(block_or_loop=l99, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l99, ann_key="pragma_unroll_explicit", ann_val=0)
  b102 = sch.get_block(name="C_rf", func_name="main")
  l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113, l114, l115, l116 = sch.get_loops(block=b102)
  b117 = sch.decompose_reduction(block=b102, loop=l110)
  b118 = sch.get_block(name="C", func_name="main")
  l119, l120, l121 = sch.get_loops(block=b118)
  b122 = sch.decompose_reduction(block=b118, loop=l121)
