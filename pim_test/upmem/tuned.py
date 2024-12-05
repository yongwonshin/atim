import tvm
from tvm.script import tir as T
from bench import *


def va_1048576_1_1_Tuned(M, dtype="int64", **kwargs):
    sch = tvm.tir.Schedule(upmem_va_factory(M, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSSS")
    (l2,) = sch.get_loops(block=b0)
    v3, v4, v5, v6, v7 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1024, 8, 1, 128, 1],
    )
    l8, l9, l10, l11, l12 = sch.split(
        loop=l2, factors=[v3, v4, v5, v6, v7], preserve_unit_iters=True
    )
    sch.reorder(l8, l9, l10, l11, l12)
    sch.bind(loop=l8, thread_axis="blockIdx.x")
    sch.bind(loop=l9, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l8, ann_key="bank", ann_val=1)
    b13 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b13, loop=l10, preserve_unit_loops=True, index=-1)
    b14 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="local", consumer_blocks=[b0])
    sch.compute_at(block=b14, loop=l10, preserve_unit_loops=True, index=-1)
    v15 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b14, ann_key="meta_schedule.cooperative_fetch", ann_val=v15)
    b16 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="local", consumer_blocks=[b0])
    sch.compute_at(block=b16, loop=l10, preserve_unit_loops=True, index=-1)
    # v17 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    # sch.annotate(block_or_loop=b16, ann_key="meta_schedule.cooperative_fetch", ann_val=v17)
    # v18 = sch.sample_categorical(
    #     candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    # )
    # sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v18)
    # b19 = sch.get_block(name="root", func_name="main")
    # sch.annotate(block_or_loop=b19, ann_key="meta_schedule.optimization_level", ann_val=4)
    # sch.enter_postproc()
    # b20 = sch.get_block(name="root", func_name="main")
    # sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.unroll_implicit")
    # b21, b22, b23, b24 = sch.get_child_blocks(b20)
    # l25, l26, l27, l28 = sch.get_loops(block=b21)
    # sch.annotate(block_or_loop=l25, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    # sch.annotate(block_or_loop=l25, ann_key="pragma_unroll_explicit", ann_val=0)
    # l29, l30, l31, l32 = sch.get_loops(block=b22)
    # sch.annotate(block_or_loop=l29, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    # sch.annotate(block_or_loop=l29, ann_key="pragma_unroll_explicit", ann_val=0)
    # l33, l34, l35, l36, l37 = sch.get_loops(block=b23)
    # sch.annotate(block_or_loop=l33, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    # sch.annotate(block_or_loop=l33, ann_key="pragma_unroll_explicit", ann_val=0)
    # l38, l39, l40, l41 = sch.get_loops(block=b24)
    # sch.annotate(block_or_loop=l38, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    # sch.annotate(block_or_loop=l38, ann_key="pragma_unroll_explicit", ann_val=0)
    return sch


def red_524288_1_1_Tuned(M, dtype="int64", **kwargs):
    sch = tvm.tir.Schedule(upmem_red_factory(M, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    (l2,) = sch.get_loops(block=b0)
    v3, v4 = sch.sample_perfect_tile2(
        loop=l2, n=2, min_n_splits=2, max_n_splits=2048, decision=[1024, 512]
    )
    l5, l6 = sch.split(loop=l2, factors=[v3, v4], preserve_unit_iters=True)
    b7 = sch.rfactor(loop=l5, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b7, ann_key="meta_schedule.meta_schedule_rfactor_producer_block", ann_val=1
    )
    sch.annotate(
        block_or_loop=b0, ann_key="meta_schedule.meta_schedule_rfactor_consumer_block", ann_val=1
    )
    b8 = sch.get_block(name="C_rf", func_name="main")
    l9, l10 = sch.get_loops(block=b8)
    v11, v12 = sch.sample_perfect_tile2(
        loop=l10, n=2, min_n_splits=2, max_n_splits=24, decision=[16, 32]
    )
    l13, l14 = sch.split(loop=l10, factors=[v11, v12], preserve_unit_iters=True)
    b15 = sch.rfactor(loop=l13, factor_axis=0, mem_scope="shared")
    l16, l17, l18 = sch.get_loops(block=b15)
    sch.reverse_compute_at(block=b8, loop=l17, preserve_unit_loops=False, index=-1)
    sch.unannotate(block_or_loop=b8, ann_key="meta_schedule.meta_schedule_rfactor_producer_block")
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_cross_thread_reduction_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b15, ann_key="meta_schedule.meta_schedule_rfactor_producer_block", ann_val=1
    )
    b19 = sch.get_block(name="C_rf_rf", func_name="main")
    sch.reorder_block_iter_var(block=b19, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b19, ann_key="meta_schedule.tiling_structure", ann_val="SRRR")
    l20, l21, l22 = sch.get_loops(block=b19)
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l22, n=3, max_innermost_factor=256, min_innermost_factor=1, decision=[2, 2, 128]
    )
    l26, l27, l28 = sch.split(loop=l22, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.bind(loop=l20, thread_axis="blockIdx.x")
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b19, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l26, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(
        block=b19, read_buffer_index=0, storage_scope="local", consumer_blocks=[b19]
    )
    sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
    l31, l32, l33, l34, l35 = sch.get_loops(block=b19)
    b36 = sch.decompose_reduction(block=b19, loop=l33)
    v37 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v37)
    b38 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b38, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b39 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b39, ann_key="meta_schedule.unroll_implicit")
    b40, b41, b42, b43, b44, b45 = sch.get_child_blocks(b39)
    l46, l47 = sch.get_loops(block=b40)
    sch.annotate(block_or_loop=l46, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l46, ann_key="pragma_unroll_explicit", ann_val=0)
    l48, l49, l50, l51, l52 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l48, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l48, ann_key="pragma_unroll_explicit", ann_val=0)
    l53, l54, l55, l56, l57 = sch.get_loops(block=b42)
    sch.annotate(block_or_loop=l53, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l53, ann_key="pragma_unroll_explicit", ann_val=0)
    l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b43)
    sch.annotate(block_or_loop=l58, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l58, ann_key="pragma_unroll_explicit", ann_val=0)
    l64, l65 = sch.get_loops(block=b44)
    sch.annotate(block_or_loop=l64, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l64, ann_key="pragma_unroll_explicit", ann_val=0)
    (l66,) = sch.get_loops(block=b45)
    sch.annotate(block_or_loop=l66, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l66, ann_key="pragma_unroll_explicit", ann_val=0)
    b67 = sch.get_block(name="C", func_name="main")
    (l68,) = sch.get_loops(block=b67)
    b69 = sch.decompose_reduction(block=b67, loop=l68)
    return sch


# basic
def mtv_8192_1_8192_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[16, 512]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[128, 16, 2, 1, 2],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 32, 8],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l20, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l27, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b37)
    l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b38)
    l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b39)
    l68, l69, l70, l71, l72, l73 = sch.get_loops(block=b40)
    l74, l75 = sch.get_loops(block=b41)
    b76 = sch.get_block(name="C_rf", func_name="main")
    l77, l78, l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b76)
    b86 = sch.decompose_reduction(block=b76, loop=l81)
    b87 = sch.get_block(name="C", func_name="main")
    l88, l89 = sch.get_loops(block=b87)
    b90 = sch.decompose_reduction(block=b87, loop=l89)
    return sch


# higher dim
def mmtv_256_512_512_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 16, 2, 2, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 16],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


# gpu 6b
def mtv_12288_1_4096_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3 = sch.get_loops(block=b1)
    v4, v5, v6, v7, v8 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 1024, 3, 1, 2],
    )
    l9, l10, l11, l12, l13 = sch.split(
        loop=l2, factors=[v4, v5, v6, v7, v8], preserve_unit_iters=True
    )
    v14, v15, v16 = sch.sample_perfect_tile(
        loop=l3,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 32, 16],
    )
    l17, l18, l19 = sch.split(loop=l3, factors=[v14, v15, v16], preserve_unit_iters=True)
    sch.reorder(l9, l10, l11, l17, l12, l18, l13, l19)
    sch.bind(loop=l9, thread_axis="blockIdx.x")
    sch.bind(loop=l10, thread_axis="blockIdx.y")
    sch.reorder(l11)
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l9, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    b20 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b20, loop=l11, preserve_unit_loops=True, index=-1)
    b21 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b21, loop=l17, preserve_unit_loops=True, index=-1)
    v22 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b21, ann_key="meta_schedule.cooperative_fetch", ann_val=v22)
    b23 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b23, loop=l17, preserve_unit_loops=True, index=-1)
    v24 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b23, ann_key="meta_schedule.cooperative_fetch", ann_val=v24)
    v25 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v25)
    b26 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b26, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b27 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.unroll_implicit")
    b28, b29, b30, b31 = sch.get_child_blocks(b27)
    l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b28)
    l38, l39, l40, l41, l42 = sch.get_loops(block=b29)
    l43, l44, l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b30)
    l51, l52, l53, l54 = sch.get_loops(block=b31)
    b55 = sch.get_block(name="C", func_name="main")
    l56, l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b55)
    b64 = sch.decompose_reduction(block=b55, loop=l59)
    return sch


def mtv_4096_1_4096_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[32, 128]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 8, 1, 2, 4],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 4, 16],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l21, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l26, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l26, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48 = sch.get_loops(block=b37)
    l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b38)
    l55, l56, l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b39)
    l64, l65, l66, l67, l68, l69, l70, l71 = sch.get_loops(block=b40)
    l72, l73 = sch.get_loops(block=b41)
    b74 = sch.get_block(name="C_rf", func_name="main")
    l75, l76, l77, l78, l79, l80, l81, l82, l83 = sch.get_loops(block=b74)
    b84 = sch.decompose_reduction(block=b74, loop=l79)
    b85 = sch.get_block(name="C", func_name="main")
    l86, l87 = sch.get_loops(block=b85)
    b88 = sch.decompose_reduction(block=b85, loop=l87)
    return sch


def mtv_16384_1_4096_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 512]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 16, 2, 2, 1],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 8, 4],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l21, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l26, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l26, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48 = sch.get_loops(block=b37)
    l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b38)
    l55, l56, l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b39)
    l64, l65, l66, l67, l68, l69, l70, l71 = sch.get_loops(block=b40)
    l72, l73 = sch.get_loops(block=b41)
    b74 = sch.get_block(name="C_rf", func_name="main")
    l75, l76, l77, l78, l79, l80, l81, l82, l83 = sch.get_loops(block=b74)
    b84 = sch.decompose_reduction(block=b74, loop=l79)
    b85 = sch.get_block(name="C", func_name="main")
    l86, l87 = sch.get_loops(block=b85)
    b88 = sch.decompose_reduction(block=b85, loop=l87)
    return sch


def mtv_4096_1_16384_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[64, 256]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[32, 16, 1, 1, 8],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 2, 2],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l20, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l27, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b37)
    l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b38)
    l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b39)
    sch.annotate(block_or_loop=l59, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l59, ann_key="pragma_unroll_explicit", ann_val=0)
    l68, l69, l70, l71, l72, l73 = sch.get_loops(block=b40)
    l74, l75 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l74, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l74, ann_key="pragma_unroll_explicit", ann_val=0)
    b76 = sch.get_block(name="C_rf", func_name="main")
    l77, l78, l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b76)
    b86 = sch.decompose_reduction(block=b76, loop=l81)
    b87 = sch.get_block(name="C", func_name="main")
    l88, l89 = sch.get_loops(block=b87)
    b90 = sch.decompose_reduction(block=b87, loop=l89)
    return sch


# gpu 30b
def mtv_21504_1_7168_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[32, 224]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 21, 4, 4, 1],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[7, 2, 16],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l21, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l26, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l26, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48 = sch.get_loops(block=b37)
    l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b38)
    l55, l56, l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b39)
    l64, l65, l66, l67, l68, l69, l70, l71 = sch.get_loops(block=b40)
    l72, l73 = sch.get_loops(block=b41)
    b74 = sch.get_block(name="C_rf", func_name="main")
    l75, l76, l77, l78, l79, l80, l81, l82, l83 = sch.get_loops(block=b74)
    b84 = sch.decompose_reduction(block=b74, loop=l79)
    b85 = sch.get_block(name="C", func_name="main")
    l86, l87 = sch.get_loops(block=b85)
    b88 = sch.decompose_reduction(block=b85, loop=l87)
    return sch


def mtv_7168_1_7168_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[16, 448]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[112, 16, 2, 2, 1],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 1, 56],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l20, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l27, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b37)
    l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b38)
    l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b39)
    sch.annotate(block_or_loop=l59, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l59, ann_key="pragma_unroll_explicit", ann_val=0)
    l68, l69, l70, l71, l72, l73 = sch.get_loops(block=b40)
    l74, l75 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l74, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l74, ann_key="pragma_unroll_explicit", ann_val=0)
    b76 = sch.get_block(name="C_rf", func_name="main")
    l77, l78, l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b76)
    b86 = sch.decompose_reduction(block=b76, loop=l81)
    b87 = sch.get_block(name="C", func_name="main")
    l88, l89 = sch.get_loops(block=b87)
    b90 = sch.decompose_reduction(block=b87, loop=l89)
    return sch


def mtv_28672_1_7168_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[32, 224]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[56, 16, 1, 16, 2],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 7, 16],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l21, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l27, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b37)
    l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b38)
    l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b39)
    l68, l69, l70, l71, l72, l73, l74, l75 = sch.get_loops(block=b40)
    l76, l77 = sch.get_loops(block=b41)
    b78 = sch.get_block(name="C_rf", func_name="main")
    l79, l80, l81, l82, l83, l84, l85, l86, l87 = sch.get_loops(block=b78)
    b88 = sch.decompose_reduction(block=b78, loop=l83)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91 = sch.get_loops(block=b89)
    b92 = sch.decompose_reduction(block=b89, loop=l91)
    return sch


def mtv_7168_1_28672_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mtv_factory(M, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3 = sch.get_loops(block=b0)
    v4, v5 = sch.sample_perfect_tile2(
        loop=l3, n=2, min_n_splits=2, max_n_splits=256, decision=[32, 896]
    )
    l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)
    b8 = sch.rfactor(loop=l6, factor_axis=0, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b8,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b9 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b9, new_order=[1, 0, 2])
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l10, l11, l12 = sch.get_loops(block=b9)
    v13, v14, v15, v16, v17 = sch.sample_perfect_tile(
        loop=l10,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 14, 4, 1, 2],
    )
    l18, l19, l20, l21, l22 = sch.split(
        loop=l10, factors=[v13, v14, v15, v16, v17], preserve_unit_iters=True
    )
    v23, v24, v25 = sch.sample_perfect_tile(
        loop=l12,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 64, 14],
    )
    l26, l27, l28 = sch.split(loop=l12, factors=[v23, v24, v25], preserve_unit_iters=True)
    sch.reorder(l11, l18, l19, l20, l26, l21, l27, l22, l28)
    sch.bind(loop=l11, thread_axis="blockIdx.x")
    sch.bind(loop=l18, thread_axis="blockIdx.y")
    sch.reorder(l19)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l11, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l18, ann_key="bank", ann_val=1)
    b29 = sch.cache_write(block=b9, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b29, loop=l20, preserve_unit_loops=True, index=-1)
    b30 = sch.cache_read(block=b9, read_buffer_index=0, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b30, loop=l27, preserve_unit_loops=True, index=-1)
    v31 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b30, ann_key="meta_schedule.cooperative_fetch", ann_val=v31)
    b32 = sch.cache_read(block=b9, read_buffer_index=1, storage_scope="local", consumer_blocks=[b9])
    sch.compute_at(block=b32, loop=l27, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    v34 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v34)
    b35 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b36 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b36, ann_key="meta_schedule.unroll_implicit")
    b37, b38, b39, b40, b41 = sch.get_child_blocks(b36)
    l42, l43, l44, l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b37)
    l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b38)
    l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b39)
    l68, l69, l70, l71, l72, l73 = sch.get_loops(block=b40)
    l74, l75 = sch.get_loops(block=b41)
    b76 = sch.get_block(name="C_rf", func_name="main")
    l77, l78, l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b76)
    b86 = sch.decompose_reduction(block=b76, loop=l81)
    b87 = sch.get_block(name="C", func_name="main")
    l88, l89 = sch.get_loops(block=b87)
    b90 = sch.decompose_reduction(block=b87, loop=l89)
    return sch


# gpt 6b
def mmtv_16_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 32]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 1, 2],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 4, 4],
    )
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
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l39, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l39, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    (
        l54,
        l55,
        l56,
        l57,
        l58,
        l59,
        l60,
        l61,
        l62,
        l63,
        l64,
        l65,
        l66,
        l67,
    ) = sch.get_loops(block=b49)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b50)
    (
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
        l89,
        l90,
        l91,
        l92,
        l93,
        l94,
    ) = sch.get_loops(block=b51)
    l95, l96, l97, l98, l99, l100, l101, l102, l103, l104 = sch.get_loops(block=b52)
    l105, l106, l107 = sch.get_loops(block=b53)
    b108 = sch.get_block(name="C_rf", func_name="main")
    (
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
        l120,
        l121,
        l122,
    ) = sch.get_loops(block=b108)
    b123 = sch.decompose_reduction(block=b108, loop=l116)
    b124 = sch.get_block(name="C", func_name="main")
    l125, l126, l127 = sch.get_loops(block=b124)
    b128 = sch.decompose_reduction(block=b124, loop=l127)
    return sch


def mmtv_16_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 32]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 2, 2],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 2, 16],
    )
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
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l39, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l39, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    (
        l54,
        l55,
        l56,
        l57,
        l58,
        l59,
        l60,
        l61,
        l62,
        l63,
        l64,
        l65,
        l66,
        l67,
    ) = sch.get_loops(block=b49)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b50)
    (
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
        l89,
        l90,
        l91,
        l92,
        l93,
        l94,
    ) = sch.get_loops(block=b51)
    l95, l96, l97, l98, l99, l100, l101, l102, l103, l104 = sch.get_loops(block=b52)
    l105, l106, l107 = sch.get_loops(block=b53)
    b108 = sch.get_block(name="C_rf", func_name="main")
    (
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
        l120,
        l121,
        l122,
    ) = sch.get_loops(block=b108)
    b123 = sch.decompose_reduction(block=b108, loop=l116)
    b124 = sch.get_block(name="C", func_name="main")
    l125, l126, l127 = sch.get_loops(block=b124)
    b128 = sch.decompose_reduction(block=b124, loop=l127)
    return sch


def mmtv_16_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[16, 16]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 1, 4, 1],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 2, 4],
    )
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
    sch.reverse_compute_at(block=b41, loop=l33, preserve_unit_loops=True, index=-1)
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l38, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l38, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b49)
    l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b50)
    (
        l75,
        l76,
        l77,
        l78,
        l79,
        l80,
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
    ) = sch.get_loops(block=b51)
    sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=0)
    l89, l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b52)
    l102, l103, l104 = sch.get_loops(block=b53)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=0)
    b105 = sch.get_block(name="C_rf", func_name="main")
    (
        l106,
        l107,
        l108,
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
    ) = sch.get_loops(block=b105)
    b120 = sch.decompose_reduction(block=b105, loop=l113)
    b121 = sch.get_block(name="C", func_name="main")
    l122, l123, l124 = sch.get_loops(block=b121)
    b125 = sch.decompose_reduction(block=b121, loop=l124)
    return sch


def mmtv_16_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[2, 128]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 1, 2, 1, 256],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 2, 8],
    )
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
    sch.reverse_compute_at(block=b41, loop=l33, preserve_unit_loops=True, index=-1)
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l38, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l38, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b49)
    l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b50)
    (
        l75,
        l76,
        l77,
        l78,
        l79,
        l80,
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
    ) = sch.get_loops(block=b51)
    sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=0)
    l89, l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b52)
    l102, l103, l104 = sch.get_loops(block=b53)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=0)
    b105 = sch.get_block(name="C_rf", func_name="main")
    (
        l106,
        l107,
        l108,
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
    ) = sch.get_loops(block=b105)
    b120 = sch.decompose_reduction(block=b105, loop=l113)
    b121 = sch.get_block(name="C", func_name="main")
    l122, l123, l124 = sch.get_loops(block=b121)
    b125 = sch.decompose_reduction(block=b121, loop=l124)
    return sch


def mmtv_64_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 8, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 4, 8],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l28, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b39)
    l53, l54, l55, l56, l57, l58, l59, l60, l61 = sch.get_loops(block=b40)
    l62, l63, l64, l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l62, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l62, ann_key="pragma_unroll_explicit", ann_val=0)
    l75, l76, l77, l78, l79, l80, l81, l82 = sch.get_loops(block=b42)
    b83 = sch.get_block(name="C", func_name="main")
    l84, l85, l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b83)
    b97 = sch.decompose_reduction(block=b83, loop=l90)
    return sch


def mmtv_64_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[4, 64]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 2, 2],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 1, 8],
    )
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
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l38, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l38, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b49)
    l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b50)
    (
        l75,
        l76,
        l77,
        l78,
        l79,
        l80,
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
    ) = sch.get_loops(block=b51)
    l89, l90, l91, l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b52)
    l99, l100, l101 = sch.get_loops(block=b53)
    b102 = sch.get_block(name="C_rf", func_name="main")
    (
        l103,
        l104,
        l105,
        l106,
        l107,
        l108,
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
    ) = sch.get_loops(block=b102)
    b117 = sch.decompose_reduction(block=b102, loop=l110)
    b118 = sch.get_block(name="C", func_name="main")
    l119, l120, l121 = sch.get_loops(block=b118)
    b122 = sch.decompose_reduction(block=b118, loop=l121)
    return sch


def mmtv_64_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(
        block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR"
    )
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 16, 1, 2, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 32, 2],
    )
    l28, l29, l30 = sch.split(
        loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True
    )
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(
        block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1]
    )
    sch.compute_at(block=b32, loop=l28, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(
        block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33
    )
    b34 = sch.cache_read(
        block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1]
    )
    sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(
        block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35
    )
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(
        block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4
    )
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b39)
    sch.annotate(block_or_loop=l43, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l43, ann_key="pragma_unroll_explicit", ann_val=0)
    l53, l54, l55, l56, l57, l58, l59, l60, l61 = sch.get_loops(block=b40)
    sch.annotate(block_or_loop=l53, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l53, ann_key="pragma_unroll_explicit", ann_val=0)
    l62, l63, l64, l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(
        block=b41
    )
    sch.annotate(block_or_loop=l62, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l62, ann_key="pragma_unroll_explicit", ann_val=0)
    l75, l76, l77, l78, l79, l80, l81, l82 = sch.get_loops(block=b42)
    sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=0)
    b83 = sch.get_block(name="C", func_name="main")
    l84, l85, l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(
        block=b83
    )
    b97 = sch.decompose_reduction(block=b83, loop=l90)
    return sch


def mmtv_64_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[4, 64]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 1, 8, 1],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 1, 8],
    )
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
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l39, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l39, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    (
        l54,
        l55,
        l56,
        l57,
        l58,
        l59,
        l60,
        l61,
        l62,
        l63,
        l64,
        l65,
        l66,
        l67,
    ) = sch.get_loops(block=b49)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b50)
    (
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
        l89,
        l90,
        l91,
        l92,
        l93,
        l94,
    ) = sch.get_loops(block=b51)
    l95, l96, l97, l98, l99, l100, l101, l102, l103, l104 = sch.get_loops(block=b52)
    l105, l106, l107 = sch.get_loops(block=b53)
    b108 = sch.get_block(name="C_rf", func_name="main")
    (
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
        l120,
        l121,
        l122,
    ) = sch.get_loops(block=b108)
    b123 = sch.decompose_reduction(block=b108, loop=l116)
    b124 = sch.get_block(name="C", func_name="main")
    l125, l126, l127 = sch.get_loops(block=b124)
    b128 = sch.decompose_reduction(block=b124, loop=l127)
    return sch


def mmtv_256_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 16],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l23, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88, l89, l90, l91 = sch.get_loops(block=b42)
    b92 = sch.get_block(name="C", func_name="main")
    (
        l93,
        l94,
        l95,
        l96,
        l97,
        l98,
        l99,
        l100,
        l101,
        l102,
        l103,
        l104,
        l105,
    ) = sch.get_loops(block=b92)
    b106 = sch.decompose_reduction(block=b92, loop=l99)
    return sch


def mmtv_256_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 1, 2, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 1, 32],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l68, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l68, ann_key="pragma_unroll_explicit", ann_val=0)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_256_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[32, 1, 8],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_256_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 16, 1, 1, 4],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[32, 2, 4],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l68, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l68, ann_key="pragma_unroll_explicit", ann_val=0)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


# gpt 30b
def mmtv_28_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    return sch


def mmtv_28_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    return sch


def mmtv_28_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    return sch


def mmtv_28_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[256, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 16, 1, 1, 4],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[32, 2, 4],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l68, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l68, ann_key="pragma_unroll_explicit", ann_val=0)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_28_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[4, 64]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[28, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 1, 2],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 4, 16],
    )
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
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l39, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l39, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    (
        l54,
        l55,
        l56,
        l57,
        l58,
        l59,
        l60,
        l61,
        l62,
        l63,
        l64,
        l65,
        l66,
        l67,
    ) = sch.get_loops(block=b49)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b50)
    (
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
        l89,
        l90,
        l91,
        l92,
        l93,
        l94,
    ) = sch.get_loops(block=b51)
    l95, l96, l97, l98, l99, l100, l101, l102, l103, l104 = sch.get_loops(block=b52)
    l105, l106, l107 = sch.get_loops(block=b53)
    b108 = sch.get_block(name="C_rf", func_name="main")
    (
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
        l120,
        l121,
        l122,
    ) = sch.get_loops(block=b108)
    b123 = sch.decompose_reduction(block=b108, loop=l116)
    b124 = sch.get_block(name="C", func_name="main")
    l125, l126, l127 = sch.get_loops(block=b124)
    b128 = sch.decompose_reduction(block=b124, loop=l127)
    return sch


def mmtv_28_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 32]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[28, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 2, 1, 2],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 2, 4],
    )
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
    sch.reverse_compute_at(block=b41, loop=l33, preserve_unit_loops=True, index=-1)
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l38, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l38, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b49)
    l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b50)
    (
        l75,
        l76,
        l77,
        l78,
        l79,
        l80,
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
    ) = sch.get_loops(block=b51)
    l89, l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b52)
    l102, l103, l104 = sch.get_loops(block=b53)
    b105 = sch.get_block(name="C_rf", func_name="main")
    (
        l106,
        l107,
        l108,
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
    ) = sch.get_loops(block=b105)
    b120 = sch.decompose_reduction(block=b105, loop=l113)
    b121 = sch.get_block(name="C", func_name="main")
    l122, l123, l124 = sch.get_loops(block=b121)
    b125 = sch.decompose_reduction(block=b121, loop=l124)
    return sch


def mmtv_28_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 32]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[28, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 8, 1],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 8, 4],
    )
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
    sch.reverse_compute_at(block=b41, loop=l33, preserve_unit_loops=True, index=-1)
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l38, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l38, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b49)
    l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b50)
    (
        l75,
        l76,
        l77,
        l78,
        l79,
        l80,
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
    ) = sch.get_loops(block=b51)
    sch.annotate(block_or_loop=l75, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l75, ann_key="pragma_unroll_explicit", ann_val=0)
    l89, l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b52)
    l102, l103, l104 = sch.get_loops(block=b53)
    sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=0)
    b105 = sch.get_block(name="C_rf", func_name="main")
    (
        l106,
        l107,
        l108,
        l109,
        l110,
        l111,
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
    ) = sch.get_loops(block=b105)
    b120 = sch.decompose_reduction(block=b105, loop=l113)
    b121 = sch.get_block(name="C", func_name="main")
    l122, l123, l124 = sch.get_loops(block=b121)
    b125 = sch.decompose_reduction(block=b121, loop=l124)
    return sch


def mmtv_28_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6 = sch.sample_perfect_tile2(
        loop=l4, n=2, min_n_splits=2, max_n_splits=256, decision=[8, 32]
    )
    l7, l8 = sch.split(loop=l4, factors=[v5, v6], preserve_unit_iters=True)
    b9 = sch.rfactor(loop=l7, factor_axis=1, mem_scope="global")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
    sch.annotate(
        block_or_loop=b9,
        ann_key="meta_schedule.meta_schedule_rfactor_producer_block",
        ann_val=1,
    )
    sch.annotate(
        block_or_loop=b0,
        ann_key="meta_schedule.meta_schedule_rfactor_consumer_block",
        ann_val=1,
    )
    b10 = sch.get_block(name="C_rf", func_name="main")
    sch.reorder_block_iter_var(block=b10, new_order=[1, 2, 0, 3])
    sch.annotate(block_or_loop=b10, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l11, l12, l13, l14 = sch.get_loops(block=b10)
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l11,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[28, 1, 1, 1, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l11, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27, v28, v29 = sch.sample_perfect_tile(
        loop=l12,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 4, 1, 2],
    )
    l30, l31, l32, l33, l34 = sch.split(
        loop=l12, factors=[v25, v26, v27, v28, v29], preserve_unit_iters=True
    )
    v35, v36, v37 = sch.sample_perfect_tile(
        loop=l14,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 1, 8],
    )
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
    sch.reverse_compute_at(block=b41, loop=l33, preserve_unit_loops=True, index=-1)
    b42 = sch.cache_read(
        block=b10, read_buffer_index=0, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b42, loop=l39, preserve_unit_loops=True, index=-1)
    v43 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b42, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(
        block=b10, read_buffer_index=1, storage_scope="local", consumer_blocks=[b10]
    )
    sch.compute_at(block=b44, loop=l39, preserve_unit_loops=True, index=-1)
    v45 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
    v46 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v46)
    b47 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b47, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b48 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b48, ann_key="meta_schedule.unroll_implicit")
    b49, b50, b51, b52, b53 = sch.get_child_blocks(b48)
    (
        l54,
        l55,
        l56,
        l57,
        l58,
        l59,
        l60,
        l61,
        l62,
        l63,
        l64,
        l65,
        l66,
        l67,
    ) = sch.get_loops(block=b49)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b50)
    (
        l81,
        l82,
        l83,
        l84,
        l85,
        l86,
        l87,
        l88,
        l89,
        l90,
        l91,
        l92,
        l93,
        l94,
    ) = sch.get_loops(block=b51)
    (
        l95,
        l96,
        l97,
        l98,
        l99,
        l100,
        l101,
        l102,
        l103,
        l104,
        l105,
        l106,
        l107,
    ) = sch.get_loops(block=b52)
    l108, l109, l110 = sch.get_loops(block=b53)
    b111 = sch.get_block(name="C_rf", func_name="main")
    (
        l112,
        l113,
        l114,
        l115,
        l116,
        l117,
        l118,
        l119,
        l120,
        l121,
        l122,
        l123,
        l124,
        l125,
    ) = sch.get_loops(block=b111)
    b126 = sch.decompose_reduction(block=b111, loop=l119)
    b127 = sch.get_block(name="C", func_name="main")
    l128, l129, l130 = sch.get_loops(block=b127)
    b131 = sch.decompose_reduction(block=b127, loop=l130)
    return sch


def mmtv_112_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[112, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 16],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_112_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[112, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 1, 16],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_112_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[112, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 8, 8],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l68, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l68, ann_key="pragma_unroll_explicit", ann_val=0)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_112_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[112, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[16, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[32, 4, 2],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l28, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b39)
    l53, l54, l55, l56, l57, l58, l59, l60, l61 = sch.get_loops(block=b40)
    l62, l63, l64, l65, l66, l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b41)
    l75, l76, l77, l78, l79, l80, l81, l82 = sch.get_loops(block=b42)
    b83 = sch.get_block(name="C", func_name="main")
    l84, l85, l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b83)
    b97 = sch.decompose_reduction(block=b83, loop=l90)
    return sch


def mmtv_448_64_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[448, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[2, 16, 1, 2, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 64, 4],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_448_128_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[448, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 1, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[8, 4, 8],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_448_256_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[448, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 2, 1, 2],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[64, 2, 2],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l22, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    l81, l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b42)
    b89 = sch.get_block(name="C", func_name="main")
    l90, l91, l92, l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b89)
    b103 = sch.decompose_reduction(block=b89, loop=l96)
    return sch


def mmtv_448_512_256_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_mmtv_factory(M, N, K, dtype))
    b0 = sch.get_block(name="root", func_name="main")
    b1 = sch.get_block(name="C", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRSRSR")
    l2, l3, l4 = sch.get_loops(block=b1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[448, 1, 1, 1, 1],
    )
    l10, l11, l12, l13, l14 = sch.split(
        loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
    )
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
        loop=l3,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[4, 16, 1, 8, 1],
    )
    l20, l21, l22, l23, l24 = sch.split(
        loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
    )
    v25, v26, v27 = sch.sample_perfect_tile(
        loop=l4,
        n=3,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1, 2, 128],
    )
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l13, l23, l29, l14, l24, l30)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l20, thread_axis="blockIdx.y")
    sch.reorder(l21, l11)
    sch.bind(loop=l21, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l10, ann_key="bank", ann_val=1)
    sch.annotate(block_or_loop=l20, ann_key="bank", ann_val=1)
    b31 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b31, loop=l23, preserve_unit_loops=True, index=-1)
    b32 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b32, loop=l29, preserve_unit_loops=True, index=-1)
    v33 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b32, ann_key="meta_schedule.cooperative_fetch", ann_val=v33)
    b34 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local", consumer_blocks=[b1])
    sch.compute_at(block=b34, loop=l29, preserve_unit_loops=True, index=-1)
    v35 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v35)
    v36 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_implicit", ann_val=v36)
    b37 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b37, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b38 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b38, ann_key="meta_schedule.unroll_implicit")
    b39, b40, b41, b42 = sch.get_child_blocks(b38)
    l43, l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b39)
    l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b40)
    l68, l69, l70, l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b41)
    sch.annotate(block_or_loop=l68, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l68, ann_key="pragma_unroll_explicit", ann_val=0)
    l81, l82, l83, l84, l85, l86, l87, l88, l89, l90, l91 = sch.get_loops(block=b42)
    b92 = sch.get_block(name="C", func_name="main")
    (
        l93,
        l94,
        l95,
        l96,
        l97,
        l98,
        l99,
        l100,
        l101,
        l102,
        l103,
        l104,
        l105,
    ) = sch.get_loops(block=b92)
    b106 = sch.decompose_reduction(block=b92, loop=l99)
    return sch


# basic
def va_67108864_1_1_Tuned(M, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_va_factory(M, dtype))
    return sch


def red_33554432_1_1_Tuned(M, dtype="int64", **kwargs):
    sch = tvm.tir.Schedule(upmem_red_factory(M, dtype))
    return sch


# def dot_33554432_1_1_Tuned(M, dtype="int32", **kwargs):
#     sch = tvm.tir.Schedule(upmem_dot_factory(M, dtype))
#     return sch


# higher dim
def ta_256_512_512_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_ta_factory(M, N, K, dtype))
    return sch


def ttv_256_512_512_Tuned(M, N, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_ttv_factory(M, N, K, dtype))
    return sch


# poly
def poly_gemv1_8192_1_8192_Tuned(M, K, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_poly_gemv1_factory(M, K, dtype))
    return sch


def poly_va_67108864_1_1_Tuned(M, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_poly_va_factory(M, dtype))
    return sch


def poly_va_1048576_1_1_Tuned(M, dtype="int32", **kwargs):
    sch = tvm.tir.Schedule(upmem_poly_va_factory(M, dtype))
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSSS")
    (l2,) = sch.get_loops(block=b0)
    v3, v4, v5, v6, v7 = sch.sample_perfect_tile(
        loop=l2,
        n=5,
        max_innermost_factor=256,
        min_innermost_factor=1,
        decision=[1024, 16, 1, 32, 2],
    )
    l8, l9, l10, l11, l12 = sch.split(
        loop=l2, factors=[v3, v4, v5, v6, v7], preserve_unit_iters=True
    )
    sch.reorder(l8, l9, l10, l11, l12)
    sch.bind(loop=l8, thread_axis="blockIdx.x")
    sch.bind(loop=l9, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=l8, ann_key="bank", ann_val=1)
    b13 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b13, loop=l10, preserve_unit_loops=True, index=-1)
    b14 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="local", consumer_blocks=[b0])
    sch.compute_at(block=b14, loop=l10, preserve_unit_loops=True, index=-1)
    v15 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b14, ann_key="meta_schedule.cooperative_fetch", ann_val=v15)
    b16 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="local", consumer_blocks=[b0])
    sch.compute_at(block=b16, loop=l10, preserve_unit_loops=True, index=-1)
    v17 = sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)
    sch.annotate(block_or_loop=b16, ann_key="meta_schedule.cooperative_fetch", ann_val=v17)
    v18 = sch.sample_categorical(
        candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=2
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_implicit", ann_val=v18)
    b19 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b19, ann_key="meta_schedule.optimization_level", ann_val=4)
    sch.enter_postproc()
    b20 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b20, ann_key="meta_schedule.unroll_implicit")
    b21, b22, b23, b24 = sch.get_child_blocks(b20)
    l25, l26, l27, l28 = sch.get_loops(block=b21)
    sch.annotate(block_or_loop=l25, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l25, ann_key="pragma_unroll_explicit", ann_val=0)
    l29, l30, l31, l32 = sch.get_loops(block=b22)
    sch.annotate(block_or_loop=l29, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l29, ann_key="pragma_unroll_explicit", ann_val=0)
    l33, l34, l35, l36, l37 = sch.get_loops(block=b23)
    sch.annotate(block_or_loop=l33, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l33, ann_key="pragma_unroll_explicit", ann_val=0)
    l38, l39, l40, l41 = sch.get_loops(block=b24)
    sch.annotate(block_or_loop=l38, ann_key="pragma_auto_unroll_max_step", ann_val=64)
    sch.annotate(block_or_loop=l38, ann_key="pragma_unroll_explicit", ann_val=0)
    return sch
