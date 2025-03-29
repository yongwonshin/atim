from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_cpu_red_8388608_1_1:
    @T.prim_func
    def main(A: T.Buffer((8388608,), "int64"), B: T.Buffer((1,), "int64")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A"], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        B_rf = T.alloc_buffer((131072, 1), "int64")
        for i_0_fused in T.parallel(131072):
            with T.block("C_rf_init"):
                vi_0 = T.axis.spatial(131072, i_0_fused)
                T.reads()
                T.writes(B_rf[vi_0, 0])
                B_rf[vi_0, 0] = T.int64(0)
            for i_1 in range(64):
                with T.block("C_rf_update"):
                    vi_0, vi_1 = T.axis.remap("SR", [i_0_fused, i_1])
                    T.reads(B_rf[vi_0, 0], A[vi_0 * 64 + vi_1])
                    T.writes(B_rf[vi_0, 0])
                    B_rf[vi_0, 0] = B_rf[vi_0, 0] + A[vi_0 * 64 + vi_1]
        with T.block("C_init"):
            T.reads()
            T.writes(B[0])
            T.block_attr({"meta_schedule.random_compute_producer": T.int64(1)})
            B[0] = T.int64(0)
        for i_0 in range(131072):
            with T.block("C_update"):
                vi_0 = T.axis.reduce(131072, i_0)
                T.reads(B[0], B_rf[vi_0, 0])
                T.writes(B[0])
                T.block_attr({"meta_schedule.random_compute_producer": T.int64(1)})
                B[0] = B[0] + B_rf[vi_0, 0]
# from tvm import tir
def apply_trace_cpu_red_8388608_1_1(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  l2, = sch.get_loops(block=b0)
  v3, v4 = sch.sample_perfect_tile(loop=l2, n=2, max_innermost_factor=64, decision=[131072, 64])
  l5, l6 = sch.split(loop=l2, factors=[v3, v4], preserve_unit_iters=True)
  b7 = sch.rfactor(loop=l5, factor_axis=0)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=1536)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
  v8 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v8)
  sch.enter_postproc()
  b9 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b9, ann_key="meta_schedule.parallel")
  sch.unannotate(block_or_loop=b9, ann_key="meta_schedule.vectorize")
  sch.unannotate(block_or_loop=b9, ann_key="meta_schedule.unroll_explicit")
  b10, b11 = sch.get_child_blocks(b9)
  l12, l13 = sch.get_loops(block=b10)
  l14 = sch.fuse(l12, preserve_unit_iters=True)
  sch.parallel(loop=l14)
  l15, = sch.get_loops(block=b11)
  b16 = sch.get_block(name="C_rf", func_name="main")
  l17, l18 = sch.get_loops(block=b16)
  b19 = sch.decompose_reduction(block=b16, loop=l18)
  b20 = sch.get_block(name="C", func_name="main")
  l21, = sch.get_loops(block=b20)
  b22 = sch.decompose_reduction(block=b20, loop=l21)
