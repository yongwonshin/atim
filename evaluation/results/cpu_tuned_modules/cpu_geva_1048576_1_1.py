from tvm.script import ir as I
from tvm.script import tir as T
from tvm import tir
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class module_cpu_geva_1048576_1_1:
    @T.prim_func
    def main(A: T.Buffer((1048576,), "int32"), B: T.Buffer((1048576,), "int32"), C: T.Buffer((1048576,), "int32"), ALPHA: T.Buffer((1,), "int32"), BETA: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main", "pragma_explicit_h2d": ["A", "B"], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        alpha_val: T.int32 = ALPHA[0]
        beta_val: T.int32 = BETA[0]
        for i_fused_0 in T.parallel(16384):
            for i_fused_1 in T.vectorized(64):
                with T.block("C"):
                    v_i = T.axis.spatial(1048576, i_fused_0 * 64 + i_fused_1)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = alpha_val * A[v_i] + beta_val * B[v_i]
# from tvm import tir
def apply_trace_cpu_geva_1048576_1_1(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.parallel", ann_val=1536)
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.vectorize", ann_val=64)
  sch.enter_postproc()
  b1 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b1, ann_key="meta_schedule.parallel")
  sch.unannotate(block_or_loop=b1, ann_key="meta_schedule.vectorize")
  b2, = sch.get_child_blocks(b1)
  l3, = sch.get_loops(block=b2)
  l4 = sch.fuse(l3, preserve_unit_iters=True)
  l5, l6 = sch.split(loop=l4, factors=[None, 64], preserve_unit_iters=True)
  sch.parallel(loop=l5)
  sch.vectorize(loop=l6)
