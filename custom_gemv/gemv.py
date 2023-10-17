import tvm
import tvm.testing
from tvm import te
import numpy as np

M = 2048
K = 2048
bn = 32

dtype = "int32"
target = tvm.target.Target(target="upmem", host="llvm")

k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), dtype, "A")
B = te.placeholder((K,), dtype, "B")
C = te.compute((M,), lambda y: te.sum(A[y, k] * B[k], axis=k), name="C")

s = te.create_schedule(C.op)

AL = s.cache_read(A, "local", [C])
BL = s.cache_read(B, "local", [C])
CL = s.cache_write(C, "local")

yb, = s[C].op.axis
yb, yo = s[C].split(yb, nparts=4)
yo, yi = s[C].split(yo, nparts=16)
yi, yc = s[C].split(yi, 2)
s[C].reorder(yo, yi, yc)

s[C].bind(yb, te.thread_axis("blockIdx.x"))
s[C].bind(yo, te.thread_axis("threadIdx.x"))

s[CL].compute_at(s[C], yi)
xo, xi = s[CL].split(s[CL].op.reduce_axis[0], 64)
s[CL].reorder(s[CL].op.axis[0], xo, xi)
s[AL].compute_at(s[CL], xo)
s[BL].compute_at(s[CL], xo)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
# tvm.lower(s, [A, B, C], simple_mode=True)

# dev = tvm.device(target.kind.name, 0)
# a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
# b = tvm.nd.array(np.random.rand(K,).astype(dtype), dev)
# c = tvm.nd.array(np.zeros((M,), dtype=dtype), dev)
# func(a, b, c)

# answer = np.dot(a.numpy(), b.numpy())
# tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
# evaluator = func.time_evaluator(func.entry_name, dev, number=10)
# print("%s: %f" % ("block caching", evaluator(a, b, c).mean))

# if target.kind.name == "cuda" or target.kind.name == "rocm" or target.kind.name.startswith("opencl"):
#     dev_module = func.imported_modules[0]
#     print("-----GPU code-----")
#     print(dev_module.get_source())

# print(func.get_source())