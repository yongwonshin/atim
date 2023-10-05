import tvm
import tvm.testing
from tvm import te
import numpy as np

M = 8192
K = 8192
bn = 32

dtype = "float32"
target = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(target.kind.name, 0)

k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), dtype, "A")
B = te.placeholder((K,), dtype, "B")
C = te.compute((M,), lambda y: te.sum(A[y, k] * B[k], axis=k), name="C")

s = te.create_schedule(C.op)

AS = s.cache_read(A, "shared", [C])
BS = s.cache_read(B, "shared", [C])
AL = s.cache_read(AS, "local", [C])
BL = s.cache_read(BS, "local", [C])
xr, _ = s[C].split(C.op.reduce_axis[0], nparts=4)
CF = s.rfactor(C, xr)
CS = s.cache_write(CF, "shared")
CL = s.cache_write(CS, "local")

xb, yb = s[CF].op.axis
yb, yo = s[CF].split(yb, nparts=4)
yo, yi = s[CF].split(yo, nparts=16)
yi, yc = s[CF].split(yi, 2)
b = s[CF].fuse(xb, yb)
s[CF].reorder(b, yo, yi, yc)

s[CF].bind(b, te.thread_axis("blockIdx.x"))
s[CF].bind(yo, te.thread_axis("threadIdx.x"))

s[CL].compute_at(s[CF], yi)
s[CS].compute_at(s[CF], b)
xo, xi = s[CL].split(s[CL].op.reduce_axis[0], 64)
s[CL].reorder(s[CL].op.axis[0], xo, xi)
s[AL].compute_at(s[CL], xo)
s[BL].compute_at(s[CL], xo)
s[AS].compute_at(s[CF], b)
s[BS].compute_at(s[CF], b)

# func = tvm.build(s, [A, B, C], target=target, name="mmult")
# tvm.lower(s, [A, B, C], simple_mode=True)
print(tvm.lower(s, [A, B, C], simple_mode=True))


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