import tvm
import tvm.testing
from tvm import te
import numpy as np

M = 2048
K = 2048

N_XB = 4
N_YB = 4
N_YT = 16
N_CACHE = 64

dtype = "int32"
target = tvm.target.Target(target="upmem", host="llvm")

k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), dtype, "A")
B = te.placeholder((K,), dtype, "B")
C = te.compute((M,), lambda y: te.sum(A[y, k] * B[k], axis=k), name="C")

s = te.create_schedule(C.op)

xb, _ = s[C].split(k, nparts = N_XB)
CF = s.rfactor(C, xb)

AL = s.cache_read(A, "local", [CF])
BL = s.cache_read(B, "local", [CF])
CL = s.cache_write(CF, "local")

xb, yb = s[CF].op.axis
yb, yo = s[CF].split(yb, nparts=N_YB)
yo, yi = s[CF].split(yo, nparts=N_YT)
yi, yc = s[CF].split(yi, 2) # 8 / sizeof(dtype)
s[CF].reorder(yo, yi, yc)

s[CF].bind(yb, te.thread_axis("blockIdx.x"))
s[CF].bind(xb, te.thread_axis("blockIdx.y"))
s[CF].bind(yo, te.thread_axis("threadIdx.x"))

s[CL].compute_at(s[CF], yi)
xo, xi = s[CL].split(s[CL].op.reduce_axis[0], N_CACHE)
s[CL].reorder(s[CL].op.axis[0], xo, xi)
s[AL].compute_at(s[CL], xo)
s[BL].compute_at(s[CL], xo)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
# tvm.lower(s, [A, B, C], simple_mode=True)

dev = tvm.device(target.kind.name, 0)
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
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