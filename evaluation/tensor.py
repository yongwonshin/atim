from functools import lru_cache
import os
import numpy as np

MAX_BYTES = 2 ** 29  # 512MB

@lru_cache(maxsize=4)
def host_array(dim, dtype, intdist=50, ones=False):
    if isinstance(dim, int):
        dim = (dim,)
    if ones:
        return np.ones(dim, dtype=dtype)

    maxdim = MAX_BYTES // np.dtype(dtype).itemsize
    if np.prod(dim) <= maxdim and dtype in ["int32", "int64"]:
        fname = f"./data/{dtype}_array.bin"
        if not os.path.exists(fname):
            np.random.randint(0, intdist, (maxdim,)).astype(dtype).tofile(fname)
        raw = np.fromfile(fname, dtype=dtype).reshape((maxdim))
        return raw[:np.prod(dim)].reshape(dim)

    if dtype[:5] == "float":
        return np.random.rand(*dim).astype(dtype)
    else:
        return np.random.randint(0, intdist, dim).astype(dtype)

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")
    np.random.randint(0, 50, MAX_BYTES // 4).astype("int32").tofile("./data/int32_array.bin")
    np.random.randint(0, 50, MAX_BYTES // 8).astype("int64").tofile("./data/int64_array.bin")