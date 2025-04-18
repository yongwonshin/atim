from functools import lru_cache
import os
import numpy as np
import argparse
from glob import glob
from pathlib import Path

# long_int32_arr = np.fromfile("../data/int32_4194304.bin", dtype="int32").reshape((4194304,))


@lru_cache(maxsize=4)
def host_array(dim, dtype, intdist=50, index=0, new=False, ones=False):
    if isinstance(dim, int):
        dim = (dim,)

    if ones:
        return np.ones(dim, dtype=dtype)

    if len(dim) == 1 and dim[0] <= 403200000 and dtype == "int64":
        long_int64_arr = np.fromfile("../data/int64_403200000.bin", dtype="int64").reshape(
            (403200000,)
        )
        return long_int64_arr[: dim[0]]
    if np.prod(dim) <= 163840 * 4096 and dtype == "int32":
        return (
            np.fromfile("../data/int32_163840_4096.bin", dtype="int32")
            .reshape((163840 * 4096))[: np.prod(dim)]
            .reshape(dim)
        )

    dimjoin = "_".join(map(str, dim))
    index_suffix = f"_v{index}" if index > 0 else ""
    fname = f"../data/{dtype}_{dimjoin}{index_suffix}.bin"

    if not os.path.exists("../data"):
        os.makedirs("../data")

    if os.path.exists(fname) and intdist == 50 and not new:
        return np.fromfile(fname, dtype=dtype).reshape(dim)
    if dtype[:5] == "float":
        return np.random.rand(*dim).astype(dtype)
    else:
        return np.random.randint(0, intdist, dim).astype(dtype)


def save_array(dim, dtype, intdist=50, unique=False):
    dimjoin = "_".join(map(str, dim))
    fname = f"../data/{dtype}_{dimjoin}"

    if unique:
        max_version = 0
        for f in glob(fname + "*"):
            version = Path(f).stem.split("_")[-1]
            if version.startswith("v"):
                max_version = max(max_version, int(version[1:]))
        fname += f"_v{max_version + 1}"
    fname += ".bin"

    host_array(dim, dtype, intdist).tofile(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim", type=str, required=True, help="Dimensions of the array. example: 8192,8192"
    )
    parser.add_argument(
        "--dtype", type=str, default="int32", help="Data type of the array (default: int32)"
    )
    parser.add_argument("--intdist", type=int, default=50, help="Integer distribution range")

    parser.add_argument(
        "-unique",
        action="store_true",
        default=False,
        help="Make unique array if array with same configuration already exists",
    )

    args = parser.parse_args()
    dim = tuple(map(int, args.dim.split(",")))
    save_array(dim, args.dtype, args.intdist, args.unique)
