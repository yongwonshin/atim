from batched_gemv import BGEMV, bgemvTile
from gemv import GEMV, gemvRCTile
from va import VA, vaTile
import numpy as np
import math

def sens_mtv():
    shape_expand = [72, 91, 123, 145, 164, 196, 212, 245]
    shapes = [(256, m) for m in shape_expand] + [(m, 256) for m in shape_expand] + [(m, m) for m in shape_expand]
    confs = [
        {'M': m, 'K': k, 'dtype': 'int32', 'n_xb': 1, 'n_yb': 1, 'n_cache': 32, 'n_yt': 16, 'n_rt': 16}
        for m, k in shapes
    ]

    gemv = GEMV(repeat=10, warmup=1000, verbose=-1)
    for c in confs:
        tt = [gemv.benchmark(**c)[1]]
        for opt in [0,1,2,4]:
            gemv.opt_level = opt
            tt.append(gemv.test(gemvRCTile, **c))
        print("\t".join([str(t) for t in tt]))

def sens_va():
    shape = [i * 100000 for i in [1, 2, 3, 4, 5, 6, 7, 8]]
    confs = [{"L": l, "n_b": 32, "n_t": 16, "n_c": 64} for l in shape]

    va = VA(repeat=10, warmup=1000, verbose=-1)
    for c in confs:
        print(c["L"])
        tt = [va.benchmark(**c)[1]]
        tt = []
        for opt in [0,1,2,4]:
            va.opt_level = opt
            tt.append(va.test(vaTile, **c))
        print("\t".join([str(t) for t in tt]))

sens_mtv()
sens_va()
