from batched_gemv import BGEMV, bgemvTile
from gemv import GEMV, gemvRCTile
import numpy as np
import math

def sens_mtv():
    shape_expand = [55, 72, 91, 123, 145, 164, 196, 212, 245, 269, 290]
    shapes = [(200, m) for m in shape_expand] + [(m, 200) for m in shape_expand]
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

def sens_mmtv():
    confs = [
        {
            "N": b,
            "M": m,
            "K": k,
            "dtype": dtype,
            "n_bb": nb,
            "n_xb": 1,
            "n_yb": yb,
            "n_cache": nc,
            "n_yt": nt,
            "n_rt": 16,
        }
        for b, m, k, nb, yb, nt, nc, dtype in [
            (64, 26, 256, 64, 1, 8, 64, "int32"),
            (64, 69, 256, 64, 2, 16, 16, "int32"),
            (64, 109, 256, 64, 4, 8, 16, "int32"),
            (64, 126, 256, 64, 4, 8, 64, "int32"),
            (64, 153, 256, 64, 4, 16, 64, "int32"),
            (64, 180, 256, 64, 4, 16, 16, "int32"),
            (64, 232, 256, 64, 4, 16, 128, "int32"),
            (64, 255, 256, 64, 8, 8, 128, "int32"),

            # (64, 272, 256, 64, 16, 8, 128, "int32"),
            # (64, 321, 256, 64, 8, 16, 16, "int32"),
            # (64, 345, 256, 64, 16, 8, 16, "int32"),
            # (64, 380, 256, 64, 16, 8, 16, "int32"),
            # (64, 399, 256, 64, 8, 16, 32, "int32"),
            # (64, 429, 256, 64, 16, 8, 16, "int32"),
            # (64, 473, 256, 64, 16, 8, 128, "int32"),
            # (64, 501, 256, 64, 16, 8, 64, "int32"),

            (64, 26, 26, 64, 1, 4, 64, "int32"),
            (64, 69, 69, 64, 4, 8, 32, "int32"),
            (64, 109, 109, 64, 4, 8, 32, "int32"),
            (64, 126, 126, 64, 4, 8, 32, "int32"),
            (64, 153, 153, 64, 4, 16, 128, "int32"),
            (64, 180, 180, 64, 4, 16, 16, "int32"),
            (64, 232, 232, 64, 4, 16, 32, "int32"),
            (64, 255, 255, 64, 4, 16, 128, "int32"),

            # (64, 272, 272, 64, 8, 16, 32, "int32"),
            # (64, 321, 321, 64, 8, 16, 16, "int32"),
            # (64, 345, 345, 64, 8, 16, 128, "int32"),
            # (64, 380, 380, 64, 8, 16, 16, "int32"),
            # (64, 399, 399, 64, 8, 16, 64, "int32"),
            # (64, 429, 429, 64, 8, 16, 128, "int32"),
            # (64, 473, 473, 64, 8, 16, 16, "int32"),
            # (64, 501, 501, 64, 8, 16, 64, "int32"),
        ]
    ]

    bgemv = BGEMV(warmup=10, repeat=1000, verbose=-1, bench=True)

    for conf in confs:
        bm = bgemv.benchmark(**conf)
        tl = [bm[1]]
        for opt in [0,1,2,4]:
            bgemv.opt_level = opt
            tt = bgemv.test(bgemvTile, **conf)
            tl.append(tt)
        print("\t".join([str(t) for t in tl]))

sens_mtv()
sens_mmtv()