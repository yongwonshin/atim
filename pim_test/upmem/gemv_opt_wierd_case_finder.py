# 실행하는 코드
import numpy as np
import pandas as pd
from io import StringIO
import os, sys
sys.path.append("/root/dev/tvm/python")
os.chdir("/root/dev/tvm/pim_test/upmem")
from gemv import GEMV, gemvRCTile
import tvm

header = "IDX\tO0\tO1\tO2\tO3\tO4\tO5\tO6\tM\tK\tBM\tBK\tT\tR\tC\tD\tWT"
with open('/root/dev/tvm/pim_test/upmem/gemv_opt_experiment_1015_unformatted.txt', 'r') as file:
    content = file.read()
    raw_datas = content.split('\n\n')
    raw_datas[0] = raw_datas[0].split("\n")[1]
    raw_datas = "\n".join([header] + [data.rsplit('\n', 2)[0] for data in raw_datas])
    df = pd.read_csv(StringIO(raw_datas), sep='\t')
    df['M'] = df['M'].replace('.', method='ffill')
    df['K'] = df['K'].replace('.', method='ffill')
    df[['O2', 'O3', 'O4', 'O1']] = df[['O2', 'O3', 'O4', 'O1']].apply(pd.to_numeric, errors='coerce').fillna(np.inf)

def extract_cases(df, col1, col2):
    case1 = df[(df[col1] / df[col2] > 1.2) & (df[col1] / df[col2] < 2) & (df[col1] != np.inf) & (df[col2] != np.inf)]
    case2 = df[(df[col1] / df[col2] > 2) & (df[col1] != np.inf) & (df[col2] != np.inf)]
    return case1, case2
o12_slower, o12_wierd = extract_cases(df, 'O2', 'O1')
o23_slower, o23_wierd = extract_cases(df, 'O3', 'O2')
o34_slower, o34_wierd = extract_cases(df, 'O4', 'O3')
# (54, 53), (47, 75), (56, 80)

configs = [o12_slower, o23_slower, o34_slower, o12_wierd, o23_wierd, o34_wierd]
instance = GEMV(warmup=1, repeat=100)
instance.automatic_set_fname = False
for i, config in enumerate(configs):
    opt1 = i % 3 + 1
    opt2 = i % 3 + 2
    wierd = i >= 3
    wstr = "wierd" if wierd else "slower"
    prefix = f"o{opt1}{opt2}{wstr}"

    print(prefix)
    for j, row in config.iterrows():
        j = int(j)
        m, k, bm, bk, t, rt, cc, dtype = int(row['M']), int(row['K']), int(row['BM']), int(row['BK']), int(row['T']), int(row['R']), int(row['C']), row['D']
        with tvm.transform.PassContext(config={"tir.UpmemKernelOptimize": opt1}):
            instance.fname = f"{prefix}_{j:02d}_opt{opt1}_{m}_{k}_{bm}_{bk}_{cc}"
            opt1_result = instance.test(gemvRCTile, M=m, K=k, n_xb=bm, n_yb=bk, n_rt=t, n_rc=rt, n_cache=cc, dtype=dtype)
        with tvm.transform.PassContext(config={"tir.UpmemKernelOptimize": opt2}):
            instance.fname = f"{prefix}_{j:02d}_opt{opt1}_{m}_{k}_{bm}_{bk}_{cc}"
            opt2_result = instance.test(gemvRCTile, M=m, K=k, n_xb=bm, n_yb=bk, n_rt=t, n_rc=rt, n_cache=cc, dtype=dtype)
        try:
            opt2_result = float(opt2_result)
            opt1_result = float(opt1_result)
            ratio = opt2_result / opt1_result
            if ratio >= 2:
                color_code = '\033[94m'  # blue
            elif ratio >= 1.2:
                color_code = '\033[92m'  # green
            else:
                color_code = '\033[0m'   # black
            print(f'{color_code}{m}\t{k}\t{bm}\t{bk}\t{cc}\t{opt1_result:.2f}\t{opt2_result:.2f}\t{ratio:.2f}\033[0m')
        except ValueError:
            print(f'\033[91m{m}\t{k}\t{bm}\t{bk}\t{cc}\t{opt1_result}\t{opt2_result}\tERROR\033[0m')

