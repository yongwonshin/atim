import os
import math
import subprocess
from tqdm import tqdm
import re
from gemv import GEMV
os.chdir("/root/dev/tvm/pim_test/upmem/baseline/GEMV_SAMSUNG/")

def run(m, n, dpus, ntile, bl=None, test=False):
    bl = bl if bl else min(6, int(math.log2(m / ntile) + 2))
    tasklet_mult = n * ntile / 2 / dpus
    tasklets = 1
    for i in range(16, 1, -1):
        if tasklet_mult % i == 0:
            tasklets = i
            break
    warmup, repeat = 10, 100 if not test else 0, 1
    os.system(f"make clean >/dev/null 2>/dev/null")
    os.system(f"NR_DPUS={dpus} NR_TASKLETS={tasklets} TYPE=INT32 BL={bl} make {'>/dev/null 2> /dev/null' if not test else ''} ")
    result = subprocess.check_output(f"""./bin/gemv_host -m {m} -n {n} -w {warmup} -e {repeat} -N {ntile}""", shell=True,)
    result = result.decode("utf-8")
    pattern = r"CPU-DPU Time \(ms\): (\d+\.\d+).*?DPU Kernel Time \(ms\): (\d+\.\d+).*?DPU-CPU Time \(ms\): (\d+\.\d+).*?Reduction Time \(ms\): (\d+\.\d+)"
    matches = re.findall(pattern, result)
    if matches:
        cpu_dpu_time, dpu_kernel_time, dpu_cpu_time, reduction_time = map(float, matches[-1])
        total_time = cpu_dpu_time + dpu_kernel_time + dpu_cpu_time + reduction_time
        tqdm.write(f"Total time: {total_time}\t{cpu_dpu_time}\t{dpu_kernel_time}\t{dpu_cpu_time}\t{reduction_time}\t{m}\t{n}\t{dpus}\t{ntile}\t{bl}\t{tasklets}")
    else:
        tqdm.write("No matches found")
        print()

workload = GEMV(warmup=10, repeat=1000, bench=True)

#3.3.1. 1DPU일때 캐시사이즈에 따라 실행 시간이 달라질 수 있음
for cache in [4, 8, 16, 32, 64, 128, 256, 512]:
    bl = math.log2(cache) + 2
    run(512, 512, 1, 1, bl)

"""
0.113929	17.820093	0.1293
0.115682	13.053944	0.132568
0.111821	12.464743	0.132134
0.106422	15.170619	0.128368
0.115852	15.169728	0.132131
0.116973	15.46373	0.13516
0.104026	16.184401	0.131696
"""

#3.3.2. DPU 타일링에 따라 실행 시간이 달라질 수 있음
for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    run(8192, 8192, 2048, i)
# 1 CPU Version Time (ms): 334.803000	CPU-DPU Time (ms): 6.459645	DPU Kernel Time (ms): 10.438159	DPU-CPU Time (ms): 0.389471	CPU-DPU Time (ms): 0.140654	DPU Kernel Time (ms): 10.421061	DPU-CPU Time (ms): 0.383729	Reduction Time (ms): 0.000034
# 2 CPU Version Time (ms): 333.225000	CPU-DPU Time (ms): 3.381316	DPU Kernel Time (ms): 5.404234	DPU-CPU Time (ms): 0.394269	CPU-DPU Time (ms): 0.138816	DPU Kernel Time (ms): 5.380288	DPU-CPU Time (ms): 0.388789	Reduction Time (ms): 0.065707
# 4 CPU Version Time (ms): 328.700000	CPU-DPU Time (ms): 1.816842	DPU Kernel Time (ms): 2.893623	DPU-CPU Time (ms): 0.406813	CPU-DPU Time (ms): 0.141462	DPU Kernel Time (ms): 2.870578	DPU-CPU Time (ms): 0.393418	Reduction Time (ms): 0.103639
# 8 CPU Version Time (ms): 330.786000	CPU-DPU Time (ms): 1.048500	DPU Kernel Time (ms): 2.172499	DPU-CPU Time (ms): 0.453161	CPU-DPU Time (ms): 0.139757	DPU Kernel Time (ms): 2.161759	DPU-CPU Time (ms): 0.433805	Reduction Time (ms): 0.160893
# 16 CPU Version Time (ms): 332.347000	CPU-DPU Time (ms): 0.683338	DPU Kernel Time (ms): 2.177447	DPU-CPU Time (ms): 0.554938	CPU-DPU Time (ms): 0.147612	DPU Kernel Time (ms): 2.156627	DPU-CPU Time (ms): 0.480926	Reduction Time (ms): 0.415086
# 32 CPU Version Time (ms): 346.363000	CPU-DPU Time (ms): 0.685231	DPU Kernel Time (ms): 2.178242	DPU-CPU Time (ms): 0.541694	CPU-DPU Time (ms): 0.144965	DPU Kernel Time (ms): 2.158502	DPU-CPU Time (ms): 0.476333	Reduction Time (ms): 0.410767
# 64 CPU Version Time (ms): 329.639000	CPU-DPU Time (ms): 0.680335	DPU Kernel Time (ms): 2.187026	DPU-CPU Time (ms): 0.739175	CPU-DPU Time (ms): 0.142261	DPU Kernel Time (ms): 2.187717	DPU-CPU Time (ms): 0.594475	Reduction Time (ms): 1.014166
# 128 CPU Version Time (ms): 385.686000	CPU-DPU Time (ms): 0.680941	DPU Kernel Time (ms): 2.190483	DPU-CPU Time (ms): 0.975069	CPU-DPU Time (ms): 0.158228	DPU Kernel Time (ms): 2.189066	DPU-CPU Time (ms): 0.686786	Reduction Time (ms): 2.102138
# 256 CPU Version Time (ms): 333.196000	CPU-DPU Time (ms): 0.485364	DPU Kernel Time (ms): 2.192969	DPU-CPU Time (ms): 1.051692	CPU-DPU Time (ms): 0.146999	DPU Kernel Time (ms): 2.217737	DPU-CPU Time (ms): 0.720530	Reduction Time (ms): 4.257458
# 512 CPU Version Time (ms): 331.109000	CPU-DPU Time (ms): 0.356226	DPU Kernel Time (ms): 2.418479	DPU-CPU Time (ms): 2.621765	CPU-DPU Time (ms): 0.147384	DPU Kernel Time (ms): 2.452687	DPU-CPU Time (ms): 1.813583	Reduction Time (ms): 18.715486
# 1024 CPU Version Time (ms): 332.056000	CPU-DPU Time (ms): 0.386426	DPU Kernel Time (ms): 2.725021	DPU-CPU Time (ms): 4.638853	CPU-DPU Time (ms): 0.162803	DPU Kernel Time (ms): 2.734353	DPU-CPU Time (ms): 3.182686	Reduction Time (ms): 39.063133
# 2048 CPU Version Time (ms): 332.056000	CPU-DPU Time (ms): 0.386426	DPU Kernel Time (ms): 2.725021	DPU-CPU Time (ms): 4.638853	CPU-DPU Time (ms): 0.162803	DPU Kernel Time (ms): 2.734353	DPU-CPU Time (ms): 3.182686	Reduction Time (ms): 39.063133
# #3.3.3. 텐서가 작을 경우  dpu를 많이 쓰는게 안좋을 수도 있음


# 3.3.3.
run(512, 512, 2048, 8, False)
run(512, 512, 2048, 16, False)
run(512, 512, 2048, 32, False)
run(512, 512, 2048, 64, False)
run(512, 512, 2048, 128, False)
run(512, 512, 2048, 256, False)

run(512, 512, 1024, 4, False)
run(512, 512, 1024, 8, False)
run(512, 512, 1024, 16, False)
run(512, 512, 1024, 32, False)
run(512, 512, 1024, 64, False)
run(512, 512, 1024, 128, False)
run(512, 512, 1024, 256, False)


run(512, 512, 512, 2, False)
run(512, 512, 512, 4, False)
run(512, 512, 512, 8, False)
run(512, 512, 512, 16, False)
run(512, 512, 512, 32, False)
run(512, 512, 512, 64, False)
run(512, 512, 512, 128, False)
run(512, 512, 512, 256, False)

run(512, 512, 256, 1, False)
run(512, 512, 256, 2, False)
run(512, 512, 256, 4, False)
run(512, 512, 256, 8, False)
run(512, 512, 256, 16, False)
run(512, 512, 256, 32, False)
run(512, 512, 256, 64, False)
run(512, 512, 256, 128, False)
run(512, 512, 256, 256, False)


run(512, 512, 128, 1, False)
run(512, 512, 128, 2, False)
run(512, 512, 128, 4, False)
run(512, 512, 128, 8, False)
run(512, 512, 128, 16, False)
run(512, 512, 128, 32, False)
run(512, 512, 128, 64, False)
run(512, 512, 128, 128, False)

with tqdm(total=5*9) as pbar:
    for B in [128]:
        for T in [64, 32, 16, 8, 4, 2, 1]:
            tqdm.write(f"{B}, {T}")
            run(8192, 8192, B, T, False)
            pbar.update(1)

"""
512 512 2048 8 6 1
Total time: 0.8667790000000001 ms 0.141809 0.359523 0.353068 0.012379

512 512 2048 16 6 2
Total time: 0.8572909999999999 ms 0.143483 0.335575 0.354382 0.023851

512 512 2048 32 6 4
Total time: 0.886261 ms 0.143038 0.322144 0.367772 0.053307

512 512 2048 64 5 8
Total time: 0.9419249999999999 ms 0.13989 0.320015 0.376955 0.105065

512 512 2048 128 4 16
Total time: 1.121979 ms 0.153801 0.325968 0.422262 0.219948

512 512 2048 2
56 3 16
Total time: 1.343384 ms 0.141733 0.332299 0.462492 0.40686

512 512 1024 16 6 4
Total time: 0.584782 ms 0.07504 0.233172 0.258361 0.018209

512 512 1024 32 6 8
Total time: 0.618626 ms 0.076784 0.227868 0.269807 0.044167

512 512 1024 64 5 16
Total time: 0.713384 ms 0.085316 0.232473 0.301413 0.094182

512 512 1024 128 4 16
Total time: 0.860126 ms 0.079738 0.239081 0.35129 0.190017

512 512 1024 256 3 16
Total time: 1.1797360000000001 ms 0.078077 0.246852 0.456888 0.397919

512 512 512 16 6 8
Total time: 0.462046 ms 0.041 0.180247 0.221031 0.019768

512 512 512 32 6 16
Total time: 0.529514 ms 0.042268 0.191658 0.248851 0.046737

512 512 512 64 5 16
Total time: 0.6239430000000001 ms 0.040895 0.190427 0.295479 0.097142

512 512 512 128 4 16
Total time: 0.832319 ms 0.041395 0.201118 0.392398 0.197408

512 512 512 256 3 16
Total time: 1.035897 ms 0.041163 0.217427 0.388243 0.389064

512 512 256 32 6 16
Total time: 0.481736 ms 0.027629 0.214077 0.188964 0.051066

512 512 256 64 5 16
Total time: 0.590359 ms 0.027259 0.219244 0.238871 0.104985

512 512 256 128 4 16
Total time: 0.679184 ms 0.026542 0.23935 0.238112 0.17518

512 512 256 256 3 16
Total time: 0.890779 ms 0.027587 0.273987 0.256338 0.332867

512 512 128 64 5 16
Total time: 0.583882 ms 0.015997 0.274046 0.212262 0.081577

512 512 128 128 4 16
Total time: 0.69877 ms 0.016024 0.31029 0.206896 0.16556
"""