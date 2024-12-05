#!/bin/bash
function run {
  python bench_autotune.py --op_type=$1 --M=$2 --N=$3 --K=$4 --workdir=isca_$1_$2_$3_$4_rev --reuse_cost_model &> isca_$1_$2_$3_$4.txt
}

function run_postfix {
  python bench_autotune.py --op_type=$1 --M=$2 --N=$3 --K=$4 --workdir=isca_$1_$2_$3_$4_rev_$5 &> isca_$1_$2_$3_$4_r$5.txt
}

function run_commented {
  python bench_autotune.py --op_type=$1 --M=$2 --N=$3 --K=$4 --workdir=isca_$5_$1_$2_$3_$4_$6_rev &> isca_$5_$1_$2_$3_$4_$6.txt
}

# run_postfix mtv 1024 1 1024 99

# run_postfix mmtv 256 512 512 99
run_postfix mmtv 256 512 512 199

# run mmtv 32 64 256
# run mmtv 32 128 256
# run mmtv 32 256 256
# run mmtv 32 512 256
# run_postfix mmtv 32 64 256 2
# run_postfix mmtv 32 128 256 2
# run_postfix mmtv 32 256 256 2
# run_postfix mmtv 32 512 256 2

# run mmtv 56 64 256
# run mmtv 56 128 256
# run mmtv 56 256 256
# run mmtv 56 512 256
# run_postfix mmtv 56 64 256 2
# run_postfix mmtv 56 128 256 2
# run_postfix mmtv 56 256 256 2
# run_postfix mmtv 56 512 256 2

# run va 8388608 1 1
# run_postfix va 8388608 1 1 2
# run red 4194304 1 1
# run mtv 2048 1 4096

# run ttv 64 64 2048
# run_postfix ttv 64 64 2048 2
# run mmtv 128 128 512
# run_postfix mmtv 128 128 512 2
# run polyva 8388608 1 1 2
# run_postfix polyva 8388608 1 1 2
# run polygemv1 2048 1 4096

# # basic
# # run_postfix va 67108864 1 1 2 # 64 * 1M
# run va 1048576 1 1 # 1M
# run_postfix va 1048576 1 1 2 # 1M
# # run dot 33554432 1 1 # 32 * 1M
# run_postfix red 33554432 1 1 2 # 32 * 1M
# run_postfix red 33554432 1 1 3 # 32 * 1M
# run_postfix red 524288 1 1 2 # 32 * 1M
# run_postfix red 524288 1 1 3 # 32 * 1M
# # run mtv 8192 1 8192 # 64 * 1M
# run mtv 1024 1 1024 # 1M

# # higher dim
# run ta 256 512 512 # 64 * 1M
# run_postfix ta 256 512 512 2 # 64 * 1M
# run ta 32 64 512 # 64 * 1M
# run_postfix ta 32 64 512 2 # 64 * 1M
# # run_postfix innerprod 256 256 512 2 # 32 * 1M
# # run ttv 256 512 512 # 64 * 1M
# run ttv 32 64 512 # 64 * 1M
# run_postfix ttv 64 128 8192 2 # 64 * 1M
# run_postfix ttv 64 128 8192 3 # 64 * 1M
# run ttv 64 128 8192 # 64 * 1M
# run ttv 16 32 2048 # 64 * 1M
# run_postfix ttv 32 64 512 2 # 64 * 1M

# # poly
# run_postfix polyva 67108864 1 1 2 # 64 * 1M
# run polyva 1048576 1 1 # 64 * 1M
# run_postfix polyva 1048576 1 1 2 # 64 * 1M
# run_postfix polygemv1 8192 1 8192 2 # 64 * 1M
# run polygemv1 1024 1 1024 # 64 * 1M
# run_postfix polygemv1 1024 1 1024 2 # 64 * 1M
# run_postfix polygemv1 1024 1 1024 3 # 64 * 1M
# run polymixed 8192 8192 1
# run_postfix polymixed 8192 8192 1 2
# run polymixed 1024 1024 1
# run_postfix polymixed 1024 1024 1 2

# mmtv
# run mmtv 256 512 512 # 64 * 1M
# run mmtv 32 64 512 # 64 * 1M
# run_postfix mmtv 32 64 512 2 # 64 * 1M
# run_postfix mmtv 256 512 512 2 # 64 * 1M
# run_postfix mmtv 256 512 512 3 # 64 * 1M

# gpt-j 6B: fc
# run_postfix mtv 12288 1 4096 2
# run mtv 4096 1 4096
# run mtv 16384 1 4096
# run mtv 4096 1 16384

# gpt-j 6B: mha
# run mmtv 16 64 256
# run mmtv 16 128 256
# run_postfix mmtv 16 256 256 2
# run_postfix mmtv 16 512 256 2
# run mmtv 64 64 256
# run mmtv 64 128 256
# run_postfix mmtv 64 256 256 2
# run mmtv 64 512 256
# run mmtv 256 64 256
# run_postfix mmtv 256 128 256 2
# run mmtv 256 256 256
# run_postfix mmtv 256 512 256 2

# gpt-j 30B: fc
# run mtv 21504 1 7168
# run mtv 7168 1 7168
# run mtv 28672 1 7168
# run mtv 7168 1 28672

# gpt-j 30B: mha
# run mmtv 28 64 256
# run mmtv 28 128 256
# run mmtv 28 256 256
# run mmtv 28 512 256
# run mmtv 112 64 256
# run mmtv 112 128 256
# run mmtv 112 256 256
# run_postfix mmtv 112 512 256 2
# run_postfix mmtv 448 64 256 2
# run_postfix mmtv 448 128 256 3
# run_postfix mmtv 448 256 256 2
# run mmtv 448 512 256

# unaligned
# run_commented mtv 763 1 763 unaligned "+opt"
# run_commented mtv 763 1 2007 unaligned "+opt"
# run_commented mtv 2007 1 763 unaligned "+opt"
# run_commented mtv 763 1 8114 unaligned "+opt"
# run_commented mtv 8114 1 763 unaligned "+opt"
# run_commented mtv 2007 1 2007 unaligned "+opt"
# run_commented mtv 2007 1 8114 unaligned "+opt"
# run_commented mtv 8114 1 2007 unaligned "+opt"
# run_commented mtv 8114 1 8114 unaligned "+opt"
# run_commented mtv 2007 1 2007 unaligned "-opt"
# run_commented mtv 2007 1 8114 unaligned "-opt"
# run_commented mtv 8114 1 2007 unaligned "-opt"
# run_commented mtv 8114 1 8114 unaligned "-opt"
