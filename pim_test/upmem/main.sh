#!/bin/bash
function run_prim_va {
  python main_result.py --op_type=$1 --schedule="vaTile" --M=$2 --N=$3 --K=$4 --warmup=10 --repeat=1000 &> "$1_$2_$3_$4.txt"
}

function run_prim_mtv {
  # python main_result.py --op_type=$1 --schedule="$1RCTile" --M=$2 --N=$3 --K=$4 --n_cache=64 --warmup=10 --repeat=1000 &> "$1RCTile_$2_$3_$4.txt"
  python main_result.py --op_type=$1 --schedule="$1RTile" --M=$2 --N=$3 --K=$4 --warmup=10 --repeat=1000 &> "$1RTile_$2_$3_$4.txt"
}

function run_prim_mmtv {
  # python main_result.py --op_type=$1 --schedule="$1RCTile" --M=$2 --N=$3 --K=$4 --n_cache=64 --warmup=10 --repeat=100 &> "$1RCTile_$2_$3_$4.txt"
  python main_result.py --op_type=$1 --schedule="$1RTile" --M=$2 --N=$3 --K=$4 --warmup=10 --repeat=100 &> "$1RTile_$2_$3_$4.txt"
}

function run_tuned {
  python main_result.py --op_type=$1 --schedule="$1_$2_$3_$4_Tuned" --M=$2 --N=$3 --K=$4 --warmup=10 --repeat=100 &> "isca_$1_Tuned_$2_$3_$4.txt"
}
function run_tuned_dtype {
  python main_result.py --op_type=$1 --schedule="$1_$2_$3_$4_Tuned" --M=$2 --N=$3 --K=$4 --dtype=$5 --warmup=10 --repeat=100 &> "isca_$1_Tuned_$2_$3_$4.txt"
}

### tuned
# # run_tuned va 67108864 1 1
# # run_tuned_dtype red 33554432 1 1 int64
# run_tuned mtv 8192 1 8192
# # run_tuned ta 256 512 512 # 64 * 1M
# # run_tuned ttv 256 512 512 # 64 * 1M
# run_tuned mmtv 256 512 512
# # run_tuned poly_va 67108864 1 1 # 64 * 1M
# # run_tuned poly_gemv1 8192 1 8192 # 64 * 1M

# # gpt 6b
run_tuned mtv 12288 1 4096
# run_tuned mtv 4096 1 4096
# run_tuned mtv 16384 1 4096
# run_tuned mtv 4096 1 16384
# # gpt 30b
# run_tuned mtv 21504 1 7168
# run_tuned mtv 7168 1 7168
# run_tuned mtv 28672 1 7168
# run_tuned mtv 7168 1 28672

# # gpt 6b
# run_tuned mmtv 16 64 256
# run_tuned mmtv 16 128 256
run_tuned mmtv 16 256 256
run_tuned mmtv 16 512 256
# run_tuned mmtv 64 64 256
# run_tuned mmtv 64 128 256
# run_tuned mmtv 64 256 256
# run_tuned mmtv 64 512 256
# run_tuned mmtv 256 64 256
run_tuned mmtv 256 128 256
# run_tuned mmtv 256 256 256
run_tuned mmtv 256 512 256
# # gpt 30b
# run_tuned mmtv 28 64 256
# run_tuned mmtv 28 128 256
# run_tuned mmtv 28 256 256
# run_tuned mmtv 28 512 256
# run_tuned mmtv 112 64 256
# run_tuned mmtv 112 128 256
# run_tuned mmtv 112 256 256
run_tuned mmtv 112 512 256
run_tuned mmtv 448 64 256
run_tuned mmtv 448 128 256
run_tuned mmtv 448 256 256
# run_tuned mmtv 448 512 256


# basic
# run_prim_va va 67108864 1 1 # 64 * 1M
# run_prim dot 33554432 1 1 # 32 * 1M
# run_prim_mtv mtv 8192 1 8192 # 64 * 1M

# poly
# run_prim polyva 67108864 1 1 # 64 * 1M
# run_prim polygemv1 8192 1 8192 # 64 * 1M

# higher dim
# run_prim ta 256 512 512 # 64 * 1M
# run_prim innerprod 256 256 512 # 32 * 1M
# run_prim ttv 256 256 1024 # 64 * 1M

# mmtv
# run_prim_mmtv mmtv 256 512 512 # 64 * 1M

# gpt-j 6B: fc
# run_prim_mtv mtv 12288 1 4096
# run_prim_mtv mtv 4096 1 4096
# run_prim_mtv mtv 16384 1 4096
# run_prim_mtv mtv 4096 1 16384

# gpt-j 6B: mha
# run_prim mmtv 16 64 256 # batch: 1
# run_prim mmtv 16 128 256 # batch: 1
# run_prim mmtv 16 256 256 # batch: 1
# run_prim mmtv 16 512 256 # batch: 1
# run_prim_mmtv mmtv 64 64 256
# run_prim_mmtv mmtv 64 128 256
# run_prim_mmtv mmtv 64 256 256
# run_prim_mmtv mmtv 64 512 256
# run_prim mmtv 256 64 256
# run_prim mmtv 256 128 256
# run_prim mmtv 256 256 256
# run_prim mmtv 256 512 256

# gpt-j 30B: fc
# run_prim mtv 21504 1 7168
# run_prim mtv 7168 1 7168
# run_prim mtv 28672 1 7168
# run_prim mtv 7168 1 28672

# gpt-j 30B: mha
# run_prim mmtv 28 64 256 # batch: 1
# run_prim mmtv 28 128 256 # batch: 1
# run_prim mmtv 28 256 256 # batch: 1
# run_prim mmtv 28 512 256 # batch: 1
# run_prim mmtv 112 64 256
# run_prim mmtv 112 128 256
# run_prim mmtv 112 256 256
# run_prim mmtv 112 512 256
# run_prim mmtv 448 64 256
# run_prim mmtv 448 128 256
# run_prim mmtv 448 256 256
# run_prim mmtv 448 512 256
