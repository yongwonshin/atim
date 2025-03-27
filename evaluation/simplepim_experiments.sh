#!/bin/bash

pushd baseline/simplepim/benchmarks

search_va() {
    local L=$1
    cd va
    for dpus in 512 1024 1536 2048; do
        rm -rf ./bin
        NR_DPUS=$dpus L=$L make > /dev/null 2> /dev/null
        ./bin/host
    done
    cd ..
}

search_reduction() {
    local L=$1
    cd reduction
    for dpus in 512 1024 1536 2048; do
        rm -rf ./bin
        NR_DPUS=$dpus L=$L make > /dev/null 2> /dev/null
        ./bin/host
    done
    cd ..
}

# Example usage:
search_va 1048576
search_red 524288

search_va 16777216
search_red 8388608

search_va 67108864
search_red 34554432

search_red 67108864

popd