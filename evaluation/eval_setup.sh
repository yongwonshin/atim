#!/bin/bash
python3 tensor.py

cd tvm_cputest
rm -rf build
mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. -G Ninja
ninja