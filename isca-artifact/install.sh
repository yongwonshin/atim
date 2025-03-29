#!/bin/bash
set -e

# Install llvm
wget --no-check-certificate https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 16 all

# Download ATiM
git clone -b artifact https://github.com/yongwonshin/atim.git
cd atim
git submodule update --init --recursive

# Install other dependencies
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f isca-artifact/atim-env.yml
conda activate atim-venv

# Install ATiM from source
rm -rf build; mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. -G Ninja
ninja
