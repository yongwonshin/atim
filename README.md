# Prerequisites
## Hardware Dependencies
We recommend using UPMEM systems with at least 16 PIM modules (2048 DPUs) to ensure sufficient inter-DPU parallelism. Our experiments used DDR4-2400 PIM modules, which are currently the only available UPMEM PIM modules.

## Software Dependencies
We implemented and tested our codes on the Ubuntu 20.04 x86-64 system with UPMEM SDK version 2021.3.0.
We assume the server is properly equipped with BIOS and MCU firmware, driver, backends (communication library), and DPU runtime provided by the UPMEM SDK.

### Using the Pre-built Docker Image
To set up the full environment quickly, run the following commands. After executing the installation script, proceed directly to the Tuning section:

```bash
docker run -it --privileged yongwonshin/atim:v0.1
./install.sh
cd atim
```
### Building the Docker Image Manually
If you prefer to build the Docker image yourself, use the Dockerfile provided:
```bash
docker build -t atim -f isca-artifact/Dockerfile isca-artifact
docker run -it --privileged atim
./install.sh
```

### Building ATiM Without Docker Image
Follow these steps to install the required dependencies and build ATiM from source.

#### 1. Install Required Ubuntu Packages

Update your package lists and install the necessary Ubuntu packages:

```bash
sudo apt-get update && \
sudo apt-get install -y --no-install-recommends \
    build-essential \
    lsb-release \
    software-properties-common \
    gnupg \
    ninja-build \
    wget \
    git
```

#### 2. Install LLVM

Download and execute the LLVM installation script:

```bash
wget --no-check-certificate https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 16 all
```

#### 3. Download ATiM

Clone the ATiM repository and initialize its submodules:

```bash
git clone -b artifact https://github.com/yongwonshin/atim.git
cd atim
git submodule update --init --recursive
```

#### 4. Install Additional Dependencies

Set up the Conda environment to install extra system and Python dependencies:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f isca-artifact/atim-env.yml
conda activate atim-venv
```

#### 5. Build ATiM from Source

With all dependencies in place, build ATiM:

```bash
rm -rf build
mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. -G Ninja
ninja
```

# Tuning
Before experiments, we need to perform tuning for CPU-autotuned, PrIM+Search, and ATiM for tensor programs.

```bash
# Step 0: prepare for autotuning
export PYTHONPATH="$(realpath .)/python:$PYTHONPATH"
conda activate atim-venv
cd evaluation
./eval_setup.sh

# Step 1: perform autotuning for CPU-autotune
python cpu_autotune.py

# Step 2: find optimal parameters for PrIM
python prim_autotune.py

# Step 3: find optimal parameters for PrIM+Search
python prim_search_autotune.py

# Step 4: find optimal parameter for SimplePIM
python simplepim_autotune.py

# Step 5: perform autotuning for ATiM
python atim_autotune.py
```

# Evaluation
After completing autotuning, we evaluate the execution times of tensor programs optimized with CPU-autotuned, PrIM/(E), PrIM+Search, SimplePIM, and ATiM.
We also evaluate different optimization levels of ATiM's PIM-aware strategies to demonstrate their effectiveness.

```bash
# Evaluate tuned binaries/modules for tensor programs
python cpu_eval.py # CPU-autotuned
python prim_eval.py # PrIM/(E) and PrIM+Search
python simplepim_eval.py # SimplePIM
python atim_eval.py # ATiM

# Evaluate ATiM's PIM-aware optimizations
python atim_branch_opt.py
```

Finally, generate graphs corresponding to Fig. 9, Fig. 10, and Fig. 12 in the paper:

```bash
cd graph
python plot.py
```

## Experiment customization
ATiM supports tuning tensor programs with various workloads and shapes. To test different workload sizes or to add new workloads, users can modify:

- `evaluation/bench.py`: Define new workloads in TIR.
- `evaluation/tasks.py`: Configure workload sizes.
- `evaluation/workloads.py`: Register workloads to run experiments.
