Prerequisites
-----
### Hardware Dependencies
We recommend using UPMEM systems with at least 16 PIM modules (2048 DPUs) to ensure sufficient inter-DPU parallelism. Our experiments used DDR4-2400 PIM modules, which are currently the only available UPMEM PIM modules.

### Software Dependencies
We implemented and tested our codes on the Ubuntu 20.04 x86-64 system with UPMEM SDK version 2021.3.0.
We assume the server is properly equipped with BIOS and MCU firmware, driver, backends (communication library), and DPU runtime provided by the UPMEM SDK.

The following commands install required Ubuntu packages.

```bash
sudo apt-get update &&
sudo apt-get install -y --no-install-recommends \
                    build-essential \
                    ninja-build \
                    wget \
                    git
```

Additionally, we assume conda environment to install additional system and python dependencies.

```bash
conda env create -f isca-artifact/atim-env.yml
conda activate atim-venv
export PYTHONPATH="$(realpath .)/python:$PYTHONPATH"
```

Lastly, users should install submodules unless recursively cloning this repository.

```bash
git submodule update --init --recursive
```

Tuning
-----
Before experiments, we need to perform tuning for CPU-autotuned, PrIM+Search, and ATiM for tensor programs.

```bash
# Step 0: prepare for evaluation
cd evaluation
python tensor.py

# Step 1: perform autotuning for CPU-autotune
python cpu_search.py

# Step 2: find optimal parameters for PrIM+Search
python prim_search.py

# Step 3: perform autotuning for ATiM
python atim_search.py
```

Evaluation
-----
After completing autotuning, we evaluate the execution times of tensor programs optimized with CPU-autotuned, PrIM/(E), PrIM+Search, SimplePIM, and ATiM.
We also evaluate different optimization levels of ATiM's PIM-aware strategies to demonstrate their effectiveness.

```bash
# Evaluate tuned binaries/modules for tensor programs
python cpu_eval.py
python prim_eval.py
python simplepim_eval.py
python atim_eval.py

# Evaluate ATiM's PIM-aware optimizations
python atim_branch_opt.py
\end{minted}
```

Finally, generate graphs corresponding to Fig. 9, Fig. 10, and Fig. 12 in the paper:

```bash
cd graph
python plot.py
```

Experiment customization
-----
ATiM supports tuning tensor programs with various workloads and shapes. To test different workload sizes or to add new workloads, users can modify:

- `evaluation/bench.py`: Define new workloads in TIR.
- `evaluation/tasks.py`: Configure workload sizes.
- `evaluation/workloads.py`: Register workloads to run experiments.
