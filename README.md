# m3: Precise Estimation of Flow-Level Performance via Machine Learning

This GitHub repository houses the scripts and guidance needed to replicate the experiments presented in our paper, "m3: Precise Estimation of Flow-Level Performance via Machine Learning". It offers all necessary tools to reproduce the experimental results documented in sections 5.2, 5.3, and 5.4 of our study.

## Contents

- [Setup Instructions](#setup-instructions)
- [Repository Structure](#repository-structure)
- [Citation Information](#citation-information)
- [Acknowledgments](#acknowledgments)
- [Getting in Touch](#getting-in-touch)

## Setup Instructions

Before you begin, ensure you have installed: Python 2 and 3, Rust, Cargo, gcc (gcc version 9.4.0), and gcc-5. Use the `environment_py27.yml` and `environment_py39.yml` conda environment files for Python setup, and follow additional instructions for other packages.

```bash
conda env create -f environment_py27.yml
conda env create -f environment_py39.yml
```

1. To install m3, execute: 
```bash
git clone https://github.com/netiken/m3.git
cd m3
```

2. To build the C libraries for m3:
```bash     
cd clibs
make run
cd ..
```

3. To initialize the submodules, including parsimon, parsimon-eval, and HPCC:

```bash
git submodule update --init --recursive
```

4. For setting up the HPCC repository for data generation, follow the detailed instructions in `parsimon/backends/High-Precision-Congestion-Control/simulation/README.md`:

```bash
cd parsimon/backends/High-Precision-Congestion-Control
cd simulation
CC='gcc-5' CXX='g++-5' ./waf configure --build-profile=optimized
```

4. To generate data for m3, adjust generation parameters in `gen_path/src/main.rs` (lines 17-28) and data parameters in `simulation/consts.py`. Then, run:

```bash
cd gen_path
cargo run --release
```

5. For training the model, ensure you're using the Python 3 environment and configure settings in `config/train_config_path.yaml`. Then execute:

```bash
cd m3
python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m3
```

6. To create checkpoints for the end-to-end m3 pipeline:
```bash
cd m3
python gen_ckpt.py
```

7. To replicate paper results, run the following in the `parsimon-eval/expts/fig_8 directory`:

```bash
# section 5.2
cargo run --release -- --root=./data --mixes spec/all_config_2.mix.json ns3-config
cargo run --release -- --root=./data --mixes spec/all_config_2.mix.json pmn-m
cargo run --release -- --root=./data --mixes spec/all_config_2.mix.json mlsys
# section 5.4
cargo run --release -- --root=./data --mixes spec/all_config_1.mix.json ns3-config
cargo run --release -- --root=./data --mixes spec/all_config_1.mix.json mlsys
```

# Repository Structure

```bash
├── ckpts          # Checkpoints of Llama-2 and 2-layer MLP used in m3
├── clibs          # C libraries for running the path-level simulation in m3
├── config         # Configuration files for training and testing m3
├── parsimon       # Core functionalities of Parsimon and m3
│   └── backends
│       └── High-Precision-Congestion-Control   # HPCC repository for data generation
├── parsimon-eval  # Scripts to reproduce m3 experiments and comparisons
├── util           # Utility functions for m3, including data loader and ML model implementations
├── gen_ckpts.py   # Script to generate checkpoints for m3
└── main_path.py   # Main script for training and testing m3
```

# Citation Information
If our work assists in your research, kindly cite our paper as follows:
```bibtex
@inproceedings{m3,
    author = {Li, Chenning and Nasr-Esfahany, Arash and Zhao, Kevin and Noorbakhsh, Kimia and Goyal, Prateesh and Alizadeh, Mohammad and Anderson, Thomas},
    title = {m3: Accurate Flow-Level Performance Estimation using Machine Learning},
    year = {2024},
}
```

# Acknowledgments

Special thanks to Kevin Zhao and Thomas Anderson for their insights shared in the NSDI'23 paper [Scalable Tail Latency Estimation for Data Center Networks](https://www.usenix.org/conference/nsdi23/presentation/zhao-kevin). The source codes can be found in [Parsimon](https://github.com/netiken/parsimon).

# Getting in Touch
For further inquiries, reach out to Chenning Li at lichenni@mit.edu

