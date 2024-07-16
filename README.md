# m3: Accurate Flow-Level Performance Estimation using Machine Learning

This repository houses the scripts and guidance needed to replicate the experiments presented in our paper, "m3: Accurate Flow-Level Performance Estimation using Machine Learning". It offers all necessary tools to reproduce the experimental results documented in sections 5.2, 5.3, and 5.4 of our paper.

- [Quick Reproduction](#quick-reproduction)
- [From Scratch](#from-scratch)
- [Train your own model](#train-your-own-model)
- [Repository Structure](#repository-structure)
- [Citation Information](#citation-information)
- [Acknowledgments](#acknowledgments)
- [Getting in Touch](#getting-in-touch)

## Quick Reproduction
The following steps provide a quick guide to reproduce the results in the paper.

1. To replicate paper results in Section 5.2, run the script `parsimon-eval/expts/fig_8/analysis/analysis_dctcp.ipynb.ipynb`.

2. To replicate paper results in Section 5.3, run the script `parsimon-eval/expts/fig_7/analysis/analysis.ipynb`.

3. To replicate paper results in Section 5.4, run the script `parsimon-eval/expts/fig_8/analysis/analysis_counterfactual.ipynb`.

## From Scratch

Before you begin, ensure you have installed: Python 3, Rust, Cargo (nightly Version), gcc-9 (for compiling the flowSim and inference), and gcc-5 (for running ns-3). For Python setup, use the `environment.yml` conda environment file, and follow the additional instructions for installing the other packages.

```bash
conda env create -f environment.yml
```

1. To install m3, execute: 
```bash
git clone https://github.com/netiken/m3.git
cd m3
```

2. To build the C libraries for m3 via gcc-9:
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
cd parsimon/backends/High-Precision-Congestion-Control/simulation
CC='gcc-5' CXX='g++-5' ./waf configure --build-profile=optimized
```
5. The checkpotins for the end-to-end m3 pipeline are available in the `ckpts` directory. You can use them directly for the following steps. Please refer to the section [Train your own model](#train-your-own-model) for training the model from scratch.

6. To replicate paper results in Section 5.2, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./data --mixes spec/all_dctcp.mix.json ns3-config
cargo run --release -- --root=./data --mixes spec/all_dctcp.mix.json pmn-m
cargo run --release -- --root=./data --mixes spec/all_dctcp.mix.json mlsys
```

Then reproduce the results in the script `parsimon-eval/expts/fig_8/analysis/analysis_dctcp.ipynb.ipynb`.
Note that ns3-config is time-consuming and may take 1-7 days to complete.

7. To replicate paper results in Section 5.3, run the following in the `parsimon-eval/expts/fig_7` directory:

```bash
cargo run --release -- --root=./data --mix spec/0.mix.json ns3
cargo run --release -- --root=./data --mix spec/0.mix.json pmn-m
cargo run --release -- --root=./data --mix spec/0.mix.json mlsys

cargo run --release -- --root=./data --mix spec/1.mix.json ns3
cargo run --release -- --root=./data --mix spec/1.mix.json pmn-m
cargo run --release -- --root=./data --mix spec/1.mix.json mlsys

```

Then reproduce the results in the script `parsimon-eval/expts/fig_7/analysis/analysis.ipynb`.

8. To replicate paper results in Section 5.4, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./data_hpcc --mixes spec/all_counterfactual_hpcc.mix.json ns3-config
cargo run --release -- --root=./data_hpcc --mixes spec/all_counterfactual_hpcc.mix.json pmn-m
cargo run --release -- --root=./data_hpcc --mixes spec/all_counterfactual_hpcc.mix.json mlsys

cargo run --release -- --root=./data_window --mixes spec/all_counterfactual_window.mix.json ns3-config
cargo run --release -- --root=./data_window --mixes spec/all_counterfactual_window.mix.json pmn-m
cargo run --release -- --root=./data_window --mixes spec/all_counterfactual_window.mix.json mlsys
```

Then reproduce the results in the script `parsimon-eval/expts/fig_8/analysis/analysis_counterfactual.ipynb`.

# Train your own model

1. To generate data for training and testing your own model, run:

```bash
cd gen_path
cargo run --release -- --python-path {path_to_python} --output-dir {dir_to_save_data}

e.g., 
cargo run --release -- --python-path /data1/lichenni/software/anaconda3/envs/py39/bin/python --output-dir /data1/lichenni/m3/parsimon/backends/High-Precision-Congestion-Control/gen_path/data
```
Note to adjust generation parameters in `parsimon/backends/High-Precision-Congestion-Control/gen_path/src/main.rs` (lines 34-37) and `simulation/consts.py`. 


2. For training the model, ensure you're using the Python 3 environment and configure settings in `config/train_config_path.yaml`. Then execute:

```bash
cd m3
python main_train.py --train_config=./config/train_config_path.yaml --mode=train --dir_input={dir_to_save_data} --dir_output={dir_to_save_ckpts}

e.g., 
python main_train.py --train_config=./config/train_config_path.yaml --mode=train --dir_input=/data1/lichenni/m3/parsimon/backends/High-Precision-Congestion-Control/gen_path/data --dir_output=/data1/lichenni/m3/ckpts
```
Note to change the gpu_id in `config/train_config_path.yaml` to the desired GPU ID you wish to use. For example, we set it to [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], which means we use GPUs-[0,1,2,3] with 4 processes on each GPU. FYI, the default PytorchLightning does not support multi-worker training on a single GPU, which requries specific modifications.
Also, change the configurations for the dataset or model for your specific use case.

3. To create checkpoints for the end-to-end m3 pipeline:
```bash
cd m3
python gen_ckpt.py --dir_output={dir_to_save_ckpts}

e.g., 
python gen_ckpt.py --dir_output=/data1/lichenni/m3/ckpts
```
Note the checkpoints will be saved in the `ckpts` directory, one is for the Llama-2 model and the other is for the 2-layer MLP model.

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
└── main_train.py   # Main script for training and testing m3
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

