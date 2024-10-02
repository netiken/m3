# m3: Accurate Flow-Level Performance Estimation using Machine Learning

This repository houses the scripts and guidance needed to replicate the experiments presented in our paper, "m3: Accurate Flow-Level Performance Estimation using Machine Learning". It offers all necessary tools to reproduce the experimental results documented in sections 5.2, 5.3, and 5.4 of our paper.

- [Quick Reproduction](#quick-reproduction)
- [From Scratch](#from-scratch)
- [Train your own model](#train-your-own-model)
- [Repository Structure](#repository-structure)
- [Citation Information](#citation-information)
- [Acknowledgments](#acknowledgments)
- [Getting in Touch](#getting-in-touch)

First, clone the repository and install the necessary dependencies. To install m3, execute: 
```bash
git clone https://github.com/netiken/m3.git
cd m3
# Initialize the submodules, including parsimon, parsimon-eval, and HPCC
git submodule update --init --recursive
```

## Quick Reproduction
The following steps provide a quick guide to reproduce the results in the paper.

1. To replicate paper results in Section 5.2, run the notebook `parsimon-eval/expts/fig_8/analysis/analysis_dctcp.ipynb`.

2. To replicate paper results in Section 5.3, run the notebook `parsimon-eval/expts/fig_7/analysis/analysis.ipynb`.

3. To replicate paper results in Section 5.4, run the notebook `parsimon-eval/expts/fig_8/analysis/analysis_counterfactual.ipynb`.

## From Scratch

1. Ensure you have installed: Python 3, Rust, Cargo (nightly Version), gcc-9 (for compiling the flowSim and inference), and gcc-5 (for running ns-3). For example, use the `environment.yml` conda environment file, and follow the additional instructions for installing the other packages. 

```bash
# Create a new conda environment for Python 3.9
conda env create -f environment.yml
conda activate m3
```

```bash
# Install Rust and Cargo, https://www.rust-lang.org/tools/install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# setup the path and check the installation using rustc --version. Then switch to nightly version
rustup install nightly
rustup default nightly
```

```bash
# Install gcc-5 via https://askubuntu.com/questions/1235819/ubuntu-20-04-gcc-version-lower-than-gcc-7
sudo vim /etc/apt/sources.list
# Add the following lines in the sources.list file
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
# Update the package list
sudo apt update
# Install gcc-5
sudo apt install gcc-5 g++-5
```

```bash
# Install gcc-9
sudo apt-get install gcc-9 g++-9
```

2. To build the C libraries for m3 via gcc-9:
```bash     
cd clibs
make run
cd ..
```

3. For setting up the HPCC repository for data generation, follow the detailed instructions in `parsimon/backends/High-Precision-Congestion-Control/simulation/README.md`:

```bash
cd parsimon/backends/High-Precision-Congestion-Control/simulation
CC='gcc-5' CXX='g++-5' CXXFLAGS='-std=c++11' ./waf configure --build-profile=optimized
```

4. The checkpotins for the end-to-end m3 pipeline are available in the `ckpts` directory. You can use them directly for the following steps. Please refer to the section [Train your own model](#train-your-own-model) for training the model from scratch.

5. To replicate paper results in Section 5.2, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
# all_dctcp.mix.json provides 1 simulation configuration. The entire list of configurations is available in all_dctcp_full.mix.json
cargo run --release -- --root=./data --mixes spec/all_dctcp.mix.json ns3-config
cargo run --release -- --root=./data --mixes spec/all_dctcp.mix.json pmn-m
cargo run --release -- --root=./data --mixes spec/all_dctcp.mix.json mlsys
```
**Note:** m3 uses the [HPCC Codebase](https://github.com/alibaba-edu/High-Precision-Congestion-Control) to run ns-3. Most errors you may encounter are likely related to the ns-3 setup. If so, please check if the directory `m3/parsimon-eval/expts/fig_8/data/0/ns3-config/2/` exists. You should find the bug details in `m3/parsimon-eval/expts/fig_8/data/0/ns3-config/2/output.txt`. If the file m3/parsimon-eval/expts/fig_8/data/0/ns3-config/2/fct_topology_flows_dctcp.txt is continuously updating, it means everything is working correctly. Each line in the file represents a completed flow.

Then reproduce the results in the notebook `parsimon-eval/expts/fig_8/analysis/analysis_dctcp.ipynb`.
Note that ns3-config is time-consuming and may take 1-7 days to complete.

6. To replicate paper results in Section 5.3, run the following in the `parsimon-eval/expts/fig_7` directory:

```bash
cargo run --release -- --root=./data --mix spec/0.mix.json ns3
cargo run --release -- --root=./data --mix spec/0.mix.json pmn-m
cargo run --release -- --root=./data --mix spec/0.mix.json mlsys

cargo run --release -- --root=./data --mix spec/1.mix.json ns3
cargo run --release -- --root=./data --mix spec/1.mix.json pmn-m
cargo run --release -- --root=./data --mix spec/1.mix.json mlsys

```

Then reproduce the results in the notebook `parsimon-eval/expts/fig_7/analysis/analysis.ipynb`.

7. To replicate paper results in Section 5.4, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./data_hpcc --mixes spec/all_counterfactual_hpcc.mix.json ns3-config
cargo run --release -- --root=./data_hpcc --mixes spec/all_counterfactual_hpcc.mix.json pmn-m
cargo run --release -- --root=./data_hpcc --mixes spec/all_counterfactual_hpcc.mix.json mlsys

cargo run --release -- --root=./data_window --mixes spec/all_counterfactual_window.mix.json ns3-config
cargo run --release -- --root=./data_window --mixes spec/all_counterfactual_window.mix.json pmn-m
cargo run --release -- --root=./data_window --mixes spec/all_counterfactual_window.mix.json mlsys
```

Then reproduce the results in the notebook `parsimon-eval/expts/fig_8/analysis/analysis_counterfactual.ipynb`.

# Train your own model

* Please use the demo data in `data` in the main directory to test the training process.

1. To generate data for training and testing your own model, run:

```bash
cd parsimon/backends/High-Precision-Congestion-Control/gen_path
cargo run --release -- --python-path {path_to_python} --output-dir {dir_to_save_data}

e.g., 
cargo run --release -- --python-path /data1/lichenni/software/anaconda3/envs/py39/bin/python --output-dir /data1/lichenni/m3/data
```
Note to adjust generation parameters in `parsimon/backends/High-Precision-Congestion-Control/gen_path/src/main.rs` (lines 34-37) and `parsimon/backends/High-Precision-Congestion-Control/simulation/consts.py`. 

```rust
// parsimon/backends/High-Precision-Congestion-Control/gen_path/src/main.rs
shard: (0..2000).collect(), // number of diverse workloads
n_flows: vec![20000], // number of flows per (src, dst) host pair
n_hosts: vec![3, 5, 7], // type of multi-hop paths, e.g., 2-hop, 4-hop, 6-hop
shard_cc: (0..20).collect(), // number of diverse network configurations, such different CCAs, CCA parameters, etc.

// We use the following parameters to generate the demo data to test
shard: (0..100).collect(), // number of diverse workloads
n_flows: vec![100], // number of flows per (src, dst) host pair
n_hosts: vec![3], // type of multi-hop paths, e.g., 2-hop, 4-hop, 6-hop
shard_cc: (0..1).collect(), // number of diverse network configurations, such different CCAs, CCA parameters, etc.
```

```python
# parsimon/backends/High-Precision-Congestion-Control/simulation/consts.py
bfsz=[20,50,10] # The buffer size in KB is uniformly sampled from [bfsz[0], bfsz[1]] * bfsz[2]
fwin=[5, 30,1000] # The window size in Byte is uniformly sampled from [fwin[0], fwin[1]] * fwin[2]
enable_pfc=[0,1] # Enable PFC or not

```

2. For training the model, ensure you're using the Python 3 environment and configure settings in `config/train_config_path.yaml`. See the detailed configurations in `config/train_config_path.yaml` for the dataset, model, and training parameters. Then execute:

```bash
cd m3
python main_train.py --train_config=./config/train_config_path.yaml --mode=train --dir_input={dir_to_save_data} --dir_output={dir_to_save_ckpts}

e.g., 
python main_train.py --train_config=./config/train_config_path_demo.yaml --mode=train --dir_input=/data1/lichenni/m3/data --dir_output=/data1/lichenni/m3/
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

