watch -c nvidia-htop.py -c

git fetch origin main
git fetch origin arjunvb
git fetch origin tracker
git merge origin/arjunvb

git submodule update --init --recursive
git submodule update --recursive
git rm --cached xxx
git submodule status --recursive

ps aux | head -1; ps aux | grep ^lichenni| sort -rnk 4 | more

tensorboard --logdir /data2/lichenni/output --port 8009 --bind_all

gcc -O3 -o run run.c -lm -fopenmp

python metric_simulation.py --test_config=./config/test_config.yaml --enable_correction --note=ddp
python metric_simulation.py --test_config=./config/test_config.yaml --note=seg
python metric_simulation_fairsharing.py --test_config=./config/test_config_fairsharing.yaml --note=fairsharing

cc -O3 -fPIC -shared -o gen_weight.so gen_weight.c -fopenmp
git filter-branch --tree-filter 'rm -f plot_simulation.ipynb' HEAD

# train
python main.py --train_config=./config/train_config.yaml --mode=train --note=train

python main.py --train_config=./config/train_config.yaml --mode=train --note=train_cwnd

python main.py --train_config=./config/train_config_param.yaml --mode=train --note=train_param

python main.py --train_config=./config/train_config_val.yaml --mode=train --note=train_val

python main.py --train_config=./config/train_config.yaml --mode=train --note=train_incast --train_on_incast

python main.py --train_config=./config/train_config.yaml --mode=train --note=train_incast

python main.py --train_config=./config/train_config_topo.yaml --mode=train --note=tc_all_pp

python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m1
python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m2
python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m3_gaussian --ckpt_path=/data2/lichenni/output/m3_gaussian_shard3000_nflows1_nhosts7_lr10Gbps/version_0/checkpoints/last.ckpt

python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m3_tc --ckpt_path=/data2/lichenni/output/m3_tc_shard3000_nflows1_nhosts7_lr10Gbps/version_0/checkpoints/last.ckpt

python main_link.py --train_config=./config/train_config_link.yaml --mode=train --note=link_m3

python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=baseline

# test
python main.py --test_config=./config/test_config.yaml --mode=test --note=train --version_id 0 --test_on_train

python main.py --test_config=./config/test_config.yaml --mode=test --note=train_cwnd --version_id 2 --test_on_train

python main.py --test_config=./config/test_config_val.yaml --mode=test --note=train_val --version_id 7 --test_on_train

python main.py --test_config=./config/test_config_param.yaml --mode=test --note=train_param --version_id 5 --test_on_train

python main.py --test_config=./config/test_config.yaml --mode=test --note=train --version_id 0 --test_on_empirical

python main.py --test_config=./config/test_config.yaml --mode=test --note=train --version_id 0 --test_on_manual

python main.py --test_config=./config/test_config_topo.yaml --mode=test --note=train_topo --version_id 0 --test_on_manual

python main.py --test_config=./config/test_config_topo.yaml --mode=test --note=tc_all --version_id 0 --test_on_empirical

python main_path.py --test_config=./config/test_config_path.yaml --mode=test --note=m3_pmn --version_id 0 --test_on_train

python main_link.py --test_config=./config/test_config_link.yaml --mode=test --note=link_m3 --version_id 4 --test_on_train

# tune
sleep 3h && CUDA_VISIBLE_DEVICES="2,3" python main_tune.py --train_config=./config/train_config.yaml --tune_config=./config/tune_config.yaml --note=tune

sleep 3h && CUDA_VISIBLE_DEVICES="1" python main_tune.py --train_config=./config/train_config.yaml --tune_config=./config/tune_config_tmp1.yaml --note=bs

sleep 3h && CUDA_VISIBLE_DEVICES="2" python main_tune.py --train_config=./config/train_config.yaml --tune_config=./config/tune_config_tmp2.yaml --note=bt

sleep 3h && CUDA_VISIBLE_DEVICES="3" python main_tune.py --train_config=./config/train_config.yaml --tune_config=./config/tune_config_tmp3.yaml --note=hd

ray stop --force

# cwnd
python main_flowsim_cwnd_perflow.py --tag pareto_lognorm --avg_util 0.5 --sizesigma 30000 --iasigma 3.0 --shard 0 --n_flows 100000 --cwnd_perflow 0.05

python main_flowsim_cwnd_sum.py --tag pareto_lognorm --avg_util 0.5 --sizesigma 30000 --iasigma 3.0 --shard 0 --n_flows 100000 --cwnd_sum 0.5

watch -n 60 echo hello world

# data gen
python input_exp_size_lognormal_ia.py --avg_util 0.5 --sizesigma 20000 --iasigma 2.0 --n_flows 10000 --enable_auxi

git submodule add git@github.com:liecn/fast-mmf-fattree.git
git submodule add git@github.com:arashne/linksim-data-gen.git
git submodule add git@github.com:liecn/High-Precision-Congestion-Control.git


scp -r tardy.csail.mit.edu:/data1/lichenni/projects/flow_simulation/data_test ./

scp -r kasra.csail.mit.edu:/data1/lichenni/projects/High-Precision-Congestion-Control/simulation/data/input/*_shard0_nflows20000_nhosts6_ntc4_lr10Gbps ./

scp -r kasra.csail.mit.edu:/data1/lichenni/projects/High-Precision-Congestion-Control/log/* ./

scp -r liangyu.csail.mit.edu:/mnt/data0/lichenni/path_pmn_tc_cc ./ 

^((?!test/).)*$
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /data1/lichenni/software/anaconda3/envs/py39/lib/libstdc++.so.6

cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28  /data1/lichenni/software/anaconda3/lib

ln -s /data1/lichenni/software/anaconda3/envs/py39/lib/libstdc++.so.6.0.29 /data1/lichenni/software/anaconda3/envs/py39/lib/libstdc++.so.6
strings /data1/lichenni/software/anaconda3/envs/py39/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29   

../gcc-5.3.0/configure --prefix=/data1/lichenni/software/gcc-5.3.0 --disable-nls --enable-languages=c,c++ --with-ld=/bin/ld --with-nm=/bin/nm --with-as=/usr/bin/as --disable-multilib


/data1/lichenni/software/anaconda3/envs/py27/bin/python /data1/lichenni/projects/flow_simulation/High-Precision-Congestion-Control/traffic_gen/traffic_gen_by_n_empirical_path.py --shard 0 -f 20 -n 8 -b 10G -o /data1/lichenni/projects/flow_simulation/High-Precision-Congestion-Control/simulation/data/input_test/shard23_nflows20000_nhosts2_lr10Gbps


setw synchronize-panes on
set -g mouse on

./waf --run 'scratch/third mix_parsimon/config_topology_flows_dctcp.txt'
./waf -d debug -out=debug.txt --run 'scratch/test'

export NS_LOG=FlowPathExample=info

./waf --run 'scratch/third /data2/lichenni/path_pmn/shard0_nflows1000_nhosts3_lr10Gbps/config_topo-pl-3_flows_dctcp.txt'

# Path: projects/flow_simulation/run.sh
python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m3_bt1_num_p30

python main_path.py --train_config=./config/train_config_path_param.yaml --mode=train --note=m3_k --ckpt_path=/data2/lichenni/output/m3_k_shard2000_nflows1_nhosts3_lr10Gbps/version_0/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=2,3 python main_path.py --train_config=./config/train_config_path.yaml --mode=train --note=m3 --ckpt_path=/data2/lichenni/output/m3_shard2000_nflows1_nhosts3_nsamples20_lr10Gbps/version_0/checkpoints/last.ckpt

python main_path.py --train_config=./config/train_config_path_cc.yaml --mode=train --note=m3_cc

python main_path.py --test_config=./config/test_config_path.yaml --mode=test --note=m3_bdp_bt10_zero --version_id 0 --test_on_train


https://github.com/liecn/flow_simulation/compare/160dca368323868e14c7302052262bae87d4442d...364f6fb0fde3d8afa0e725f645bcc54cea8d5cfe

https://github.com/kwzhao/High-Precision-Congestion-Control/compare/13958423c9b7e666b8b51bdb889816ec3f52d79a...aacaa652b0fd2c5f57b84531b29078cc66a28a9f

https://github.com/liecn/fast-mmf-fattree/compare/fdf3eb55df832a921bb2f01571a2de33df1271e1...39962ee50fddac10b93159c82c36c9be7a52f5b1

https://github.com/netiken/parsimon-eval/compare/87269a618651fd289bf5ac5ced404b77a17c9227...9547004da83cdde97707914f29ec1967d3e02d2e

git add -A . ; git commit -m "add args"; git push


/data1/lichenni/software/anaconda3/envs/py39/lib/python3.9/site-packages/lightning_fabric/utilities/device_parser.py

git remote set-url origin git@github.com:netiken/parsimon.git