./fast_mmf.out xgft 3 4 4 3 1 2 2 1 1 1 OPT trafficfile
./fast_mmf.out xgft 2 4 4 1 2 1 1 OPT trafficfile
./fast_mmf.out xgft 2 2 2 1 2 1 1 OPT trafficfile

./fast_mmf.out xgft 3 12 12 24 1 12 12 1 1 1 OPT trafficfile

./fast_mmf.out xgft 2 4 4 1 2 1 1 PF trafficfile
./get_fct_mmf_bi.out 8 1 2 1 5
./get_fct_mmf.out 8 1 2 1 4
./get_fct_mmf.out 8 1 1 4
./get_fct_mmf_fattree.out 8 1 11

# for one_layer test
./get_fct_mmf.out 8 2 2 1 1
./get_fct_mmf.out 8 2 2 1 1

gcc -g -O3 -fPIC -shared -o get_fct_mmf.so -fopenmp  -mcmodel=large \
xgft_tr.c \
model_utils.c \
xgft_utils.c  \
cplex_engine_mmf.c  \
mmf_engine_nonlp.c  \
get_fct_mmf.c

gcc -g -O3 -fPIC -shared -o get_fct_mmf.so -fopenmp  -mcmodel=large topo.c get_fct_mmf.c

gcc -O3 -fPIC -shared -o get_fct_mmf.so -fopenmp  -mcmodel=large topo.c get_fct_mmf.c

gcc -g -Wall -O3 -o get_fct_mmf.out -fopenmp  -mcmodel=large topo.c get_fct_mmf.c -lm

gcc -O3 -o get_fct_mmf.out -fopenmp  -mcmodel=large topo.c get_fct_mmf.c -lm

gcc -g -O0 -o get_fct_mmf_fattree.out -fopenmp  -mcmodel=large topo_fattree.c get_fct_mmf_fattree.c -lm

/data1/lichenni/software/anaconda3/envs/py27/bin/python /data1/lichenni/projects/flow_simulation/parsimon/backends/High-Precision-Congestion-Control/traffic_gen/traffic_gen_by_n_synthetic.py --shard 0 -f 2000 -n 7 -b 10G -o /data1/lichenni/projects/flow_simulation/data_test/data_lr10Gbps

/data1/lichenni/software/anaconda3/envs/py39/bin/python /data1/lichenni/projects/flow_simulation/fast-mmf-fattree/main_flowsim_mmf.py --root /data1/lichenni/projects/flow_simulation/data_test/data_lr10Gbps_3_large -b 10 --nhost 3 --cc dctcp

make run
time run ../ckpts/model_llama_all_e240.bin ../ckpts/model_mlp_all_e240.bin ../data_test/data_lr10Gbps_5_small -b 10 -e 576 -n 5 -p 18000 -t 1 -c 3


time run ../ckpts/model_llama_e271.bin ../ckpts/model_mlp_e271.bin ../data_test/data_lr10Gbps_7_small -b 10 -e 576 -n 7 -t 1 -f 30 -k 18000 -p 1 -c 0 -x 30 

time run ../ckpts/model_llama.bin ../ckpts/model_mlp.bin ../ckpts/data_lr10Gbps_7 -b 10 -e 576 -n 7 -t 1 -f 30 -k 18000 -p 1 -c 0 -x 30 