import numpy as np
import yaml
import sys
import os 

sys.path.insert(0, "./util")
from util.consts import (
    P99_PERCENTILE_LIST,
    PERCENTILE_METHOD,
    MTU,
    HEADER_SIZE,
    BYTE_TO_BIT,
    BDP_DICT,
    LINK_TO_DELAY_DICT,
    UNIT_G,
    get_base_delay_pmn,
    get_size_bucket_list,
    get_size_bucket_list_output,
    EPS
)
from util.utils import (
    fix_seed,
)
from util.model import FlowSimTransformer_Path
import torch
from util.arg_parser import create_config
import json
PARAM_VEC_INIT=np.array([0,30,18,1,1,0,0,0,30,0,0,0,0,0,0])

args = create_config()
fix_seed(args.shard)
DEVICE = torch.device(args.device)

# set parameters
model_trained_dir=f"{args.dir_output}/m3_shard2000_nflows1_nhosts3_nsamples20_lr10Gbps/version_0"
output_dir=f"./ckpts"
model_id="_config_e421"
# model_id="_e466"
# model_id="_hpcc_e447"
class m3_inference:
    def __init__(self):
        self.bucket_thold = 1
        self.dir_input = output_dir
        self.dir_train = model_trained_dir
        f = open(f"{self.dir_train}/data_list.json", "r")
        data_list=json.loads(f.read())
        # [["shard1191_nflows20000_nhosts7_lr10Gbpsparam_k30", [0, 6], "_topo-pl-7_dctcp"],...]
        self.data_list = data_list["test"]
        with open(f'./config/test_config_path.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            dataset_config = config["dataset"]
            model_config = config["model"]
            training_config = config["training"]
        n_params=dataset_config["n_params"]
        model = FlowSimTransformer_Path.load_from_checkpoint(
            f"{self.dir_train}/checkpoints/best{model_id}.ckpt",
            map_location=DEVICE,
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            block_size=model_config["block_size"],
            vocab_size=model_config["vocab_size"],
            dropout=model_config["dropout"],
            compile=model_config["compile"],
            loss_fn_type=model_config["loss_fn_type"],
            weight_decay=training_config["weight_decay"],
            learning_rate=training_config["learning_rate"],
            betas=training_config["betas"],
            batch_size=training_config["batch_size"],
            enable_val=training_config["enable_val"],
            enable_dist=training_config["enable_dist"],
            enable_masked_loss=training_config["enable_masked_loss"],
            enable_weighted_loss=training_config["enable_weighted_loss"],
            enable_context=dataset_config.get("enable_context", False),
            hidden_sizes=model_config["hidden_dims"],
            enable_position=model_config["enable_position"],
            enable_log=training_config["enable_log"],
            n_params=n_params,
            save_dir=output_dir,
        )
        
        model.eval()
        self.model=model
        self.lr=10
        # self.bucket_thold = dataset_config["bucket_thold"]
        self.enable_context = dataset_config.get("enable_context", False)
        self.enable_log=training_config["enable_log"]
        bdp_dict_db = {}
        bdp_dict_db_output = {}
        for n_hosts in [3,5,7]:
            BDP = 10 * MTU
            bdp_dict_db[n_hosts] = get_size_bucket_list(mtu=MTU, bdp=BDP)
            bdp_dict_db_output[n_hosts] = get_size_bucket_list_output(mtu=MTU, bdp=BDP)
        self.bdp_dict_db = bdp_dict_db
        self.bdp_dict_db_output = bdp_dict_db_output
        
        model.export_to_bin_llama_v0(filepath=f"{output_dir}/model_llama{model_id}.bin")
        model.export_to_bin_mlp(filepath=f"{output_dir}/model_mlp{model_id}.bin")
        
    def run_inference(self,idx):
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        n_hosts = int(spec.split("_")[2][6:])

        size_bucket_list = self.bdp_dict_db[n_hosts]
        size_bucket_list_output = self.bdp_dict_db_output[n_hosts]
        print(f"spec-1: {spec}, {topo_type}")   
        spec=f'data_lr10Gbps_{n_hosts}'
        print(f"spec-2: {spec}")
        
        dir_input_tmp = f"{self.dir_input}/{spec}"
        param_data = PARAM_VEC_INIT
        if param_data[3]==1.0:
            param_data=np.insert(param_data,4,0)
        else:
            param_data=np.insert(param_data,4,1)
            
        param_data=np.insert(param_data,0,[0,0,0])
        param_data[n_hosts//2-1]=1.0
        param_data[3]=BDP_DICT[n_hosts]/MTU
        print(f"param_data: {param_data}")
        
        with open(f"{dir_input_tmp}/flows.txt", "r") as f:
            num_lines = int(f.readline().strip())
            flow_id = []
            sizes = []
            fats = []
            flow_src_dst = []
            for _ in range(num_lines):
                data = f.readline().strip().split()
                flow_id.append(int(data[0]))
                flow_src_dst.append([int(data[1]), int(data[2])])
                sizes.append(int(data[5]))
                fats.append(int(float(data[6]) * UNIT_G))
        # fid = np.array(flow_id).astype("int32")
        sizes_flowsim = np.array(sizes).astype("int64")
        flow_src_dst_flowsim = np.array(flow_src_dst).astype("int32")
        sizes=sizes_flowsim
        flow_src_dst=flow_src_dst_flowsim
        
        flow_idx_target_flowsim = np.logical_and(
            flow_src_dst_flowsim[:, 0] == src_dst_pair_target[0],
            flow_src_dst_flowsim[:, 1] == src_dst_pair_target[1],
        )
        flow_idx_nontarget_flowsim=~flow_idx_target_flowsim
        flow_idx_nontarget_internal_flowsim=np.logical_and(
            flow_src_dst_flowsim[:, 0] != src_dst_pair_target[0],
            flow_src_dst_flowsim[:, 1] != src_dst_pair_target[1],
        )
        n_links_passed = abs(flow_src_dst_flowsim[:, 0] - flow_src_dst_flowsim[:, 1])+flow_idx_nontarget_flowsim+flow_idx_nontarget_internal_flowsim
        delay_comp=LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,0]]+LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,1]]
        DELAY_PROPAGATION_PERFLOW = get_base_delay_pmn(
            sizes=sizes, n_links_passed=n_links_passed, lr_bottleneck=self.lr,flow_idx_target=flow_idx_target_flowsim,flow_idx_nontarget_internal=flow_idx_nontarget_internal_flowsim
        )+delay_comp
        
        fcts_flowsim = (
            np.load(f"{dir_input_tmp}/fct_flowsim.npy") + DELAY_PROPAGATION_PERFLOW
        )
        i_fcts_flowsim = (
            sizes + np.ceil(sizes / MTU) * HEADER_SIZE
        ) * BYTE_TO_BIT / self.lr + DELAY_PROPAGATION_PERFLOW
        sldns_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
        sldns_flowsim = np.clip(sldns_flowsim, a_max=None, a_min=1.0)
        
        sldns=sldns_flowsim
        sldns_list = []
        bins = []
        x_len = len(size_bucket_list) + 1
        y_len = len(P99_PERCENTILE_LIST)

        # add the target flow
        # print("size_bucket_list: ",size_bucket_list)
        sldns_flowsim_target = sldns_flowsim[flow_idx_target_flowsim]
        sldns_list.append(sldns_flowsim_target)
        bins_target = np.digitize(sizes_flowsim[flow_idx_target_flowsim], size_bucket_list)
        bins.append(bins_target)
        
        if self.enable_context:
            for link_idx_internal in range(
                src_dst_pair_target[0], src_dst_pair_target[1]
            ):
                flow_idx_selected = np.logical_and(
                    flow_src_dst_flowsim[:, 0] <= link_idx_internal,
                    flow_src_dst_flowsim[:, 1] > link_idx_internal,
                )
                flow_idx_selected = np.logical_and(flow_idx_selected, ~flow_idx_target_flowsim)
                sizes_perlink = sizes_flowsim[flow_idx_selected]
                sldns_flowsim_perlink = sldns_flowsim[flow_idx_selected]
                
                sldns_list.append(sldns_flowsim_perlink)
                bins.append(np.digitize(sizes_perlink, size_bucket_list))

        n_sldns_list = len(sldns_list)
        sizebucket_to_sldn = np.zeros((n_sldns_list, x_len, y_len))
        num_flows_per_cell = np.zeros((n_sldns_list, x_len, y_len))
        n_sizes_effective = np.ones((n_sldns_list, 1))

        for sldns_idx in range(n_sldns_list):
            if len(bins[sldns_idx]) == 0:
                continue
            for x_idx in range(x_len):
                sldn_idx_target = np.nonzero(bins[sldns_idx] == x_idx)[0]
                if len(sldn_idx_target) < self.bucket_thold:
                    continue
                
                sldns_tmp = sldns_list[sldns_idx][sldn_idx_target]
                sizebucket_to_sldn[sldns_idx, x_idx] = np.percentile(
                    sldns_tmp, P99_PERCENTILE_LIST,
                    method=PERCENTILE_METHOD
                )
                num_flows_per_cell[sldns_idx, x_idx] = len(sldn_idx_target)
                n_sizes_effective[sldns_idx] += len(sldn_idx_target)
        res = sizebucket_to_sldn.reshape((n_sldns_list, -1)).astype(np.float32)
        
        print("sizebucket_to_sldn: ",res.shape)
        for i in range(len(sldns_list)):
            print(f"Sort Bucket {i}: {num_flows_per_cell[i,:,0]}")
            print(f"feat-input-{i}: {res[i,0]}, {res[i,-1]}")
        
        num_flows_per_cell = num_flows_per_cell.reshape((n_sldns_list, -1)).astype(
            np.float32
        )
        
        num_flows_per_cell = np.divide(num_flows_per_cell, n_sizes_effective)

        # find foreground/background flows for gt
        flow_idx_target = np.logical_and(
            flow_src_dst[:, 0] == src_dst_pair_target[0],
            flow_src_dst[:, 1] == src_dst_pair_target[1],
        )
        # output/ground truth
        sldns_output = sldns[flow_idx_target]
        bins_output = np.digitize(sizes[flow_idx_target], size_bucket_list_output)
        x_len_output = len(size_bucket_list_output) + 1
        sizebucket_to_sldn_output = np.ones((x_len_output, y_len))
        num_flows_per_cell_output = np.zeros((x_len_output, y_len))
        n_sizes_effective_output = 0

        for x_idx in range(x_len_output):
            sldn_idx_target = np.nonzero(bins_output == x_idx)[0]
            if len(sldn_idx_target) < self.bucket_thold:
                continue

            sldns_tmp = sldns_output[sldn_idx_target]
            sizebucket_to_sldn_output[x_idx] = np.percentile(
                sldns_tmp, P99_PERCENTILE_LIST,
                method=PERCENTILE_METHOD
            )
            num_flows_per_cell_output[x_idx] = len(sldn_idx_target)
            n_sizes_effective_output += len(sldn_idx_target)
        res_output = sizebucket_to_sldn_output.reshape((-1)).astype(np.float32)

        num_flows_per_cell_output = num_flows_per_cell_output.reshape((-1)).astype(
            np.float32
        )
        num_flows_per_cell_output = np.divide(
            num_flows_per_cell_output, n_sizes_effective_output
        )

        # [size_bucket,percentile]
        n_input = n_sldns_list
        # res -= 1.0
        assert (res>=0).all()
        res=np.insert(res, res.shape[1], param_data[:,None], axis=1)
        # src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])
        
        # np.savetxt(f'./fast-mmf-fattree/feat_map_py.txt', res.flatten(), fmt='%f',newline=', ')
        
        sizebucket_to_sldn_flowsim=torch.tensor(res).to(DEVICE)
        num_flows_per_cell_flowsim=torch.tensor(num_flows_per_cell).to(DEVICE)
        sizebucket_to_sldn=torch.tensor(res_output).to(DEVICE)
        num_flows_per_cell=torch.tensor(num_flows_per_cell_output)
        sizebucket_to_sldn_flowsim_idx=[n_input]
        src_dst_pair_target=np.array(src_dst_pair_target)
        spec=[spec]
        
        with torch.no_grad():
            if self.model.enable_const_opt:
                num_flows_per_cell_flowsim=num_flows_per_cell_flowsim.reshape((num_flows_per_cell_flowsim.shape[0],-1,self.model.y_len))
                num_flows_per_cell_flowsim=num_flows_per_cell_flowsim.mean(dim=-1)
                for idx_1 in range(num_flows_per_cell_flowsim.shape[0]):
                    for idx_2 in range(num_flows_per_cell_flowsim.shape[1]):
                        if num_flows_per_cell_flowsim[idx_1,idx_2]<EPS:
                            sizebucket_to_sldn_flowsim[idx_1,idx_2*self.model.y_len:(idx_2+1)*self.model.y_len]=self.model.const_tensor
                        
            if self.model.enable_context:
                idx_start = 0
                sizebucket_to_sldn_foreground = sizebucket_to_sldn.new(
                    len(spec), self.model.feat_dim
                )
                sizebucket_to_sldn_context = sizebucket_to_sldn.new(
                    len(spec), self.model.n_embd
                )
                for i in range(len(spec)):
                    sizebucket_to_sldn_foreground[i] = sizebucket_to_sldn_flowsim[
                        idx_start
                    ]
                    idx_interval = sizebucket_to_sldn_flowsim_idx[i]
                    tmp = sizebucket_to_sldn_flowsim[
                        idx_start + 1 : idx_start + idx_interval
                    ]
                    # tmp=torch.flatten(tmp).long()
                    sizebucket_to_sldn_background, _ = self.model.model_transformer(
                        tmp[None, :]
                    )
                    # print("sizebucket_to_sldn_background: ",sizebucket_to_sldn_background.shape)
                    # for j in range(sizebucket_to_sldn_background.shape[1]):
                    #     print(f"logit-{i}-{j}: {tmp[j,0]}, {tmp[j,-2]}")
                    #     print(f"logit-{i}-{j}: {sizebucket_to_sldn_background[0, j,0]}, {sizebucket_to_sldn_background[0, j,-1]}")
                    sizebucket_to_sldn_context[i] = torch.mean(
                        sizebucket_to_sldn_background, dim=1
                    )
                    # print(f"sizebucket_to_sldn_context-{i}: {sizebucket_to_sldn_context[i,0]}, {sizebucket_to_sldn_context[i,-1]}")
                    idx_start += idx_interval

                sizebucket_to_sldn_input = torch.cat(
                    [sizebucket_to_sldn_foreground, sizebucket_to_sldn_context], dim=-1
                )
                print(f"sizebucket_to_sldn_input: {sizebucket_to_sldn_input[0,0]}, {sizebucket_to_sldn_input[0,300]}, {sizebucket_to_sldn_input[0,301]}, {sizebucket_to_sldn_input[0,-1]}")
            else:
                sizebucket_to_sldn_foreground = sizebucket_to_sldn_flowsim[:, 0, :]
                sizebucket_to_sldn_input = sizebucket_to_sldn_foreground
            # sizebucket_to_sldn_input=torch.cat([sizebucket_to_sldn_input, src_dst_pair_target], dim=-1)
            sizebucket_to_sldn_est = self.model.model_mlp(sizebucket_to_sldn_input)
            sizebucket_to_sldn_est.add_(1.0)
            
            print(f"sizebucket_to_sldn_est: {sizebucket_to_sldn_est[0,0]},  {sizebucket_to_sldn_est[0,-1]}")
            
            test_dir = f"{self.model.save_dir}/{spec[0]}"
            # logging.info(f"save to {test_dir}")
            os.makedirs(test_dir, exist_ok=True)
            sizebucket_to_sldn_flowsim = sizebucket_to_sldn_flowsim.cpu().numpy()[0]
            sizebucket_to_sldn_input = sizebucket_to_sldn_input.cpu().numpy()[0]
            sizebucket_to_sldn_est = sizebucket_to_sldn_est.cpu().numpy()[0]
            sizebucket_to_sldn = sizebucket_to_sldn.cpu().numpy()
            num_flows_per_cell = num_flows_per_cell.cpu().numpy()
            
            np.savetxt(f'{output_dir}/{spec[0]}/feat_output_py.txt', sizebucket_to_sldn_input, fmt='%f',newline=' ')
            # sizebucket_to_sldn_est=sizebucket_to_sldn_est.reshape(x_len_output,y_len)
            # sizebucket_to_sldn=sizebucket_to_sldn.reshape(x_len_output,y_len)
            
            # Concatenate arrays vertically
            # concatenated_array = np.vstack((sizebucket_to_sldn_est, sizebucket_to_sldn))
            # l2_norm = np.linalg.norm(concatenated_array)
            # print(f"{spec[0]}: {l2_norm}")
            # Save the concatenated array to a text file
            # np.savetxt(f'{test_dir}/m3_python.txt', concatenated_array, fmt='%f')
            
my_model=m3_inference()
my_model.run_inference(7)

