from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch
from .consts import (
    SRC2BTL,
    BTL2DST,
    SIZE_BUCKET_LIST,
    P99_PERCENTILE_LIST,
    PERCENTILE_METHOD,
    MTU,
    HEADER_SIZE,
    BYTE_TO_BIT,
    DELAY_PROPAGATION,
    DELAY_PROPAGATION_BASE,
    SIZEDIST_LIST_EMPIRICAL,
    UTIL_LIST,
    IAS_LIST,
    BDP_DICT,
    LINK_TO_DELAY_DICT,
    BDP,
    get_size_bucket_list,
    get_size_bucket_list_output,
    get_base_delay_pmn,
)
from .utils import decode_dict
import json
import logging
import os

def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    sizebucket_to_sldn_flowsim = [item[0] for item in batch]
    sizebucket_to_sldn_flowsim = np.concatenate(sizebucket_to_sldn_flowsim, 0)

    num_flows_per_cell_flowsim = [item[1] for item in batch]
    num_flows_per_cell_flowsim = np.concatenate(num_flows_per_cell_flowsim, 0)

    sizebucket_to_sldn = np.array([item[2] for item in batch])
    num_flows_per_cell = np.array([item[3] for item in batch])
    spec = np.array([item[4] for item in batch])
    sizebucket_to_sldn_flowsim_idx = np.array([item[5] for item in batch])
    src_dst_pair_target_str = np.array([item[6] for item in batch])
    # src_dst_pair_target = np.array([item[7] for item in batch])
    return (
        torch.tensor(sizebucket_to_sldn_flowsim),
        torch.tensor(num_flows_per_cell_flowsim),
        torch.tensor(sizebucket_to_sldn),
        torch.tensor(num_flows_per_cell),
        spec,
        sizebucket_to_sldn_flowsim_idx,
        src_dst_pair_target_str,
        # torch.tensor(src_dst_pair_target),
    )

class PathDataModule(LightningDataModule):
    def __init__(
        self,
        dir_input,
        shard_list,
        n_flows_list,
        n_hosts_list,
        sample_list,
        batch_size,
        num_workers,
        train_frac,
        dir_output,
        lr,
        bucket_thold,
        mode="train",
        test_on_train=False,
        test_on_empirical=False,
        test_on_manual=False,
        enable_context=False,
        topo_type="",
    ) -> None:
        """
        Initializes a new instance of the class with the specified parameters.

        Args:
            positive_ratio (float, optional): The ratio of positive to negative samples to use for training.
                Defaults to 0.8.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.dir_input = dir_input
        self.dir_output = dir_output
        data_list = []
        if mode == "train":
            for shard in shard_list:
                for n_flows in n_flows_list:
                    for n_hosts in n_hosts_list:
                        topo_type_cur = topo_type.replace(
                            "-x_", f"-{n_hosts}_"
                        )
                        spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                        for sample in sample_list:
                            data_list.append(
                                (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}")
                            )
                            
            np.random.shuffle(data_list)
        self.data_list = data_list
        self.lr = lr
        self.bucket_thold = bucket_thold
        self.test_on_train = test_on_train
        self.test_on_empirical = test_on_empirical
        self.test_on_manual = test_on_manual
        self.enable_context = enable_context
        self.topo_type = topo_type
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders.

        Args:
            stage (str): The current stage of the training process. Either "fit" or "test".

        Returns:
            None
        """
        if stage == "fit":
            self.train_list, self.val_list = self.__random_split_list(
                self.data_list,
                self.train_frac,
            )
            num_train, num_val = (
                len(self.train_list),
                len(self.val_list),
            )
            logging.info(f"#tracks: train-{num_train}, val-{num_val}")
            self.train = self.__create_dataset(
                self.train_list,
                self.dir_input,
            )
            self.val = self.__create_dataset(
                self.val_list,
                self.dir_input,
            )

            self.__dump_data_list(self.dir_output)

        if stage == "test":
            if self.test_on_manual:
                data_list_test = []
                for shard in np.arange(0, 3000):
                    for n_flows in [30000]:
                        for n_hosts in [2, 3, 4, 5, 6, 7, 8]:
                            topo_type_cur = self.topo_type.replace(
                                "x-x", f"{n_hosts}-{n_hosts}"
                            )
                            spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{self.lr}Gbps"
                            dir_input_tmp = f"{self.dir_input}/{spec}"
                            if not os.path.exists(f"{dir_input_tmp}/flow_src_dst.npy"):
                                continue
                            flow_src_dst = np.load(f"{dir_input_tmp}/flow_src_dst.npy")
                            stats = decode_dict(
                                np.load(
                                    f"{dir_input_tmp}/stats.npy",
                                    allow_pickle=True,
                                    encoding="bytes",
                                ).item()
                            )

                            n_flows_total = stats["n_flows"]
                            if len(flow_src_dst) == n_flows_total:
                                target_idx = stats["host_pair_list"].index(
                                    (0, n_hosts - 1)
                                )
                                size_dist = stats["size_dist_candidates"][
                                    target_idx
                                ].decode("utf-8")
                                if size_dist != "gaussian":
                                    continue
                                data_list_test.append(
                                    (spec, (0, n_hosts - 1), topo_type_cur)
                                )
            else:
                if self.test_on_empirical:
                    data_list_test = []
                    for shard in np.arange(10000, 10200):
                        for n_flows in [30000]:
                            for n_hosts in [2, 3, 4, 5, 6, 7, 8]:
                                topo_type_cur = self.topo_type.replace(
                                    "x-x", f"{n_hosts}-{n_hosts}"
                                )
                                spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{self.lr}Gbps"
                                dir_input_tmp = f"{self.dir_input}/{spec}"
                                if not os.path.exists(
                                    f"{dir_input_tmp}/flow_src_dst.npy"
                                ):
                                    continue
                                flow_src_dst = np.load(
                                    f"{dir_input_tmp}/flow_src_dst.npy"
                                )
                                stats = decode_dict(
                                    np.load(
                                        f"{dir_input_tmp}/stats.npy",
                                        allow_pickle=True,
                                        encoding="bytes",
                                    ).item()
                                )
                                n_flows_total = stats["n_flows"]
                                if (
                                    n_flows_total < 5000000
                                    and len(flow_src_dst) == n_flows_total
                                ):
                                    data_list_test.append(
                                        (spec, (0, n_hosts - 1), topo_type_cur)
                                    )
                                  
                else:
                    data_list = self.__read_data_list(self.dir_output)
                    if self.test_on_train:
                        data_list_test = data_list["train"]
                    else:
                        data_list_test = data_list["test"]
            self.test = self.__create_dataset(
                data_list_test,
                self.dir_input,
            )
            logging.info(f"#tracks: test-{len(data_list_test)}")

    def switch_to_other_epochs_logic(self):
        self.train.use_first_epoch_logic = False
        
    def train_dataloader(self):
        """
        Returns a PyTorch DataLoader for the training data.

        :return: A PyTorch DataLoader object.
        :rtype: torch.utils.data.DataLoader
        """

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=None
            if (not self.enable_context)
            else my_collate,
        )

    def val_dataloader(self):
        """
        Returns a PyTorch DataLoader for the validation set.

        :return: A PyTorch DataLoader object.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None
            if (not self.enable_context)
            else my_collate,
        )

    # Create test dataloader
    def test_dataloader(self):
        """
        Returns a PyTorch DataLoader object for the test dataset.

        :return: DataLoader object with test dataset
        :rtype: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None
            if (not self.enable_context)
            else my_collate,
        )

    def __random_split_list(self, lst, percentage):
        length = len(lst)
        split_index = int(length * percentage / self.batch_size) * self.batch_size

        train_part = lst[:split_index]
        test_part = lst[split_index:]

        return train_part, test_part

    def __create_dataset(self, data_list, dir_input):
        return PathDataset_Context(
            data_list,
            dir_input,
            lr=self.lr,
            bucket_thold=self.bucket_thold,
            enable_context=self.enable_context,
        )

    def __dump_data_list(self, path):
        with open(f"{path}/data_list.json", "w") as fp:
            data_dict = {
                "train": self.train_list,
                "val": self.val_list,
                "test": self.val_list,
            }
            json.dump(data_dict, fp)

    def __read_data_list(self, path):
        f = open(f"{path}/data_list.json", "r")
        return json.loads(f.read())


class PathDataset_Context(Dataset):
    def __init__(
        self,
        data_list,
        dir_input,
        lr,
        bucket_thold,
        enable_context,
    ):
        self.data_list = data_list
        self.use_first_epoch_logic = True
        
        self.dir_input = dir_input
        self.lr = lr
        self.bucket_thold = bucket_thold
        self.enable_context = enable_context
        logging.info(
            f"call PathDataset_Context: bucket_thold={bucket_thold}, enable_context={enable_context}, data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}"
        )
        bdp_dict_db = {}
        bdp_dict_db_output = {}
        for n_hosts in [3,5,7]:
            BDP = 10 * MTU
            bdp_dict_db[n_hosts] = get_size_bucket_list(mtu=MTU, bdp=BDP)
            bdp_dict_db_output[n_hosts] = get_size_bucket_list_output(mtu=MTU, bdp=BDP)
        self.bdp_dict_db = bdp_dict_db
        self.bdp_dict_db_output = bdp_dict_db_output

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])
        
        # load data
        dir_input_tmp = f"{self.dir_input}/{spec}"
        feat_path=f"{dir_input_tmp}/feat{topo_type}.npz"
        
        if not os.path.exists(feat_path) or self.use_first_epoch_logic:
            n_hosts = int(spec.split("_")[2][6:])
            
            size_bucket_list = self.bdp_dict_db[n_hosts]
            size_bucket_list_output = self.bdp_dict_db_output[n_hosts]

            param_data = np.load(f"{dir_input_tmp}/param{topo_type}.npy")
            if param_data[3]==1.0:
                param_data=np.insert(param_data,4,0)
            else:
                param_data=np.insert(param_data,4,1)
                
            param_data=np.insert(param_data,0,[0,0,0])
            param_data[n_hosts//2-1]=1.0
            param_data[3]=BDP_DICT[n_hosts]/MTU
            
            fid=np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
            sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
            flow_src_dst_flowsim = np.load(f"{dir_input_tmp}/fsd.npy")
            
            sizes=sizes_flowsim[fid]
            flow_src_dst=flow_src_dst_flowsim[fid]
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
            sldns = np.divide(fcts, i_fcts)
            
            # find foreground/background flows for flowsim
            flow_idx_target_flowsim = np.logical_and(
                flow_src_dst_flowsim[:, 0] == src_dst_pair_target[0],
                flow_src_dst_flowsim[:, 1] == src_dst_pair_target[1],
            )
            flow_idx_nontarget_flowsim=~flow_idx_target_flowsim
            flow_idx_nontarget_internal_flowsim=np.logical_and(
                flow_src_dst_flowsim[:, 0] != src_dst_pair_target[0],
                flow_src_dst_flowsim[:, 1] != src_dst_pair_target[1],
            )

            # compute propagation delay
            n_links_passed = abs(flow_src_dst_flowsim[:, 0] - flow_src_dst_flowsim[:, 1])+flow_idx_nontarget_flowsim+flow_idx_nontarget_internal_flowsim
            delay_comp=LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,0]]+LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,1]]
            DELAY_PROPAGATION_PERFLOW = get_base_delay_pmn(
                sizes=sizes_flowsim, n_links_passed=n_links_passed, lr_bottleneck=self.lr,flow_idx_target=flow_idx_target_flowsim,flow_idx_nontarget_internal=flow_idx_nontarget_internal_flowsim
            )+delay_comp

            # load sldns from flowsim
            fcts_flowsim = (
                np.load(f"{dir_input_tmp}/fct_flowsim.npy") + DELAY_PROPAGATION_PERFLOW
            )
            i_fcts_flowsim = (
                sizes_flowsim + np.ceil(sizes_flowsim / MTU) * HEADER_SIZE
            ) * BYTE_TO_BIT / self.lr + DELAY_PROPAGATION_PERFLOW
            sldns_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)
            sldns_flowsim = np.clip(sldns_flowsim, a_max=None, a_min=1.0)
            
            # compute sldns from flowsim for each link
            sldns_list = []
            bins = []
            x_len = len(size_bucket_list) + 1
            y_len = len(P99_PERCENTILE_LIST)

            # add the foreground traffic
            sldns_flowsim_target = sldns_flowsim[flow_idx_target_flowsim]
            sldns_list.append(sldns_flowsim_target)
            bins_target = np.digitize(sizes_flowsim[flow_idx_target_flowsim], size_bucket_list)
            bins.append(bins_target)
            
            # add the background traffic
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
                    
            # generate the feature map
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
            
            num_flows_per_cell = num_flows_per_cell.reshape((n_sldns_list, -1)).astype(
                np.float32
            )
            # normalize the number of flows per cell
            num_flows_per_cell = np.divide(num_flows_per_cell, n_sizes_effective)

            # find foreground/background flows for gt
            flow_idx_target = np.logical_and(
                flow_src_dst[:, 0] == src_dst_pair_target[0],
                flow_src_dst[:, 1] == src_dst_pair_target[1],
            )
            # generate the ground truth feature map 
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
            
            np.savez(feat_path, res=res, num_flows_per_cell=num_flows_per_cell, res_output=res_output, num_flows_per_cell_output=num_flows_per_cell_output,n_input=n_input)
        else:
            data = np.load(feat_path)
            res = data["res"]
            num_flows_per_cell = data["num_flows_per_cell"]
            res_output = data["res_output"]
            num_flows_per_cell_output = data["num_flows_per_cell_output"]
            n_input = data["n_input"]
        
        return (
            res,
            num_flows_per_cell,
            res_output,
            num_flows_per_cell_output,
            spec,
            n_input,
            src_dst_pair_target_str,
            # np.array(src_dst_pair_target),
        )
