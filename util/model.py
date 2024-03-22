import torch
import torch.distributed as dist

from pytorch_lightning import LightningModule
import torch.nn as nn
from .consts import (
    EPS,
    SIZE_BUCKET_LIST_LABEL,
    SIZE_BUCKET_LIST_LABEL_OUTPUT,
    P99_PERCENTILE_LIST,
)
from .utils import (
    serialize_fp32
)
import numpy as np
import logging
import struct
import os

from .model_llama import Transformer, ModelArgs

class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()
        return

    def forward(self, x):
        return torch.exp(x)
class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, prediction, target, weight):
        # Calculate the weighted L1 loss
        loss = torch.abs(prediction - target) * weight

        # Calculate the mean loss
        mean_loss = torch.mean(loss)

        return mean_loss
class FlowSimTransformer_Base(LightningModule):
    def __init__(
        self,
        n_layer,
        n_head,
        n_embd,
        block_size,
        vocab_size,
        dropout,
        compile,
        loss_fn_type,
        enable_dist,
        enable_val,
        enable_position,
        save_dir=None,
    ):
        super().__init__()
        if loss_fn_type == "l1":
            # self.loss_fn = nn.L1Loss()
            self.loss_fn = WeightedL1Loss()
        elif loss_fn_type == "mse":
            self.loss_fn = nn.MSELoss()
        conf = ModelArgs(
            dim=n_embd,
            n_layers=n_layer,
            n_heads=n_head,
            vocab_size=vocab_size,
            multiple_of = 32,
            max_seq_len=block_size,
            dropout=dropout,
        )
        self.model_transformer = Transformer(conf)
        self.enable_dist = enable_dist
        self.enable_val = enable_val
        self.save_dir = save_dir
        logging.info(
            f"loss_fn: {loss_fn_type}, n_layer: {n_layer}, n_head: {n_head}, n_embd: {n_embd}, block_size: {block_size}, vocab_size: {vocab_size}, dropout: {dropout}, enable_position: {enable_position}, enable_dist: {enable_dist}, enable_val: {enable_val}"
        )
    def export_to_bin_llama_v0(self, filepath):
        """ Original export of llama2.c bin files, i.e. version v0 """
        model=self.model_transformer
        out_file = open(filepath, 'wb')

        # first write out the header
        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        p = model.params
        shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
        # legacy format uses negative/positive vocab size as a shared classifier flag
        if not shared_classifier:
            p.vocab_size = -p.vocab_size
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                        n_kv_heads, p.vocab_size, p.max_seq_len)
        out_file.write(header)

        # next write out the embedding weights
        serialize_fp32(out_file, model.tok_embeddings.weight)
        serialize_fp32(out_file, model.tok_embeddings.bias)
        
        # now all the layers
        # attention weights
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wq.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wk.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wv.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wo.weight)
        # ffn weights
        for layer in model.layers:
            serialize_fp32(out_file, layer.ffn_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w1.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w2.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize_fp32(out_file, model.norm.weight)
        # freqs_cis
        serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
        serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

        # final classifier weights
        if not shared_classifier:
            serialize_fp32(out_file, model.output.weight)

        # write to binary file
        out_file.close()
        print(f"wrote {filepath}")

    def step(self, batch, batch_idx, tag=None):
        return None

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="train")

    def validation_step(self, batch, batch_idx):
        if self.enable_val:
            return self.step(batch, batch_idx, tag="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="test")

class FlowSimTransformer_Path(FlowSimTransformer_Base):
    def __init__(
        self,
        n_layer=4,
        n_head=4,
        n_embd=64,
        block_size=64,
        vocab_size=50257,
        dropout=0.0,
        compile=False,
        loss_fn_type="l1",
        weight_decay=1e-2,
        learning_rate=6e-4,
        betas=[0.9, 0.95],
        batch_size=400,
        enable_masked_loss=False,
        enable_weighted_loss=False,
        enable_context=False,
        hidden_sizes=None,
        activation=nn.ReLU,
        output_activation=nn.Identity,
        enable_dist=False,
        enable_val=True,
        enable_position=True,
        enable_log=False,
        enable_const_opt=True,
        n_params=1,
        save_dir=None,
    ):
        feat_dim = len(SIZE_BUCKET_LIST_LABEL) * len(P99_PERCENTILE_LIST)
        feat_dim+=n_params
        super().__init__(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            vocab_size=feat_dim,
            dropout=dropout,
            compile=compile,
            loss_fn_type=loss_fn_type,
            enable_dist=enable_dist,
            enable_val=enable_val,
            enable_position=enable_position,
            save_dir=save_dir,
        )
        if enable_log:
            output_activation = ExpActivation
            logging.info(f"use ExpActivation")
        self.n_embd = n_embd
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = tuple(betas)
        self.batch_size = batch_size
        self.enable_masked_loss = enable_masked_loss
        self.enable_weighted_loss = enable_weighted_loss
        self.enable_context = enable_context
        self.enable_const_opt=enable_const_opt
        
        input_dim = feat_dim + n_embd if enable_context else feat_dim
      
        output_dim = len(P99_PERCENTILE_LIST) * len(SIZE_BUCKET_LIST_LABEL_OUTPUT)
        
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        self.feat_dim = feat_dim
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dim_list=hidden_sizes
        self.model_mlp = self.mlp(
            sizes=sizes,
            activation=activation,
            output_activation=output_activation,
            # dropout=dropout
        )
        self.y_len=len(P99_PERCENTILE_LIST)
        if enable_const_opt:
            self.const_tensor=nn.Parameter(torch.zeros(self.y_len))
        logging.info(
            f"model: {sizes}, enable_context: {enable_context},enable_const_opt:{enable_const_opt}")

    def mlp(self, sizes, activation, output_activation, dropout=None):
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            if j == 0 and dropout:
                layers += [
                    nn.Linear(sizes[j], sizes[j + 1]),
                    nn.Dropout(dropout),
                    act(),
                ]
            else:
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)
    
    def export_to_bin_mlp(self, filepath="model_mlp.bin"):
        """export the model weights in fp32 into .bin file to be read from C"""
        f = open(filepath, "wb")

        print(f"mlp: {self.input_dim}, {self.hidden_dim_list[0]}, {self.hidden_dim_list[1]}, {self.output_dim}")
        header = struct.pack(
            "iiiii", self.input_dim, self.hidden_dim_list[0], self.hidden_dim_list[1], self.output_dim, self.y_len
        )
        print(header)
        f.write(header)
        # now all the layers
        for layer in self.model_mlp:
            if isinstance(layer, nn.Linear):
                # print(f"{layer.weight.shape}, {layer.bias.shape}")
                # print(
                #     f"[{layer.weight[0,0]}, {layer.weight[-1,-1]}], [{layer.bias[0]}, {layer.bias[-1]}]"
                # )
                serialize_fp32(f,layer.weight)
                serialize_fp32(f,layer.bias)
        if self.enable_const_opt:
            print(f"const_tensor: {self.const_tensor[0]}, {self.const_tensor[-1]}")
            serialize_fp32(f,self.const_tensor)
        # write to binary file
        f.close()
        print(f"wrote {filepath}")
        
    def step(self, batch, batch_idx, tag=None):
        (
            sizebucket_to_sldn_flowsim,
            num_flows_per_cell_flowsim,
            sizebucket_to_sldn,
            num_flows_per_cell,
            spec,
            sizebucket_to_sldn_flowsim_idx,
            src_dst_pair_target_str,
            # src_dst_pair_target,
        ) = batch
        
        if self.enable_const_opt:
            num_flows_per_cell_flowsim=num_flows_per_cell_flowsim.reshape((num_flows_per_cell_flowsim.shape[0],-1,self.y_len)).mean(dim=-1)
            for idx_1 in range(num_flows_per_cell_flowsim.shape[0]):
                for idx_2 in range(num_flows_per_cell_flowsim.shape[1]):
                    if num_flows_per_cell_flowsim[idx_1,idx_2]<EPS:
                        sizebucket_to_sldn_flowsim[idx_1,idx_2*self.y_len:(idx_2+1)*self.y_len]=self.const_tensor
                                    
        if self.enable_context:
            idx_start = 0
            sizebucket_to_sldn_foreground = sizebucket_to_sldn.new(
                len(spec), self.feat_dim
            )
            sizebucket_to_sldn_context = sizebucket_to_sldn.new(
                len(spec), self.n_embd
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
                sizebucket_to_sldn_background, _ = self.model_transformer(
                    tmp[None, :]
                )
                sizebucket_to_sldn_context[i] = torch.mean(
                    sizebucket_to_sldn_background, dim=1
                )
                idx_start += idx_interval

            sizebucket_to_sldn_input = torch.cat(
                [sizebucket_to_sldn_foreground, sizebucket_to_sldn_context], dim=-1
            )
        else:
            sizebucket_to_sldn_foreground = sizebucket_to_sldn_flowsim[:, 0, :]
            sizebucket_to_sldn_input = sizebucket_to_sldn_foreground
        # Instead of sizebucket_to_sldn_est = self.model_mlp(sizebucket_to_sldn_input) + 1.0
        sizebucket_to_sldn_est = self.model_mlp(sizebucket_to_sldn_input)
        sizebucket_to_sldn_est.add_(1.0)  # In-place addition
        
        loss_weights = num_flows_per_cell > 0.0
        
        loss = self.loss_fn(
                torch.div(sizebucket_to_sldn_est, sizebucket_to_sldn),
                torch.ones_like(sizebucket_to_sldn),
                loss_weights
            )
        
        if self.enable_dist:
            self.log(
                f"{tag}_loss_sync",
                loss,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
        else:
            self.log(
                f"{tag}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
        # logging.info(f"step-{batch_idx}-{tag}_loss: {loss}")

        if tag == "test":
            test_dir = f"{self.save_dir}/{spec[0]}_{src_dst_pair_target_str[0]}"
            # logging.info(f"save to {test_dir}")
            os.makedirs(test_dir, exist_ok=True)
            sizebucket_to_sldn_flowsim = sizebucket_to_sldn_flowsim.cpu().numpy()[0]
            sizebucket_to_sldn_est = sizebucket_to_sldn_est.cpu().numpy()[0]
            sizebucket_to_sldn = sizebucket_to_sldn.cpu().numpy()[0]
            num_flows_per_cell = num_flows_per_cell.cpu().numpy()[0]
            # num_flows_per_cell_flowsim_ori=num_flows_per_cell_flowsim_ori.cpu().numpy()[0]
            # num_flows_per_cell_flowsim=num_flows_per_cell_flowsim.cpu().numpy()[0]
            # error = np.divide(
            #     abs(sizebucket_to_sldn_est - sizebucket_to_sldn),
            #     sizebucket_to_sldn,
            #     out=np.zeros_like(sizebucket_to_sldn),
            #     where=sizebucket_to_sldn != 0,
            # )
            # logging.info(
            #     np.round(np.nanmin(error), 3),
            #     np.round(np.nanpercentile(error, 50), 3),
            #     np.round(np.nanmax(error), 3),
            # )
            np.savez(
                f"{test_dir}/res.npz",
                sizebucket_to_sldn_est=sizebucket_to_sldn_est,
                sizebucket_to_sldn_flowsim=sizebucket_to_sldn_flowsim,
                sizebucket_to_sldn=sizebucket_to_sldn,
                num_flows_per_cell=num_flows_per_cell,
                # num_flows_per_cell_flowsim_ori=num_flows_per_cell_flowsim_ori,
                # num_flows_per_cell_flowsim=num_flows_per_cell_flowsim,
            )
        return loss

    def configure_optimizers(self):
        optimizer = self.model_transformer.configure_optimizers(
            self.weight_decay, self.learning_rate, self.betas
        )
        # optimizer_mlp = torch.optim.Adam(
        #     self.model.parameters(), lr=self.learning_rate
        # )
        optimizer.add_param_group(
            {"params": self.model_mlp.parameters(), "weight_decay": 0.0}
        )
        if self.enable_const_opt:
            optimizer.add_param_group(
                {"params": self.const_tensor, "weight_decay": 0.0}
            )
        # return optimizer_transformer,optimizer_mlp
        return optimizer