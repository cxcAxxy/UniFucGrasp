import os
import sys
import torch
import torch.nn as nn

from model.encoder import Encoder, CvaeEncoder
from model.transformer import Transformer
from model.latent_encoder import LatentEncoder
from model.mlp import MLPKernel, ShadowHandMLP, InspireHandMLP,HnuHandMLP
from collections import defaultdict

def pad_to_30(tensor):
    if tensor.shape[1] < 30:
        pad_size = 30 - tensor.shape[1]
        pad = torch.zeros(tensor.shape[0], pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=1)
    return tensor



def create_encoder_network(emb_dim, device=torch.device('cpu')) -> nn.Module:
    encoder = Encoder(emb_dim=emb_dim)
    return encoder


class Network(nn.Module):
    def __init__(self, cfg, mode,robot_name):
        super(Network, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.robot_name = robot_name
        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim)

        # CVAE encoder
        self.point_encoder = CvaeEncoder(emb_dims=cfg.emb_dim, output_channels=2 * cfg.latent_dim, feat_dim=cfg.emb_dim)
        self.latent_encoder = LatentEncoder(in_dim=2*cfg.latent_dim, dim=4*cfg.latent_dim, out_dim=cfg.latent_dim)

        #self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)
        #根据robot_name创建
        if self.robot_name =="shadowhand_motor":
            self.ShadowHandMLP=ShadowHandMLP()
        elif self.robot_name =="inspire":
            self.InspireHandMLP=InspireHandMLP()
        elif self.robot_name =="hnu_hand":
            self.HnuHandMLP=HnuHandMLP()
        else:
            self.ShadowHandMLP = ShadowHandMLP()
            self.InspireHandMLP = InspireHandMLP()
            self.HnuHandMLP = HnuHandMLP()

    def forward(self, robot_pc, object_pc,robot_name,target_pc=None):
        #前向函数的计算
        if self.cfg.center_pc:  # zero-mean the robot point cloud
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

            # point cloud encoder
        robot_embedding = self.encoder_robot(robot_pc)
        object_embedding = self.encoder_object(object_pc)

        #创建编码器
        # point cloud transformer
        transformer_robot_outputs = self.transformer_robot(robot_embedding, object_embedding)
        transformer_object_outputs = self.transformer_object(object_embedding, robot_embedding)
        robot_embedding_tf = robot_embedding + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding + transformer_object_outputs["src_embedding"]

        # CVAE encoder
        if self.mode == 'train':
            grasp_pc = torch.cat([target_pc, object_pc], dim=1)
            grasp_emb = torch.cat([robot_embedding_tf, object_embedding_tf], dim=1)
            latent = self.point_encoder(torch.cat([grasp_pc, grasp_emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)
        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], self.cfg.latent_dim).to(robot_pc.device)

        #不是直接从z，而是应该拼接对应的原始变量进行操作，不然效果很差
        #得到z变量之后，应该是要拼接对应的手部信息和物体点云信息。
        robot_embedding_avg=torch.max(robot_embedding_tf,dim=1)[0]
        object_embedding_avg = torch.max(object_embedding_tf,dim=1)[0]

        #这个地方的处理还是有一些粗糙，后续可以加大层数或者是其他的。
        x=torch.cat([robot_embedding_avg, object_embedding_avg,z], dim=1) #(B,2176)

        grouped_data = defaultdict(list)

        for name, data in zip(robot_name, x):
            grouped_data[name].append(data)

        name_to_tensor = {}

        for k in grouped_data:
            grouped_data[k] = torch.stack(grouped_data[k])
        if len(grouped_data["shadowhand_motor"])> 0:
            grouped_data["shadowhand_motor"]=torch.tensor(grouped_data["shadowhand_motor"]).to(robot_pc.device)
            trj_shadowhand_motor = self.ShadowHandMLP(grouped_data["shadowhand_motor"])
            trj_shadowhand_motor = pad_to_30(trj_shadowhand_motor)
            name_to_tensor["shadowhand_motor"] = trj_shadowhand_motor

        if len(grouped_data["inspire"])>0:
            grouped_data["inspire"]=torch.tensor(grouped_data["inspire"]).to(robot_pc.device)
            trj_inspire = self.InspireHandMLP(grouped_data["inspire"])
            trj_inspire = pad_to_30(trj_inspire)
            name_to_tensor["inspire"] = trj_inspire

        if len(grouped_data["hnu_hand"])>0:
            grouped_data["hnu_hand"] = torch.tensor(grouped_data["hnu_hand"]).to(robot_pc.device)
            trj_hnuhand = self.HnuHandMLP(grouped_data["hnu_hand"])
            trj_hnuhand = pad_to_30(trj_hnuhand)
            name_to_tensor["hnu_hand"] = trj_hnuhand


        # 重新按 robot_name 顺序拼接
        index_counters = defaultdict(int)
        final_list = []

        for name in robot_name:
            if name not in name_to_tensor or name_to_tensor[name].numel() == 0:
                continue  # 如果该手类型没有数据，则跳过

            idx = index_counters[name]
            if idx >= name_to_tensor[name].shape[0]:
                continue  # 防止索引越界（保险检查）

            final_list.append(name_to_tensor[name][idx])
            index_counters[name] += 1


        final_tensor = torch.stack(final_list)  # shape [4, 30]


        outputs = {
            'trj': final_tensor,
            'mu': mu,
            'logvar': logvar,
        }
        return outputs

def create_network(cfg, mode,robot_name):
    network = Network(
        cfg=cfg,
        mode=mode,
        robot_name=robot_name
    )
    return network
