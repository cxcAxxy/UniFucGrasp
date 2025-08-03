import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchgen.executorch.api.et_cpp import return_type

from HandModel.handmodel import create_hand

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def get_pred_hand_pt(trj_batch,indexs,robot_name):
    # 获取唯一名字（保持原顺序）
    seen = set()
    unique_names = []
    for name in robot_name:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    # 调用 create_hand
    for name in unique_names:
        if name == "shadowhand_motor":
            hand_shaowhand=create_hand("shadowhand_motor")
        elif name == "inspire":
            hand_inspire=create_hand("inspire")
        elif name =="hnu_hand":
            hand_hnuhand=create_hand("hnu_hand")
    hand_points=[]
    for i,trj in enumerate(trj_batch):
        robot_name_=robot_name[i]
        if robot_name_ == "shadowhand_motor":
            trj=trj[:29]
            hand_ = hand_shaowhand.get_transformed_links_pc(trj)
        elif robot_name_ == "inspire":
            trj=trj[:19]
            hand_=hand_inspire.get_transformed_links_pc(trj)
        elif robot_name_ == "hnu_hand":
            trj=trj[:27]
            hand_=hand_hnuhand.get_transformed_links_pc(trj)
        hand_points.append(hand_)

    stack_hand_pt=torch.stack(hand_points)

    stack_hand_pt=stack_hand_pt[:,indexs,:]
    return stack_hand_pt



def split_sample_dict(sample_dict):
    # 获取样本数 N
    N = next(iter(sample_dict.values())).shape[0]

    # 初始化 N 个空字典
    result = [{} for _ in range(N)]

    for link_name, tensor in sample_dict.items():
        for i in range(N):
            result[i][link_name] = tensor[i]  # shape: (P, 3)
    return result


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, network, epoch_idx,robot_name):
        super().__init__()
        self.cfg = cfg
        self.network = network
        self.epoch_idx = epoch_idx
        self.lr = cfg.lr
        self.robot_name = robot_name
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        object_pc = batch['object_pc']
        target_robot_pc = batch['target_robot_pc']
        object_pc = batch['object_pc']
        robot_pc = batch['robot_pc']
        target_q = batch['target_q']
        robot_name=batch['robot_name']

        #需要一个由对应的trj得到重建的函数

        network_output = self.network(
            robot_pc,
            object_pc,
            robot_name,
            target_robot_pc
        )

        trj = network_output['trj']
        mu = network_output['mu']
        logvar = network_output['logvar']

        #通过trj去进行重构损失的建设,测试阶段，随便一点
        loss = 0.

        if self.cfg.loss_kl:
            loss_kl = - 0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            loss_kl = torch.sqrt(1 + loss_kl ** 2) - 1
            loss_kl = loss_kl * self.cfg.loss_kl_weight
            self.log('loss_kl', loss_kl, prog_bar=True)
            loss += loss_kl


        if self.cfg.loss_recon:
            indexs = [200, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950]
            pred_control_points=get_pred_hand_pt(trj,indexs,robot_name)
            gt_control_points=target_robot_pc[:,indexs,:]
            loss_re = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
            loss_re = torch.mean(loss_re)
            self.log('loss_re', loss_re, prog_bar=True)
            loss += loss_re
        #加上一个对应的loss
        if self.cfg.loss_trj:

            #填充的都是0，0-0还是0，对损失没有影响，所以这个地方那个可以不处理
            target_t=target_q[:,:3]
            target_r=target_q[:,3:7]
            target_j=target_q[:,7:]
            pred_t=trj[:,:3]
            pred_r=trj[:,3:7]
            pred_j=trj[:,7:]

            loss_t=nn.L1Loss(reduction='sum')(target_t,pred_t)
            self.log('loss_t', loss_t, prog_bar=True)
            loss_r=nn.L1Loss(reduction='sum')(target_r,pred_r)
            self.log('loss_r', loss_r, prog_bar=True)
            loss_j=nn.L1Loss(reduction='sum')(target_j,pred_j)
            self.log('loss_j', loss_j, prog_bar=True)

            loss_q=10 * loss_j + 1.5 * loss_r + 100 * loss_t
            self.log('loss_trj', loss_q, prog_bar=True)
            loss+=loss_q

        self.log("loss", loss, prog_bar=True)
        return loss


#这些是轻量化库自己调用的

    def on_after_backward(self):
        """
        For unknown reasons, there is a small chance that the gradients in CVAE may become NaN during backpropagation.
        In such cases, skip the iteration.
        """
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = None

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % self.cfg.save_every_n_epoch == 0:
            self.ddp_print(f"Saving state_dict at epoch: {self.epoch_idx}")
            torch.save(self.network.state_dict(), f'{self.cfg.save_dir}/epoch_{self.epoch_idx}.pth')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

