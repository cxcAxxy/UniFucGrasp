import os
import numpy as np
import sys
import json
import math
import hydra
import random
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from HandModel.handmodel import create_hand

shadowhand_motor_joints_name = ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
                   'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
                   'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
                   'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
                   'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']

def modify_hand_pose(hand_pose, ori_joint_order, target_joint_order):
    """
    Change the order of hand pose from original hand to target hand

    Parameters
    ----------
    hand_pose:(B, `n_dofs`) torch.FloatTensor
        hand joints in original order
    ori_joint_order: list of str
        original joint order
    target_joint_order: list of str
        target joint order

    Returns
    -------
    target_hand_pose: (B, `n_dofs`) torch.FloatTensor
    """
    assert (len(ori_joint_order) == hand_pose.shape[1] and len(target_joint_order) == hand_pose.shape[
        1]), "the shape is different"
    if isinstance(hand_pose, np.ndarray):
        target_hand_pose = np.zeros_like(hand_pose)
    else:
        target_hand_pose = torch.zeros_like(hand_pose, device=hand_pose.device)
    for i in range(len(target_joint_order)):
        flag = False
        for j, joint_name in enumerate(ori_joint_order):
            if target_joint_order[i][-4:] == joint_name[-4:]:
                target_hand_pose[:, i] = hand_pose[:, j]
                flag = True
                break
        assert flag, f"joint name {target_joint_order[i]} not found"
    return target_hand_pose

#上层封装，构建
class HandDataset(Dataset):
    def __init__(
            self,
            robot_name = "shadowhand_motor",
            is_trian =True,
            num_points =1024,
            object_pc_type= 'random',
    ):

        self.robot_name=robot_name

        self.is_train=is_trian
        self.num_points=num_points
        self.object_pc_type=object_pc_type
        if self.robot_name == "shadowhand_motor":
            self.hand_shadowhand_motor=create_hand(robot_name)              #构建对应的hand模型
        elif self.robot_name == "inspire":
            self.hand_inspire=create_hand(robot_name)
        elif self.robot_name == "hnu_hand":
            self.hand_hnu_hand=create_hand(robot_name)
        else:
            self.hand_shadowhand_motor=create_hand("shadowhand_motor")
            self.hand_inspire=create_hand("inspire")
            self.hand_hnu_hand=create_hand("hnu_hand")

        #data地址路径，目前改用绝对路径，后期改变做上层封装
        if self.is_train:
            if self.robot_name == 'shadowhand_motor':
                dataset_path = "/home/cxc/Desktop/cxc_hand/Grasps_Dataset/train/shadowhand_train"
            elif self.robot_name == 'inspire':
                dataset_path = "/home/cxc/Desktop/cxc_hand/Grasps_Dataset/train/inspire_train"
            elif self.robot_name=='hnu_hand':
                dataset_path="/home/cxc/Desktop/cxc_hand/Grasps_Dataset/train/hunand_train"
            else:
                dataset_path="/home/cxc/Desktop/cxc_hand/Grasps_Dataset/train/all"
        else:
            if self.robot_name == 'shadowhand_motor':
                dataset_path = "/home/cxc/Desktop/cxc_hand/Grasps_Dataset/test/shadowhand_test"
            elif self.robot_name == 'inspire':
                dataset_path = "/home/cxc/Desktop/cxc_hand/Grasps_Dataset/test/inspire_test"
            elif self.robot_name=='hnu_hand':
                dataset_path="/home/cxc/Desktop/cxc_hand/Grasps_Dataset/test/hnuhand_test"

        self.mesh_root="/home/cxc/Desktop/cxc_hand/Grasps_Dataset/Obj_Data"
        self.metadata=[]
        self.object_mesh_path=[]
        self.dataname=[]
        self.datalengths=[]
        #抓取数据和对应的object_grasp_point的作用
        for subdir, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(subdir, file)
                    data=np.load(full_path, allow_pickle=True)
                    # 获取上级目录名
                    obj_class=data[0]['obj_class']
                    obj_name=data[0]['obj_name']
                    data_rtj=data[1]['rtj']
                    hand_name = data[0]['hand_name']
                    rtj_length = data[0]['rtj_length']
                    # 拼接 mesh 路径（比如换个目录结构或扩展名）
                    mesh_file=os.path.join(obj_class, obj_name + ".obj")
                    if hand_name == "shadowhand_motor":  # 由于数据集的问题，需要对数据的顺序做一些处理，满足
                        for grasp in data_rtj:  # 对数据进行处理修改
                            grasp=grasp[:rtj_length]
                            quat = grasp[:4]  # wxyz
                            pos = grasp[4:7] * 0.001
                            joints = grasp[7:]
                            quat = torch.from_numpy(quat).float()
                            pos = torch.from_numpy(pos).float()
                            joints = torch.from_numpy(joints)
                            joints = modify_hand_pose(joints.unsqueeze(0), shadowhand_motor_joints_name,
                                                      self.hand_shadowhand_motor.robot.get_joint_parameter_names()).squeeze(0)
                            for_batch=torch.zeros(1)
                            hand_pose = torch.cat([pos, quat, joints,for_batch],dim=0)
                            self.dataname.append(hand_name)
                            self.metadata.append(hand_pose)
                            self.object_mesh_path.append(mesh_file)
                    elif hand_name == "inspire":
                        for grasp in data_rtj:
                            hand_pose = torch.from_numpy(grasp).float()
                            self.metadata.append(hand_pose)
                            self.object_mesh_path.append(mesh_file)
                            self.dataname.append(hand_name)
                    elif hand_name == "hnu_hand":
                        for grasp in data_rtj:
                            hand_pose = torch.from_numpy(grasp).float()
                            self.metadata.append(hand_pose)
                            self.object_mesh_path.append(mesh_file)
                            self.dataname.append(hand_name)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        "Train: sample a batch of data"
        if self.is_train:
            target_q=self.metadata[index]
            #找到对应的target
            mesh_path=self.object_mesh_path[index]
            hand_name=self.dataname[index]
            #找到file.obj
            mesh_path_all=os.path.join(self.mesh_root,mesh_path)

            object_mesh=trimesh.load_mesh(mesh_path_all)
            object_pc=object_mesh.sample(self.num_points)

            if hand_name == "shadowhand_motor":
                target_q_true=target_q[:29]
                robot_pc = self.hand_shadowhand_motor.get_transformed_links_pc()
                target_robot_pc = self.hand_shadowhand_motor.get_transformed_links_pc(q=target_q_true)
            elif hand_name == "inspire":
                target_q_true=target_q[:19]
                robot_pc = self.hand_inspire.get_transformed_links_pc()
                target_robot_pc = self.hand_inspire.get_transformed_links_pc(q=target_q_true)
            elif hand_name == "hnu_hand":
                target_q_true = target_q[:27]
                robot_pc = self.hand_hnu_hand.get_transformed_links_pc()
                target_robot_pc = self.hand_hnu_hand.get_transformed_links_pc(q=target_q_true)


            if isinstance(object_pc, np.ndarray):
                object_pc = torch.from_numpy(object_pc).float()
            elif not isinstance(object_pc, torch.Tensor):
                object_pc = torch.tensor(object_pc).float()

            return {
                'robot_name': hand_name,
                'object_pc': object_pc,
                'robot_pc': robot_pc,
                'target_q': target_q,                #注意这个地方的target_q是有有效部分和无效部分的
                'target_robot_pc': target_robot_pc
            }
        else:
            #构建的就是没有target_pc,对应的就是
            mesh_path = self.object_mesh_path[index]
            #random的方式构建point
            object_mesh = trimesh.load_mesh(mesh_path)
            object_pc, _ = object_mesh.sample(self.num_points)
            robot_pc = self.hand.get_transformed_links_pc()
            if isinstance(object_pc, np.ndarray):
                object_pc = torch.from_numpy(object_pc).float()
            elif not isinstance(object_pc, torch.Tensor):
                object_pc = torch.tensor(object_pc).float()
            return {
                'robot_name': self.robot_name,
                'object_pc': object_pc,
                'robot_pc': robot_pc,
            }
def create_dataloader(cfg, mode="train"):

    if mode == "train":
        is_train=True
    else:
        is_train=False

    dataset_hand=HandDataset(robot_name=cfg.robot_name,
                        is_trian=is_train,num_points=cfg.num_points,
                        object_pc_type=cfg.object_pc_type)

    dataloader=DataLoader(
        dataset_hand,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=is_train,
        drop_last=True
    )
    return dataloader

def main():
    #测试data数据集的构建
    HandDataset_=HandDataset("shadowhand_motor")

if __name__ == '__main__':
    main()