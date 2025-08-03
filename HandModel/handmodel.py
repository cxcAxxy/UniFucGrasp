
import os

import json
import pytorch_kinematics as pk
import torch
import torch.nn
import transforms3d
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
from utils.utils_hand.rot6d import *
from utils.utils_hand.utils_math import *
from utils.utils_hand.func_utils import *

#对于shadowhand的测试操作
view_mat_mesh = torch.tensor([[0.0, 0.0, 1.0],
                              [-1.0, 0, 0.0],
                              [0.0, -1.0, 0]], device='cuda:0')

class HandModel:
    def __init__(self,robot_name,urdf_path,mesh_path,hand_points_path,link_num_point,remove_links_list,device,scale):
        self.robot_name = robot_name
        self.device = device
        #构建运动学模型
        self.robot = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_path)
        visual = URDF.from_xml_string(open(urdf_path).read())
        self.mesh_verts = {}
        self.mesh_faces = {}
        self.scale=scale
        self.link_num_point = link_num_point
        self.mesh_points={}         #主要的作用是采样

        if os.path.exists(hand_points_path):  # In case of generating robot links pc, the file doesn't exist.
            links_pc_data = torch.load(hand_points_path, map_location=device)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = {}

        #构建mesh
        for i_link, link in enumerate(visual.links):
            #print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                # print(link.visuals[0])，由于urdf可能定义的不同，为了不改变urdf，在这个地方进行一个处理
                if robot_name == 'inspire':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                elif robot_name == 'shadowhand':
                    filename = link.visuals[0].geometry.filename.split('/', 1)[1]
                elif robot_name == 'shadowhand_motor':
                    filename = link.visuals[0].geometry.filename
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)

                if link.name not in remove_links_list and not os.path.exists(hand_points_path):
                    points_sample=mesh.sample(link_num_point)       #采样，均匀采样link_num_point点
                # 这个地方为什么会出现这样的情况，还有待后面验证
                if (type(mesh) == tm.scene.scene.Scene):
                    continue

            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if link.name not in remove_links_list and not os.path.exists(hand_points_path):
                #for train down sample
                self.mesh_points[link.name] = np.array(points_sample) * scale

            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            if link.name not in remove_links_list and not os.path.exists(hand_points_path):
                self.mesh_points[link.name] = np.matmul(rotation, self.mesh_points[link.name].T).T + translation
                self.links_pc[link.name] = np.matmul(rotation, self.mesh_points[link.name].T).T + translation


            self.mesh_faces[link.name] = np.array(mesh.faces)

    def update_kinematics(self, q,robot_name):
        #构建运动学，q==rtj
        #测试专用，后面要改
        #q=torch.tensor(q).to(self.device)
        q = q.clone().detach().to(self.device)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        self.global_translation = q[:, :3]
        self.global_rotation = compute_rotation_matrix_from_quaternion(q[:,3:7])     #注意这个地方，四元数是wxyz的顺序
        if robot_name == 'shadowhand_motor':
            self.global_rotation = self.global_rotation.matmul(view_mat_mesh.to(self.device))
        self.current_status = self.robot.forward_kinematics(q[:, 7:])

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_trimesh_from_q(self, q=None,i=0):
        if q is not None:
            self.update_kinematics(q,self.robot_name)
        scene = tm.Scene()
        parts = {}
        for link_name in self.mesh_verts:
            # 获取当前链节顶点
            v = self.mesh_verts[link_name]  # (N, 3)
            # 变换矩阵，从current_status里取
            trans_matrices = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrices[min(len(trans_matrices) - 1, i)].detach().cpu().numpy()
            # 顶点齐次坐标
            v_homo = np.concatenate([v, np.ones((len(v), 1))], axis=-1)  # (N,4)
            # 应用当前链节的变换矩阵
            transformed_v = (trans_matrix @ v_homo.T).T[..., :3]  # (N,3)
            # 叠加全局旋转和平移
            global_rot = self.global_rotation[i].detach().cpu().numpy()
            global_trans = self.global_translation[i].detach().cpu().numpy()
            transformed_v = (global_rot @ transformed_v.T).T + global_trans
            # 缩放
            transformed_v *= self.scale
            # 获取面信息
            f = self.mesh_faces[link_name]
            # 生成当前链节变换后的三角网格
            part_mesh = tm.Trimesh(vertices=transformed_v, faces=f)
            parts[link_name] = part_mesh
            # 添加到scene
            scene.add_geometry(part_mesh)
        # 合并所有mesh的顶点和面，生成完整的机械手mesh
        vertices = []
        faces = []
        vertex_offset = 0  #ertex_offset是用来记录当前累积的顶点数的
        for geom in scene.geometry.values():
            if isinstance(geom, tm.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)
        return {
            'visual': tm.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }

    def get_transformed_links_pc(self, q=None, links_pc=None):
        """
        Use robot link pc & q value to get point cloud.
        :param q: (7 + DOF,), joint values (quat representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (N, 4), with link index
        """
        if q is None:
            #使用canonical_q来操作
            q=self.get_canonical_q()
        self.update_kinematics(q,self.robot_name)
        if links_pc is None:
            links_pc = self.links_pc
        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]
            se3 = self.current_status[link_name].get_matrix()[0].to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)
        all_pc_se3 = torch.cat(all_pc_se3, dim=0)
        all_pc_se3 = all_pc_se3[:,:3]  #返回xyz即可
        return all_pc_se3

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.robot.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q=torch.cat([torch.zeros(7,dtype=canonical_q.dtype,device=canonical_q.device),canonical_q])  #进行一个拼接操作,7是xyz+wxyz
        return canonical_q

    def get_sampled_pc(self, q=None, num_points=2000):
        """
        :param q: (7 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()
        sampled_pc = self.get_transformed_links_pc(q)
        return farthest_point_sampling(sampled_pc, num_points)

def create_hand(hand_name,scale=1,link_num_points=512):

    hand_assert=json.loads(open("/home/cxc/Desktop/cxc_hand/data/data_hand/hand_assert.json","r").read())
    urdf_path=hand_assert["urdf_path"][hand_name]
    mesh_path=hand_assert["meshes_path"][hand_name]
    hand_points_path=hand_assert["hand_points_path"][hand_name]
    remove_link_list=hand_assert["remove_links"][hand_name]


    hand = HandModel(hand_name,urdf_path,mesh_path,hand_points_path,link_num_points,remove_link_list,'cuda',scale)
    return hand