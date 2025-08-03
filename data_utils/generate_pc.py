import os
import sys
import argparse
import time
import viser
import torch
import trimesh
from HandModel.handmodel import *

def generate_robot_pc(robot_name,links_num_points_all):
    output_path="/home/cxc/Desktop/cxc_hand/data/Pointhand_down"
    hand_name_file=f"{robot_name}.pth"
    output_path=os.path.join(output_path,hand_name_file)
    hand = create_hand(robot_name,1,links_num_points_all)         #创建的时候，保持一致

    server = viser.ViserServer(host='127.0.0.1', port=8080)
    links_pc = hand.mesh_points
    sampled_pc, sampled_pc_index = hand.get_sampled_pc(num_points=links_num_points_all)

    filtered_links_pc = {}
    for link_index, (link_name, points) in enumerate(links_pc.items()):
        mask = [i % links_num_points_all for i in sampled_pc_index
                if link_index * links_num_points_all <= i < (link_index + 1) * links_num_points_all]
        links_pc[link_name] = torch.tensor(points, dtype=torch.float32)
        filtered_links_pc[link_name] = torch.tensor(points[mask], dtype=torch.float32)
        print(f"[{link_name}] original shape: {links_pc[link_name].shape}, filtered shape: {filtered_links_pc[link_name].shape}")

    data = {
        'original': links_pc,
        'filtered': filtered_links_pc
    }
    torch.save(data, output_path)
    print("\nGenerating robot point cloud finished.")

    server.scene.add_point_cloud(
        'point cloud',
        sampled_pc[:, :3].numpy(),
        point_size=0.001,
        point_shape="circle",
        colors=(0, 0, 200)
    )
    while True:
        time.sleep(1)

if __name__ == '__main__':
    generate_robot_pc("hnu_hand",2000)      #注意这个地方的512是总的采样个数，根据逻辑来看，应该是和mesh采样的保持一致，都设置为512
