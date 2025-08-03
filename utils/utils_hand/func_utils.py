import os
import sys
import torch

def farthest_point_sampling(point_cloud, num_points=1024):
    """
    :param point_cloud: (N, 3) or (N, 4), point cloud (with link index)
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 4), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] == 4:
        point_cloud = point_cloud[:, :3]

    selected_indices = [0]
    distances = torch.norm(point_cloud - point_cloud[selected_indices[-1]], dim=1)
    for _ in range(num_points - 1):
        farthest_point_idx = torch.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = torch.norm(point_cloud - point_cloud[farthest_point_idx], dim=1)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    return sampled_point_cloud, selected_indices
