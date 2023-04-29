import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_rays, safe_normalize

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    segs = [trimesh.load_path(np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])) for pos, dir in zip(poses, dirs)]
    
    for seg, dir in zip(segs, dirs):
        seg.colors = DIR_COLORS[[dir]].repeat(len(seg.entities), 0)
        objects.append(seg)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)

    targets = 0

    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs

def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)

    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs    

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        self.opt = opt
        self.device = device
        self.type = type
        self.H = H
        self.W = W
        self.size = size
        self.training = self.type in ['train', 'all']
        self.cx = self.H / 2
        self.cy = self.W / 2

    def collate(self, index):
        B = len(index)
        if self.training:
            poses, dirs = rand_poses(B, self.device, radius_range=self.opt.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate)
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
        else:
            phi = (index[0] / self.size) * 360
            poses, dirs = circle_poses(self.device, radius=self.opt.radius_range[1] * 1.2, theta=60, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
            fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        
        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            loader = torch.nn.DataParallel(loader, device_ids=list(range(torch.cuda.device_count())))
        
        return loader