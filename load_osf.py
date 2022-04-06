"""Data loader for OSF data."""

import os
import torch
import numpy as np
import imageio 
import json

import cam_utils


trans_t = lambda t: torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]
], dtype=torch.float)

rot_phi = lambda phi: torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]
], dtype=torch.float)

rot_theta = lambda th: torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]
], dtype=torch.float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def coordinates_to_c2w(x, y, z, r=2.5):
    theta = np.arccos(z / r)
    phi = np.arctan2(x, -y)
    Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    R = Rz @ Rx
    c2w = R.tolist()
    c2w[0].append(x)
    c2w[1].append(y)
    c2w[2].append(z)
    c2w.append([0., 0., 0., 1.])
    #c2w = np.array(c2w).astype(np.float32)
    return c2w

def convert_cameras_to_nerf_format(anno):
    """
    Args:
        anno: List of annotations for each example. Each annotation is represented by a
            dictionary that must contain the key `RT` which is the world-to-camera
            extrinsics matrix with shape [3, 4], in [right, down, forward] coordinates.

    Returns:
        c2w: [N, 4, 4] np.float32. Array of camera-to-world extrinsics matrices in
            [right, up, backwards] coordinates.
    """
    c2w_list = []
    for a in anno:
        # Convert from w2c to c2w.
        w2c = np.array(a['RT'] + [[0.0, 0.0, 0.0, 1.0]])
        c2w = cam_utils.w2c_to_c2w(w2c)

        # Convert from [right, down, forwards] to [right, up, backwards]
        c2w[:3, 1] *= -1  # down -> up
        c2w[:3, 2] *= -1  # forwards -> back
        c2w_list.append(c2w)
    c2w = np.array(c2w_list)
    print("c2w: ", c2w)
    return c2w


def load_osf_data(test_file_path):

    all_poses = []
    all_metadata = []
    counts = [0]
    test_file = np.load(test_file_path)
    N = test_file.shape[0]
    for i in range(N):
        cx, cy, cz, lx, ly, lz = test_file[i]
        poses = coordinates_to_c2w(cx, cy, cz)
        metadata = np.array([[lx, ly, lz]]).astype(np.float32)
        all_poses.append(poses)
        all_metadata.append(metadata)

    poses = np.array(all_poses).astype(np.float32)

    metadata = np.concatenate(all_metadata, 0)
    counts.append(N)
    i_split = [np.arange(counts[0], counts[1])]

    H, W, focal = 256, 256, 355.5555419921875

    return poses, [H, W, focal], i_split, metadata
