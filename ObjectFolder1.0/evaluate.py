import os
import datetime
import sys
import torch
import numpy as np
import imageio
import json
import random
import time
from tqdm import tqdm, trange
from scipy.spatial import KDTree

import indirect_utils
from load_osf import load_osf_data
from intersect import compute_object_intersect_tensors
from ray_utils import transform_rays
from run_osf_helpers import *
from scatter import scatter_coarse_and_fine
import shadow_utils

import VisionNet_utils
import AudioNet_utils
import AudioNet_model
import TouchNet_utils
import TouchNet_model

from scipy.io.wavfile import write
import librosa
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():
    import configargparse
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    #parser = configargparse.ArgumentParser()
    parser.add_argument("--object_file_path", type=str, required=True, help='ObjectFile path')
    parser.add_argument('--config', is_config_file=True, help='config file path')

    # VisionNet options
    parser.add_argument("--vision_test_file_path", default='data/vision_demo.npy', help='The path of the testing file for vision, which should be a npy file.')
    parser.add_argument("--vision_results_dir", type=str, default='./results/vision/', help='The path of the vision results directory to save rendered images.')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # AudioNet options
    parser.add_argument('--audio_vertices_file_path', default='./data/audio_demo_vertices.npy', help='The path of the testing vertices file for audio, which should be a npy file.')
    parser.add_argument('--audio_forces_file_path', default='./data/forces.npy', help='The path of forces file for audio, which should be a npy file.')
    parser.add_argument('--audio_batchSize', type=int, default=10000, help='input batch size')
    parser.add_argument('--audio_results_dir', type=str, default='./results/audio/', help='The path of audio results directory to save rendered impact sounds as .wav files.')

    # TouchNet options
    parser.add_argument('--touch_vertices_file_path', default='./data/touch_demo_vertices.npy', help='The path of the testing vertices file for touch, which should be a npy file.')
    parser.add_argument('--touch_batchSize', type=int, default=10000, help='input batch size')
    parser.add_argument('--touch_results_dir', type=str, default='./results/touch/', help='The path of the touch results directory to save rendered tactile RGB images.')

    return parser


def VisionNet_eval(args):

    args.secondary_chunk = args.chunk

    metadata, render_metadata = None, None
    near = 0.01
    far = 4

    poses, hwf, i_split, metadata = load_osf_data(args.vision_test_file_path)
    i_test = i_split[0]

    render_poses = np.array(poses[i_test])

    # Create dummy metadata if not loaded from dataset.
    if metadata is None:
        metadata = torch.tensor([[0, 0, 1]] * len(images), dtype=torch.float)  # [N, 3]
    if render_metadata is None:
        render_metadata = metadata

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models, optimizer = VisionNet_utils.create_nerf(
        args, metadata, render_metadata)
    global_step = start

    bds_dict = {
        'near': torch.tensor(near).float(),
        'far': torch.tensor(far).float(),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_kwargs_test['render_metadata'] = metadata

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    with torch.no_grad():
        images = None

        testsavedir = args.vision_results_dir
        os.makedirs(testsavedir, exist_ok=True)
        print('Begin rendering images in ', testsavedir)
        rgbs, _ = VisionNet_utils.render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                gt_imgs=images, savedir=testsavedir,
                                c2w_staticcam=None, render_start=None,
                                render_end=None)
        print('Done rendering images in ', testsavedir)


def AudioNet_eval(args):
    dim = 32
    audio_sampling_rate = 16000
    audio_window_size = 400
    audio_hop_size = 160
    audio_stft_time_dim = 201
    audio_stft_freq_dim = 257
    audio_network_depth = 8
    xyz = np.load(args.audio_vertices_file_path)
    forces = np.load(args.audio_forces_file_path)
    xyz = xyz[0:1,:]
    forces = forces[0:1]
    N = xyz.shape[0]
    print(N)
    #N: number of data
    #D: number of dimension
    #C: number of channels, real and img
    #F: number of frequency
    #T: number of timestamps
    N, D, C, F, T = N, 3, 2, audio_stft_freq_dim, audio_stft_time_dim

    checkpoint = torch.load(args.object_file_path)
    normalizer_dic = checkpoint['AudioNet']['normalizer']
    voxel_vertex = checkpoint['AudioNet']['voxel_vertex']
    vert_tree = KDTree(voxel_vertex)
    translation = checkpoint['AudioNet']['translation']
    scale = checkpoint['AudioNet']['scale']

    k = 4 # Average over 4 nearest neighbors
    xyz_in_voxel = np.zeros((4, N, 3))
    for i in range(N):
        obj_coordinates = xyz[i]
        binvox_coordinates = AudioNet_utils.transform_mesh_collision_binvox(obj_coordinates, translation, scale)
        coordinates_in_voxel = binvox_coordinates * dim
        voxel_verts_index = vert_tree.query(coordinates_in_voxel, k)[1]
        for j in range(k):
            xyz_in_voxel[j, i] = voxel_vertex[voxel_verts_index[j]]

    xyz_in_voxel = np.repeat(xyz_in_voxel.reshape((4, N, 1, 3)), F * T, axis=2)
    #normalize xyz_in_voxel to [-1, 1]
    xyz_in_voxel_min = xyz_in_voxel.min()
    xyz_in_voxel_max = xyz_in_voxel.max()
    xyz_in_voxel = (xyz_in_voxel - xyz_in_voxel_min) / (xyz_in_voxel_max - xyz_in_voxel_min)

    spec_comps_f1_min = normalizer_dic['f1_min']
    spec_comps_f1_max = normalizer_dic['f1_max']
    spec_comps_f2_min = normalizer_dic['f2_min']
    spec_comps_f2_max = normalizer_dic['f2_max']
    spec_comps_f3_min = normalizer_dic['f3_min']
    spec_comps_f3_max = normalizer_dic['f3_max']

    #initialize frequency and time features
    freq_feats = np.repeat(np.repeat(np.arange(F).reshape((F, 1)), T, axis=1).reshape((1, 1, F, T)), N, axis=0)
    time_feats = np.repeat(np.repeat(np.arange(T).reshape((1, T)), F, axis=0).reshape((1, 1, F, T)), N, axis=0)

    #normalize frequency and time features to [-1, 1]
    freq_feats_min = freq_feats.min()
    freq_feats_max = freq_feats.max()
    time_feats_min = time_feats.min()
    time_feats_max = time_feats.max()
    freq_feats = (freq_feats - freq_feats_min) / (freq_feats_max - freq_feats_min)
    time_feats = (time_feats - time_feats_min) / (time_feats_max - time_feats_min)

    data_x = np.concatenate((freq_feats, time_feats), axis=1)
    data_y = np.concatenate((freq_feats, time_feats), axis=1)
    data_z = np.concatenate((freq_feats, time_feats), axis=1)
    data_x = np.transpose(data_x.reshape((N, 2, -1)), axes = [0, 2, 1])
    data_y = np.transpose(data_y.reshape((N, 2, -1)), axes = [0, 2, 1])
    data_z = np.transpose(data_z.reshape((N, 2, -1)), axes = [0, 2, 1])
    data_x = np.repeat(data_x.reshape((1, N, -1, 2)), k, axis=0)
    data_y = np.repeat(data_y.reshape((1, N, -1, 2)), k, axis=0)
    data_z = np.repeat(data_z.reshape((1, N, -1, 2)), k, axis=0)

    #Now concatenate xyz and feats to get final feats matrix as [x, y, z, f, t, real, img] 
    feats_x = np.concatenate((xyz_in_voxel, data_x), axis=3).reshape((-1, 5))
    feats_y = np.concatenate((xyz_in_voxel, data_y), axis=3).reshape((-1, 5))
    feats_z = np.concatenate((xyz_in_voxel, data_z), axis=3).reshape((-1, 5))

    embed_fn, input_ch = AudioNet_model.get_embedder(10, 0)
    model = AudioNet_model.AudioNeRF(D = audio_network_depth, input_ch = input_ch)
    state_dic = checkpoint['AudioNet']["model_state_dict"]
    state_dic = AudioNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = nn.DataParallel(model).to(device)
    model.eval()
    loss_fn = torch.nn.MSELoss(reduction='mean')

    start_time = time.time()
    preds_x = np.zeros((feats_x.shape[0], 2))
    preds_y = np.zeros((feats_y.shape[0], 2))
    preds_z = np.zeros((feats_z.shape[0], 2))
    N_rand = args.audio_batchSize

    print("Begin rendering impact sounds in ", args.audio_results_dir)
    for i in trange(feats_x.shape[0] // N_rand + 1):
        curr_feats_x = torch.Tensor(feats_x[i*N_rand:(i+1)*N_rand]).to(device)
        curr_feats_y = torch.Tensor(feats_y[i*N_rand:(i+1)*N_rand]).to(device)
        curr_feats_z = torch.Tensor(feats_z[i*N_rand:(i+1)*N_rand]).to(device)
        embedded_x = embed_fn(curr_feats_x)
        embedded_y = embed_fn(curr_feats_y)
        embedded_z = embed_fn(curr_feats_z)
        results_x, results_y, results_z = model(embedded_x, embedded_y, embedded_z)
        
        preds_x[i*N_rand:(i+1)*N_rand, :] = results_x.detach().cpu().numpy()
        preds_y[i*N_rand:(i+1)*N_rand, :] = results_y.detach().cpu().numpy()
        preds_z[i*N_rand:(i+1)*N_rand, :] = results_z.detach().cpu().numpy()

    preds_x = preds_x * (spec_comps_f1_max - spec_comps_f1_min) + spec_comps_f1_min
    preds_y = preds_y * (spec_comps_f2_max - spec_comps_f2_min) + spec_comps_f2_min
    preds_z = preds_z * (spec_comps_f3_max - spec_comps_f3_min) + spec_comps_f3_min
    preds_x = np.transpose(preds_x.reshape((k, N, -1, 2)), axes = [0, 1, 3, 2]).reshape((k, N, 1, C, F, T))
    preds_y = np.transpose(preds_y.reshape((k, N, -1, 2)), axes = [0, 1, 3, 2]).reshape((k, N, 1, C, F, T))
    preds_z = np.transpose(preds_z.reshape((k, N, -1, 2)), axes = [0, 1, 3, 2]).reshape((k, N, 1, C, F, T))

    #save evaluation results
    os.makedirs(args.audio_results_dir, exist_ok=True)

    for i in trange(N):
        force_x, force_y, force_z = forces[i]
        signal = np.zeros(audio_sampling_rate*2)
        for j in range(k):
            spec_x = preds_x[j, i, 0, 0, :, :] + preds_x[j, i, 0, 1, :, :] * 1j
            signal_x = librosa.istft(spec_x, hop_length=audio_hop_size, win_length=audio_window_size, length=audio_sampling_rate*2)
            spec_y = preds_y[j, i, 0, 0, :, :] + preds_y[j, i, 0, 1, :, :] * 1j
            signal_y = librosa.istft(spec_y, hop_length=audio_hop_size, win_length=audio_window_size, length=audio_sampling_rate*2)
            spec_z = preds_z[j, i, 0, 0, :, :] + preds_z[j, i, 0, 1, :, :] * 1j
            signal_z = librosa.istft(spec_z, hop_length=audio_hop_size, win_length=audio_window_size, length=audio_sampling_rate*2)
            temp = signal_x * force_x + signal_y * force_y + signal_z * force_z
            signal += temp
        signal = signal / np.abs(signal).max()
        end_time = time.time()
        print(end_time - start_time)
        # Write WAV file
        output_path = os.path.join(args.audio_results_dir, str(i+1) + '.wav')
        write(output_path, audio_sampling_rate, signal.astype(np.float32))
    print('Done rendering impact sounds in ', args.audio_results_dir)


def TouchNet_eval(args):
    touch_network_depth = 8

    xyz = np.load(args.touch_vertices_file_path)

    #N: number of data
    #C: channels
    #W: Width dimension
    #H: Height dimension
    #N, C, W, H = touch_images.shape
    N, C, W, H = xyz.shape[0], 3, 160, 120

    #initialize frequency and time features
    w_feats = np.repeat(np.repeat(np.arange(W).reshape((W, 1)), H, axis=1).reshape((1, 1, W, H)), N, axis=0)
    h_feats = np.repeat(np.repeat(np.arange(H).reshape((1, H)), W, axis=0).reshape((1, 1, W, H)), N, axis=0)

    checkpoint = torch.load(args.object_file_path)

    #normalize frequency and time features to [-1, 1]
    w_feats_min = w_feats.min()
    w_feats_max = w_feats.max()
    h_feats_min = h_feats.min()
    h_feats_max = h_feats.max()
    w_feats = 2 * ((w_feats - w_feats_min) / w_feats_max) - 1
    h_feats = 2 * ((h_feats - h_feats_min) / h_feats_max) - 1

    data_x = np.concatenate((w_feats, h_feats), axis=1)
    data_x = np.transpose(data_x.reshape((N, 2, -1)), axes = [0, 2, 1])

    xyz = np.repeat(xyz.reshape((N, 1, 3)), W * H, axis=1)

    #normalize xyz to [-1, 1]
    xyz_min = xyz.min()
    xyz_max = xyz.max()
    xyz = 2 * ((xyz - xyz_min) / xyz_max) - 1

    #Now concatenate xyz and feats to get final feats matrix as [x, y, z, w, h, r, g, b] 
    data= np.concatenate((xyz, data_x), axis=2).reshape((-1, 5))
    feats = data

    embed_fn, input_ch = TouchNet_model.get_embedder(10, 0)
    model = TouchNet_model.NeRF(D = touch_network_depth, input_ch = input_ch, output_ch = 3)
    state_dic = checkpoint['TouchNet']["model_state_dict"]
    state_dic = TouchNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = nn.DataParallel(model).to(device)
    model.eval()
    loss_fn = torch.nn.MSELoss(reduction='mean')

    preds = np.zeros((feats.shape[0], 3))
    N_rand = args.touch_batchSize

    print("Begin rendering tactile images in ", args.touch_results_dir)
    start_time = time.time()
    for i in trange(feats.shape[0] // N_rand + 1):
        curr_feats = torch.Tensor(feats[i*N_rand:(i+1)*N_rand]).to(device)
        embedded = embed_fn(curr_feats)
        results = model(embedded)            
        preds[i*N_rand:(i+1)*N_rand, :] = results.detach().cpu().numpy()
    end_time = time.time()
    print(end_time - start_time)
    
    preds = (((preds + 1) / 2) * 255)
    preds = np.transpose(preds.reshape((N, -1, 3)), axes = [0, 2, 1]).reshape((N, C, W, H))
    preds = np.clip(np.rint(preds), 0, 255).astype(np.uint8)
    preds = preds.transpose(0,2,3,1)

    os.makedirs(args.touch_results_dir, exist_ok=True)
    #save evaluation results
    for i in trange(N):
        filename = os.path.join(args.touch_results_dir, '{}.png'.format(i+1))
        imageio.imwrite(filename, preds[i])
    print("Done rendering tactile images in ", args.touch_results_dir)

if __name__ =='__main__':
    parser = config_parser()
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    VisionNet_eval(args=args)
    AudioNet_eval(args=args)
    TouchNet_eval(args=args)
