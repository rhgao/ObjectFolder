import os
import datetime
import sys
import torch
import torch.nn as nn
import numpy as np
import imageio
import json
import random
import time
from tqdm import tqdm, trange
import scipy
import librosa
from scipy.io.wavfile import write
from scipy.spatial import KDTree
import ddsp_torch as ddsp
import itertools
from taxim_render import TaximRender
from PIL import Image
import argparse
from load_osf import load_osf_data
import AudioNet_utils
import AudioNet_model
import TouchNet_utils
import TouchNet_model
import VisionNet_utils
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="vision,audio,touch")
    parser.add_argument("--object_file_path", type=str, default="demo/ObjectFile.pth", help='ObjectFile path')
    parser.add_argument("--KiloOSF", action="store_true")

    # VisionNet options
    parser.add_argument("--vision_test_file_path", default='data/vision_demo.npy', help='The path of the testing file for vision, which should be a npy file.')
    parser.add_argument("--vision_results_dir", type=str, default='./results/vision/', help='The path of the vision results directory to save rendered images.')

    # AudioNet options
    parser.add_argument('--audio_vertices_file_path', default='./data/audio_demo_vertices.npy', help='The path of the testing vertices file for audio, which should be a npy file.')
    parser.add_argument('--audio_forces_file_path', default='./data/forces.npy', help='The path of forces file for audio, which should be a npy file.')
    parser.add_argument("--audio_results_dir", type=str, default='./results/audio/', help='The path of the audio results directory to save impact sounds.')

    # TouchNet options
    parser.add_argument('--touch_vertices_file_path', default='./data/touch_demo_vertices.npy', help='The path of the testing vertices file for touch, which should be a npy file.')
    parser.add_argument('--touch_gelinfo_file_path', default='./data/touch_demo_gelinfo.npy', help='The path of the gel configurations for touch, which should be a npy file.')
    parser.add_argument('--touch_results_dir', type=str, default='./results/touch/', help='The path of the touch results directory to save rendered tactile RGB images.')

    return parser


def VisionNet_eval(args):

    checkpoint = torch.load(args.object_file_path)
    cfg = checkpoint['VisionNet']['cfg']

    metadata, render_metadata = None, None
    background_color = torch.ones(3, dtype=torch.float, device=device)

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
    intrinsics = CameraIntrinsics(int(H), int(W), focal, focal, W * .5, H * .5)

    render_kwargs_train = {
        'perturb' : cfg['perturb'],
        'N_samples' : cfg['num_samples_per_ray'],
        'N_importance' : cfg['num_importance_samples_per_ray'],
        'use_viewdirs': True,
        'use_lightdirs': True,
        'white_bkgd' : cfg['blender_white_background'],
        'raw_noise_std' : cfg['raw_noise_std'],
        'near' : cfg['near'],
        'far' : cfg['far'],
        'metadata': metadata,
        'render_metadata': metadata,
        'random_direction_probability': cfg['random_direction_probability'],
        'von_mises_kappa': cfg['von_mises_kappa'],
        'background_color': background_color,
        'lightdirs_method': 'metadata',
        'cfg': cfg
    }

    ConfigManager.init(cfg)

    if args.KiloOSF:
        import kilonerf_cuda
        from local_distill import create_multi_network_fourier_embedding, has_flag, create_multi_network
        kilonerf_cuda.init_stream_pool(16)
        kilonerf_cuda.init_magma()

        position_num_input_channels, position_fourier_embedding = create_multi_network_fourier_embedding(1, cfg['num_frequencies'])
        direction_num_input_channels, direction_fourier_embedding = create_multi_network_fourier_embedding(1, cfg['num_frequencies_direction'])
        light_num_input_channels, light_fourier_embedding = create_multi_network_fourier_embedding(1, cfg['num_frequencies_light'])

        root_nodes = occupancy_grid = None

        res = cfg['fixed_resolution']
        network_resolution = torch.tensor(res, dtype=torch.long, device=torch.device('cpu'))
        num_networks = res[0] * res[1] * res[2]
        model = multi_network = create_multi_network(num_networks, position_num_input_channels, direction_num_input_channels, light_num_input_channels, 4, 'multimatmul_differentiable', cfg).to(device)

        global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(torch.device('cpu'))
        global_domain_size = global_domain_max - global_domain_min
        network_voxel_size = global_domain_size / network_resolution

        # Determine bounding boxes (domains) of all networks. Required for global to local coordinate conversion.
        domain_mins = []
        domain_maxs = []
        for coord in itertools.product(*[range(r) for r in res]):
            coord = torch.tensor(coord, device=torch.device('cpu'))
            domain_min = global_domain_min + network_voxel_size * coord
            domain_max = domain_min + network_voxel_size
            domain_mins.append(domain_min.tolist())
            domain_maxs.append(domain_max.tolist())
        domain_mins = torch.tensor(domain_mins, device=device)
        domain_maxs = torch.tensor(domain_maxs, device=device)
        occupancy_grid = checkpoint['VisionNet']['occupancy_grid']

        additional_kwargs = {
            'root_nodes': root_nodes,
            'position_fourier_embedding': position_fourier_embedding,
            'direction_fourier_embedding': direction_fourier_embedding,
            'light_fourier_embedding': light_fourier_embedding,
            'multi_network': multi_network,
            'domain_mins': domain_mins,
            'domain_maxs': domain_maxs,
            'occupancy_grid': occupancy_grid,
            'debug_network_color_map': None
        }
    else:
        model, embed_fn, embeddirs_fn, embedlights_fn = create_nerf(cfg)
        model = model.to(device)
        network_query_fn = lambda inputs, viewdirs, lightdirs, network_fn : VisionNet_utils.run_network(inputs, viewdirs, lightdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    embedlights_fn=embedlights_fn,
                                                                    netchunk=cfg['network_chunk_size'])

        additional_kwargs = {
            'network_query_fn' : network_query_fn,
            'network_fn' : model
        }

    render_kwargs_train.update(additional_kwargs)
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = cfg['llff_lindisp']

    render_kwargs_test = render_kwargs_train.copy()
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['random_direction_probability'] = -1
    render_kwargs_test['von_mises_kappa'] = -1

    model.load_state_dict(checkpoint['VisionNet']['model_state_dict'])

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    model.eval()
    with torch.no_grad():
        images = None
        testsavedir = args.vision_results_dir
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, _ = VisionNet_utils.render_path(render_poses, intrinsics, cfg['chunk_size'], render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=cfg['render_factor'])

def AudioNet_eval(args):

    checkpoint = torch.load(args.object_file_path)
    normalizer_dic = checkpoint['AudioNet']['normalizer']
    gains_f1_min = normalizer_dic['f1_min']
    gains_f1_max = normalizer_dic['f1_max']
    gains_f2_min = normalizer_dic['f2_min']
    gains_f2_max = normalizer_dic['f2_max']
    gains_f3_min = normalizer_dic['f3_min']
    gains_f3_max = normalizer_dic['f3_max']
    xyz_min = normalizer_dic['xyz_min']
    xyz_max = normalizer_dic['xyz_max']
    freqs = checkpoint['AudioNet']['frequencies']
    damps = checkpoint['AudioNet']['dampings']

    forces = np.load(args.audio_forces_file_path)

    xyz = np.load(args.audio_vertices_file_path).reshape((-1, 3))
    # normalize xyz to [-1, 1]
    xyz = (xyz - xyz_min) / (xyz_max - xyz_min)

    N = xyz.shape[0]
    G = freqs.shape[0]

    embed_fn, input_ch = AudioNet_model.get_embedder(10, 0)
    model = AudioNet_model.AudioNeRF(D=8, input_ch=input_ch, output_ch=G)
    state_dic = checkpoint['AudioNet']["model_state_dict"]
    state_dic = AudioNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = nn.DataParallel(model).to(device)
    model.eval()

    preds_gain_x = torch.zeros((N, G)).to(device)
    preds_gain_y = torch.zeros((N, G)).to(device)
    preds_gain_z = torch.zeros((N, G)).to(device)

    batch_size = 1024

    for i in trange(N // batch_size + 1):
        curr_x = torch.Tensor(xyz[i*batch_size:(i+1)*batch_size]).to(device)
        curr_y = torch.Tensor(xyz[i*batch_size:(i+1)*batch_size]).to(device)
        curr_z = torch.Tensor(xyz[i*batch_size:(i+1)*batch_size]).to(device)
        embedded_x = embed_fn(curr_x)
        embedded_y = embed_fn(curr_y)
        embedded_z = embed_fn(curr_z)
        results_x, results_y, results_z = model(embedded_x, embedded_y, embedded_z)

        preds_gain_x[i*batch_size:(i+1)*batch_size] = results_x
        preds_gain_y[i*batch_size:(i+1)*batch_size] = results_y
        preds_gain_z[i*batch_size:(i+1)*batch_size] = results_z

    preds_gain_x = preds_gain_x * (gains_f1_max - gains_f1_min) + gains_f1_min
    preds_gain_y = preds_gain_y * (gains_f2_max - gains_f2_min) + gains_f2_min
    preds_gain_z = preds_gain_z * (gains_f3_max - gains_f3_min) + gains_f3_min
    preds_gain = torch.cat((preds_gain_x[:, None, :], preds_gain_y[:, None, :], preds_gain_z[:, None, :]), 1)

    freqs = torch.Tensor(freqs).to(device)
    damps = torch.Tensor(damps).to(device)

    testsavedir = args.audio_results_dir
    os.makedirs(testsavedir, exist_ok=True)

    for i in trange(N):
        preds_gain_x_i = preds_gain[i, 0, :]
        preds_gain_y_i = preds_gain[i, 1, :]
        preds_gain_z_i = preds_gain[i, 2, :]
        force_x, force_y, force_z = forces[i]
        combined_preds_gain = force_x * preds_gain_x_i + force_y * preds_gain_y_i + force_z * preds_gain_z_i
        combined_preds_gain = combined_preds_gain.unsqueeze(0)
        modal_fir = torch.unsqueeze(ddsp.get_modal_fir(combined_preds_gain, freqs, damps), axis=1)
        impulse = torch.reshape(torch.Tensor(scipy.signal.unit_impulse(44100*3)).to(device), (1, -1)).repeat(modal_fir.shape[0], 1)
        result = ddsp.fft_convolve(impulse, modal_fir)
        signal = result[0, :].detach().cpu().numpy()
        signal = signal / np.abs(signal).max()
        # write wav file
        output_path = os.path.join(testsavedir, str(i+1) + '.wav')
        write(output_path, 44100, signal.astype(np.float32))


def TouchNet_eval(args):

    checkpoint = torch.load(args.object_file_path)

    rotation_max = 15
    depth_max = 0.04
    depth_min = 0.0339
    displacement_min = 0.0005
    displacement_max = 0.0020
    depth_max = 0.04
    depth_min = 0.0339
    rgb_width = 120
    rgb_height = 160
    network_depth = 8

    #TODO load object...
    vertex_min = checkpoint['TouchNet']['xyz_min']
    vertex_max = checkpoint['TouchNet']['xyz_max']

    vertex_coordinates = np.load(args.touch_vertices_file_path)
    N = vertex_coordinates.shape[0]
    gelinfo_data = np.load(args.touch_gelinfo_file_path)
    theta, phi, displacement = gelinfo_data[:, 0], gelinfo_data[:, 1], gelinfo_data[:, 2]
    phi_x = np.cos(phi)
    phi_y = np.sin(phi)

    # normalize theta to [-1, 1]
    theta = (theta - np.radians(0)) / (np.radians(rotation_max) - np.radians(0))

    #normalize displacement to [-1,1]
    displacement_norm = (displacement - displacement_min) / (displacement_max - displacement_min)

    #normalize coordinates to [-1,1]
    vertex_coordinates = (vertex_coordinates - vertex_min) / (vertex_max - vertex_min)

    #initialize horizontal and vertical features
    w_feats = np.repeat(np.repeat(np.arange(rgb_width).reshape((rgb_width, 1)), rgb_height, axis=1).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    h_feats = np.repeat(np.repeat(np.arange(rgb_height).reshape((1, rgb_height)), rgb_width, axis=0).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    #normalize horizontal and vertical features to [-1, 1]
    w_feats_min = w_feats.min()
    w_feats_max = w_feats.max()
    h_feats_min = h_feats.min()
    h_feats_max = h_feats.max()
    w_feats = (w_feats - w_feats_min) / (w_feats_max - w_feats_min)
    h_feats = (h_feats - h_feats_min) / (h_feats_max - h_feats_min)
    w_feats = torch.FloatTensor(w_feats)
    h_feats = torch.FloatTensor(h_feats)

    theta = np.repeat(theta.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_x = np.repeat(phi_x.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_y = np.repeat(phi_y.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    displacement_norm = np.repeat(displacement_norm.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    vertex_coordinates = np.repeat(vertex_coordinates.reshape((N, 1, 3)), rgb_width * rgb_height, axis=1)

    data_wh = np.concatenate((w_feats, h_feats), axis=1)
    data_wh = np.transpose(data_wh.reshape((N, 2, -1)), axes=[0, 2, 1])
    #Now get final feats matrix as [x, y, z, theta, phi_x, phi_y, displacement, w, h]
    data = np.concatenate((vertex_coordinates, theta, phi_x, phi_y, displacement_norm, data_wh), axis=2).reshape((-1, 9))

    #checkpoint = torch.load(args.object_file_path)
    embed_fn, input_ch = TouchNet_model.get_embedder(10, 0)
    model = TouchNet_model.NeRF(D = network_depth, input_ch = input_ch, output_ch = 1)
    state_dic = checkpoint['TouchNet']['model_state_dict']
    state_dic = TouchNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = nn.DataParallel(model).to(device)
    model.eval()

    preds = np.empty((data.shape[0], 1))

    batch_size = 1024

    testsavedir = args.touch_results_dir
    os.makedirs(testsavedir, exist_ok=True)

    for i in trange(data.shape[0] // batch_size + 1):
        inputs = torch.Tensor(data[i*batch_size:(i+1)*batch_size]).to(device)
        embedded = embed_fn(inputs)
        results = model(embedded)
        preds[i*batch_size:(i+1)*batch_size, :] = results.detach().cpu().numpy()

    preds = preds  * (depth_max - depth_min) + depth_min
    preds = np.transpose(preds.reshape((N, -1, 1)), axes = [0, 2, 1]).reshape((N, rgb_width, rgb_height))
    taxim = TaximRender("./calibs/")
    for i in trange(N):
        height_map, contact_map, tactile_map = taxim.render(preds[i], displacement[i])
        tactile_map = Image.fromarray(tactile_map.astype(np.uint8), 'RGB')
        filename = os.path.join(testsavedir, '{}.png'.format(i+1))
        tactile_map.save(filename)

if __name__ =='__main__':
    parser = config_parser()
    args = parser.parse_args()
    modalities = args.modality.strip().split(",") 

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if "vision" in modalities:
        VisionNet_eval(args=args)
    if "audio" in modalities:
        AudioNet_eval(args=args)
    if "touch" in modalities:
        TouchNet_eval(args=args)
