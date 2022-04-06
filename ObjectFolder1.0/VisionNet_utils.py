import os
import datetime
import sys
import torch
import numpy as np
import imageio
import json
import random
import time

import indirect_utils
from load_osf import load_osf_data
from intersect import compute_object_intersect_tensors
from ray_utils import transform_rays
from run_osf_helpers import *
from scatter import scatter_coarse_and_fine
import shadow_utils

from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(fn, chunk):
    """Construct a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)
    return ret


def run_network(inputs, viewdirs, lightdirs, fn, embed_fn, embedviews_fn, embedlights_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.

    Args:
        inputs: [R, S, 3] float tensor. Sampled points per day.
        viewdirs: [R, S, 3] float tensor. Viewing directions.
        lightdirs: [R, S, 3] float tensor. Light directions.
        fn: Network function.
        embed_fn: Function to embed input points.
        embedviews_fn: Function to embed viewdirs.
        embedlights_fn: Function to embed viewdirs.
        netchunk: Batch size to feed into the network.
    """
    def prepare_and_embed(embeddirs_fn, dirs):
        """
        Args:
            embeddirs_fn: Function to embed directions.
            dirs: [R, S, 3] float tensor. Per-sample directions.

        Returns:
            edirs: [RS, L] float tensor. Embedded directions.
        """
        dirs_flat = torch.reshape(dirs, [-1, dirs.shape[-1]])  # [RS, 3]
        edirs = embeddirs_fn(dirs_flat)  # [RS, L]
        return edirs

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        embedded_dirs = prepare_and_embed(embedviews_fn, viewdirs)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    if lightdirs is not None:
        embedded_dirs = prepare_and_embed(embedlights_fn, lightdirs)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def evaluate_single_object(
        ray_batch, params, other_params=None, ret_ray_batch=False, **kwargs):
    """Evaluate rays for a single object.
    Args:
        ray_batch: [R, M] float tensor. All information necessary for sampling along a
            ray, including: ray origin, ray direction, min dist, max dist, and
            unit-magnitude viewing direction, and iamge ID, all in world coordinate
            frame.
        params: Dictionary of object parameters that must contain the following items:
                network_fn: Coarse network.
                network_fine: Fine network.
                intersect_bbox: bool. If true, intersect rays with object bbox. Only
                    rays that intersect are evaluated, and sampling bounds are
                    determined by the intersection bounds.
            Additional parameters that are only required if intersect_bbox = True:
                box_center: List of length 3 containing the (x, y, z) center of the bbox.
                box_dims: List of length 3 containing the x, y, z dimensions of the bbox.
            Optional params:
                translation: List of length 3 containing the (x, y, z) object-to-world
                    translation.
        other_params: List of params in the same format as `params` representing objects
            that we want to cast shadows from onto the current object. Only required if
            `render_shadows` is true.
        **kwargs: Dict. Additional arguments.
    Returns:
        ret0: Coarse network outputs.
        ret: Fine network outputs.
    """
    # Store the original number of rays before intersection.
    num_rays = ray_batch.shape[0]
    num_rays_to_eval = num_rays

    # Transform rays from world to object coordinate frame.
    ray_batch_obj = ray_utils.transform_rays(
        ray_batch, params, kwargs['use_viewdirs'], inverse=True)
    metadata_obj = ray_utils.transform_dirs(kwargs['metadata'], params, inverse=True)
    # ray_batch_obj = ray_batch
    # metadata_obj = kwargs['metadata']

    # Potentially only select a subset of intersecting rays to evaluate.
    ray_batch_obj_to_eval = ray_batch_obj
    if params['intersect_bbox']:
        # Compute ray bbox intersections. Update the batch of rays to only be the set of
        # rays that intersect with the object, and near/far bounds correspond to the
        # intersection bounds.
        ray_batch_obj_to_eval, indices, intersect_mask = compute_object_intersect_tensors(
            ray_batch_obj, box_center=params['box_center'], box_dims=params['box_dims'])  # [R?, M], [R?,]
        num_rays_to_eval = len(indices)

    # Sample points along rays and evaluate the network on the points. We skip this if
    # there are no intersections  because the network/concat operations throw errors on
    # zero batch size.
    ret0, ret = None, None
    if num_rays_to_eval > 0:
        metadata = kwargs.pop('metadata')
        ret0, ret = run_single_object(
                ray_batch_obj_to_eval, metadata=metadata_obj, network_fn=params['network_fn'],
                #network_fine=params['network_fine'], lightdirs_method=params['lightdirs_method'], **kwargs)
                network_fine=params['network_fine'], lightdirs_method='metadata', **kwargs)
        kwargs['metadata'] = metadata
        if ret_ray_batch:
            ret['ray_batch'] = ray_batch
        num_samples = ret['pts'].shape[1]  # S

        # Convert from object back into world coordinate space.
        ret['pts'] = ray_utils.transform_points_into_world_coordinate_frame(ret['pts'], params)
        ray_batch_to_eval = ray_utils.transform_rays(
                ray_batch_obj_to_eval, params, kwargs['use_viewdirs'])
        # ray_batch_to_eval = ray_batch_obj_to_eval

        # Optionally compute shadows or indirect illumination from other objects onto
        # the current object.
        if kwargs['render_shadows']:
            # Only keep objects that have bboxes.
            other_params = [p for p in other_params if p['intersect_bbox']]
        if len(other_params) > 0:
            rays_i = ray_batch_obj_to_eval[:, -1:]  # [R?, 1]
            #if kwargs['render_shadows']:
            #    shadow_trans = compute_shadows(
            #            ray_batch_to_eval, ret['pts'], other_params, **kwargs)
            #    ret['rgb'] *= shadow_trans  # [R?, S, 3]
            #elif kwargs['render_indirect']:
            #    indirect_radiance = compute_indirect_illumination(
            #            rays_i, ret['pts'], other_params, **kwargs)
            #    ret['rgb'] *= indirect_radiance  # [R?, S, 3]
        if 'light_rgb' in params:
            light_rgb = torch.tensor(params['light_rgb'], dtype=tf.float32)[None, None, :]
            ret0['rgb'] *= light_rgb
            ret['rgb'] *= light_rgb
        # ret['alpha'] = tf.ones_like(ret['alpha']) * 0.01

    # Scatter the results from intersecting rays back to the original set of rays.
    if params['intersect_bbox']:
        ret0, ret = scatter_coarse_and_fine(ret0, ret, indices, num_rays, **kwargs)
        if ret_ray_batch:
            ret['ray_batch'] = torch.cat((ray_batch, intersect_mask[:, None].float()), dim=1)
    # Only save shadow rays for objects that don't require bbox intersections (o/w we
    # need to scatter shadow rays first)
    # if not params['intersect_bbox'] and shadow_ray_batch is not None:
    #     if ret_ray_batch:
    #         ret['shadow_ray_batch'] = tf.reshape(
    #                 shadow_ray_batch, [num_rays, num_samples, -1])  # [R, S, M*O]
    return ret0, ret


def evaluate_multiple_objects(ray_batch, object_params, **kwargs):
    """Evaluates multiple objects and sorts them into a single set of results.
    Args:
        ray_batch: [R, M] float  tensor. All information necessary for sampling along a
            ray, including: ray origin, ray direction, min dist, max dist, and
            unit-magnitude viewing direction, and iamge ID, all in world coordinate
            frame.
        object_params: List of object parameters that must contain the following items:
                network_fn: Coarse network.
                network_fine: Fine network.
                intersect_bbox: bool. If true, intersect rays with object bbox. Only
                    rays that intersect are evaluated, and sampling bounds are
                    determined by the intersection bounds.
            Additional parameters that are only required if intersect_bbox = True:
                box_center: List of length 3 containing the (x, y, z) center of the bbox.
                box_dims: List of length 3 containing the x, y, z dimensions of the bbox.
            Optional params:
                translation: List of length 3 containing the (x, y, z) object-to-world
                    translation.
        **kwargs: Dict. Additional arguments.
    Returns:
        ret0: Coarse network results.
        ret: Fine network results.
    """
    if len(object_params) == 0:
        return {}, {}

    # Evaluate set of objects.
    id2ret0, id2ret = {}, {}
    for i, params in enumerate(object_params):
        other_params = [p for j, p in enumerate(object_params) if j != i]
        id2ret0[i], id2ret[i] = evaluate_single_object(
                ray_batch, params, other_params=other_params, **kwargs)

    # Combine results across objects by sorting. Num samples is multiplied by the
    # number of objects.
    ret0 = combine_multi_object_results(id2ret0)
    ret = combine_multi_object_results(id2ret)
    return ret0, ret


def render_rays(ray_batch, **kwargs):
    """Volumetric rendering.
    Args:
        ray_batch: [R, M] float tensor. All information necessary for sampling along a
            ray, including: ray origin, ray direction, min dist, max dist, and
            unit-magnitude viewing direction, and iamge ID, all in world coordinate
            frame.
        white_bkgd: bool. If True, assume a white background.
        **kwargs: Dict. Additional arguments.
    Returns:
        ret: [Dict]. Contains the following key-value pairs:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray. From fine model.
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray. From fine model.
            raw: [num_rays, num_samples, 4]. Raw predictions from model.
            rgb0: See rgb_map. Output for coarse model.
            disp0: See disp_map. Output for coarse model.
            acc0: See acc_map. Output for coarse model.
            z_std: [num_rays]. Standard deviation of sample distances along each ray.
    """
    ret0, ret = evaluate_multiple_objects(ray_batch, **kwargs)

    # Compose points along rays into the final rendered result in pixel space.
    outputs0 = compose_outputs(z_vals=ret0['z_vals'], rgb=ret0['rgb'],
            alpha=ret0['alpha'], white_bkgd=kwargs['white_bkgd'])
    outputs = compose_outputs(z_vals=ret['z_vals'], rgb=ret['rgb'], alpha=ret['alpha'],
            white_bkgd=kwargs['white_bkgd'])

    # Add composed outputs into the dictionaries we are returning.
    ret0.update(outputs0)
    ret.update(outputs)

    # Merge coarse results into the main dictionary we are returning.
    if kwargs['N_importance'] > 0:
        ret['rgb0'] = ret0['rgb_map']
        ret['disp0'] = ret0['disp_map']
        ret['acc0'] = ret0['acc_map']
        # ret['z_std'] = tf.math.reduce_std(ret['z_samples'], -1)  # [N_rays]

    return ret


def batchify_rays(rays_flat, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    chunk = kwargs['chunk']
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, rays=None, c2w=None, ndc=True,
           near=0., far=1., c2w_staticcam=None, img_id=0,
           **kwargs):
    """Render rays
    Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        focal: float. Focal length of pinhole camera.
        chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
        rays: array of shape [3, batch_size, 3]. Ray origin, direction, and image ID for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        translation: List of length 3 containing the (x, y, z) object-to-world
            translation to apply to the object.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
        camera while using other c2w argument for viewing directions.
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d, rays_i = get_rays(H, W, focal, c2w, img_id=img_id)
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    else:
        # use provided ray batch
        rays_o, rays_d, rays_i = rays

    if kwargs['use_viewdirs']:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d, rays_i = get_rays(H, W, focal, c2w_staticcam, img_id=img_id)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, torch.tensor([1.], dtype=torch.float), rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    rays_i = torch.reshape(rays_i[..., :1], [-1, 1]).float()  # [N, 1]
    rays_near, rays_far = near * \
        torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat((rays_o, rays_d, rays_near, rays_far), dim=-1)
    if kwargs['use_viewdirs']:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = torch.cat((rays, viewdirs), dim=-1)
    rays = torch.cat((rays, rays_i), dim=-1)

    # Render and reshape
    metadata_original = kwargs.pop('metadata')
    if c2w is not None:  # Special metadata for cases when we're rendering full image.
        metadata = kwargs['render_metadata']
    else:
        metadata = metadata_original
    ### try
    #metadata = metadata_original
    all_ret = batchify_rays(
            rays, metadata=metadata, near=near, far=far, **kwargs)
    kwargs['metadata'] = metadata_original

    for k in all_ret:
        #tf.compat.v1.debugging.assert_equal(
        #    all_ret[k].shape[0], tf.math.reduce_prod(sh[:-1]), message=f'k: {k}, {all_ret[k].shape[0]}, {tf.math.reduce_prod(sh[:-1])}')
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None,
        c2w_staticcam=None, render_start=None,
        render_end=None):

    H, W, focal = hwf

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):

        c2w = c2w[:3, :4]

        rgb, disp, acc, extras = render(
            H, W, focal, chunk=chunk, c2w=c2w, img_id = -render_poses.shape[0] + i, **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{}.png'.format(i+1))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, metadata, render_metadata):
    """Instantiate NeRF's MLP model.

    Args:
        args: Arguments from the arg parser.
        metadata: [N, 3] float tensor. Metadata about each image. Currently only light
            position is provided.
        metadata: [N, 3] float tensor. Metadata about render poses. Currently only light
            position is provided.

    Returns:
        render_kwargs_train: Training render arguments.
        render_kwargs_test: Testing render arguments.
        start: int. Iteration to start training at.
        grad_vars: Variables to apply gradients to.
        models: Dict of models.
    """
    # If object params are not provided, we initialize the params for a single object,
    # and by default intersect_bbox is False.
    #if args.object_params is None:
    args.object_params = [{'intersect_bbox': False}]

    # Currently we only support multi-object rendering during inference.
    n_objects = len(args.object_params)
    if n_objects > 1:
        assert args.render_only

    # The iteration to start at, which can change if an earlier model is loaded later.
    start = 0
    for i, params in enumerate(args.object_params):
        embed_fn, input_ch = get_embedder(multires=10, i=0)

        input_ch_views, input_ch_lights = 0, 0
        embedviews_fn = None
        embedlights_fn = None
        #if args.use_viewdirs:
        embedviews_fn, input_ch_views = get_embedder(
            multires=4, i=0)
        #if args.use_lightdirs:
        embedlights_fn, input_ch_lights = get_embedder(
            multires=4, i=0)
        output_ch = 4
        skips = [4]

        model = NeRF(
            D=8, W=256,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, input_ch_lights=input_ch_lights,
            use_viewdirs=True, use_lightdirs=True)
        grad_vars = list(model.parameters())

        model_fine = None
        #if args.N_importance > 0:
        model_fine = NeRF(
            D=8, W=256,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, input_ch_lights=input_ch_lights,
            use_viewdirs=True, use_lightdirs=True)
        grad_vars += list(model_fine.parameters())

        def network_query_fn(inputs, viewdirs, lightdirs, network_fn): return run_network(
            inputs, viewdirs, lightdirs, network_fn,
            embed_fn=embed_fn,
            embedviews_fn=embedviews_fn,
            embedlights_fn=embedlights_fn,
            netchunk=args.netchunk)

        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=5e-4, betas=(0.9, 0.999))

        start = 0
        #basedir = args.basedir
        #expname = args.expname

        # Load model weights, if available.
        """
        def get_ckpts(exp_dir):
            # Find all the coarse model checkpoints.
            return [os.path.join(exp_dir, f) for f in sorted(os.listdir(exp_dir)) if 'tar' in f]
                    #('model_' in f and 'fine' not in f and 'optimizer' not in f)]
        if 'exp_dir' in args.object_params[i]:
            ckpts = get_ckpts(params['exp_dir'])
        elif args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else:
            ckpts = get_ckpts(os.path.join(args.basedir, args.expname))
        """
        #ckpts = [args.object_file_path]
        #print('Found ckpts', ckpts)
        #if len(ckpts) > 0 and not args.no_reload:
        #ckpt_path = ckpts[-1]
        ckpt_path = args.object_file_path
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['VisionNet']['global_step']
        optimizer.load_state_dict(ckpt['VisionNet']['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['VisionNet']['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['VisionNet']['network_fine_state_dict'])

        args.object_params[i]['network_fn'] = model
        args.object_params[i]['network_fine'] = model_fine

    # This models dictionary will contain all the model components (coarse, fine,
    # optimizer) that will be saved during training. Multi-object saving is currently
    # not supported.
    models = {}
    if n_objects == 1:
        models['model'] = model
        if model_fine is not None:
            models['model_fine'] = model_fine

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': 1.,
        'N_importance': 128,
        'N_samples': 64,
        'use_viewdirs': True,
        'use_lightdirs': True,
        'shadow_lightdirs_method': None,
        'white_bkgd': True,
        'raw_noise_std': 0.,
        'scaled_sigmoid': True,
        'object_params': args.object_params,
        'metadata': metadata,
        'render_metadata': render_metadata,
        #'render_metadata': metadata,
        'render_shadows': False,
        'render_indirect': False,
        'secondary_chunk': args.secondary_chunk,
    }

    # NDC only good for LLFF-style forward facing data
    #if args.dataset_type != 'llff' or args.no_ndc:
    #    print('Not ndc!')
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = False

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, optimizer


def save_extras_as_npy(savedir, extras, i, tag):
    for k in extras.keys():
        extras_dir = os.path.join(savedir, 'extras', k, tag)
        os.makedirs(extras_dir, exist_ok=True)
        extras_path = os.path.join(extras_dir, f'{i:06d}.npy')
        np.save(open(extras_path, 'wb'), extras[k].detach().cpu().numpy())
