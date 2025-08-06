#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
from os import makedirs
from time import time, perf_counter
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm
import imageio
import numpy as np
from pathlib import Path

from scene import Scene
from scene.dataset_readers import loadCameras
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.pose_utils import get_tensor_from_camera
from utils.loss_utils import l1_loss, ssim, l1_loss_mask, ssim_loss_mask
from utils.sfm_utils import save_time
# from utils.camera_utils import generate_interpolated_path
from utils.camera_trajectory import generate_interpolated_path
from utils.camera_utils import visualizer
from arguments import ModelParams, PipelineParams, get_combined_args
import cv2
import shutil
from PIL import Image

def save_interpolate_pose(model_path, iter, n_views, source_path, traj):
    depth_path = os.path.join(source_path, "depth")
    depth = []
    for file in os.listdir(depth_path):
        if file.endswith(".npy"):
            tmp_depth = np.load(os.path.join(depth_path, file))
            depth.append(tmp_depth)
    depth_min = min([depth[i].min() for i in range(len(depth))])
    depth_max = max([depth[i].max() for i in range(len(depth))])
    rot_weight = depth_min
    org_pose = np.load(model_path / f"pose/ours_{iter}/pose_optimized.npy")
    # visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/ours_{iter}/poses_optimized.png")
    n_interp = int(10 * 20 / n_views)
    all_inter_pose = []
    selected_method = traj
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp, method=selected_method, rot_weight=rot_weight, depth=depth)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    # visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/ours_{iter}/poses_interpolated_{selected_method}.png")
    np.save(model_path / f"pose/ours_{iter}/pose_interpolated_{traj}.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30, trajs=None, trajs_name=None, is_depth=None):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    import os
    import cv2
    import imageio
    # First pass: compute min dimensions
    min_h, min_w = float('inf'), float('inf')
    valid_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in valid_files:
        image_path = os.path.join(image_folder, filename)
        image = imageio.imread(image_path)
        h, w = image.shape[0], image.shape[1]
        min_h = min(min_h, h)
        min_w = min(min_w, w)
    
    images = []
    cnt = 100
    path = os.path.dirname(image_folder)
    clip_traj = []
    clip_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(output_video_path)))) + f"_{trajs_name}"
    for idx, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            image = cv2.resize(image, (min_w, min_h))
            images.append(image)
        if (idx+1) % cnt ==0 and idx != 0 and trajs is not None:
            print(idx)
            if is_depth:
                imageio.mimwrite(os.path.join(path, clip_name + f"_{int(idx/(cnt-1))}_depth.mp4"), images, fps=fps)
            else:
                imageio.mimwrite(os.path.join(path, clip_name + f"_{int(idx/(cnt-1))}.mp4"), images, fps=fps)
                pose_file = os.path.join(path, clip_name + f"_{int(idx/(cnt-1))}.txt")
                with open(pose_file, '+w') as f:
                    f.write('\n')
                    clip_traj = trajs[(idx - (cnt - 1)):(idx+1)]
                    print(len(clip_traj), "start:", f"{idx-(cnt - 1)}", "end:", f"{idx+1}")
                    for i, traj in enumerate(clip_traj):
                        f.write(str(i) + ' ')
                        f.write(' '.join(str(t) for t in traj) + '\n')
            images = []


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering_result = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )
        rendering = rendering_result["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.infer_video and not dataset.eval:
        save_interpolate_pose(Path(args.model_path), iteration, args.n_views, args.source_path, args.traj)
        interp_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / f'pose_interpolated_{args.traj}.npy')
        viewpoint_stack, camera_traj = loadCameras(interp_pose, scene.getTrainCameras())
        render_set(
            dataset.model_path,
            "interp",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
        )
        image_folder = os.path.join(dataset.model_path, f'interp/ours_{iteration}/renders')
        output_video_file = os.path.join(dataset.model_path, f'interp/ours_{iteration}/rgb_video.mp4')
        images_to_video(image_folder, output_video_file, fps=20, trajs=camera_traj, trajs_name=args.traj)

        selected_method = args.traj
        selected_method_folder = os.path.join(dataset.model_path, f'interp/{selected_method}')
        shutil.rmtree(selected_method_folder, ignore_errors=True)
        os.rename(os.path.join(dataset.model_path, f'interp/ours_{iteration}'), selected_method_folder)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")    
    parser.add_argument("--optim_test_pose_iter", default=1500, type=int)
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--test_fps", action="store_true")
    parser.add_argument("--traj", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args)
