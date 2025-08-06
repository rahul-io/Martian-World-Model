import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())
ic(torch.cuda.device_count())

from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks,
                             project_points)
from PIL import Image
import cv2
from utils.scale_metric_depth_calculate import fit_scale_and_shift_multiple2
from gim.demo import gim_run
import imageio
import matplotlib.pyplot as plt
from vggt.vggt_run import vggt_run
from component.scene_util import Scene
from component.metric_3d_v2 import metric3d_depth_cal_save

def main(source_path, model_path, device, image_size, n_views, co_vis_dsp, depth_thre):

    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    train_img_files = image_files

    image_files = train_img_files
    images, org_imgs_shape, img_names = load_images(image_files, size=image_size)

    intrinsic, _, _ = vggt_run(image_files, device=device)
    fx = (intrinsic[0][0][0] + intrinsic[1][0][0]) / 2 * (images[0]['true_shape'][0][1] / (intrinsic[0,0,2] *2))
    fy = (intrinsic[0][1][1] + intrinsic[1][1][1]) / 2  * (images[0]['true_shape'][0][0] / (intrinsic[0,1,2] *2))
    scene = Scene(images, [fx, fy])

    x1, x2 = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) for img in scene.imgs]
    mkpts_0, mkpts_1, valid_matches = gim_run(x1, x2)
    # valid validation
    H0, W0 = images[0]['img'].shape[-2:]
    valid_matches_im0 = (mkpts_0[:, 0] >= 3) & (mkpts_0[:, 0] < int(W0) - 3) & (
        mkpts_0[:, 1] >= 3) & (mkpts_0[:, 1] < int(H0) - 3)
    H1, W1 = images[1]['img'].shape[-2:]
    valid_matches_im1 = (mkpts_1[:, 0] >= 3) & (mkpts_1[:, 0] < int(W1) - 3) & (
        mkpts_1[:, 1] >= 3) & (mkpts_1[:, 1] < int(H1) - 3)
    valid_matches = valid_matches & valid_matches_im0 & valid_matches_im1
    mkpts_0, mkpts_1 = mkpts_0[valid_matches], mkpts_1[valid_matches]

    focals = to_numpy(scene.get_focals())
    metric_3d_depths = metric3d_depth_cal_save(img_names, focals.repeat(2, 1), scene.get_principal_points().detach().cpu().numpy(), org_imgs_shape)

    # pnp solve pose
    extrinsics_w2c_0 = np.eye(4)
    pts3d_0 = to_numpy(scene.get_pts3d_0(inv(extrinsics_w2c_0), metric_3d_depths[0]))[0]
    pts3d_0_flat = pts3d_0.reshape(-1, 3)
    # Create mapping from 2D points to 3D points in view0
    pts0_to_3d = {}
    H, W = scene.get_depthmaps()[0].shape
    for i, (x, y) in enumerate(mkpts_0):
        if 0 <= int(y) < H and 0 <= int(x) < W:
            idx = int(y) * W + int(x)
            pts0_to_3d[(x, y)] = pts3d_0_flat[idx]
    # Get corresponding 2D points in view1 and map them to view0's 3D points
    imagePoints = []
    objectPoints = []
    for pt0, pt1 in zip(mkpts_0, mkpts_1):
        pt0_tuple = (pt0[0], pt0[1])
        if pt0_tuple in pts0_to_3d:
            imagePoints.append(pt1)
            objectPoints.append(pts0_to_3d[pt0_tuple])

    imagePoints = np.array(imagePoints, dtype=np.float32)
    objectPoints = np.array(objectPoints, dtype=np.float32)
    k = scene.get_intrinsics()[1].cpu().detach().numpy()
    res = cv2.solvePnPRansac(objectPoints, imagePoints, k, None,
                             iterationsCount=50000, 
                            confidence=0.999,
                            reprojectionError=0.9,
                            flags=cv2.SOLVEPNP_EPNP,
                            )
    success, R, T, inliers = res
    assert success

    if inliers is not None and len(inliers) > 0:
        inlier_indices = inliers.ravel()
        inlier_objectPoints = objectPoints[inlier_indices]
        inlier_imagePoints = imagePoints[inlier_indices]
        R, T = cv2.solvePnPRefineLM(inlier_objectPoints, inlier_imagePoints, k, None, R, T)
    R = cv2.Rodrigues(R)[0]  

    pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
    extrinsics_w2c_1 = np.r_[np.c_[R, T], [(0, 0, 0, 1)]] # world to cam

    # project pts3d_0 to camera 1 to get sparse depth
    points_2d_1, sparse_depth_1 = project_points(pts3d_0_flat, k, extrinsics_w2c_1)

    h, w = metric_3d_depths[1].shape
    sparse_depth_map = np.zeros((h, w), dtype=np.float32)
    sparse_depth_mask = np.zeros((h, w), dtype=bool)

    valid_points = (points_2d_1[:, 0] >= 0) & (points_2d_1[:, 0] < w) & \
                (points_2d_1[:, 1] >= 0) & (points_2d_1[:, 1] < h) & \
                (sparse_depth_1 > 0) & (~np.isnan(sparse_depth_1))

    x_indices = np.clip(np.round(points_2d_1[valid_points, 0]).astype(int), 0, w-1)
    y_indices = np.clip(np.round(points_2d_1[valid_points, 1]).astype(int), 0, h-1)

    sparse_depth_map[y_indices, x_indices] = sparse_depth_1[valid_points]
    sparse_depth_mask[y_indices, x_indices] = True

    avg_scale2 , avg_shift2 = fit_scale_and_shift_multiple2([metric_3d_depths[1].reshape(-1)], [sparse_depth_1.reshape(-1)], [sparse_depth_mask.reshape(-1)])
    another_scale_depth = metric_3d_depths[1] * avg_scale2 + avg_shift2
    pts3d_1 = to_numpy(scene.get_pts3d_0(pose, another_scale_depth))[0]

    scale_depth = [metric_3d_depths[0], another_scale_depth]
    pts3d = np.array([pts3d_0, pts3d_1])
    depthmaps = np.log(scale_depth)
    extrinsics_w2c =np.array([extrinsics_w2c_0, extrinsics_w2c_1])
    intrinsics = to_numpy(scene.get_intrinsics())
    imgs = np.array(scene.imgs)
    focals = to_numpy(scene.get_focals())

    print(f'>> Calculate the co-visibility mask...')
    sorted_conf_indices = np.arange(n_views)
    if depth_thre > 0:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None
    
    focals = np.repeat(focals[0], n_views)
    print(f'>> Saving results...')
    end_time = time()
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
    save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)
    pts_num = save_points3D(sparse_0_path, imgs, pts3d, None, overlapping_masks, use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=model_path, depth_threshold=depth_thre)
    save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)
    print(f'[INFO] Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')    
    print(f'[INFO] Number of points after downsampling: {pts_num}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--n_views', type=int, default=3, help='')
    parser.add_argument('--co_vis_dsp', action="store_true")
    parser.add_argument('--depth_thre', type=float, default=0.03, help='Depth threshold')

    args = parser.parse_args()
    main(args.source_path, args.model_path, args.device, args.image_size, args.n_views, args.co_vis_dsp, args.depth_thre)