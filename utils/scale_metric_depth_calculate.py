import numpy as np
import cv2
from scipy.optimize import least_squares
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_depth_image(depth_image, target_shape):
    return cv2.resize(depth_image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def filter_depth_by_mask(depth_image, mask):
    valid_points = depth_image[mask > 0]
    return valid_points

def fit_scale_and_shift_multiple(mast3r_depths, metric_3d_depths, masks, confs):
    scales = []
    shifts = []
    metric_depth_v2_resized_msk = []

    for mast3r_depth, metric_depth_v2, mask, conf in zip(mast3r_depths, metric_3d_depths, masks, confs):
        metric_depth_v2_resized = resize_depth_image(metric_depth_v2, mast3r_depth.shape)
        valid_mast3r_depth = filter_depth_by_mask(mast3r_depth, mask)
        valid_metric_depth_v2 = filter_depth_by_mask(metric_depth_v2_resized, mask)
        valid_conf = filter_depth_by_mask(conf, mask)
        
        assert valid_mast3r_depth.shape[0] == valid_metric_depth_v2.shape[0], "Number of valid points is inconsistent"
        metric_depth_v2_resized_msk.append(valid_metric_depth_v2)

        weights = valid_conf / (np.sum(valid_conf) + 1e-6)

        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * valid_mast3r_depth)
        sum_wy = np.sum(weights * valid_metric_depth_v2)
        sum_wxx = np.sum(weights * (valid_mast3r_depth**2))
        sum_wxy = np.sum(weights * valid_mast3r_depth * valid_metric_depth_v2)

        denom = (sum_w * sum_wxx - sum_wx**2) + 1e-8
        scale0 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
        shift0 = (sum_wy - scale0 * sum_wx) / (sum_w + 1e-8)
        
        initial_guess = [scale0, shift0]

        def residuals(params):
            scale, shift = params
            return np.sqrt(weights) * (scale * valid_mast3r_depth + shift - valid_metric_depth_v2)

        result = least_squares(
            residuals,
            x0=initial_guess,
            method='trf',
            loss='huber',
            f_scale=0.1,
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9
        )

        scale, shift = result.x
        scales.append(scale)
        shifts.append(shift)

    avg_scale = np.mean(scales)
    avg_shift = np.mean(shifts)

    test_transformation_error(mast3r_depths, metric_3d_depths, masks, avg_scale, avg_shift)
    return avg_scale, avg_shift

def construct_mask_from_confs(confs, threshold=1.0):
    return [(conf > threshold).astype(np.uint8) for conf in confs]

def test_transformation_error(mast3r_depths, metric_3d_depths, masks, avg_scale, avg_shift):
    error_metrics = []
    for i, (mast3r_depth, metric_depth_v2, mask) in enumerate(zip(mast3r_depths, metric_3d_depths, masks)):
        transformed_mast3r = avg_scale * mast3r_depth + avg_shift
        metric_depth_v2_resized = resize_depth_image(metric_depth_v2, mast3r_depth.shape)
        valid_transformed = transformed_mast3r[mask > 0]
        valid_metric = metric_depth_v2_resized[mask > 0]

        error = valid_transformed - valid_metric
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        error_metrics.append((mae, rmse))

    avg_mae = np.mean([em[0] for em in error_metrics])
    avg_rmse = np.mean([em[1] for em in error_metrics])


def update_point_cloud_scale_shift(pts3d_list, scale, shift):
    updated_list = []
    for pc in pts3d_list:
        pc_new = pc.copy()
        Z = pc_new[..., 2]
        Z_hat = scale * Z + shift
        ratio = np.zeros_like(Z, dtype=np.float32)
        ratio = Z_hat / Z
        pc_new[..., 0] *= ratio
        pc_new[..., 1] *= ratio
        pc_new[..., 2] = Z_hat
        updated_list.append(pc_new)
    return updated_list

def scale_bbox_and_points(pts3d_list, scale_factor=1.0):
    all_points = []
    shapes = []
    for arr in pts3d_list:
        shapes.append(arr.shape)
        all_points.append(arr.reshape(-1, 3))
    all_points = np.concatenate(all_points, axis=0)
    old_min = np.min(all_points, axis=0)
    old_max = np.max(all_points, axis=0)
    new_min = old_min
    new_max = old_min + scale_factor*(old_max - old_min)
    bbox_size = old_max - old_min
    new_pts3d_list = []
    idx_arr = 0
    for arr in pts3d_list:
        pts_flat = arr.reshape(-1, 3)
        alpha = (pts_flat - old_min) / bbox_size
        new_pts_flat = new_min + alpha*(new_max - new_min)
        new_pts = new_pts_flat.reshape(arr.shape)
        new_pts3d_list.append(new_pts)
    return new_pts3d_list

def fit_scale_and_shift_multiple2(mast3r_depths, metric_3d_depths, masks):
    scales = []
    shifts = []
    metric_depth_v2_resized_msk = []
    for mast3r_depth, metric_depth_v2, mask in zip(mast3r_depths, metric_3d_depths, masks):
        valid_mast3r_depth = filter_depth_by_mask(mast3r_depth, mask)
        valid_metric_depth_v2 = filter_depth_by_mask(metric_depth_v2, mask)
        assert valid_mast3r_depth.shape[0] == valid_metric_depth_v2.shape[0]
        metric_depth_v2_resized_msk.append(valid_metric_depth_v2)
        weights = np.ones_like(valid_metric_depth_v2, dtype=np.float64)
        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * valid_mast3r_depth)
        sum_wy = np.sum(weights * valid_metric_depth_v2)
        sum_wxx = np.sum(weights * (valid_mast3r_depth**2))
        sum_wxy = np.sum(weights * valid_mast3r_depth * valid_metric_depth_v2)
        denom = (sum_w * sum_wxx - sum_wx**2) + 1e-8
        scale0 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
        shift0 = (sum_wy - scale0 * sum_wx) / (sum_w + 1e-8)
        initial_guess = [scale0, shift0]
        def residuals(params):
            scale, shift = params
            return np.sqrt(weights) * (scale * valid_mast3r_depth + shift - valid_metric_depth_v2)
        result = least_squares(
            residuals,
            x0=initial_guess,
            method='trf',
            loss='huber',
            f_scale=0.1,
            max_nfev=10000,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9
        )
        scale, shift = result.x
        scales.append(scale)
        shifts.append(shift)
    avg_scale = np.mean(scales)
    avg_shift = np.mean(shifts)
    test_transformation_error2(mast3r_depths, metric_3d_depths, masks, avg_scale, avg_shift)
    return avg_scale, avg_shift

def test_transformation_error2(mast3r_depths, metric_3d_depths, masks, avg_scale, avg_shift):
    error_metrics = []
    for i, (mast3r_depth, metric_depth_v2, mask) in enumerate(zip(mast3r_depths, metric_3d_depths, masks)):
        transformed_mast3r = avg_scale * mast3r_depth + avg_shift
        valid_transformed = transformed_mast3r[mask > 0]
        valid_metric = metric_depth_v2[mask > 0]
        error = valid_transformed - valid_metric
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        error_metrics.append((mae, rmse))
    avg_mae = np.mean([em[0] for em in error_metrics])
    avg_rmse = np.mean([em[1] for em in error_metrics])