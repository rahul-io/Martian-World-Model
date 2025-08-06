import os
import torch
import cv2
import numpy as np
from scipy.ndimage import median_filter

def _resize_cv2_image(img, long_edge_size):
    H, W = img.shape[:2]
    S = max(W, H)
    if S > long_edge_size:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_CUBIC
    new_W = int(round(W * long_edge_size / S))
    new_H = int(round(H * long_edge_size / S))
    return cv2.resize(img, (new_W, new_H), interpolation=interp)

def resize_and_crop(img, size, square_ok=False):
    H1, W1 = img.shape[:2]
    

    if size == 224:

        scale = int(round(size * max(W1 / H1, H1 / W1)))
        img_resized = _resize_cv2_image(img, scale)
    else:

        img_resized = _resize_cv2_image(img, size)
    

    H, W = img_resized.shape[:2]
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)

        img_cropped = img_resized[cy - half: cy + half, cx - half: cx + half]
    else:
        halfw = ((2 * cx) // 16) * 8
        halfh = ((2 * cy) // 16) * 8
        if (not square_ok) and (W == H):
            halfh = int(3 * halfw / 4)
        img_cropped = img_resized[cy - halfh: cy + halfh, cx - halfw: cx + halfw]
    
    return img_cropped

def metric3d_depth_cal_save(rgb_file, focals, principal_points, org_imgs_shape):
    intrinsics_branch1 = np.concatenate((focals, principal_points), axis=1)
    org_width, org_height = org_imgs_shape
    pp = principal_points[0]
    focals_tmp = focals[:, 0]  
    scale_factor_x = org_width / (2 * pp[0])
    scale_factor_y = org_height / (2 * pp[1])
    intrinsic_branch2 = np.array([focals_tmp[0] * scale_factor_x, focals_tmp[1] * scale_factor_y, org_width / 2, org_height / 2])

    rgbs_branch1 = []         
    pad_infos_branch1 = []    
    rgb_origins_branch1 = []  
    intrinsics_scaled_branch1 = [] 

    rgbs_branch2 = []         
    pad_infos_branch2 = []    
    rgb_origins_branch2 = []  
    intrinsics_scaled_branch2 = []

    input_size = (616, 1064)  
    padding_val = [123.675, 116.28, 103.53]
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    for i, file in enumerate(rgb_file):

        rgb_img1 = cv2.imread(file)[:, :, ::-1]
        rgb_img1 = resize_and_crop(rgb_img1, 512, square_ok=False)
        intrinsic1 = intrinsics_branch1[i]
        h1, w1 = rgb_img1.shape[:2]
        scale1 = min(input_size[0] / h1, input_size[1] / w1)
        rgb1 = cv2.resize(rgb_img1, (int(w1 * scale1), int(h1 * scale1)), interpolation=cv2.INTER_LINEAR)

        intrinsic1_scaled = [intrinsic1[0] * scale1, intrinsic1[1] * scale1, intrinsic1[2] * scale1, intrinsic1[3] * scale1]
        # padding
        h_res1, w_res1 = rgb1.shape[:2]
        pad_h1 = input_size[0] - h_res1
        pad_w1 = input_size[1] - w_res1
        pad_h_half1 = pad_h1 // 2
        pad_w_half1 = pad_w1 // 2
        rgb1_padded = cv2.copyMakeBorder(rgb1, pad_h_half1, pad_h1 - pad_h_half1, pad_w_half1, pad_w1 - pad_w_half1,
                                         cv2.BORDER_CONSTANT, value=padding_val)
        pad_info1 = [pad_h_half1, pad_h1 - pad_h_half1, pad_w_half1, pad_w1 - pad_w_half1]

        rgb1_tensor = torch.from_numpy(rgb1_padded.transpose((2, 0, 1))).float()
        rgb1_tensor = torch.div((rgb1_tensor - mean) , std)
        rgbs_branch1.append(rgb1_tensor[None, :, :, :].cuda())
        pad_infos_branch1.append(pad_info1)
        rgb_origins_branch1.append(rgb_img1)
        intrinsics_scaled_branch1.append(intrinsic1_scaled)

        rgb_img2 = cv2.imread(file)[:, :, ::-1]
        h2, w2 = rgb_img2.shape[:2]
        scale2 = min(input_size[0] / h2, input_size[1] / w2)
        rgb2 = cv2.resize(rgb_img2, (int(w2 * scale2), int(h2 * scale2)), interpolation=cv2.INTER_LINEAR)
        intrinsic2_scaled = [intrinsic_branch2[0] * scale2, intrinsic_branch2[1] * scale2,
                             intrinsic_branch2[2] * scale2, intrinsic_branch2[3] * scale2]
        h_res2, w_res2 = rgb2.shape[:2]
        pad_h2 = input_size[0] - h_res2
        pad_w2 = input_size[1] - w_res2
        pad_h_half2 = pad_h2 // 2
        pad_w_half2 = pad_w2 // 2
        rgb2_padded = cv2.copyMakeBorder(rgb2, pad_h_half2, pad_h2 - pad_h_half2, pad_w_half2, pad_w2 - pad_w_half2,
                                         cv2.BORDER_CONSTANT, value=padding_val)
        pad_info2 = [pad_h_half2, pad_h2 - pad_h_half2, pad_w_half2, pad_w2 - pad_w_half2]
        rgb2_tensor = torch.from_numpy(rgb2_padded.transpose((2, 0, 1))).float()
        rgb2_tensor = (rgb2_tensor - mean) / std
        rgbs_branch2.append(rgb2_tensor.unsqueeze(0).cuda())
        pad_infos_branch2.append(pad_info2)
        rgb_origins_branch2.append(rgb_img2)
        intrinsics_scaled_branch2.append(intrinsic2_scaled)

    batch_branch1 = torch.cat(rgbs_branch1, dim=0)
    batch_branch2 = torch.cat(rgbs_branch2, dim=0)
    combined_batch = torch.cat([batch_branch1, batch_branch2], dim=0)

    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True, trust_repo=True)
    model.cuda().eval()
    with torch.no_grad():
        pred_depths_all, _, _ = model.inference({'input': combined_batch})

    num_imgs = len(rgb_file)
    pred_depths_branch1 = pred_depths_all[:num_imgs]
    pred_depths_branch2 = pred_depths_all[num_imgs:]

    depth_results_branch1 = [] 
    for pred_depth, intrinsic_scaled, pad_info, rgb_img in zip(pred_depths_branch1,
                                                                intrinsics_scaled_branch1,
                                                                pad_infos_branch1,
                                                                rgb_origins_branch1):
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0]-pad_info[1],
                                pad_info[2]: pred_depth.shape[1]-pad_info[3]]
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :],
                                                     size=rgb_img.shape[:2],
                                                     mode='bilinear').squeeze()
        canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)
        depth_results_branch1.append(pred_depth.detach().cpu().numpy())
    return_depth = []
    for metric_depth in depth_results_branch1:
        tmp = median_filter(metric_depth, size=9)
        tmp2 = median_filter(tmp, size=13)
        return_depth.append(tmp2)

    for pred_depth_np, rgb_path in zip(depth_results_branch1, rgb_file):
        base_path = os.path.dirname(os.path.dirname(rgb_path))
        depth_dir = os.path.join(base_path, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        epsilon = 1e-6
        pred_inv_depth = 1.0 / (pred_depth_np + epsilon)
        inv_min = np.min(pred_inv_depth)
        inv_max = np.max(pred_inv_depth)
        pred_inv_depth_norm = (pred_inv_depth - inv_min) / (inv_max - inv_min + epsilon)
        colored_depth = cv2.applyColorMap((pred_inv_depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(depth_dir, f"{filename}_branch1.png"), colored_depth)

    for pred_depth, intrinsic_scaled, pad_info, rgb_img, rgb_path in zip(pred_depths_branch2,
                                                       intrinsics_scaled_branch2,
                                                       pad_infos_branch2,
                                                       rgb_origins_branch2,
                                                       rgb_file):
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0]-pad_info[1],
                                pad_info[2]: pred_depth.shape[1]-pad_info[3]]
        pred_depth = torch.nn.functional.interpolate(pred_depth.unsqueeze(0).unsqueeze(0),
                                                     size=rgb_img.shape[:2],
                                                     mode='bilinear').squeeze()
        canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)
        pred_depth_np = pred_depth.detach().cpu().numpy()
        base_path = os.path.dirname(os.path.dirname(rgb_path))
        depth_dir = os.path.join(base_path, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        np.save(os.path.join(depth_dir, f'{filename}.npy'), pred_depth_np)
        epsilon = 1e-6
        pred_inv_depth = 1.0 / (pred_depth_np + epsilon)
        inv_min = np.min(pred_inv_depth)
        inv_max = np.max(pred_inv_depth)
        pred_inv_depth_norm = (pred_inv_depth - inv_min) / (inv_max - inv_min + epsilon)
        colored_depth = cv2.applyColorMap((pred_inv_depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(depth_dir, f"{filename}_branch2.png"), colored_depth)

    return return_depth