import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import numpy as np
import imageio
import os
from plyfile import PlyData, PlyElement

def vggt_run(image_file_names, device="cpu"):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("./facebook/VGGT-1B").to(device)
    images = load_and_preprocess_images(image_file_names).to(device)
    images = images[None]

    with torch.no_grad() and torch.amp.autocast(dtype=dtype, device_type=device):
        # scene_dir_name = os.path.basename(os.path.dirname(os.path.dirname(image_names[0])))
        # output_dir = f"output/{scene_dir_name}"
        # os.makedirs(output_dir, exist_ok=True)

        aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # np.save(os.path.join(output_dir, "intrinsic.npy"), intrinsic.detach().cpu().numpy().squeeze(0))

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    return intrinsic.detach().cpu().numpy().squeeze(0), extrinsic.detach().cpu().numpy().squeeze(0), depth_map.detach().cpu().numpy().squeeze(0)