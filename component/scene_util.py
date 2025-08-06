import torch
import torch.nn as nn
import numpy as np
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_numpy, to_cpu
from dust3r.utils.image import rgb
class Scene():
    def __init__(self, images, focal_lengths, device='cuda'):
        self.device = device
        self.n_imgs = 2  # Based on usage in init_geo.py, seems to work with 2 images
        self.imshapes = []
        for i in range(len(images)):
            self.imshapes.append((images[i]['img'].shape[-2], images[i]['img'].shape[-1]))
        self.focal_break = 20
        
        # Initialize image parameters
        self.im_focals = nn.ParameterList([
            torch.FloatTensor([self.focal_break*np.log(max(H, W))]) 
            for H, W in self.imshapes
        ])
        
        # Initialize principal points based on actual image shapes
        # im_pp: learnable offset parameters for principal points (set to zero since no optimization)
        self.im_pp = nn.ParameterList([torch.zeros((2,)) for _ in range(len(self.imshapes))])
        
        # Initialize depth maps and confidence maps
        self.im_depthmaps = nn.ParameterList([
            torch.randn(H, W)/10-3 for H, W in self.imshapes
        ])
        
        self.im_conf = nn.ParameterList([
            torch.ones(H, W) * 0.8 for H, W in self.imshapes
        ])
        
        # _pp: base principal points (image center coordinates)
        # Since no optimization, this directly represents the actual principal points
        self._pp = torch.tensor([(W/2, H/2) for H, W in self.imshapes], dtype=torch.float32)
        
        # Disable gradient computation for principal point offsets since no optimization
        for pp_param in self.im_pp:
            pp_param.requires_grad_(False)
        
        # Grid for each image
        self._grid = [xy_grid(W, H, device=device) for H, W in self.imshapes]
        
        # (H,W,3)
        self.imgs = [rgb(img['img']).squeeze(0) for img in images]
        
        # Initialize intrinsics
        self.intrinsics = self._compute_default_intrinsics()
        self.preset_focal(np.repeat(focal_lengths, self.n_imgs), msk=None)

    def _compute_default_intrinsics(self):
        """Compute default intrinsic matrices"""
        num_imgs = len(self.imshapes)
        K = torch.zeros((num_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K
    
    def preset_focal(self, known_focals, msk=None):
        """Set known focal lengths"""
        indices = self._get_msk_indices(msk)
        for idx, focal in zip(indices, known_focals):
            self._set_focal(idx, focal, force=True)
        self.im_focals.requires_grad_(False)
        # Update intrinsics after setting focals
        self.intrinsics = self._compute_default_intrinsics()
    
    def _get_msk_indices(self, msk):
        """Get mask indices"""
        num_imgs = len(self.imshapes)
        if msk is None:
            return range(num_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif hasattr(msk, 'dtype') and msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == num_imgs
            return np.where(msk)[0]
        elif hasattr(msk, 'dtype') and np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            return range(num_imgs)
    
    def _set_focal(self, idx, focal, force=False):
        """Set focal length for a specific image"""
        param = self.im_focals[idx]
        if param.requires_grad or force:
            param.data[:] = self.focal_break * np.log(focal)
        return param
    
    def get_focals(self):
        """Get focal lengths"""
        log_focals = torch.stack(list(self.im_focals), dim=0)
        focals = (log_focals / self.focal_break).exp()
        return focals.to(self.device)
    
    def get_principal_points(self):
        """Get principal points"""
        pp = self._pp + 10 * torch.stack(list(self.im_pp), dim=0)
        return pp.to(self.device)
    
    def get_intrinsics(self):
        """Get intrinsic matrices"""
        return self.intrinsics.to(self.device)
    
    def get_depthmaps(self, raw=False):
        """Get depth maps"""
        res = [dm.exp().to(self.device) for dm in self.im_depthmaps]
        if not raw:
            return res  # Already in H,W format
        else:
            return [dm.view(-1) for dm in res]  # Flatten for raw format
    
    def get_conf(self, mode=None):
        """Get confidence maps"""
        return [conf.to(self.device) for conf in self.im_conf]
    
    def get_masks(self):
        """Get masks based on confidence threshold"""
        return [(conf.to(self.device) > self.min_conf_thr) for conf in self.im_conf]
    
    def _fast_depthmap_to_pts3d(self, depth, pixel_grid, focal, pp):
        """Convert depth map to 3D points"""
        pp = pp.unsqueeze(0).unsqueeze(0)  # Add batch and spatial dims
        focal = focal.unsqueeze(0).unsqueeze(0)
        
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)  # Add batch dim
        if pixel_grid.dim() == 2:
            pixel_grid = pixel_grid.unsqueeze(0)  # Add batch dim
            
        depth = depth.unsqueeze(-1)
        return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
    
    def get_pts3d_0(self, extrinsic, depth_map):
        """Get 3D points for single image with given extrinsic and depth"""
        # Convert inputs to tensors
        if isinstance(extrinsic, np.ndarray):
            extrinsic = torch.tensor(extrinsic, dtype=torch.float32, device=self.device)
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.tensor(depth_map, dtype=torch.float32, device=self.device)
        
        # Get camera parameters
        focal = self.get_focals()[0]  # Use first camera's focal length
        pp = self.get_principal_points()[0]  # Use first camera's principal point
        grid = self._grid[0]  # Use first camera's grid
        
        # Convert depth to 3D points in camera frame
        if depth_map.dim() == 2:
            H, W = depth_map.shape
            pixel_grid = grid.view(H, W, 2)
        else:
            pixel_grid = grid
        
        # Convert depth map to 3D points
        rel_pts = self._fast_depthmap_to_pts3d(depth_map, pixel_grid, focal, pp)
        
        # Transform to world coordinates
        if extrinsic.shape == (3, 4):
            # Add homogeneous row
            extrinsic = torch.cat([extrinsic, torch.tensor([[0, 0, 0, 1]], device=self.device)], dim=0)
        
        # Apply transformation
        pts3d = geotrf(extrinsic.unsqueeze(0), rel_pts)
        
        return pts3d