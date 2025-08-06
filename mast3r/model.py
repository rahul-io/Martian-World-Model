# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

# import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa
import numpy as np
from scipy.spatial import KDTree
from dust3r.utils.device import to_numpy, todevice  # noqa
import math
inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device='cuda', block_size=None, dist='l2'):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                # dists_blk = dists[i * block_size:(i+1)*block_size, j * block_size:(j+1)*block_size]
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size)
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy()
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B
class cdistMatcher:
    def __init__(self, db_pts, device='cuda'):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = bruteforce_reciprocal_nns(queries, self.db_pts, device=self.device, **kw)
        dis = None
        return dis, nnA


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != 'y_x':
            xy1 = xy1[0].base[:, ::-1]
            xy2 = xy2[0].base[:, ::-1]

    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2


def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', **matcher_kw):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
        max_iter = 10
    else:
        x1, y1 = subsample_or_initxy1
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        if isinstance(y1, torch.Tensor):
            y1 = y1.cpu().numpy()
        max_iter = 1

    xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    if 'dist' in matcher_kw or 'block_size' in matcher_kw \
            or (isinstance(device, str) and device.startswith('cuda')) \
            or (isinstance(device, torch.device) and device.type.startswith('cuda')):
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        tree1 = cdistMatcher(pts1, device=device)
        tree2 = cdistMatcher(pts2, device=device)
    else:
        pts1, pts2 = to_numpy((pts1, pts2))
        tree1 = KDTree(pts1)
        tree2 = KDTree(pts2)

    notyet = np.ones(len(xy1), dtype=bool)
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

    niter = 0
    # n_notyet = [len(notyet)]
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        if not ret_basin:
            notyet &= (old_xy2 != xy2)  # remove points that have converged

        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)  # remove points that have converged

        # n_notyet.append(notyet.sum())
        niter += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1

    # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    if pixel_tol > 0:
        # in case we only want to match some specific points
        # and still have some way of checking reciprocity
        old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
        new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
        dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2