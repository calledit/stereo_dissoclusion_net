import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

import dinov3
import color_oklab
import subprocess
import sys
import time
import math
from model2 import CrossFrameAttentionUNet
sys.path.append("C:\\Users\\calle\\projects\\metric_depth_video_toolbox")
import depth_frames_helper
MODEL_maxOUTPUT_depth = 100.0

#for raft
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

##if epoch is low give -1 and -2 if high -1,-2,-3
global_layer_ids = [-1, -2, -3]

# -----------------------------
# Model: Gated UNet
# -----------------------------

class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.feature = nn.Conv2d(in_ch, out_ch, kernel, stride,
                                 padding=padding, dilation=dilation)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel, stride,
                              padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.feature(x)
        gate = torch.sigmoid(self.gate(x))
        return self.act(feat) * gate


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = GatedConv(ch, ch)
        self.conv2 = GatedConv(ch, ch)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class GatedUNet(nn.Module):
    def __init__(self, in_ch=15, out_ch=3, base=24):
        super().__init__()

        self.enc1 = nn.Sequential(
            GatedConv(in_ch, base),
            ResBlock(base),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            GatedConv(base, base * 2),
            ResBlock(base * 2),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            GatedConv(base * 2, base * 4),
            ResBlock(base * 4),
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            GatedConv(base * 4, base * 8),
            ResBlock(base * 8),
        )

        self.bottleneck = nn.Sequential(
            GatedConv(base * 8, base * 8, dilation=2),
            ResBlock(base * 8),
        )

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base*8, base*4, kernel_size=3, padding=1),  # 384 → 192
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base*4)                                      # now safe
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base*2)
        )

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base)
        )

        self.out = nn.Sequential(
            GatedConv(base, base),
            nn.Conv2d(base, out_ch, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


import torch.nn.functional as F

def cv2_to_torch_affine(M_cv, H, W, device=None, dtype=torch.float32):
    """
    M_cv: 2x3 OpenCV affine matrix, pixel coords, mapping src -> tgt
    H, W: image height/width (assumed same for src and tgt)
    Returns: theta (1, 2, 3) for torch.nn.functional.affine_grid
    """
    # 1. make it 3x3
    A = np.eye(3, dtype=np.float32)
    A[:2, :] = M_cv  # src -> tgt in pixel coords

    # 2. invert: we need tgt -> src
    A_inv = np.linalg.inv(A)  # now maps tgt_pixel -> src_pixel

    # 3. build pixel <-> norm transforms
    # norm -> pixel  (for target/output grid)
    T_out = np.array([
        [(W - 1) / 2.0, 0.0,          (W - 1) / 2.0],
        [0.0,           (H - 1) / 2.0, (H - 1) / 2.0],
        [0.0,           0.0,           1.0]
    ], dtype=np.float32)

    # pixel -> norm (for source/input sampling)
    T_in_inv = np.array([
        [2.0 / (W - 1), 0.0,           -1.0],
        [0.0,           2.0 / (H - 1), -1.0],
        [0.0,           0.0,            1.0]
    ], dtype=np.float32)

    # 4. combine:
    # v_in_norm = T_in_inv @ A_inv @ T_out @ v_out_norm
    B = T_in_inv @ A_inv @ T_out  # 3x3

    theta = B[:2, :]  # 2x3

    theta_t = torch.tensor(theta, dtype=dtype)
    if device is not None:
        theta_t = theta_t.to(device)
    return theta_t  # (1, 2, 3)


def warp_torch_image(img, M, padding_mode = "zeros"):  # img: B×C×H×W # border
    B, C, H, W = img.shape

    # Expand matrix for batch
    #M = M.expand(B, -1, -1)  # (B, 2, 3)

    # Create sampling grid
    grid = F.affine_grid(M, img.size(), align_corners=False)

    # Warp using grid_sample (differentiable)
    warped = F.grid_sample(img, grid, align_corners=False, padding_mode=padding_mode)
    return warped

def invert_affine_matrix_batch(M):   # M: (B,2,3)
    R = M[:, :, :2]                  # (B,2,2)
    t = M[:, :, 2].unsqueeze(-1)     # (B,2,1)

    R_inv = torch.inverse(R)         # (B,2,2)
    t_inv = -torch.bmm(R_inv, t)     # (B,2,1)

    M_inv = torch.cat([R_inv, t_inv], dim=-1)  # (B,2,3)
    return M_inv

def scale_affine_matrix_batch_old(M, alpha):   # M: (B,2,3), alpha in [0,1]
    # 1. Split R and t
    R = M[:, :, :2]                  # (B,2,2)
    t = M[:, :, 2]                   # (B,2)

    # 2. Identity rotation
    I = torch.eye(2, device=M.device).unsqueeze(0).expand_as(R)

    # 3. Interpolate rotation (Slerp-like, linear ok for small rotations)
    R_scaled = (1 - alpha) * I + alpha * R

    # 4. Interpolate translation
    t_scaled = alpha * t

    # 5. Rebuild affine matrix
    M_scaled = torch.cat([R_scaled, t_scaled.unsqueeze(-1)], dim=-1)
    return M_scaled
    
def scale_affine_matrix_batch(M, alpha):
    """
    M: (B,2,3) affine matrix (R | t)
    alpha: scalar or tensor in [0,1]  scale % of transform strength
    returns scaled affine matrix, preserving PURE rotation + translation
    """
    B = M.shape[0]
    
    # --- Extract rotation and translation ---
    R = M[:, :, :2]      # (B,2,2)
    t = M[:, :, 2]       # (B,2)
    
    # --- Get rotation angle from matrix ---
    # atan2 gives correct signed angle (CCW positive)
    theta = torch.atan2(R[:,1,0], R[:,0,0])       # (B,)

    # --- Interpolate rotation angle ---
    theta_scaled = alpha * theta                  # (B,)

    # --- Rebuild valid rotation matrix ---
    cos = torch.cos(theta_scaled)
    sin = torch.sin(theta_scaled)

    R_scaled = torch.zeros_like(R)
    R_scaled[:,0,0] = cos
    R_scaled[:,0,1] = -sin
    R_scaled[:,1,0] = sin
    R_scaled[:,1,1] = cos

    # --- Scale translation ---
    t_scaled = alpha * t                          # (B,2)

    # --- Combine back into affine matrix ---
    M_scaled = torch.cat([R_scaled, t_scaled.unsqueeze(-1)], dim=-1)  # (B,2,3)
    return M_scaled

def octahedral_encode_normals(normal_uint8):
    """
    normal_uint8: array of shape (..., 3), dtype uint8
                  XYZ normal in [0..255] encoding.

    Returns: array (..., 2) float32, octahedral-encoded normals in [0..1].
    """

    # Convert from [0..255] → [-1..1]
    n = normal_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0

    # Normalize to unit vectors
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    n = n / norm

    x, y, z = n[..., 0], n[..., 1], n[..., 2]

    # Octahedral projection
    denom = np.abs(x) + np.abs(y) + np.abs(z) + 1e-8
    u = x / denom
    v = y / denom

    # Fold back where z < 0
    mask = (z < 0)

    u_fold = (1 - np.abs(v)) * np.sign(u)
    v_fold = (1 - np.abs(u)) * np.sign(v)

    u = np.where(mask, u_fold, u)
    v = np.where(mask, v_fold, v)

    # Convert from [-1..1] → [0..1]
    u01 = (u * 0.5) + 0.5
    v01 = (v * 0.5) + 0.5
    
    out = np.stack([u01, v01], axis=-1).astype(np.float32)
    #print(out.shape)
    return out

import kornia as K
import kornia.feature as KF

from rembg import remove, new_session

session = None

def rembg_mask_from_rgb(rgb_np: np.ndarray) -> np.ndarray:
    global session
    if session is None:
        session = new_session('isnet-general-use')
    mask_img = remove(
        rgb_np,
        only_mask=True,
        session=session
    )
    return np.asarray(mask_img < 1, dtype=np.uint8)*255




def estimateRigidTransform_no_scale(pts1, pts2):
    # pts1, pts2 = (N, 2) numpy float32

    # 1. Compute centroids
    c1 = pts1.mean(axis=0)
    c2 = pts2.mean(axis=0)

    # 2. Subtract centroid
    X = pts1 - c1
    Y = pts2 - c2

    # 3. Solve R via SVD  (pure rotation)
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt

    # Fix possible reflection
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # 4. Get translation
    t = c2 - R @ c1

    # 5. Build affine matrix 2×3
    M = np.hstack([R, t.reshape(2, 1)])
    return M

matcher = None

def align_affine_2d(
        img_src, img_tgt,
        depth_src, depth_tgt,
        mask_src, mask_tgt,
        dist_thresh=5
    ):
    
    global matcher
    # 1. Create background masks (only use far areas)
    mask_src = cv2.bitwise_and(mask_src.astype(np.uint8) * 255,
                               rembg_mask_from_rgb(img_src))#depth_src.mean()+depth_src.std() 
                               
    mask_src = cv2.bitwise_and(mask_src, (depth_src > depth_src.std()).astype(np.uint8) * 255)

    mask_tgt = cv2.bitwise_and(mask_tgt.astype(np.uint8) * 255,
                               rembg_mask_from_rgb(img_tgt))#depth_tgt.mean()+depth_tgt.std()(depth_tgt > 0).astype(np.uint8) * 255
    
    mask_src = cv2.bitwise_and(mask_tgt, (depth_tgt > depth_tgt.std()).astype(np.uint8) * 255)
                               
    #show_imgs([img_src, mask_src, img_tgt, mask_tgt])

    # 2. Distance transform: how far from mask boundary
    dist_src = cv2.distanceTransform(mask_src, cv2.DIST_L2, 5)
    dist_tgt = cv2.distanceTransform(mask_tgt, cv2.DIST_L2, 5)

    # 3. Extract masked background
    src_bg = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    tgt_bg = cv2.bitwise_and(img_tgt, img_tgt, mask=mask_tgt)

    gray_src = cv2.cvtColor(src_bg, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(tgt_bg, cv2.COLOR_BGR2GRAY)

    # 4. Convert to tensors for LoFTR
    t_src = torch.from_numpy(gray_src / 255.).float()[None, None]   # [1,1,H,W]
    t_tgt = torch.from_numpy(gray_tgt / 255.).float()[None, None]
    t_src, t_tgt = t_src.cuda(), t_tgt.cuda()  # optional, if using CUDA
    if matcher is None:
        matcher = KF.LoFTR(pretrained="outdoor")
        matcher = matcher.cuda()
        matcher.eval()
    # 5. LoFTR matching
    input_dict = {"image0": t_src, "image1": t_tgt}
    with torch.no_grad():
        preds = matcher(input_dict)

    if preds["keypoints0"].shape[0] < 10:
        raise RuntimeError("Too few LoFTR matches")

    pts1 = preds["keypoints0"].cpu().numpy()  # (N,2)
    pts2 = preds["keypoints1"].cpu().numpy()
    
    if pts1 is None or pts2 is None:
        raise RuntimeError("ERROR: LoFTR produced no valid points after filtering")

    if pts1.shape[0] < 10 or pts2.shape[0] < 10:
        raise RuntimeError(f"Too few matches: pts1={pts1.shape}, pts2={pts2.shape}")

    # Ensure both arrays are float32
    pts1_raw = np.asarray(pts1, dtype=np.float32)
    pts2_raw = np.asarray(pts2, dtype=np.float32)
    
    

    # Ensure same number of matches
    #min_len = min(len(pts1), len(pts2))
    #pts1 = pts1[:min_len]
    #pts2 = pts2[:min_len]

    keep_1 = []
    keep_2 = []

    for (x1, y1), (x2, y2) in zip(pts1_raw, pts2_raw):
        xi1, yi1 = int(x1), int(y1)
        xi2, yi2 = int(x2), int(y2)
        # Both must be valid & far from edge
        if dist_src[yi1, xi1] > dist_thresh and dist_tgt[yi2, xi2] > dist_thresh:
            keep_1.append([x1, y1])
            keep_2.append([x2, y2])

    pts1 = np.array(keep_1, dtype=np.float32)
    pts2 = np.array(keep_2, dtype=np.float32)

    if pts1 is None or pts2 is None or len(pts1) < 10 or len(pts2) < 10:
        raise RuntimeError("Too few valid LoFTR matches after filtering")
    

    #print(pts1, pts2)

    # 7. Compute affine transform (rotation + translation)
    #M = estimateRigidTransform_no_scale(pts1, pts2)#cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

    # 8. Apply transform
    rgb_aligned = cv2.warpAffine(img_src, M, (img_tgt.shape[1], img_tgt.shape[0]))

    return rgb_aligned, M


def align_affine_2d_(
        img_src, img_tgt,
        depth_src, depth_tgt,
        mask_src, mask_tgt,
        dist_thresh=10,      # pixels from mask edge to reject ORB keypoints
        n_features=3000      # ORB feature count
    ):

    # 1. Create background masks (only use far areas)
    mask_src = cv2.bitwise_and(mask_src.astype(np.uint8) * 255,
                               (depth_src > depth_src.mean()).astype(np.uint8) * 255)

    mask_tgt = cv2.bitwise_and(mask_tgt.astype(np.uint8) * 255,
                               (depth_tgt > depth_tgt.mean()).astype(np.uint8) * 255)

    # 2. Distance transform: how far from mask boundary
    dist_src = cv2.distanceTransform(mask_src, cv2.DIST_L2, 5)
    dist_tgt = cv2.distanceTransform(mask_tgt, cv2.DIST_L2, 5)

    # 3. Extract masked background
    src_bg = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    tgt_bg = cv2.bitwise_and(img_tgt, img_tgt, mask=mask_tgt)

    gray_src = cv2.cvtColor(src_bg, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(tgt_bg, cv2.COLOR_BGR2GRAY)

    # 4. ORB features
    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(gray_src, None)
    kp2, des2 = orb.detectAndCompute(gray_tgt, None)

    # Handle failure case
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise RuntimeError("Not enough features in masked background")

    # 5. Remove keypoints near mask boundary
    def filter_keypoints(kp, des, dist_map):
        kp_f = []
        des_f = []
        for k, d in zip(kp, des):
            x, y = int(k.pt[0]), int(k.pt[1])
            if dist_map[y, x] > dist_thresh:     # keep only points far from edge
                kp_f.append(k)
                des_f.append(d)
        return kp_f, np.array(des_f) if len(des_f) > 0 else ([], None)

    kp1, des1 = filter_keypoints(kp1, des1, dist_src)
    kp2, des2 = filter_keypoints(kp2, des2, dist_tgt)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise RuntimeError("Too few valid background features after filtering")

    # 6. Matching
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:150]  # best features

    # 7. Extract matched coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)



    print(pts1, pts2)

    # 8. Compute transform — rotation + translation ONLY
    M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

    # 9. Apply transform
    rgb_aligned   = cv2.warpAffine(img_src,   M, (img_tgt.shape[1], img_tgt.shape[0]))

    return rgb_aligned, M

def oklab_similarity_numpy(pred_oklab, gt_oklab, alpha=15.0):
    """
    pred_oklab, gt_oklab = numpy arrays of shape (H, W, 3), float32 in OKLab space
    alpha = controls sensitivity (higher → stricter similarity)

    Returns:
        similarity map ∈ [0,1], shape (H, W)
    """
    # ΔL, Δa, Δb
    diff = pred_oklab - gt_oklab  # (H,W,3)

    # Euclidean distance in OKLab
    delta = np.sqrt(np.sum(diff * diff, axis=2) + 1e-6)  # (H,W)

    # convert **distance → similarity**
    similarity = np.exp(-alpha * delta)  # (H,W), ∈ (0,1]

    return similarity.astype(np.float32)

def show_imgs(list_of_imgs, titles=None, cols=3, figsize=(150, 100)):
    """
    Displays a list of images in a grid using Matplotlib.

    Args:
        list_of_imgs (list): A list where each element is an image represented 
                             as a NumPy array (e.g., loaded via PIL, OpenCV, or mpimg).
        titles (list, optional): A list of strings for subplot titles. 
                                 Must be the same length as list_of_imgs.
        cols (int, optional): The number of columns in the display grid. Defaults to 4.
        figsize (tuple, optional): The overall size of the figure (width, height) in inches.
    """
    import math
    import matplotlib.pyplot as plt
    
    n_images = len(list_of_imgs)
    cols = min(cols, n_images)
    if n_images == 0:
        print("The image list is empty.")
        return

    # Calculate the required number of rows dynamically
    rows = math.ceil(n_images / cols)
    rows = min(rows, n_images)

    # Create the figure and the axes grid
    fig, axes = plt.subplots(rows, cols)

    # Flatten the axes array for easier iteration if it's a multi-row grid
    # If there's only 1 row/col, axes might not be an array, so we handle that
    if rows == 1 and cols == 1:
        axes = [axes] # Wrap in list to make it iterable
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()


    # Iterate through the images and the axes
    for i, img in enumerate(list_of_imgs):
        ax = axes[i]
        ax.imshow(img)
        
        # Set title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i])
        
        # Hide axis ticks for cleaner display
        ax.axis('off')

    # Turn off any remaining empty subplots if n_images doesn't fill the grid perfectly
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



raft_model = None
raft_device = None
raft_trans = None

def pad8_batch(t1, t2):
    # RAFT requires H,W divisible by 8
    _, _, H, W = t1.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return t1, t2, (H, W, 0, 0)
    t1p = F.pad(t1, (0, pad_w, 0, pad_h))  # (l,r,t,b)
    t2p = F.pad(t2, (0, pad_w, 0, pad_h))
    return t1p, t2p, (H, W, pad_h, pad_w)

def crop_flow(flow, meta):
    H, W, pad_h, pad_w = meta
    return flow[:, :, :H, :W] if (pad_h or pad_w) else flow
    
    
def flow_to_grid(flow):
    """
    Converts flow [B,2,H,W] (u,v) in pixel units
    into a normalized sampling grid for F.grid_sample.
    """
    B, _, H, W = flow.shape

    # Base coordinates in normalized coordinates [-1,1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing='ij'
    )
    base_grid = torch.stack((x, y), dim=-1)   # [H, W, 2]

    # Normalize flow to [-1,1] range for grid_sample
    flow_norm_x = flow[:, 0] / ((W - 1) / 2)
    flow_norm_y = flow[:, 1] / ((H - 1) / 2)
    flow_norm = torch.stack((flow_norm_x, flow_norm_y), dim=-1)  # [B,H,W,2]

    # Add base grid → get final sampling coords
    grid = base_grid + flow_norm

    return grid

# -----------------------------
# Dataset
# -----------------------------

class StereoDisocclusionDataset(Dataset):
    """
    Expects structure like:

    training_data/
      0/
        foo.png
        foo.png_gen.png
        foo.png_gen_normal.png
        foo.png_gen_depth.png
      a/
        ...

    We treat each '*_gen.png' as a sample anchor.
    """

    def __init__(self, resolution=256, file_list=None, preload=False):
        """
        roots: list of root folders (e.g. ['training_data'])
        resolution: resize shortest side to this, then center-crop square
        file_list: optional list of (base_path) strings for train/val split
        """
        self.resolution = resolution

        if file_list is not None:
            self.samples =  file_list
        else:
            self.samples = self._scan()
        
        
            
        self.preload = preload
        self.cache = {}

        if preload:
            print(f"Preloading {len(self.samples)} samples into RAM...")
            for base in self.samples:
                self.cache[str(base)] = self._load_sample(base)
            print("Preload complete.")

    def _scan(self):
        raise Exception("dont use")

        
        
        print(f"Found {len(samples)} samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_rgb(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _load_depth(path):
        depthdat = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depthdat is None:
            raise RuntimeError(f"Could not read depth: {path}")
        depthdat = cv2.cvtColor(depthdat, cv2.COLOR_BGR2RGB)
        depth = depth_frames_helper.decode_rgb_depth_frame(depthdat, MODEL_maxOUTPUT_depth, True)
        depth = depth/MODEL_maxOUTPUT_depth #We encode the depth as 0-1 for the network. People say net's like values betwen 0 and 1
        return depth

    @staticmethod
    def _resize_and_center_crop(img, target):
        """Resize to (target,target)."""
        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)


        return img
    
    @staticmethod    
    def get_transform_and_sim(orig, warped, depth, mask, warped_next, warped_next_depth, mask_next):
        warp_next_align = warped_next.copy()
        warp_next_align_mask = warped_next.copy()
        warp_next_align_mask[mask_next] = 1
        warp_next_align_mask[~mask_next] = 0
        warp_next_align[~mask_next] = 0 #set hole to 0

        
        try:
            aligned_next_rgb, M_next_2_target = align_affine_2d(warp_next_align, orig, warped_next_depth, depth, mask_next, mask)
        except RuntimeError:
            aligned_next_rgb = warp_next_align
            M_next_2_target = np.array([
                [1, 0, 0],   # x' = 1*x + 0*y + 0
                [0, 1, 0]    # y' = 0*x + 1*y + 0
            ], dtype=np.float32)
            
        mask_aligned = cv2.warpAffine(warp_next_align_mask, M_next_2_target, (warp_next_align_mask.shape[1], warp_next_align_mask.shape[0]))
        next_2_target = cv2_to_torch_affine(M_next_2_target, warped_next.shape[0], warped_next.shape[1])
        
        
        
        #Generate frame similarity mask
        gray_alimask = cv2.cvtColor(mask_aligned, cv2.COLOR_RGB2GRAY)
        
        #mask_warp = aligned_depth == 0
        similarity = oklab_similarity_numpy(color_oklab.srgb_to_oklab(orig.astype(np.float32)/255), color_oklab.srgb_to_oklab(aligned_next_rgb.astype(np.float32)/255))
        #print(similarity.min(), similarity.mean(), similarity.max())
        
        
        #We dont care about similary where we already have data (non infill areas)
        similarity[mask] = 0
        
        # also dont care about areas where the next frame is stretched(has holes)
        similarity[gray_alimask == 0] = 0
        
        #Apply threshold and make boolean
        same_thres = 0.75#Threashold found using experimentation
        same = similarity > same_thres
        
        
        #filled = warped.copy()
        #filled[same] = aligned_next_rgb[same]
        #show_imgs([orig, warped, warp_next_align, aligned_next_rgb, filled, (same.astype(np.uint8)*255)])
        
        return next_2_target, same
    
    @staticmethod    
    def calc_sim2(orig, warp_non_hole_mask, other_frame_warped, other_frame_hole, grid):
        
        other_frame_warped_t = torch.from_numpy(other_frame_warped).permute(2, 0, 1).unsqueeze(0).float()
        warp_aligned = F.grid_sample(
            other_frame_warped_t, 
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        other_hole_mask = np.ones_like(other_frame_hole)
        gr = cv2.cvtColor(other_frame_hole, cv2.COLOR_BGR2GRAY) != 0
        other_hole_mask[gr] = 0
        other_frame_hole_t = torch.from_numpy(other_hole_mask).permute(2, 0, 1).unsqueeze(0).float()
        other_frame_hole_aligned = F.grid_sample(
            other_frame_hole_t, 
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        other_frame_non_holes = cv2.cvtColor(other_frame_hole_aligned, cv2.COLOR_BGR2GRAY) != 0
        
        #mask_warp = aligned_depth == 0
        similarity = oklab_similarity_numpy(color_oklab.srgb_to_oklab(orig.astype(np.float32)/255), color_oklab.srgb_to_oklab(warp_aligned.astype(np.float32)/255))
        #print(similarity.min(), similarity.mean(), similarity.max())
        
        
        #We dont care about similary where we already have data (non infill areas)
        similarity[~warp_non_hole_mask] = 0
        
        # also dont care about areas where the next frame is stretched(has holes)
        similarity[~other_frame_non_holes] = 0
        
        #Apply threshold and make boolean
        same_thres = 0.75#Threashold found using experimentation
        same = similarity > same_thres
        
        
        #filled = warped.copy()
        #filled[same] = aligned_next_rgb[same]
        #show_imgs([orig, warped, warp_next_align, aligned_next_rgb, filled, (same.astype(np.uint8)*255)])
        
        return same
    
    @staticmethod    
    def save_sim(mask, png_path):
        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        # Save mask (convert to 0–255 uint8)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        cv2.imwrite(png_path, mask_uint8)
    
    @staticmethod
    def load_sim(png_path):

        # Load mask
        mask_png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_png > 127)     # back to bool

        return mask
    
    
    @staticmethod    
    def save_transform_and_mask(transform, mask, png_path, npy_path):
        """
        transform: torch.Tensor (M transform)
        mask: numpy array H×W, dtype=bool or 0/1
        png_path: where to save mask as PNG
        npy_path: where to save transform as .npy
        """
        
        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        # Save mask (convert to 0–255 uint8)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        cv2.imwrite(png_path, mask_uint8)

        # Save transform (numpy array)
        np.save(npy_path, transform.detach().cpu().numpy())
        
    @staticmethod
    def load_transform_and_mask(png_path, npy_path, device="cpu"):
        """
        Returns:
            transform: torch.Tensor
            mask: numpy bool array H×W
        """

        # Load mask
        mask_png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_png > 127)     # back to bool

        # Load transform
        transform = torch.from_numpy(np.load(npy_path)).to(device)

        return transform, mask
    
    @staticmethod
    def add_gaussian_noise(img, std=0.01):
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = img + noise
        return np.clip(noisy, 0.0, 1.0)   # if your images are in [0,1]
        
    @staticmethod
    def add_sp_noise(img, prob=0.01):
        noisy = img.copy()
        mask = np.random.rand(*img.shape[:2])

        noisy[mask < prob/2] = 0.0
        noisy[mask > 1 - prob/2] = 1.0

        return noisy
    
    @staticmethod
    def add_lowfreq_noise(img, scale=0.1, size=32):
        h, w = img.shape[:2]
        noise = np.random.randn(size, size).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise *= scale
        return np.clip(img + noise[...,None], 0.0, 1.0)
    
    @staticmethod
    def augment_noise(img):
        # float32 numpy image in [0,1]
        if np.random.rand() < 0.9:
            img = StereoDisocclusionDataset.add_gaussian_noise(img, std=np.random.uniform(0.005, 0.03))

        if np.random.rand() < 0.6:
            img = StereoDisocclusionDataset.add_sp_noise(img, prob=0.01)

        if np.random.rand() < 0.4:
            img = StereoDisocclusionDataset.add_lowfreq_noise(img, scale=np.random.uniform(0.05, 0.15))

        return img
    
    @staticmethod
    def blend_numpy(img1, img2, alpha=0.5):
        return img1 * alpha + img2 * (1 - alpha)
    
    @staticmethod
    def invert_affine_batched(T):
        """
        Invert batched affine transforms of shape (B, 2, 3).
        Returns (B, 2, 3).
        """

        # Accept unbatched (2,3)
        if T.ndim == 2:
            T = T.unsqueeze(0)

        B = T.shape[0]

        # Split
        A = T[..., :2]           # (B,2,2)
        b = T[..., 2:]           # (B,2,1)

        # Invert A
        A_inv = torch.linalg.inv(A)  # (B,2,2)

        # Invert translation: b_inv = -A_inv @ b
        # Shapes: (B,2,2) @ (B,2,1) → (B,2,1)
        b_inv = -torch.matmul(A_inv, b)

        # Recombine
        T_inv = torch.cat([A_inv, b_inv], dim=-1)  # (B,2,3)

        return T_inv

    @staticmethod
    def make_pytorch_affine(T_pix, H, W, invert=False):
        """
        Converts a pixel-space affine (B,2,3) to a correct PyTorch normalized affine.
        Optionally returns its inverse in normalized space.
        """
        if T_pix.ndim == 2:
            T_pix = T_pix.unsqueeze(0)

        B = T_pix.shape[0]
        device, dtype = T_pix.device, T_pix.dtype

        # Correct: allocate real memory for batch
        T_h = torch.eye(3, device=device, dtype=dtype).repeat(B,1,1)
        T_h[:, :2, :] = T_pix

        # Pixel → normalized
        S_in_one = torch.tensor([
            [2.0/W,   0.0,    -1.0],
            [0.0,     2.0/H,  -1.0],
            [0.0,     0.0,     1.0]
        ], device=device, dtype=dtype)

        S_in = S_in_one.repeat(B,1,1)

        # Normalized → pixel
        S_out_one = torch.tensor([
            [W/2.0,    0.0,   W/2.0],
            [0.0,    H/2.0,  H/2.0],
            [0.0,      0.0,    1.0]
        ], device=device, dtype=dtype)

        S_out = S_out_one.repeat(B,1,1)

        # Compute normalized transform
        T_norm = S_in @ T_h @ torch.inverse(S_out)

        if invert:
            # Invert only the 2×2 linear part and translation
            A = T_norm[:, :2, :2]
            b = T_norm[:, :2, 2:]
            A_inv = torch.linalg.inv(A)
            b_inv = -A_inv @ b
            return torch.cat([A_inv, b_inv], dim=-1)

        return T_norm[:, :2, :]

    @staticmethod
    def calc_flow(target, sources):
        global raft_model, raft_device, raft_trans
        if raft_model is None:
            raft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raft_weights = Raft_Large_Weights.DEFAULT
            raft_trans = raft_weights.transforms()
            raft_model = raft_large(weights=raft_weights).to(raft_device).eval()
        
        pil_target = Image.fromarray(target)
        
        im1_list, im2_list = [], []
        for src in sources:
            pil_src = Image.fromarray(src)
            im1, im2 = raft_trans(pil_target, pil_src)     # [3,H,W] each, float32 normalized
            im1_list.append(im1.unsqueeze(0))
            im2_list.append(im2.unsqueeze(0))
        t1 = torch.cat(im1_list, dim=0)
        t2 = torch.cat(im2_list, dim=0)

        # Move to device
        
        t1 = t1.to(raft_device, non_blocking=True)
        t2 = t2.to(raft_device, non_blocking=True)
        
        t1p, t2p, meta = pad8_batch(t1, t2)
        
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(raft_device.type == "cuda"))
        with ctx:
            out = raft_model(t1p, t2p)
            
            # Torchvision RAFT outputs vary
        if isinstance(out, (list, tuple)):
            flow = out[-1]                      # [B,2,H',W']
        elif isinstance(out, dict) and "flow" in out:
            flow = out["flow"]
        else:
            raise TypeError(f"Unexpected RAFT output type: {type(out)}")
        flow = crop_flow(flow, meta).detach().cpu()
        
        
        return flow
    
    @staticmethod
    def load_raft(png_file_path):
        """
        Load RAFT flow saved by save_raft().
        Returns torch tensor with shape (2, H, W)
        suitable for torchvision.flow_to_image().
        """

        img = Image.open(png_file_path).convert("RGB")
        arr = np.array(img)  # uint8 [H,W,3]

        R = torch.from_numpy(arr[..., 0]).float()  # up/down encoded
        G = torch.from_numpy(arr[..., 1]).float()  # left/right encoded

        # Decode:
        #   stored = flow + 127
        #   flow   = stored - 127
        fy = R - 127.0      # vertical
        fx = G - 127.0      # horizontal

        # Return in shape (2, H, W)
        flow = torch.stack([fx, fy], dim=0)
        return flow
        
    @staticmethod
    def save_raft(flow, png_file_path):
        """
        Save RAFT flow to PNG.

        flow: PyTorch tensor [H,W,2] or [2,H,W] or [1,2,H,W]
              flow[...,0] = left/right (x)
              flow[...,1] = up/down (y)
        """
        
        os.makedirs(os.path.dirname(png_file_path), exist_ok=True)
        # Normalize shape to [H,W,2]
        if flow.ndim == 4:  # [1,2,H,W]
            flow = flow.squeeze(0)
        if flow.shape[0] == 2:  # [2,H,W]
            flow = flow.permute(1, 2, 0)

        flow = flow.cpu().float()  # ensure float32 CPU

        # Extract x,y
        fx = flow[..., 0]
        fy = flow[..., 1]

        # Clamp into [-127, +127]
        fx = torch.clamp(fx, -127, 127)
        fy = torch.clamp(fy, -127, 127)

        # Encode: -127→0, 0→127, +127→254
        R = (fy + 127).byte()  # vertical flow into R
        G = (fx + 127).byte()  # horizontal flow into G
        B = torch.zeros_like(R)  # unused

        # Stack into PNG image [H,W,3]
        img = torch.stack([R, G, B], dim=-1).numpy()

        # Save PNG
        Image.fromarray(img, mode="RGB").save(png_file_path)
    
    def _load_sample(self, sample):
        """
        Loads and returns a single sample (inp_t, orig_t, mask_t)
        but does NOT convert to torch — stay as numpy.
        """
        base = Path(sample[1])
        feat_cache_folder = "D:\\nural\\cache\\"
        orig_path   = base.with_suffix(".png")
        feat_path = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_features_cache.pt"
        warped_path = Path(str(base) + "_gen.png")
        normal_path = Path(str(base) + "_gen_normal.png")
        depth_path  = Path(str(base) + "_gen_depth.png")

        # Load raw data
        orig    = self._load_rgb(orig_path)
        warped  = self._load_rgb(warped_path)
        normals = self._load_rgb(normal_path)
        depth   = self._load_depth(depth_path)

        # Resize
        orig    = self._resize_and_center_crop(orig, self.resolution)
        warped  = self._resize_and_center_crop(warped, self.resolution)
        normals = self._resize_and_center_crop(normals, self.resolution)
        depth   = self._resize_and_center_crop(depth, self.resolution)

        orig_feat = 0#dinov3.get_features(orig, feat_path, layer_ids = global_layer_ids, dest_device = 'cpu')
        # Normalize
        orig_f = orig.astype(np.float32) / 255.0
        orig_f = color_oklab.srgb_to_oklab01(orig_f)
        warped_f = color_oklab.srgb_to_oklab01(warped.astype(np.float32) / 255.0)
        normals_f = octahedral_encode_normals(normals)
        depth_f = depth.astype(np.float32)

        hole_mask = (normals != 0).any(axis=-1).astype(np.float32)
        
        warped_prev_path = Path(str(sample[0]) + "_gen.png")
        warped_prev_normal_path = Path(str(sample[0]) + "_gen_normal.png")
        warped_prev_depth_path = Path(str(sample[0]) + "_gen_depth.png")
        warped_next_path = Path(str(sample[2]) + "_gen.png")
        warped_next_normal_path = Path(str(sample[2]) + "_gen_normal.png")
        warped_next_depth_path = Path(str(sample[2]) + "_gen_depth.png")
        
        path_next_2_target = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_next_transform.npy"
        path_prev_2_target = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_prev_transform.npy"
        same_next_path = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_next_same.png"
        same_prev_path = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_prev_same.png"
        path_prev_flow = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_prev_flow.png"
        path_next_flow = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_next_flow.png"
        same_next_path2 = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_next_same2.png"
        same_prev_path2 = feat_cache_folder + base.parent.name + os.sep + str(base.name) + "_prev_same2.png"
        
        warped_prev  = self._load_rgb(warped_prev_path)
        normals_prev = self._load_rgb(warped_prev_normal_path)
        
        warped_next  = self._load_rgb(warped_next_path)
        normals_next = self._load_rgb(warped_next_normal_path)
        
        warped_prev  = self._resize_and_center_crop(warped_prev, self.resolution)
        normals_prev = self._resize_and_center_crop(normals_prev, self.resolution)
        
        warped_next  = self._resize_and_center_crop(warped_next, self.resolution)
        normals_next = self._resize_and_center_crop(normals_next, self.resolution)
        
        #The hole mask for next and prev are True where there is valid pixels
        hole_mask_prev = (normals_prev == 0).any(axis=-1).astype(np.float32)
        hole_mask_next = (normals_next == 0).any(axis=-1).astype(np.float32)
        
        warped_f_prev = color_oklab.srgb_to_oklab01(warped_prev.astype(np.float32) / 255.0)
        warped_f_next = color_oklab.srgb_to_oklab01(warped_next.astype(np.float32) / 255.0)
        if False:
            if not os.path.exists(path_next_2_target) or not os.path.exists(path_prev_2_target) or not os.path.exists(same_next_path) or not os.path.exists(same_prev_path):
                mask = cv2.cvtColor(normals, cv2.COLOR_BGR2GRAY) == 0
                mask_next = cv2.cvtColor(normals_next, cv2.COLOR_BGR2GRAY) == 0
                warped_next_depth   = self._load_depth(warped_next_depth_path)
                warped_next_depth   = self._resize_and_center_crop(warped_next_depth, self.resolution)
                
                mask_prev = cv2.cvtColor(normals_prev, cv2.COLOR_BGR2GRAY) == 0
                warped_prev_depth   = self._load_depth(warped_prev_depth_path)
                warped_prev_depth   = self._resize_and_center_crop(warped_prev_depth, self.resolution)
                
                next_2_target, next_same = self.get_transform_and_sim(orig, warped, depth, mask, warped_next, warped_next_depth, mask_next)
                self.save_transform_and_mask(next_2_target, next_same, same_next_path, path_next_2_target)
                
                prev_2_target, prev_same = self.get_transform_and_sim(orig, warped, depth, mask, warped_prev, warped_prev_depth, mask_prev)
                self.save_transform_and_mask(prev_2_target, prev_same, same_prev_path, path_prev_2_target)
            else:
                prev_2_target, prev_same = self.load_transform_and_mask(same_prev_path, path_prev_2_target)
                next_2_target, next_same = self.load_transform_and_mask(same_next_path, path_next_2_target)
        
        if not os.path.exists(path_next_flow) or not os.path.exists(path_prev_flow):
            flow_prev, flow_next = self.calc_flow(orig, [warped_prev, warped_next])
            self.save_raft(flow_prev, path_prev_flow)
            self.save_raft(flow_next, path_next_flow)
        else:
            flow_prev = self.load_raft(path_prev_flow)
            flow_next = self.load_raft(path_next_flow)
            
        
        grid_prev = flow_to_grid(flow_prev.unsqueeze(0)).squeeze(0)
        grid_next = flow_to_grid(flow_next.unsqueeze(0)).squeeze(0)
        
        if not os.path.exists(same_next_path2) or not os.path.exists(same_prev_path2):
            orig_non_hole_mask = cv2.cvtColor(normals, cv2.COLOR_BGR2GRAY) != 0
            #mask_next_hole = cv2.cvtColor(normals_next, cv2.COLOR_BGR2GRAY) == 0
            #mask_prev_hole = cv2.cvtColor(normals_prev, cv2.COLOR_BGR2GRAY) == 0
            next_sim = self.calc_sim2(orig, orig_non_hole_mask, warped_next, normals_next, grid_next)
            prev_sim = self.calc_sim2(orig, orig_non_hole_mask, warped_prev, normals_prev, grid_prev)
            
            self.save_sim(next_sim, same_next_path2)
            self.save_sim(prev_sim, same_prev_path2)
        else:
            next_sim = self.load_sim(same_next_path2)
            prev_sim = self.load_sim(same_prev_path2)
            
        if False:
            from torchvision.utils import flow_to_image
            img = flow_to_image(flow_prev)
            out_imgs = [warped_prev, orig]
            #for img in imgs:
            out_imgs.append(img.permute(1, 2, 0).numpy())
            warped_prev_t = torch.from_numpy(warped_prev).permute(2, 0, 1).unsqueeze(0).float()
            prev_aligned = F.grid_sample(
                warped_prev_t, 
                grid_prev.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            ).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            out_imgs.append(prev_aligned)
            
            alpha = 0.50
            
            blended = cv2.addWeighted(warped_prev, alpha, orig, 1 - alpha, 0)
            out_imgs.append(blended)
            
            blended_align = cv2.addWeighted(prev_aligned, alpha, orig, 1 - alpha, 0)
            out_imgs.append(blended_align)
            
            out_imgs.append(warped)
            
            patched = warped.copy()
            patched[prev_sim] = prev_aligned[prev_sim]
            out_imgs.append(patched)
            out_imgs.append(prev_sim)
            
            show_imgs(out_imgs, titles = ['warped_prev', 'orig', 'flow', 'aligned', 'blended', 'aligned_blended', 'input', 'patchced', 'similarity'])
        
        #print(flow)
        #exit()
        
        #prev_same = prev_same.astype(np.float32)
        #next_same = next_same.astype(np.float32)
        if False:
            #Generate _2_target and same
            
            
            #prev_same = (normals != 0).any(axis=-1).astype(np.float32)
            #next_same = np.repeat(next_same[:, :, None] , 3, axis=2) # (H,W,3)
            #print(f"{hole_mask.shape=} {next_same.shape=}")
            
            
            
            #In pretraning we feed the network with a warped version of the ground truth to help it out.
            if True:
                
                ground_truth_warp = orig_f.copy()
                if True:
                    ground_truth_warp = self.augment_noise(ground_truth_warp)
                
                    next_2_target = random_affine().squeeze(0)
                    
                    prev_2_target = random_affine().squeeze(0)
                
                percent_of_groun_truth = 1.0
                
                #One out of 100 times we show the acctual images
                if np.random.rand() < 0.01:
                    percent_of_groun_truth = 0.0
                
                scale_value = 1.0 # Keep growing untill it works with 1 as of now it works with 0.3
                #print(next_2_target.shape, ground_truth_warp.shape)
                scaled_next_2_target = scale_affine_matrix_batch(next_2_target.unsqueeze(0), scale_value).squeeze(0)
                scaled_target_2_next = self.invert_affine_batched(scaled_next_2_target.unsqueeze(0)).squeeze(0)
                
                #scaled_target_2_next = invert_affine_grid_sample(scaled_next_2_target.unsqueeze(0), 256, 256).squeeze(0)
                
                warped_f_next_ = warp_torch_image(torch.from_numpy(ground_truth_warp).permute(2, 0, 1).unsqueeze(0), scaled_target_2_next.unsqueeze(0), padding_mode = "border").squeeze(0).permute(1, 2, 0)
                warped_f_next = self.blend_numpy(color_oklab.srgb_to_oklab01(warped_f_next_), warped_f_next, alpha=percent_of_groun_truth)
                
                #next_2_target = scaled_next_2_target
                
                
                scaled_prev_2_target = scale_affine_matrix_batch(prev_2_target.unsqueeze(0), scale_value).squeeze(0)
                scaled_target_2_prev = self.invert_affine_batched(scaled_prev_2_target.unsqueeze(0)).squeeze(0)
                
                #scaled_target_2_prev = invert_affine_grid_sample(scaled_prev_2_target.unsqueeze(0), 256, 256).squeeze(0)
                #scaled_target_2_prev = self.make_pytorch_affine(scaled_prev_2_target, 256, 256, invert=True)
                
                warped_f_prev_ = warp_torch_image(torch.from_numpy(ground_truth_warp).permute(2, 0, 1).unsqueeze(0), scaled_target_2_prev.unsqueeze(0), padding_mode = "border").squeeze(0).permute(1, 2, 0)
                warped_f_prev = self.blend_numpy(color_oklab.srgb_to_oklab01(warped_f_prev_), warped_f_prev, alpha=percent_of_groun_truth)
                
                #prev_2_target = scaled_prev_2_target
        
        #prev_2_target = 0
        #next_2_target = 0
        #prev_same = 0
        #next_same = 0
        
        inp = np.concatenate([
            warped_f,
            hole_mask[..., None],
            normals_f,
            depth_f[..., None],
            warped_f_prev,
            hole_mask_prev[..., None],
            warped_f_next,
            hole_mask_next[..., None]
        ], axis=-1)

        return inp, orig_f, grid_prev, prev_sim, grid_next, next_sim, hole_mask

    def __getitem__(self, idx):
        base = self.samples[idx]
        key = str(base)

        if self.preload:
            raise Exception("dont use")
            inp, orig, grid_prev, prev_same, grid_next, next_same, mask = self.cache[key]
        else:
            # fallback: load on the fly
            inp, orig, grid_prev, prev_same, grid_next, next_same, mask = self._load_sample(base)

        # Convert to tensors here
        inp_t = torch.from_numpy(inp).permute(2, 0, 1)
        orig_t = torch.from_numpy(orig).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)[None, ...]
        
        prev_same = torch.from_numpy(prev_same)[None, ...]
        next_same = torch.from_numpy(next_same)[None, ...]

        return inp_t, orig_t, grid_prev, prev_same, grid_next, next_same, mask_t

def random_affine(batch_size=1,
                  max_deg=20.0,
                  max_scale=0.2,
                  max_shift=0.25,
                  device="cpu",
                  dtype=torch.float32):
    """
    Returns (B,2,3) normalized affine matrices for affine_grid/grid_sample.

    max_deg   : maximum rotation in degrees
    max_scale : maximum scale change (0.1 → ±10%)
    max_shift : max translation in normalized coords (0.1 → ±10% of image)
    """

     # rotation
    deg = (torch.rand(1, device=device) * 2 - 1) * max_deg
    rad = deg * math.pi / 180.0
    c = torch.cos(rad)
    s = torch.sin(rad)

    # uniform scale around 1.0
    scale = 1.0 + (torch.rand(1, device=device) * 2 - 1) * max_scale

    # translation in normalized coordinates
    tx = (torch.rand(1, device=device) * 2 - 1) * max_shift
    ty = (torch.rand(1, device=device) * 2 - 1) * max_shift

    # build (1, 2, 3)
    M = torch.zeros(1, 2, 3, device=device, dtype=dtype)

    # rotation * scale
    M[:, 0, 0] =  scale * c
    M[:, 0, 1] =  scale * s
    M[:, 1, 0] = -scale * s
    M[:, 1, 1] =  scale * c

    # translation
    M[:, 0, 2] = tx
    M[:, 1, 2] = ty

    return M

def affine_norm_to_pix(M, H, W):
    # Convert normalized translation into pixel translation
    # For align_corners=False
    S = torch.tensor([
        [(W-1)/2,      0     ],
        [0,        (H-1)/2   ]
    ], dtype=M.dtype, device=M.device)

    S_inv = torch.inverse(S)

    A = M[..., :2]                 # (B,2,2)
    b = M[..., 2:]                 # (B,2,1)

    # Convert linear + translation
    A_pix = S @ A @ S_inv
    b_pix = S @ b

    return torch.cat([A_pix, b_pix], dim=-1)

def invert_affine_grid_sample(M, H, W):
    # Convert to pixel affine
    M_pix = affine_norm_to_pix(M, H, W)

    # Invert pixel affine
    A = M_pix[..., :2]
    b = M_pix[..., 2:]
    A_inv = torch.linalg.inv(A)
    b_inv = -A_inv @ b
    M_pix_inv = torch.cat([A_inv, b_inv], dim=-1)

    # Convert back to normalized form
    return affine_pix_to_norm(M_pix_inv, H, W)

def affine_pix_to_norm(M_pix, H, W):
    """
    Convert a pixel-space affine matrix (B,2,3)
    into a normalized affine grid matrix for PyTorch affine_grid
    when align_corners=False.

    Parameters:
      M_pix: (B,2,3) pixel-space affine
      H, W: image height, width

    Returns:
      M_norm: (B,2,3) normalized affine usable in affine_grid/grid_sample
    """

    if M_pix.ndim == 2:
        M_pix = M_pix.unsqueeze(0)

    A_pix = M_pix[..., :2]      # (B,2,2)
    b_pix = M_pix[..., 2:]      # (B,2,1)

    # Conversion scale matrices
    # S maps normalized → pixel
    S = torch.tensor([
        [(W-1)/2,      0     ],
        [     0,   (H-1)/2   ]
    ], dtype=M_pix.dtype, device=M_pix.device)

    # S_inv maps pixel → normalized
    S_inv = torch.inverse(S)

    # Convert linear part
    A_norm = S_inv @ A_pix @ S

    # Convert translation
    b_norm = S_inv @ b_pix

    M_norm = torch.cat([A_norm, b_norm], dim=-1)
    return M_norm


# -----------------------------
# Losses
# -----------------------------

def masked_l1(pred, gt, mask):
    # mask = (B,1,H,W), 1 inside hole
    diff = torch.abs(pred - gt) * mask
    return (diff.sum() / (mask.sum() * pred.shape[1] + 1e-6))


def identity_l1(pred, warped_oklab, mask):
    # keep original pixels where mask==0
    keep = 1.0 - mask
    return (torch.abs(pred - warped_oklab) * keep).mean()

def rgb_loss(pred, warped_oklab, mask):
    # keep original pixels where mask==0
    
    return (torch.abs(color_oklab.oklab01_to_srgb(pred) - color_oklab.oklab01_to_srgb(warped_rgb)) * mask).mean()

# Global Oklab bounds for converting 0..1 back to true Oklab
A_MIN, A_MAX = -0.233, 0.276
B_MIN, B_MAX = -0.311, 0.198

def oklab01_to_oklab_torch(x):
    """ x: (B,3,H,W) in 0..1 (L,a01,b01) → true Oklab """
    L  = x[:, 0:1]
    a01 = x[:, 1:2]
    b01 = x[:, 2:3]

    a = a01 * (A_MAX - A_MIN) + A_MIN
    b = b01 * (B_MAX - B_MIN) + B_MIN

    return torch.cat([L, a, b], dim=1)


def oklab01_perceptual_loss(pred, gt, mask=None):
    """
    pred, gt : (B, 3, H, W) in Oklab01 (0..1)
    mask     : (B,1,H,W) optional. 1 = include pixel
    """

    # Convert Oklab01 to true Oklab
    pred_oklab = oklab01_to_oklab_torch(pred)
    gt_oklab   = oklab01_to_oklab_torch(gt)

    # ΔL, Δa, Δb
    diff = pred_oklab - gt_oklab

    # Euclidean distance in true Oklab
    delta = torch.sqrt(torch.sum(diff * diff, dim=1, keepdim=True) + 1e-6)  # (B,1,H,W)

    if mask is not None:
        mask = mask.float()
        delta = delta * mask
        return delta.sum() / (mask.sum() + 1e-6)

    return delta.mean()


def gradient_loss(pred, gt, mask):
    # simple first-order gradient loss on masked region
    def grads(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    dx_p, dy_p = grads(pred)
    dx_g, dy_g = grads(gt)

    mask_x = mask[:, :, :, 1:]
    mask_y = mask[:, :, 1:, :]

    loss = (torch.abs(dx_p - dx_g) * mask_x).mean() + \
           (torch.abs(dy_p - dy_g) * mask_y).mean()
    return loss


class DINOFeatureLoss(torch.nn.Module):
    def __init__(self, layer_ids=[-1, -5], feature_weight=1.0):
        super().__init__()
        self.layer_ids = layer_ids
        self.feature_weight = feature_weight

    def forward(self, pred_feat, target_feat):
        loss = 0.0
        
        for idx in self.layer_ids:
            
            f_pred   = pred_feat[idx]      # [B, 1, C, H, W]
            f_target = target_feat[idx]    # [B, 1, C, H, W]

            # Flatten layer dimension for normalization + cosine
            # → [B, C, H, W]
            f_pred   = f_pred.squeeze(1)
            f_target = f_target.squeeze(1)

            # Normalize over channels
            f_pred   = F.normalize(f_pred, dim=1)
            f_target = F.normalize(f_target, dim=1)

            # Cosine similarity over channel dimension
            feat_loss = 1 - F.cosine_similarity(f_pred, f_target, dim=1).mean()
            loss += feat_loss

        return loss / len(self.layer_ids) * self.feature_weight


# -----------------------------
# Train / Val split helper
# -----------------------------

def build_sample_list_old(data_root):
    roots = [str(Path(data_root).resolve())]
    bases = []
    for root in roots:
        for subdir, _, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith("_gen.png"):
                    full = Path(subdir) / fname
                    base = str(full)[:-len("_gen.png")]
                    bases.append(base)
    bases.sort() #bases.sort(key=lambda x: zlib.crc32(x.encode('utf-8')))
    return bases


def build_sample_list(data_root):
    roots = [str(Path(data_root).resolve())]
    bases = []
    
    vid_samples = {}
    for root in roots:
        for subdir, _, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith("_gen.png"):
                    full = Path(subdir) / fname
                    file_name = os.path.basename(full)
                    file_parts = str(file_name).split("_vid_")
                    vid_name = file_parts[0]
                    if vid_name not in vid_samples:
                        vid_samples[vid_name] = []
                    base = str(full)[:-len("_gen.png")]
                    vid_samples[vid_name].append(base)
                    #base_fram1 = 
                    #base = str(full)[:-len("_gen.png")]
                    #samples
                    #bases.append(base)
                    
    #bases.sort() #bases.sort(key=lambda x: zlib.crc32(x.encode('utf-8')))
    out_samples = []
    for i, key in enumerate(vid_samples):
        if len(vid_samples[key]) == 3:
            out_samples.append(sorted(vid_samples[key], key=lambda p: int(p.split("vid_")[1].split(".")[0])))
    return out_samples

# -----------------------------
# Training
# -----------------------------
def get_rgb(img):
    return (color_oklab.oklab01_to_srgb(img.permute(1,2,0).detach().cpu().numpy())*255).astype(np.uint8)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build sample list and split

    all_samples = build_sample_list(args.data_root)
    
    print(f"Total samples: {len(all_samples)}")
    
    #for debuging
    #all_samples = all_samples[:5000]
    
    train_samples = []
    val_samples = []
    
    

    for base in all_samples:
        p = Path(base[0])
        # parent folder name (0..9, a..f)
        folder = p.parent.name.lower()

        if folder == "f":
            val_samples.append(base)
        else:
            train_samples.append(base)
            
    if len(val_samples) == 0: # if we  dont have any validation we take the first 50 from the training data
        val_samples = train_samples[:50]
        train_samples = train_samples[50:]
    
    #DEBUG we only need 300 images to validate; more than that is just a waste
    val_samples = val_samples[:400]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = StereoDisocclusionDataset(
        resolution=args.resolution,
        file_list=train_samples,
        preload = args.preload
    )
    val_ds = StereoDisocclusionDataset(
        resolution=args.resolution,
        file_list=val_samples,
        preload = args.preload
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    #model = GatedUNet().to(device)
    model = CrossFrameAttentionUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = GradScaler(device)

    start_epoch = 1  # default
    
    if args.auto_resume and args.resume is None:
        ckpts = sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"[Auto-Resume] checkpoint: {latest}")
            args.resume = latest
        else:
            start_epoch = 1

    # --- RESUME TRAINING ---
    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])

        # AMP scaler is NOT saved in your checkpoint, so we just keep a fresh one.
        start_epoch = ckpt["epoch"] + 1

        print(f"Resuming from epoch {ckpt['epoch']} → starting at epoch {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    using_feature_based_loss = False
    feature_warmup_epochs = 7
    
    if using_feature_based_loss:
        dino_loss_fn = DINOFeatureLoss(layer_ids=global_layer_ids, feature_weight=1.0).to(device)

    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (inp, target, grid_prev, prev_same, grid_next, next_same, mask) in enumerate(train_dl):
            inp = inp.to(device, non_blocking=True)
            pixel_target = target.to(device, non_blocking=True)
            
            prev_same = prev_same.to(device, non_blocking=True)
            grid_next = grid_next.to(device, non_blocking=True)
            grid_prev = grid_prev.to(device, non_blocking=True)
            next_same = next_same.to(device, non_blocking=True)
            
            
            blended_target = False
            #we have a warmup time wher the network is trained to learn the features of dinov3
            if using_feature_based_loss:
                target_feat = {k: v.to(device, non_blocking=True) for k, v in target_feat.items()}
                n_epoch = epoch-1
                if n_epoch < feature_warmup_epochs:
                    alpha = max(0.0, 1.0 - max(0, n_epoch-2) / (feature_warmup_epochs-2))
                    feature_representation = dinov3.get_visual_representation_batch_torch(target_feat[-3], args.resolution, args.resolution)
                    target = alpha * feature_representation + (1-alpha) * pixel_target
                    blended_target = True
                else:
                    target = pixel_target
            else:
                target = pixel_target
            #mask = mask.to(device, non_blocking=True)
            mask = inp[:, 3:4, :, :]
            
            #use_features

            warped_rgb = inp[:, :3, :, :]
            next_warped_rgb = inp[:, 11:14, :, :]
            next_warped_mask = inp[:, 14:15, :, :]
            prev_warped_rgb = inp[:, 7:10, :, :]
            prev_warped_mask = inp[:, 10:11, :, :]

            opt.zero_grad(set_to_none=True)
            #with autocast(device.type):
            if True:
                pred = model(inp)
                
                if torch.isnan(pred).any():
                    print("NaNs in pred, aborting")
                    raise SystemExit
                
                pred = pred.clamp(0.0, 1.0)
                
                loss_id = identity_l1(pred, warped_rgb, mask)
                
                
                where_use_next = next_same * mask# commented out as i cant use this next mask is in the unaligned format * next_warped_mask
                where_use_prev = prev_same * mask# commented out as i cant use this next mask is in the unalingned format * prev_warped_mask
                
                where_use_target = mask * (1 - where_use_next)
                where_use_target = where_use_target * (1 - where_use_prev)
                
                loss_hole = masked_l1(pred, target, where_use_target) * args.hole_weight                
                loss_grad = gradient_loss(pred, target, where_use_target)
                
                
                
                #multiply the next/prev_warped_mask with next/prev_warped_rgb before aligning this shold not have any effect on the comparision but hopfully can be difirentiated to allow the loss gradient to  be followed
                
                #next_aligned = warp_torch_image(next_warped_rgb*next_warped_mask, next_2_target)
                next_aligned = F.grid_sample(
                    next_warped_rgb*next_warped_mask, 
                    grid_next,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False
                )
                next_loss = masked_l1(pred, next_aligned, where_use_next)
                
                #prev_aligned = warp_torch_image(prev_warped_rgb*prev_warped_mask, prev_2_target)
                prev_aligned = F.grid_sample(
                    prev_warped_rgb*prev_warped_mask, 
                    grid_prev,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False
                )
                prev_loss = masked_l1(pred, prev_aligned, where_use_prev)
                
                
                loss = loss_hole + 0.2 * loss_grad + 0.5 * loss_id + 0.4* prev_loss + 0.4 * next_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"Epoch {epoch} [{i+1}/{len(train_dl)}] loss={avg:.4f}")
                running_loss = 0.0

        # simple val pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target, prev_2_target, prev_same, next_2_target, next_same, mask in val_dl:
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                warped_rgb = inp[:, :3, :, :]

                if True:# autocast(device.type):
                    pred = model(inp).clamp(0.0, 1.0)
                    loss_hole = masked_l1(pred, target, mask) * args.hole_weight #OLD stuff
                    #loss_hole = oklab01_perceptual_loss(pred, target, mask) * args.hole_weight
                    loss_id = identity_l1(pred, warped_rgb, mask)
                    loss_grad = gradient_loss(pred, target, mask)
                    loss = loss_hole + 0.2 * loss_grad + 0.1 * loss_id

                val_loss += loss.item()

        val_loss /= len(val_dl)
        print(f"Epoch {epoch} validation loss = {val_loss:.4f}")

        # checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "val_loss": val_loss,
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)


# -----------------------------
# Main / Argparse
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="training_data_extrin",
                        help="Root folder with 0..f subfolders")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hole_weight", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training crop size (square)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--preload', action='store_true',
                    help='if we are to preload the data (not recomended unless you have ALLOT of ram does not really effect traning speed as traning is GPU bound anyway)', required=False)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Internal watchdog flag (do not use manually)")

    args = parser.parse_args()
    
    if not args.auto_resume and args.resume is None:
        # Sometimes torch dies(from some internal crash) this makes sure that the traning continues when that happen
        # The Scaler values are not saved and restored so the loss will go up a bit after the restart but it should settle down again after a while
        
        print("[Watchdog] Starting watchdog supervisor...")
        print("[Watchdog] To disable watchdog, pass: --auto_resume or --resume")

        # Reconstruct command with --auto_resume added
        cmd = [sys.executable, sys.argv[0], "--auto_resume"] + sys.argv[1:]

        while True:
            print("[Watchdog] Launching training subprocess...")
            proc = subprocess.Popen(cmd)
            ret = proc.wait()

            if ret == 0:
                print("[Watchdog] Training finished normally. Exiting.")
                break

            print(f"[Watchdog] Training crashed (exit code {ret}). Restarting in 5 seconds...")
            time.sleep(5)

        return  # End watchdog mode
    
    #You can only use one worked if you lead everything to ram
    if args.preload:
        arg.num_workers = 1
    
    train(args)


if __name__ == "__main__":
    main()
