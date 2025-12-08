import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import os
from train import GatedUNet, CrossFrameAttentionUNet, StereoDisocclusionDataset, octahedral_encode_normals
import time
import color_oklab
import glob
import re

default_resolution = 256

device = None
model = None
# ---------------------------------------------------------------
# Create model once and load checkpoint
# ---------------------------------------------------------------
def load_model(checkpoint_path):
    
    global device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    model = CrossFrameAttentionUNet() #GatedUNet()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, device

# ---------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------
def infer(img, normal_img, depth_percent, prev_frame, prev_mask, next_frame, next_mask):
    """
    img:        (H,W,3) uint8  RGB warped image
    normal_img: (H,W,3) uint8  normals or zero outside holes
    depth:      (H,W) float32 or uint16 depth map normalized later

    Returns:
        RGB uint8 (H,W,3)
    """
    global model, device
    if model is None:
        model, device = load_model(os.path.dirname(os.path.abspath(__file__))+os.sep+"checkpoints"+os.sep+"epoch_139.pt")
    # ------------------------
    # Convert inputs
    # ------------------------
    
    org_width = img.shape[1]
    org_height = img.shape[0]
    #img = cv2.resize(img, (default_resolution, default_resolution))
    #normal_img = cv2.resize(normal_img, (default_resolution, default_resolution), interpolation=cv2.INTER_NEAREST)
    #depth_percent = cv2.resize(depth_percent, (default_resolution, default_resolution), interpolation=cv2.INTER_NEAREST)

    
    img_f = color_oklab.srgb_to_oklab01(img.astype(np.float32) / 255.0)
    normals_f = octahedral_encode_normals(normal_img)
    #print (normals_f)

    # Hole mask = normals != 0
    hole_mask = (normal_img != 0).any(axis=-1).astype(np.float32)
    
    #show_imgs([hole_mask])
    
    prev_frame_f = color_oklab.srgb_to_oklab01(prev_frame.astype(np.float32) / 255.0)
    next_frame_f = color_oklab.srgb_to_oklab01(next_frame.astype(np.float32) / 255.0)
    
    prev_mask = prev_mask.astype(np.float32)
    next_mask = next_mask.astype(np.float32)
    
    # ------------------------
    # Build 8-channel tensor
    # ------------------------
    inp = np.concatenate([
        img_f,                         # (H,W,3)
        hole_mask[...,None],           # (H,W,1)
        normals_f,                     # (H,W,3)
        depth_percent[...,None],       # (H,W,1)
        prev_frame_f,
        prev_mask[..., None],
        next_frame_f,
        next_mask[..., None]
    ], axis=-1)

    # Torch format
    inp_t = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(device)

    # ------------------------
    # Inference
    # ------------------------
    with torch.no_grad():
        pred = model(inp_t).clamp(0,1)[0]    # (3,H,W)

    
    pred_np = pred.permute(1,2,0).cpu().numpy()
    
    pred_np = color_oklab.oklab01_to_srgb(pred_np)
    
    #pred_np = cv2.resize(pred_np, (org_width, org_height))
    
    
    pred_np = (pred_np * 255).astype(np.uint8)


    return pred_np


def get_sorted_vid_files(example_path):
    """
    Given one file path, return ONLY files that match *_vid_<number>.png
    sorted numerically by the number. Returns only file paths, nothing else.
    """
    folder = os.path.dirname(example_path)

    # Extract prefix up to 'vid_'
    filename = os.path.basename(example_path)
    prefix = filename.split("vid_")[0] + "vid_"

    pattern = os.path.join(folder, prefix + "*.png")
    files = glob.glob(pattern)

    # Filter strictly and sort
    def extract_num(path):
        match = re.search(r"_vid_(\d+)\.png$", os.path.basename(path))
        return int(match.group(1)) if match else float('inf')

    return sorted(
        [f for f in files if re.search(r"_vid_\d+\.png$", os.path.basename(f))],
        key=extract_num
    )

def run_inference_on_dataset_item(checkpoint, sample, resolution, output_path):
    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------
    global model, device
    if model is None:
        model, device = load_model(checkpoint)
    
    
    
    # ----------------------------------------------------
    # Build a 1-item dataset using your existing class
    # ----------------------------------------------------
    ds = StereoDisocclusionDataset(
        resolution=resolution,
        file_list = [sample]
        #samples=[sample]       # list with ONE sample base path
    )

    inp_one, target, prev_2_target, prev_same, next_2_target, next_same, mask = ds[0]  # uses your dataset preprocessing
    
    inp = inp_one.unsqueeze(0).to(device)

    

    # ----------------------------------------------------
    # Inference
    # ----------------------------------------------------
    
    #with torch.no_grad():
    #    pred = model(inp).clamp(0, 1)
    
    #start = time.perf_counter()
    with torch.no_grad():
        pred = model(inp).clamp(0, 1)
    #end = time.perf_counter()

    #elapsed_ms = (end - start) * 1000
    #print(f"\n===== Inference time =====")
    #print(f"{elapsed_ms:.2f} ms")
    #print("==========================\n")

    pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
    
    pred_np = color_oklab.oklab01_to_srgb(pred_np)
    pred_np = (pred_np * 255).astype(np.uint8).clip(0,  255)

    # ----------------------------------------------------
    # Save output
    # ----------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)
    print(inp_one.shape)
    warped_rgb = (color_oklab.oklab01_to_srgb(inp_one[:3, :, :].permute(1, 2, 0).cpu().numpy())* 255).astype(np.uint8).clip(0,  255)
    next_warped_rgb = (color_oklab.oklab01_to_srgb(inp_one[11:14, :, :].permute(1, 2, 0).cpu().numpy())* 255).astype(np.uint8).clip(0,  255)
    prev_warped_rgb = (color_oklab.oklab01_to_srgb(inp_one[7:10, :, :].permute(1, 2, 0).cpu().numpy())* 255).astype(np.uint8).clip(0,  255)
    
    return pred_np, warped_rgb, next_warped_rgb, prev_warped_rgb


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
    return theta_t.unsqueeze(0)  # (1, 2, 3)


def warp_torch_image(img, M):  # img: B×C×H×W
    B, C, H, W = img.shape

    # Expand matrix for batch
    M = M.expand(B, -1, -1)  # (B, 2, 3)

    # Create sampling grid
    grid = F.affine_grid(M, img.size(), align_corners=False)

    # Warp using grid_sample (differentiable)
    warped = F.grid_sample(img, grid, align_corners=False)
    return warped



def align_image(target_rgb, src_rgb, warp_mode=cv2.MOTION_EUCLIDEAN):
    """
    Align `src_rgb` to `target_rgb` using ECC alignment.
    Handles both grayscale and RGB input.

    Returns:
        aligned_rgb (np.ndarray): aligned version of src_rgb
        warp_matrix (np.ndarray)
        cc (float)
    """

    # Convert to grayscale for ECC (required!)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2GRAY) \
        if target_rgb.ndim == 3 else target_rgb
    src_gray = cv2.cvtColor(src_rgb, cv2.COLOR_BGR2GRAY) \
        if src_rgb.ndim == 3 else src_rgb

    # Float32
    im1 = target_gray.astype(np.float32)
    im2 = src_gray.astype(np.float32)

    # Initialize warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # ECC criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )

    # Compute transform
    cc, warp_matrix = cv2.findTransformECC(
        im1, im2, warp_matrix, warp_mode, criteria
    )

    # Warp the *RGB* source image
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(
            src_rgb, warp_matrix, (target_rgb.shape[1], target_rgb.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    else:
        aligned = cv2.warpAffine(
            src_rgb, warp_matrix, (target_rgb.shape[1], target_rgb.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

    return aligned#, warp_matrix, cc


def vis_image(file_thing, args):
    
    
    sample = get_sorted_vid_files(file_thing)
    base_image = sample[1]
    
    ground_truth = cv2.cvtColor(cv2.imread(base_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    size = (ground_truth.shape[1], ground_truth.shape[0])
    
    prediction, warped_rgb, next_warped_rgb, prev_warped_rgb = run_inference_on_dataset_item(args.checkpoint, sample, args.resolution, args.output)
    
    #show_imgs([ground_truth, warped_rgb, next_warped_rgb, prev_warped_rgb])
    
    normals = cv2.cvtColor(cv2.imread(base_image+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    prediction = cv2.resize(prediction, size)
    warped_rgb = cv2.resize(warped_rgb, size)
    next_warped_rgb = cv2.resize(next_warped_rgb, size)
    prev_warped_rgb = cv2.resize(prev_warped_rgb, size)
    
    alpha = 0.50
    blended = cv2.addWeighted(prev_warped_rgb, alpha, warped_rgb, 1 - alpha, 0)
    alpha = 2/3
    blended = cv2.addWeighted(blended, alpha, next_warped_rgb, 1 - alpha, 0)
    
    #This basic alignment just makes the alignment worse for most images
    #You could align better using more advanced methods but i dont see any reason to have that in the visualiser
    
    #aligned_next_warped_rgb = align_image(warped_rgb, next_warped_rgb)
    #aligned_prev_warped_rgb = align_image(warped_rgb, prev_warped_rgb)

    #alpha = 0.50
    #blended_aligned = cv2.addWeighted(aligned_prev_warped_rgb, alpha, warped_rgb, 1 - alpha, 0)
    #alpha = 2/3
    #blended_aligned = cv2.addWeighted(blended_aligned, alpha, aligned_next_warped_rgb, 1 - alpha, 0)

    show_imgs([
        ground_truth,
        
        prediction,
        normals,
        warped_rgb,
        prev_warped_rgb,
        next_warped_rgb,
        blended,
    ], titles = ['ground_truth', 'infered', 'normals', 'input rgb',  'prev frame', 'next frame', 'blended'])

    return
    
    input_img = cv2.cvtColor(cv2.imread(base_image+"_gen.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    input_next_img = cv2.cvtColor(cv2.imread(sample[2]+"_gen.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    input_prev_img = cv2.cvtColor(cv2.imread(sample[0]+"_gen.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    target_depth = StereoDisocclusionDataset._load_depth(base_image+"_gen_depth.png")*100
    
    next_depth = StereoDisocclusionDataset._load_depth(sample[2]+"_gen_depth.png")*100
    prev_depth = StereoDisocclusionDataset._load_depth(sample[2]+"_gen_depth.png")*100
    
    normals = cv2.cvtColor(cv2.imread(base_image+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    next_normals = cv2.cvtColor(cv2.imread(sample[2]+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    prev_normals = cv2.cvtColor(cv2.imread(sample[0]+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    align_mask = cv2.cvtColor(normals, cv2.COLOR_BGR2GRAY) == 0
    align_mask_next = cv2.cvtColor(next_normals, cv2.COLOR_BGR2GRAY) == 0
    align_mask_prev = cv2.cvtColor(prev_normals, cv2.COLOR_BGR2GRAY) == 0
    
    #input_next_img[~align_mask_next] = 0
    
    similarity = input_next_img
    aligned_rgb = similarity
    filled = input_img.copy()
    if False:
        #try:
        #
        #    #aligned_rgb, M = align_affine_2d(input_next_img, input_img, next_depth, target_depth, align_mask_next, align_mask)
        #except:
        #    aligned_rgb = input_next_img
        #    M = np.array([
        #        [1, 0, 0],   # x' = 1*x + 0*y + 0
        #        [0, 1, 0]    # y' = 0*x + 1*y + 0
        #    ], dtype=np.float32)
            
        torch_aff, same = StereoDisocclusionDataset.get_transform_and_sim(ground_truth, input_img, target_depth, align_mask, input_next_img, next_depth, align_mask_next)
        #print(input_next_img.shape)
        #torch_aff = cv2_to_torch_affine(M, input_next_img.shape[0], input_next_img.shape[1])
        
        im_img = torch.from_numpy(input_next_img.astype(np.float32)/255).permute(2, 0, 1).unsqueeze(0)
        
        aligned_rgb = warp_torch_image(im_img, torch_aff).squeeze(0)
        aligned_rgb = (aligned_rgb.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)
        
        #gray_ali = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
        
        #mask_warp = aligned_depth == 0
        #similarity = oklab_similarity_numpy(color_oklab.srgb_to_oklab(ground_truth.astype(np.float32)/255), color_oklab.srgb_to_oklab(aligned_rgb.astype(np.float32)/255))
        #print(similarity.min(), similarity.mean(), similarity.max())
        #similarity = np.abs(cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY) - gray_ali)
        
        #similarity = (((rgb_ssim(ground_truth, aligned_rgb)+1)/2) > 0.8).astype(np.uint8) *255
        
        #We dont care about similary where we already have data (is non infill areas)
        #similarity[align_mask] = 0
        #similarity[gray_ali == 0] = 0
        
        #Apply threshold and make boolean
        #same_thres = 0.75#Threashold found using experimentation
        #similarity[similarity < same_thres] = 0
        
        #fill in image based on similar areas
        #filled[similarity > same_thres] = aligned_rgb[similarity > same_thres]
        filled[same] = aligned_rgb[same]
        
        torch_aff_prev, same_prev = StereoDisocclusionDataset.get_transform_and_sim(ground_truth, input_img, target_depth, align_mask, input_prev_img, prev_depth, align_mask_prev)
        
        
        im_img = torch.from_numpy(input_prev_img.astype(np.float32)/255).permute(2, 0, 1).unsqueeze(0)
        aligned_rgb_prev = warp_torch_image(im_img, torch_aff_prev).squeeze(0)
        aligned_rgb_prev = (aligned_rgb_prev.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)
        
        filled[same_prev] = aligned_rgb_prev[same_prev]
        
        
        alpha = 0.50
        #blended_infer = cv2.addWeighted(infered, alpha, normals, 1 - alpha, 0)
        blended_align = cv2.addWeighted(ground_truth, alpha, aligned_rgb, 1 - alpha, 0)
        #blended_align = aligned_rgb.copy()#ground_truth.copy()
        blended_align = cv2.addWeighted(blended_align, alpha, aligned_rgb_prev, 1 - alpha, 0)
        
    #except:
    #    pass
    
    """
    
    
    
    trg_mask = target_depth < target_depth.mean()
    src_mask = next_depth < next_depth.mean()
    
    aligned_next_depth = align_depth(next_depth, src_mask, target_depth, trg_mask)
    """

    #run_inference_on_dataset_item(args.checkpoint, sample, args.resolution, args.output)
    
    result = cv2.cvtColor(cv2.imread(args.output, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    infered = cv2.resize(result, (ground_truth.shape[1], ground_truth.shape[0]))
    
    
    normals = cv2.cvtColor(cv2.imread(base_image+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    alpha = 0.85
    blended_infer = cv2.addWeighted(infered, alpha, normals, 1 - alpha, 0)
    blended_ground_truth = cv2.addWeighted(ground_truth, alpha, normals, 1 - alpha, 0)
    
    
    f_img = ground_truth.astype(np.float32) / 255.0
    
    gf = ground_truth.astype(np.float32) / 255.0
    ok = color_oklab.srgb_to_oklab(gf)
    
    
    #print("L:", ok[...,0].min(), ok[...,0].max())
    #print("a:", ok[...,1].min(), ok[...,1].max())
    #print("b:", ok[...,2].min(), ok[...,2].max())

    ok_ground_truth = color_oklab.srgb_to_oklab01(gf)
    rgb_ok_ground_truth = color_oklab.oklab01_to_srgb(ok_ground_truth)
    
    show_imgs([
        ground_truth,
        blended_ground_truth,
        infered,
        blended_infer,
        warped_rgb,
        normals,
        prev_warped_rgb,
        next_warped_rgb,
        aligned_rgb,
        similarity,
        filled,
        blended_align
    ], titles = ['ground_truth', 'blended_ground_truth', 'infered', 'infered_blend', 'input rgb', 'input normal', 'prev frame', 'next frame', 'aligned_rgb', 'similarity', 'filled', 'blended_align'])

def rgb_ssim(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1 * mu1
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2 * mu2
    sigma12   = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("input",
                        help="Base path without suffix (foo.png, foo.png_gen.png, etc) or folder containing those")
    parser.add_argument("--output", default="infill.png")
    parser.add_argument("--resolution", type=int, default=default_resolution) #pick same as depth model so you are not resolution
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        print("input is folder")
        for entry in os.scandir(args.input):
            if entry.is_file() and entry.name.endswith("vid_1.png"):
                print(entry.path)   # full path to each matching .png file
                vis_image(entry.path, args)
    else:
        vis_image(args.input, args)


import cv2
import numpy as np

def align_affine_2d(
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

    # 8. Compute transform — rotation + translation ONLY
    M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

    # 9. Apply transform
    rgb_aligned   = cv2.warpAffine(img_src,   M, (img_tgt.shape[1], img_tgt.shape[0]))

    return rgb_aligned, M



def align_affine_2d_old(img_src, img_tgt, depth_src, depth_tgt, mask_src, mask_tgt):
    # 1. create background masks
    mask_src = cv2.bitwise_and(mask_src.astype(np.uint8) * 255, (depth_src > depth_src.mean()).astype(np.uint8) * 255)
    mask_tgt = cv2.bitwise_and(mask_tgt.astype(np.uint8) * 255, (depth_tgt > depth_tgt.mean()).astype(np.uint8) * 255)


    # 3. masked input for matching
    src_bg = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    tgt_bg = cv2.bitwise_and(img_tgt, img_tgt, mask=mask_tgt)

    gray_src = cv2.cvtColor(src_bg, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(tgt_bg, cv2.COLOR_BGR2GRAY)

    # 4. ORB keypoints & descriptors
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray_src, None)
    kp2, des2 = orb.detectAndCompute(gray_tgt, None)

    # 5. Matching
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:150]

    # 6. Coordinates of matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # 7. Estimate only rotation + translation (no scaling!)
    M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

    # 8. Apply transform to RGB and depth
    rgb_aligned   = cv2.warpAffine(img_src,   M, (img_tgt.shape[1], img_tgt.shape[0]))
    depth_aligned = cv2.warpAffine(depth_src, M, (img_tgt.shape[1], img_tgt.shape[0]))

    return rgb_aligned, depth_aligned, M


def align_depth(depth_src, mask_src, depth_tgt, mask_tgt):
    src = depth_src.copy()
    tgt = depth_tgt.copy()
    src[mask_src == 0] = 0
    tgt[mask_tgt == 0] = 0

    # 2. Prioritise background (farther = larger weight)
    # Normalize and square to boost background
    src_w = cv2.normalize(src, None, 0, 1, cv2.NORM_MINMAX) ** 2
    tgt_w = cv2.normalize(tgt, None, 0, 1, cv2.NORM_MINMAX) ** 2

    # 3. Use ECC image alignment (fast & robust)
    warp_matrix = np.eye(2, 3, dtype=np.float32)  # affine transform

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                80,  # max number of iterations
                1e-6)  # convergence threshold

    (cc, warp_matrix) = cv2.findTransformECC(
        src_w,        # source image (weighted)
        tgt_w,        # target image (weighted)
        warp_matrix,  # initial transform
        cv2.MOTION_AFFINE,
        criteria,
        inputMask=mask_src.astype(np.uint8)  # mask used only for source
    )

    # 4. Apply transform to original depth frame
    aligned_src = cv2.warpAffine(
        depth_src,
        warp_matrix,
        (depth_src.shape[1], depth_src.shape[0]),
        flags=cv2.INTER_LINEAR
    )
    return aligned_src

import math
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    main()
