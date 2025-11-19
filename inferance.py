import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import os
from train import GatedUNet, StereoDisocclusionDataset



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

    from train import GatedUNet  # imports your model

    model = GatedUNet(in_ch=8, out_ch=3)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, device

# ---------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------
def infer(img, normal_img, depth_percent):
    """
    img:        (H,W,3) uint8  RGB warped image
    normal_img: (H,W,3) uint8  normals or zero outside holes
    depth:      (H,W) float32 or uint16 depth map normalized later

    Returns:
        RGB uint8 (H,W,3)
    """
    global model, device
    if model is None:
        model, device = load_model(os.path.dirname(os.path.abspath(__file__))+os.sep+"checkpoints"+os.sep+"epoch_021.pt")
    # ------------------------
    # Convert inputs
    # ------------------------
    
    org_width = img.shape[1]
    org_height = img.shape[0]
    img = cv2.resize(img, (default_resolution, default_resolution))
    normal_img = cv2.resize(normal_img, (default_resolution, default_resolution), interpolation=cv2.INTER_NEAREST)
    depth_percent = cv2.resize(depth_percent, (default_resolution, default_resolution), interpolation=cv2.INTER_NEAREST)

    
    img_f = img.astype(np.float32) / 255.0
    normals_f = normal_img.astype(np.float32) / 255.0

    # Hole mask = normals != 0
    hole_mask = (normal_img != 0).any(axis=-1).astype(np.float32)

    # ------------------------
    # Build 8-channel tensor
    # ------------------------
    inp = np.concatenate([
        img_f,                         # (H,W,3)
        hole_mask[...,None],           # (H,W,1)
        normals_f,                     # (H,W,3)
        depth_percent[...,None],       # (H,W,1)
    ], axis=-1)

    # Torch format
    inp_t = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(device)

    # ------------------------
    # Inference
    # ------------------------
    with torch.no_grad():
        pred = model(inp_t).clamp(0,1)[0]    # (3,H,W)

    
    pred_np = pred.permute(1,2,0).cpu().numpy()
    pred_np = cv2.resize(pred_np, (org_width, org_height))
    
    
    pred_np = (pred_np * 255).astype(np.uint8)


    return pred_np


def run_inference(checkpoint, base_path, resolution, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------
    # Build a 1-item dataset using your existing class
    # ----------------------------------------------------
    ds = StereoDisocclusionDataset(
        roots=[str(Path(base_path).parent)],   # folder containing images
        resolution=resolution,
        file_list=[str(Path(base_path))]       # list with ONE sample base path
    )

    inp, target, mask = ds[0]  # uses your dataset preprocessing
    inp = inp.unsqueeze(0).to(device)

    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------
    model = GatedUNet(in_ch=8, out_ch=3)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # ----------------------------------------------------
    # Inference
    # ----------------------------------------------------
    with torch.no_grad():
        pred = model(inp).clamp(0, 1)

    pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
    pred_np = (pred_np * 255).astype(np.uint8)

    # ----------------------------------------------------
    # Save output
    # ----------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_base", required=True,
                        help="Base path without suffix (foo.png, foo.png_gen.png, etc)")
    parser.add_argument("--output", default="infill.png")
    parser.add_argument("--resolution", type=int, default=default_resolution) #pick same as depth model so you are not resolution
    args = parser.parse_args()

    ground_truth = cv2.cvtColor(cv2.imread(args.input_base, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    

    run_inference(args.checkpoint, args.input_base, args.resolution, args.output)
    
    result = cv2.cvtColor(cv2.imread(args.output, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    infered = cv2.resize(result, (ground_truth.shape[1], ground_truth.shape[0]))
    
    
    normals = cv2.cvtColor(cv2.imread(args.input_base+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    alpha = 0.85
    blended_infer = cv2.addWeighted(infered, alpha, normals, 1 - alpha, 0)
    
    blended_ground_truth = cv2.addWeighted(ground_truth, alpha, normals, 1 - alpha, 0)
    
    show_imgs([
        ground_truth,
        blended_ground_truth,
        infered,
        blended_infer,
        cv2.cvtColor(cv2.imread(args.input_base+"_gen.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread(args.input_base+"_gen_normal.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ], titles = ['ground_truth', 'blended_ground_truth', 'infered', 'infered_blend', 'input rgb', 'input normal'])


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
