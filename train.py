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
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append("C:\\Users\\calle\\projects\\metric_depth_video_toolbox")
import depth_frames_helper
MODEL_maxOUTPUT_depth = 100.0

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
    def __init__(self, in_ch=8, out_ch=3, base=48):
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

    def __init__(self, roots, resolution=256, file_list=None, preload=True):
        """
        roots: list of root folders (e.g. ['training_data'])
        resolution: resize shortest side to this, then center-crop square
        file_list: optional list of (base_path) strings for train/val split
        """
        self.roots = [Path(r) for r in roots]
        self.resolution = resolution

        if file_list is not None:
            self.samples = [Path(p) for p in file_list]
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
        samples = []
        for root in self.roots:
            for subdir, _, files in os.walk(root):
                for fname in files:
                    if fname.lower().endswith("_gen.png"):
                        full = Path(subdir) / fname
                        base = str(full)[:-len("_gen.png")]
                        samples.append(Path(base))
        samples.sort()
        print(f"Found {len(samples)} samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_depth(self, path):
        depthdat = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depthdat is None:
            raise RuntimeError(f"Could not read depth: {path}")
        depthdat = cv2.cvtColor(depthdat, cv2.COLOR_BGR2RGB)
        depth = depth_frames_helper.decode_rgb_depth_frame(depthdat, MODEL_maxOUTPUT_depth, True)
        depth = depth/MODEL_maxOUTPUT_depth #We encode the depth as 0-1 for the network. People say net's like values betwen 0 and 1
        return depth

    def _resize_and_center_crop(self, img, target):
        """Resize to (target,target)."""
        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)


        return img
    
    def _load_sample(self, base):
        """
        Loads and returns a single sample (inp_t, orig_t, mask_t)
        but does NOT convert to torch — stay as numpy.
        """
        orig_path   = base.with_suffix(".png")
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

        # Normalize
        orig_f = orig.astype(np.float32) / 255.0
        warped_f = warped.astype(np.float32) / 255.0
        normals_f = normals.astype(np.float32) / 255.0
        depth_f = depth.astype(np.float32)

        hole_mask = (normals != 0).any(axis=-1).astype(np.float32)

        inp = np.concatenate([
            warped_f,
            hole_mask[..., None],
            normals_f,
            depth_f[..., None]
        ], axis=-1)

        return inp, orig_f, hole_mask

    def __getitem__(self, idx):
        base = self.samples[idx]
        key = str(base)

        if self.preload:
            inp, orig, mask = self.cache[key]
        else:
            # fallback: load on the fly
            inp, orig, mask = self._load_sample(base)

        # Convert to tensors here
        inp_t = torch.from_numpy(inp).permute(2, 0, 1)
        orig_t = torch.from_numpy(orig).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)[None, ...]

        return inp_t, orig_t, mask_t
    
    def old__getitem__(self, idx):
        base = self.samples[idx]  # e.g. ".../foo.png"

        # Paths
        orig_path   = base.with_suffix(".png")
        warped_path = Path(str(base) + "_gen.png")
        normal_path = Path(str(base) + "_gen_normal.png")
        depth_path  = Path(str(base) + "_gen_depth.png")

        # Load
        orig = self._load_rgb(orig_path)
        warped = self._load_rgb(warped_path)
        normals = self._load_rgb(normal_path)  # RGB normals encoded in 0..255
        depth = self._load_depth(depth_path)   # depth is normalized by this func

        # Resize / crop all consistently
        orig    = self._resize_and_center_crop(orig, self.resolution)
        warped  = self._resize_and_center_crop(warped, self.resolution)
        normals = self._resize_and_center_crop(normals, self.resolution)
        depth   = self._resize_and_center_crop(depth, self.resolution)

        # Normalize
        orig_f   = orig.astype(np.float32) / 255.0
        warped_f = warped.astype(np.float32) / 255.0

        # Normals: map 0..255 → -1..1 (roughly)
        #normals_f = normals.astype(np.float32) / 255.0 * 2.0 - 1.0
        
        #We have the normals be mapped 0-1 since networks like values 0-1
        normals_f = normals.astype(np.float32) / 255.0 

        # Hole mask: normals != 0 (any channel non-zero)
        hole_mask = (normals != 0).any(axis=-1).astype(np.float32)  # (H,W)
        
        #If you want to hide the contents of the hole
        #warped_f[normals != 0] = 0

        # Depth already approx in [0,1]
        depth_f = depth.astype(np.float32)

        # Build 8-channel input
        # Channels: [0:3] warped RGB, [3] mask, [4:7] normals, [7] depth
        inp = np.concatenate([
            warped_f,                          # (H,W,3)
            hole_mask[..., None],              # (H,W,1)
            normals_f,                         # (H,W,3)
            depth_f[..., None],                # (H,W,1)
        ], axis=-1)

        inp_t = torch.from_numpy(inp).permute(2, 0, 1)     # (C,H,W)
        orig_t = torch.from_numpy(orig_f).permute(2, 0, 1) # (3,H,W)
        mask_t = torch.from_numpy(hole_mask)[None, ...]    # (1,H,W)

        return inp_t, orig_t, mask_t


# -----------------------------
# Losses
# -----------------------------

def masked_l1(pred, gt, mask):
    # mask = (B,1,H,W), 1 inside hole
    diff = torch.abs(pred - gt) * mask
    return diff.sum() / (mask.sum() * pred.shape[1] + 1e-6)


def identity_l1(pred, warped_rgb, mask):
    # keep original pixels where mask==0
    keep = 1.0 - mask
    return (torch.abs(pred - warped_rgb) * keep).mean()


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


# -----------------------------
# Train / Val split helper
# -----------------------------

def build_sample_list(data_root):
    roots = [str(Path(data_root).resolve())]
    bases = []
    for root in roots:
        for subdir, _, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith("_gen.png"):
                    full = Path(subdir) / fname
                    base = str(full)[:-len("_gen.png")]
                    bases.append(base)
    bases.sort()
    return bases


# -----------------------------
# Training
# -----------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build sample list and split
    all_samples = build_sample_list(args.data_root)
    print(f"Total samples: {len(all_samples)}")

    all_samples = build_sample_list(args.data_root)

    train_samples = []
    val_samples = []

    for base in all_samples:
        p = Path(base)
        # parent folder name (0..9, a..f)
        folder = p.parent.name.lower()

        if folder == "f":
            val_samples.append(base)
        else:
            train_samples.append(base)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = StereoDisocclusionDataset(
        roots=[args.data_root],
        resolution=args.resolution,
        file_list=train_samples,
        preload = args.preload
    )
    val_ds = StereoDisocclusionDataset(
        roots=[args.data_root],
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
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = GatedUNet(in_ch=8, out_ch=3, base=args.base_channels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = GradScaler()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (inp, target, mask) in enumerate(train_dl):
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            warped_rgb = inp[:, :3, :, :]

            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(inp).clamp(0.0, 1.0)

                loss_hole = masked_l1(pred, target, mask) * args.hole_weight
                loss_id = identity_l1(pred, warped_rgb, mask)
                loss_grad = gradient_loss(pred, target, mask)

                loss = loss_hole + 0.2 * loss_grad + 0.1 * loss_id

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
            for inp, target, mask in val_dl:
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                warped_rgb = inp[:, :3, :, :]

                with autocast():
                    pred = model(inp).clamp(0.0, 1.0)
                    loss_hole = masked_l1(pred, target, mask) * args.hole_weight
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
    parser.add_argument("--data_root", type=str, default="training_data",
                        help="Root folder with 0..f subfolders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hole_weight", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training crop size (square)")
    parser.add_argument("--val_fraction", type=float, default=0.02,
                        help="Fraction of samples for validation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--preload', action='store_true', help='if we are to preload the data', required=False)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    
    #You can only use one worked if you lead everything to ram
    if args.preload:
        arg.num_workers = 1
    
    train(args)


if __name__ == "__main__":
    main()
