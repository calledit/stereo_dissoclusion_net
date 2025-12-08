import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(channels: int) -> nn.GroupNorm:
    """
    Helper to create a stable GroupNorm for a given channel count.
    """
    if channels % 8 == 0:
        return nn.GroupNorm(8, channels)
    elif channels % 4 == 0:
        return nn.GroupNorm(4, channels)
    else:
        return nn.GroupNorm(1, channels)


# ----------------- Basic U-Net building blocks -----------------

class DoubleConv(nn.Module):
    """
    2× (Conv2d -> GroupNorm -> ReLU)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            make_gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            make_gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale by 2, then DoubleConv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upscale 'x' (from deeper level), filter the skip features, concat, then DoubleConv.

    - in_ch:   channels of x (from deeper level)
    - skip_ch: channels of skip connection (from encoder at same spatial level)
    - out_ch:  output channels after fusion
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # bilinear upsample then 1x1 conv to reduce channels on the up-path
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
        )

        # 1x1 conv to filter skip features (this is where we suppress normals leaking)
        self.skip_filter = nn.Conv2d(skip_ch, skip_ch, kernel_size=1)

        # initialize skip_filter as close to identity as possible
        with torch.no_grad():
            # shape: (skip_ch, skip_ch, 1, 1)
            w = torch.eye(skip_ch).view(skip_ch, skip_ch, 1, 1)
            if self.skip_filter.weight.shape == w.shape:
                self.skip_filter.weight.copy_(w)
            else:
                nn.init.kaiming_normal_(self.skip_filter.weight, nonlinearity='relu')
            nn.init.zeros_(self.skip_filter.bias)

        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # [B, in_ch//2, H_up, W_up]

        # Handle minor size mismatches
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            skip = skip[:, :,
                        diff_y // 2: diff_y // 2 + x.size(2),
                        diff_x // 2: diff_x // 2 + x.size(3)]

        # Filter skip features (suppresses direct leakage of normals/etc)
        skip = self.skip_filter(skip)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ----------------- Cross-frame attention block -----------------

class CrossFrameAttention(nn.Module):
    """
    Attention over the frame dimension: target / prev / next.

    For each spatial location (h, w), we have 3 feature vectors:
      f_t(b, :, h, w), f_p(b, :, h, w), f_n(b, :, h, w)

    We use MultiheadAttention over this length-3 sequence, with:
      query = target vector
      key/value = [target, prev, next] vectors

    Shapes:
      f_t, f_p, f_n: [B, C, H, W]
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False,  # (L, N, E)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, f_t, f_p, f_n):
        B, C, H, W = f_t.shape

        # Work in float32 for safety
        f_t32 = f_t.to(torch.float32)
        f_p32 = f_p.to(torch.float32)
        f_n32 = f_n.to(torch.float32)

        # Stack along "frame" dimension: [B, 3, C, H, W]
        stack = torch.stack([f_t32, f_p32, f_n32], dim=1)  # (B, 3, C, H, W)
        stack = stack.permute(1, 0, 3, 4, 2)               # (3, B, H, W, C)
        seq = stack.reshape(3, B * H * W, C)               # (L=3, N=B*H*W, C)

        # Query is target only, keys/values are all frames
        q = seq[0:1, ...]  # (1, N, C)
        k = seq            # (3, N, C)
        v = seq            # (3, N, C)

        attn_out, _ = self.mha(q, k, v)  # (1, N, C) in fp32

        # Residual + layer norm
        attn_out = attn_out.squeeze(0)  # (N, C)

        f_t_flat = f_t32.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (N, C)
        fused_flat = self.norm(attn_out + f_t_flat)                 # (N, C)

        fused = fused_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return fused.to(f_t.dtype)


# ----------------- Spatial mixing / local smoothing -----------------

class SpatialMixBlock(nn.Module):
    """
    Local spatial mixing / attention:
      x + GN(Conv(ReLU(Conv(x))))

    Normalization inside the residual path stabilizes activations,
    especially in low-texture regions.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = make_gn(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        res = self.norm(res)
        return x + res


# ----------------- Shared encoder for each frame -----------------

class FrameEncoder(nn.Module):
    """
    Encoder used for:
      - target block (7 ch: rgb + mask + 2d normals + depth)
      - prev block   (4 ch: rgb + mask)
      - next block   (4 ch: rgb + mask)
    """
    def __init__(self, in_ch: int, base_ch: int = 32):
        super().__init__()
        self.inc   = DoubleConv(in_ch,     base_ch)       # -> c1
        self.down1 = Down(base_ch,         base_ch * 2)   # -> c2
        self.down2 = Down(base_ch * 2,     base_ch * 4)   # -> c3
        self.down3 = Down(base_ch * 4,     base_ch * 8)   # -> c4
        self.down4 = Down(base_ch * 8,     base_ch * 8)   # -> c5 (bottleneck)

    def forward(self, x):
        x1 = self.inc(x)      # [B, c1, H,   W]
        x2 = self.down1(x1)   # [B, c2, H/2, W/2]
        x3 = self.down2(x2)   # [B, c3, H/4, W/4]
        x4 = self.down3(x3)   # [B, c4, H/8, W/8]
        x5 = self.down4(x4)   # [B, c5, H/16,W/16]
        return [x1, x2, x3, x4, x5]


# ----------------- Full model -----------------

class CrossFrameAttentionUNet(nn.Module):
    """
    Cross-Frame Attention U-Net with:
      - GroupNorm everywhere
      - Stable spatial mixing blocks
      - Bilinear upsampling (no transpose conv)
      - Skip filters to reduce normals leaking into RGB output

    Expects input [B, 15, H, W] with layout:

        0:3   -> warped_f (RGB)
        3:4   -> hole_mask
        4:6   -> normals_f (2D normals)
        6:7   -> depth_f

        7:10  -> warped_f_prev (RGB)
        10:11 -> hole_mask_prev

        11:14 -> warped_f_next (RGB)
        14:15 -> hole_mask_next

    Outputs: [B, 3, H, W] (RGB, unclamped)
    """
    def __init__(
        self,
        in_channels: int = 15,
        base_channels: int = 32,
        num_heads: int = 4,
        out_channels: int = 3,
    ):
        super().__init__()

        # Hard-coded split for your layout:
        self.target_ch = 7  # warped_f(3) + mask(1) + normals2D(2) + depth(1)
        self.prev_ch   = 4  # warped_f_prev(3) + mask_prev(1)
        self.next_ch   = 4  # warped_f_next(3) + mask_next(1)
        assert self.target_ch + self.prev_ch + self.next_ch == in_channels, \
            f"in_channels={in_channels} doesn't match expected 15"

        # Encoders
        self.encoder_target = FrameEncoder(self.target_ch, base_ch=base_channels)
        self.encoder_prev   = FrameEncoder(self.prev_ch,   base_ch=base_channels)
        self.encoder_next   = FrameEncoder(self.next_ch,   base_ch=base_channels)

        # Channel sizes at each level
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 8

        # Cross-frame attention + spatial mixing at each level
        self.cfa_blocks = nn.ModuleList([
            CrossFrameAttention(c1, num_heads=num_heads),
            CrossFrameAttention(c2, num_heads=num_heads),
            CrossFrameAttention(c3, num_heads=num_heads),
            CrossFrameAttention(c4, num_heads=num_heads),
            CrossFrameAttention(c5, num_heads=num_heads),
        ])

        self.spatial_blocks = nn.ModuleList([
            SpatialMixBlock(c1),
            SpatialMixBlock(c2),
            SpatialMixBlock(c3),
            SpatialMixBlock(c4),
            SpatialMixBlock(c5),
        ])

        # Decoder with skip filtering in each Up block
        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: [B, 15, H, W]
        """
        # Split input into target / prev / next blocks
        x_target = x[:, 0:7,   :, :]   # warped_f, mask, normals2D, depth
        x_prev   = x[:, 7:11,  :, :]   # warped_f_prev, mask_prev
        x_next   = x[:, 11:15, :, :]   # warped_f_next, mask_next

        # Encode each block
        feats_t = self.encoder_target(x_target)  # list of 5 levels
        feats_p = self.encoder_prev(x_prev)
        feats_n = self.encoder_next(x_next)

        # Apply cross-frame attention + spatial mixing at each level
        fused_feats = []
        for lvl, (cfa, spatial) in enumerate(zip(self.cfa_blocks, self.spatial_blocks)):
            f_t = feats_t[lvl]
            f_p = feats_p[lvl]
            f_n = feats_n[lvl]

            out = cfa(f_t, f_p, f_n)   # temporal fusion
            out = spatial(out)         # local spatial mixing
            fused_feats.append(out)

        f1, f2, f3, f4, f5 = fused_feats

        # Decoder with filtered skips
        x = f5
        x = self.up1(x, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        out = self.out_conv(x)  # [B, 3, H, W]

        return out
