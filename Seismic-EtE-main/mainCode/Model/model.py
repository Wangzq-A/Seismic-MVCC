# -*-codeing = utf-8 -*-
# @Author: wzq
# @Description:
# @FileName：
# -*- coding: utf-8 -*-
# Time : 2026/3/24
# Description: Core Architecture of SeismicStarSSM: End-to-End Deep Clustering Edition

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Warning: mamba_ssm not found. Using standard PyTorch fallback for SS2D (Slow, for debugging only).")

class GaussianAttention2D(nn.Module):
    def __init__(self, height=32, width=32, sigma_h=None, sigma_w=None):
        super().__init__()
        if sigma_h is None: sigma_h = height / 4.0
        if sigma_w is None: sigma_w = width / 4.0

        y = torch.arange(height).float()
        center_y = height // 2
        weight_y = torch.exp(-0.5 * ((y - center_y) / sigma_h) ** 2).view(1, 1, -1, 1)

        x = torch.arange(width).float()
        center_x = width // 2
        weight_x = torch.exp(-0.5 * ((x - center_x) / sigma_w) ** 2).view(1, 1, 1, -1)

        self.register_buffer('weights', weight_y * weight_x)

    def forward(self, x):
        return x * self.weights


# --------------------------- 3. Feature & Context Modules ---------------------------
class CAM(nn.Module):
    """ (Correlation-Aware Attribute Masking)"""

    def __init__(self, num_attributes=10, embed_dim=16, threshold=0.85):
        super().__init__()
        self.threshold = threshold
        self.num_attributes = num_attributes

        self.phi_q = nn.Linear(2, embed_dim)
        self.phi_k = nn.Linear(2, embed_dim)

        self.beta = 20.0
        self.suppress_factor = 0.2

    def forward(self, x_ref, training=True):
        B, C, H, W = x_ref.shape

        mean = x_ref.mean(dim=(2, 3))
        std = x_ref.std(dim=(2, 3), unbiased=False)
        desc = torch.stack([mean, std], dim=-1)

        Q = F.normalize(self.phi_q(desc), dim=-1)
        K = F.normalize(self.phi_k(desc), dim=-1)
        corr_matrix = torch.matmul(Q, K.transpose(1, 2))

        corr_matrix = 0.5 * (corr_matrix + corr_matrix.transpose(1, 2))
        identity = torch.eye(C, device=x_ref.device).unsqueeze(0)
        corr_matrix = corr_matrix * (1 - identity) + identity

        weights = torch.ones(B, C, device=x_ref.device)

        if training:
            redundancy = torch.sigmoid(self.beta * (corr_matrix.abs() - self.threshold)) * (1 - identity)
            red_score = redundancy.max(dim=1).values

            weights = 1.0 - red_score * (1.0 - self.suppress_factor)
            jitter = 0.8 + 0.4 * torch.rand_like(weights)
            weights = weights * jitter

        return weights.view(B, C, 1, 1), corr_matrix.mean(dim=0)


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, return_weights=False):
        b, c, _, _ = x.size()
        y_fc = self.fc(self.avg_pool(x).view(b, c))
        if return_weights:
            return y_fc
        return x * y_fc.view(b, c, 1, 1)


class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.f1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.f2 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.g = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.ReLU6()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        return self.g(x) + residual


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Warning: mamba_ssm not found. Using standard PyTorch fallback for SS2D (Slow, for debugging only).")

class GLSS2D(nn.Module):
    """
    Introduction of the Global-Local Fusion Module in Mamba SS2D (4-Directional Selective Scanning)
    """

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = math.ceil(dim / 16)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv2d = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.SiLU()


        self.x_proj = nn.Linear(dim, 4 * (d_state * 2 + self.dt_rank), bias=False)
        self.dt_projs = nn.ModuleList([nn.Linear(self.dt_rank, dim) for _ in range(4)])


        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(dim, 1)
        self.A_logs = nn.ParameterList([nn.Parameter(torch.log(A)) for _ in range(4)])
        self.Ds = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(4)])

        self.out_proj = nn.Linear(dim, dim)


        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def cross_scan(self, x):
        B, C, H, W = x.shape
        x1 = x.reshape(B, C, -1)
        x2 = torch.flip(x, dims=[2, 3]).reshape(B, C, -1)
        x3 = x.transpose(2, 3).reshape(B, C, -1)
        x4 = torch.flip(x.transpose(2, 3), dims=[2, 3]).reshape(B, C, -1)
        return [x1, x2, x3, x4]

    def cross_merge(self, ys, H, W):
        B, C, L = ys[0].shape
        y1 = ys[0].reshape(B, C, H, W)
        y2 = torch.flip(ys[1].reshape(B, C, H, W), dims=[2, 3])
        y3 = ys[2].reshape(B, C, W, H).transpose(2, 3)
        y4 = torch.flip(ys[3].reshape(B, C, W, H), dims=[2, 3]).transpose(2, 3)
        return y1 + y2 + y3 + y4

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # A. Extracting pure local features
        local_feat = self.local_conv(x)

        # B. Preparing Mamba global features
        x_flat = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm(x_flat)
        x_proj = self.in_proj(x_norm)

        # Split into the SSM branch (x_ssm) and the gated branch (z)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        x_ssm = x_ssm.permute(0, 3, 1, 2).contiguous()
        x_ssm = self.conv2d(x_ssm)
        x_ssm = self.act(x_ssm)

        # C. Cross-view Selective Scanning (SS2D)
        xs = self.cross_scan(x_ssm)  # list of (B, C, H*W)
        x_ssm_flat = x_ssm.view(B, C, -1).transpose(1, 2)  # (B, L, C)

        x_dbl = self.x_proj(x_ssm_flat)  # (B, L, 4 * (2*d_state + dt_rank))
        x_dbl = x_dbl.view(B, -1, 4, self.d_state * 2 + self.dt_rank)

        ys = []
        for i in range(4):
            x_i = xs[i]
            dt_i, B_i, C_i = torch.split(
                x_dbl[:, :, i, :],
                [self.dt_rank, self.d_state, self.d_state],
                dim=-1
            )

            dt_i = self.dt_projs[i](dt_i).transpose(1, 2).contiguous()  # (B, dim, L)
            B_i = B_i.transpose(1, 2).contiguous()  # (B, d_state, L)
            C_i = C_i.transpose(1, 2).contiguous()  # (B, d_state, L)

            A_i = -torch.exp(self.A_logs[i])  # (dim, d_state)
            D_i = self.Ds[i]  # (dim,)


            y_i = selective_scan_fn(
                    x_i, dt_i, A_i, B_i, C_i, D_i, z=None,
                    delta_bias=None, delta_softplus=True, return_last_state=False
                )

            ys.append(y_i)

        #  Mamba out
        y_merged = self.cross_merge(ys, H, W)

        # D. Gate-controlled activation
        z = self.act(z.permute(0, 3, 1, 2))
        global_feat = y_merged * z

        # Output Projection
        global_feat = self.out_proj(global_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # E. Global + Local + Residual Fusion
        return global_feat + local_feat + residual


# --------------------------- 4. Main Model ---------------------------
class SeismicMVCC_Pretrain(nn.Module):
    def __init__(self, embed_dim=128, num_attributes=10, num_slices=5):
        super().__init__()

        self.slice_fusion = nn.Sequential(
            nn.Conv2d(num_attributes * num_slices, num_attributes, 3, padding=1, groups=num_attributes, bias=False),
            nn.BatchNorm2d(num_attributes),
            nn.ReLU(inplace=True)
        )
        self.cam = CAM(num_attributes)

        mid_channels = num_attributes * 4
        self.depth_conv = nn.Conv2d(num_attributes, mid_channels, 3, padding=1, groups=num_attributes)
        self.se_attn = SEAttention(channel=mid_channels)
        self.point_conv = nn.Conv2d(mid_channels, 32, kernel_size=1)
        self.star_block = StarBlock(dim=32)
        self.mid_proj = nn.Conv2d(32, embed_dim, 3, padding=1)

        self.attn_vertical = GaussianAttention2D(height=32, width=32, sigma_h=2.0, sigma_w=5.0)
        self.attn_time = GaussianAttention2D(height=32, width=32, sigma_h=3.0, sigma_w=3.0)

        self.glss2d_inline = GLSS2D(embed_dim)
        self.glss2d_crossline = GLSS2D(embed_dim)
        self.glss2d_time = GLSS2D(embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True)
        self.cross_fusion = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

        self.projector = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 64))

    def random_shift(self, x, max_shift=2):
        if max_shift == 0: return x
        B, C, H, W = x.shape
        shift_y = random.randint(-max_shift, max_shift)
        shift_x = random.randint(-max_shift, max_shift)
        x_padded = F.pad(x, (max_shift, max_shift, max_shift, max_shift), mode='reflect')
        start_y, start_x = max_shift + shift_y, max_shift + shift_x
        return x_padded[:, :, start_y:start_y + H, start_x:start_x + W]

    def _pad_vertical(self, x):
        return F.pad(x, (0, 0, 5, 6), mode='reflect')

    def process_view(self, x, is_profile=True, return_importance=False):
        x = self.depth_conv(x)
        if return_importance:
            return self.se_attn(x, return_weights=True)
        x = self.se_attn(x)
        x = self.mid_proj(self.star_block(self.point_conv(x)))
        return self.attn_vertical(x) if is_profile else self.attn_time(x)

    def forward_features(self, x_i, x_c, x_t, apply_aug=False, return_weights=False):
        x_i_fused = self.slice_fusion(x_i)
        x_c_fused = self.slice_fusion(x_c)
        x_t_fused = self.slice_fusion(x_t)

        aug_weights, corr_mat = self.cam(x_t_fused, training=apply_aug)
        x_i = self._pad_vertical(x_i_fused * aug_weights)
        x_c = self._pad_vertical(x_c_fused * aug_weights)
        x_t = x_t_fused * aug_weights

        if return_weights:
            importance = self.process_view(x_t, is_profile=False, return_importance=True)
            importance = importance.view(importance.shape[0], 10, 4).mean(dim=-1)

        f_i = self.glss2d_inline(self.process_view(x_i, is_profile=True))
        f_c = self.glss2d_crossline(self.process_view(x_c, is_profile=True))
        f_t = self.glss2d_time(self.process_view(x_t, is_profile=False))

        B, E, H, W = f_t.shape
        memory = torch.cat([f_i.mean(dim=(2, 3)).unsqueeze(1), f_c.mean(dim=(2, 3)).unsqueeze(1)], dim=1)

        center, r = H // 2, 4
        tgt = f_t[:, :, center - r:center + r + 1, center - r:center + r + 1].permute(0, 2, 3, 1).reshape(B, -1, E)

        features = self.norm(self.cross_fusion(tgt, memory)).mean(dim=1)

        if return_weights:
            return features, importance, corr_mat
        return features

    def forward(self, x_i, x_t, x_c):
        f1 = self.forward_features(x_i, x_c, x_t, apply_aug=True)
        z1 = F.normalize(self.projector(f1), p=2, dim=1)

        x_i_shift = self.random_shift(x_i, max_shift=2)
        x_t_shift = self.random_shift(x_t, max_shift=2)
        x_c_shift = self.random_shift(x_c, max_shift=2)

        f2 = self.forward_features(x_i_shift, x_c_shift, x_t_shift, apply_aug=True)
        z2 = F.normalize(self.projector(f2), p=2, dim=1)

        return z1, z2
