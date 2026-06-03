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

def selective_scan_pytorch_fallback(u, delta, A, B, C, D):
    """A slow SSM scan implementation using pure PyTorch (for backup purposes only)"""
    batch, dim, d_state = u.shape[0], A.shape[0], A.shape[1]
    seq_len = u.shape[2]

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    x = torch.zeros((batch, dim, d_state), device=u.device)
    ys = []

    for i in range(seq_len):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)
    y = y + u * D.unsqueeze(-1)
    return y


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


class AttributeCorrelationAugmenter(nn.Module):
    def __init__(self, num_attributes=10, threshold=0.85):
        super().__init__()
        self.threshold = threshold

    def forward(self, x_ref, training=True):
        B, C, H, W = x_ref.shape
        gap = x_ref.mean(dim=[2, 3])
        if B > 1:
            corr_matrix = torch.corrcoef(gap.T)
        else:
            corr_matrix = torch.eye(C, device=x_ref.device)

        weights = torch.ones(B, C, device=x_ref.device)

        if training:
            identity = torch.eye(C, device=x_ref.device)
            mask_candidates = (corr_matrix.abs() > self.threshold) & (identity == 0)
            final_mask = torch.zeros(B, C, device=x_ref.device)
            for i in range(C):
                for j in range(i + 1, C):
                    if mask_candidates[i, j]:
                        target = i if torch.rand(1) > 0.5 else j
                        final_mask[:, target] = 1.0

            suppress_factor = 0.2
            rand_dropout = (torch.rand_like(final_mask) > 0.2).float()
            scale_factor = 1.0 - (final_mask * rand_dropout * (1.0 - suppress_factor))
            weights = weights * scale_factor

            jitter = 0.8 + 0.4 * torch.rand_like(weights)
            weights = weights * jitter

        return weights.view(B, C, 1, 1), corr_matrix


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
        input = x
        x = self.norm(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        return x + input


class GLSS2D(nn.Module):
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

        local_feat = self.local_conv(x)
        x_flat = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_flat)
        x_proj = self.in_proj(x_norm)

        x_ssm, z = x_proj.chunk(2, dim=-1)
        x_ssm = x_ssm.permute(0, 3, 1, 2).contiguous()
        x_ssm = self.conv2d(x_ssm)
        x_ssm = self.act(x_ssm)

        xs = self.cross_scan(x_ssm)
        x_ssm_flat = x_ssm.view(B, C, -1).transpose(1, 2)

        x_dbl = self.x_proj(x_ssm_flat)
        x_dbl = x_dbl.view(B, -1, 4, self.d_state * 2 + self.dt_rank)

        ys = []
        for i in range(4):
            x_i = xs[i]
            dt_i, B_i, C_i = torch.split(
                x_dbl[:, :, i, :],
                [self.dt_rank, self.d_state, self.d_state],
                dim=-1
            )

            dt_i = self.dt_projs[i](dt_i).transpose(1, 2).contiguous()
            B_i = B_i.transpose(1, 2).contiguous()
            C_i = C_i.transpose(1, 2).contiguous()

            A_i = -torch.exp(self.A_logs[i])
            D_i = self.Ds[i]

            if selective_scan_fn is not None:
                y_i = selective_scan_fn(
                    x_i, dt_i, A_i, B_i, C_i, D_i, z=None,
                    delta_bias=None, delta_softplus=True, return_last_state=False
                )
            else:
                dt_i_softplus = F.softplus(dt_i)
                y_i = selective_scan_pytorch_fallback(x_i, dt_i_softplus, A_i, B_i, C_i, D_i)

            ys.append(y_i)

        y_merged = self.cross_merge(ys, H, W)
        z = self.act(z.permute(0, 3, 1, 2))
        global_feat = y_merged * z
        global_feat = self.out_proj(global_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return global_feat + local_feat + residual


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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        if return_weights: return y
        return x * y.view(b, c, 1, 1)


class SeismicStarSSM(nn.Module):
    def __init__(self, embed_dim=128, num_attributes=10, num_slices=5, num_clusters=4):
        super().__init__()
        self.num_clusters = num_clusters

        self.slice_fusion = nn.Sequential(
            nn.Conv2d(num_attributes * num_slices, num_attributes, 3, padding=1, groups=num_attributes, bias=False),
            nn.BatchNorm2d(num_attributes),
            nn.ReLU(inplace=True)
        )
        self.augmenter = AttributeCorrelationAugmenter(num_attributes)

        mid_channels = num_attributes * 4
        self.depth_conv = nn.Conv2d(num_attributes, mid_channels, 3, padding=1, groups=num_attributes)
        self.se_attn = SEAttention(channel=mid_channels, reduction=4)
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

        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, 64))
        torch.nn.init.xavier_normal_(self.cluster_centers.data)
        self.alpha = 1.0

    def _pad_vertical(self, x):
        return F.pad(x, (0, 0, 5, 6), mode='reflect')

    def random_shift(self, x, max_shift=2):
        if max_shift == 0: return x
        B, C, H, W = x.shape
        shift_y = random.randint(-max_shift, max_shift)
        shift_x = random.randint(-max_shift, max_shift)
        x_padded = F.pad(x, (max_shift, max_shift, max_shift, max_shift), mode='reflect')
        start_y, start_x = max_shift + shift_y, max_shift + shift_x
        return x_padded[:, :, start_y:start_y + H, start_x:start_x + W]

    def process_low(self, x):
        x = self.depth_conv(x)
        x = self.se_attn(x)
        x = self.point_conv(x)
        x = self.star_block(x)
        return self.mid_proj(x)

    def forward_features(self, x_inline, x_time, x_crossline, apply_aug=False):
        x_i_fused = self.slice_fusion(x_inline)
        x_c_fused = self.slice_fusion(x_crossline)
        x_t_fused = self.slice_fusion(x_time)

        aug_weights, _ = self.augmenter(x_t_fused, training=apply_aug)
        x_i = self._pad_vertical(x_i_fused * aug_weights)
        x_c = self._pad_vertical(x_c_fused * aug_weights)
        x_t = x_t_fused * aug_weights

        feat_i = self.process_low(x_i)
        feat_c = self.process_low(x_c)
        feat_t = self.process_low(x_t)

        feat_i = self.attn_vertical(feat_i)
        feat_c = self.attn_vertical(feat_c)
        feat_t = self.attn_time(feat_t)

        feat_i = self.glss2d_inline(feat_i)
        feat_c = self.glss2d_crossline(feat_c)
        feat_t = self.glss2d_time(feat_t)

        B, E, H, W = feat_t.shape
        pool_i = feat_i.mean(dim=(2, 3)).unsqueeze(1)
        pool_c = feat_c.mean(dim=(2, 3)).unsqueeze(1)
        memory = torch.cat([pool_i, pool_c], dim=1)

        center, radius = H // 2, 4
        center_t = feat_t[:, :, center - radius: center + radius + 1, center - radius: center + radius + 1]
        tgt = center_t.permute(0, 2, 3, 1).reshape(B, -1, E)

        fused_tokens = self.cross_fusion(tgt, memory)
        return self.norm(fused_tokens).mean(dim=1)

    def get_soft_assignment(self, z):
        centers = F.normalize(self.cluster_centers, p=2, dim=1)
        dist = torch.sum(torch.pow(z.unsqueeze(1) - centers, 2), 2)
        temperature = 0.1
        dist = dist / temperature

        q = 1.0 / (1.0 + dist / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, x_inline, x_time, x_crossline):
        f1 = self.forward_features(x_inline, x_time, x_crossline, apply_aug=True)
        z1 = self.projector(f1)
        z1 = F.normalize(z1, p=2, dim=1)

        if not self.training:
            return self.get_soft_assignment(z1)

        x_i_shifted = self.random_shift(x_inline, max_shift=2)
        x_t_shifted = self.random_shift(x_time, max_shift=2)
        x_c_shifted = self.random_shift(x_crossline, max_shift=2)
        f2 = self.forward_features(x_i_shifted, x_t_shifted, x_c_shifted, apply_aug=True)
        z2 = self.projector(f2)
        z2 = F.normalize(z2, p=2, dim=1)

        q1 = self.get_soft_assignment(z1)
        return z1, z2, q1