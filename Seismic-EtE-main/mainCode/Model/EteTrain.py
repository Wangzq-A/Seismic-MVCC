# -*-codeing = utf-8 -*-
# @Author: wzq
# @Description:
# @FileName：
# -*- coding: utf-8 -*-
# Description: Main Program for Data Loading and Joint Training (Contrastive + DEC)

import os
import sys
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from kmeans_pytorch import kmeans as Kmeans_torch

from model import SeismicMVCC_Pretrain

# =========================== 1. Logging & Utils ===========================
logger = None


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    global logger
    logger = logging.getLogger("SeismicLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    logger.info(f"=== End-to-End Training Session Started: {timestamp} ===")
    return logger


def log(msg):
    if logger:
        logger.info(msg)
    else:
        print(msg)


# =========================== 2. Dataset ===========================
class SeismicDataset(Dataset):
    def __init__(self, data_root, patch_size=32, sampling_interval=4, time_depth=21):
        self.data_root = data_root
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.time_depth = time_depth
        self.n_inline = 1767
        self.n_crossline = 576
        self.global_mean = 6340984.5000
        self.global_std = 40601594.0000

        self.samples = []
        for il in range(0, self.n_inline, sampling_interval):
            for xl in range(0, self.n_crossline, sampling_interval):
                self.samples.append({'center_il': il, 'center_xl': xl})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pos = self.samples[idx]
        cil, cxl = pos['center_il'], pos['center_xl']

        x_inline = self._normalize_global(self._load_multi_slice('axis_0', cxl, cil, radius=2))
        x_crossline = self._normalize_global(self._load_multi_slice('axis_1', cil, cxl, radius=2))
        x_time = self._normalize_global(self._load_multi_slice('axis_2', 10, (cil, cxl), radius=2))
        return x_inline, x_time, x_crossline, pos

    def _normalize_global(self, x):
        return (x - self.global_mean) / (self.global_std + 1e-6)

    def _load_multi_slice(self, axis, center_idx, secondary_pos, radius=2):
        slices = []
        max_idx = self.n_crossline - 1 if axis == 'axis_0' else (self.n_inline - 1 if axis == 'axis_1' else 20)
        shape_arg = (self.time_depth, self.patch_size) if axis != 'axis_2' else (self.patch_size, self.patch_size)

        for offset in range(-radius, radius + 1):
            safe_idx = max(0, min(max_idx, center_idx + offset))
            slices.append(self._load_padded(axis, safe_idx, secondary_pos, shape_arg))

        stack_raw = torch.stack(slices, dim=0).permute(1, 0, 2, 3)
        c_new = stack_raw.shape[0] * stack_raw.shape[1]
        return stack_raw.reshape(c_new, stack_raw.shape[2], stack_raw.shape[3])

    def _load_padded(self, view_type, fixed_idx, center_pos, shape):
        file_path = os.path.join(self.data_root, view_type, f'slice_{fixed_idx:04d}.dat')
        full_shape = (1767, 21, 10) if view_type == 'axis_0' else (
            (576, 21, 10) if view_type == 'axis_1' else (576, 1767, 10))
        full_max = 1767 if view_type == 'axis_0' else 576
        data = np.memmap(file_path, dtype='float32', mode='r', shape=full_shape)

        if view_type in ['axis_0', 'axis_1']:
            start, end = center_pos - self.half_size, center_pos + self.half_size
            valid_start, valid_end = max(0, start), min(full_max, end)
            patch_raw = data[valid_start:valid_end, :, :]
            pad_left, pad_right = valid_start - start, end - valid_end
            if pad_left > 0 or pad_right > 0:
                patch_raw = np.pad(patch_raw, ((pad_left, pad_right), (0, 0), (0, 0)), mode='edge')
            return torch.from_numpy(patch_raw).permute(2, 1, 0).float()
        elif view_type == 'axis_2':
            cil, cxl = center_pos
            il_start, il_end = cil - self.half_size, cil + self.half_size
            xl_start, xl_end = cxl - self.half_size, cxl + self.half_size
            valid_il_start, valid_il_end = max(0, il_start), min(1767, il_end)
            valid_xl_start, valid_xl_end = max(0, xl_start), min(576, xl_end)
            patch_raw = data[valid_xl_start:valid_xl_end, valid_il_start:valid_il_end, :]
            pad_il_left, pad_il_right = valid_il_start - il_start, il_end - valid_il_end
            pad_xl_left, pad_xl_right = valid_xl_start - xl_start, xl_end - valid_xl_end
            if (pad_il_left + pad_il_right + pad_xl_left + pad_xl_right) > 0:
                patch_raw = np.pad(patch_raw, ((pad_xl_left, pad_xl_right), (pad_il_left, pad_il_right), (0, 0)),
                                   mode='edge')
            return torch.from_numpy(patch_raw).permute(2, 0, 1).float()


# =========================== 3. Losses ===========================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j, fn_threshold=1.0):
        batch_size = z_i.shape[0]
        features = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(features, features.T)

        labels = torch.cat([torch.arange(batch_size, device=z_i.device) + batch_size,
                            torch.arange(batch_size, device=z_i.device)], dim=0)
        self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=z_i.device)
        fn_mask = (similarity_matrix > fn_threshold) & (~self_mask)

        pos_mask = torch.zeros_like(self_mask)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True

        mask_to_ignore = fn_mask & (~pos_mask)
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix.masked_fill_(self_mask, -1e4)
        similarity_matrix.masked_fill_(mask_to_ignore, -1e4)

        return self.criterion(similarity_matrix, labels)


class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t().detach()

    def forward(self, q):
        p = self.target_distribution(q)
        return F.kl_div(q.log(), p, reduction='batchmean')


# =========================== 4. Training Engine ===========================
def train_end_to_end(model, dataloader, epochs=50, device='cuda', pretrained_path=None):
    setup_logger("/root/autodl-tmp/logs_production")

    if pretrained_path and os.path.exists(pretrained_path):
        log(f">>> Loading pretrained weights from {pretrained_path}...")
        state_dict = torch.load(pretrained_path, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if 'cluster_centers' not in k}
        model.load_state_dict(state_dict, strict=False)

    log(">>> Initializing Cluster Centers with K-means...")
    init_features = []
    model.eval()
    with torch.no_grad():
        for i, (x_i, x_t, x_c, _) in enumerate(tqdm(dataloader, desc="Extracting init features")):
            if i > 200: break
            f = model.forward_features(x_i.to(device), x_t.to(device), x_c.to(device))
            z = F.normalize(model.projector(f), p=2, dim=1)
            init_features.append(z)

    init_features = torch.cat(init_features, dim=0)
    _, centers = Kmeans_torch(X=init_features, num_clusters=model.num_clusters, distance='euclidean', device=device)
    model.cluster_centers.data = centers.to(device)
    log(">>> K-means Initialization Complete!")

    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'cluster_centers' not in n], 'lr': 1e-5},
        {'params': model.cluster_centers, 'lr': 1e-4}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    con_criterion = ContrastiveLoss(temperature=0.1).to(device)
    clu_criterion = ClusterLoss().to(device)
    scaler = GradScaler()

    log(">>> Starting End-to-End Joint Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0

        current_fn_threshold = 0.9 if epoch < 10 else 0.5
        alpha_clu = 15.0

        for x_i, x_t, x_c, _ in pbar:
            x_i, x_t, x_c = x_i.to(device), x_t.to(device), x_c.to(device)
            optimizer.zero_grad()

            with autocast():
                z1, z2, q1 = model(x_i, x_t, x_c)

                loss_con = con_criterion(z1.float(), z2.float(), fn_threshold=current_fn_threshold)
                loss_clu = clu_criterion(q1.float())

                avg_probs = q1.mean(dim=0)
                marginal_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
                cond_entropy = -torch.mean(torch.sum(q1 * torch.log(q1 + 1e-6), dim=1))

                loss = loss_con + alpha_clu * loss_clu + 0.1 * cond_entropy - 0.1 * marginal_entropy

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(
                {'con': f"{loss_con.item():.3f}", 'clu': f"{loss_clu.item():.3f}", 'ent': f"{cond_entropy.item():.3f}"})

        scheduler.step()
        log(f"Epoch {epoch} Completed. Avg Loss: {epoch_loss / len(dataloader):.5f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"/root/autodl-tmp/model_DEC_epoch{epoch}.pth")

    return model


if __name__ == "__main__":
    config = {
        "data_root": "../model_Predata/combined_3d_by_axis",
        "pretrained_con_path": "/root/autodl-tmp/model_SM_2.pth",
        "batch_size": 256,
        "epochs": 50,
        "seed": 42,
        "num_clusters": 4
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SeismicMVCC_Pretrain(embed_dim=128, num_clusters=config["num_clusters"]).to(device)

    print(">>> Phase 1: End-to-End Fine-Tuning")
    train_ds = SeismicDataset(config["data_root"], sampling_interval=4)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=10, pin_memory=True)

    model = train_end_to_end(model, train_dl, epochs=config["epochs"], device=device,
                             pretrained_path=config["pretrained_con_path"])

    # 显式保存最终的权重供 plot.py 推理使用
    torch.save(model.state_dict(), "/root/autodl-tmp/model_DEC_final.pth")
    print(">>> Training complete. Model saved to /root/autodl-tmp/model_DEC_final.pth")
