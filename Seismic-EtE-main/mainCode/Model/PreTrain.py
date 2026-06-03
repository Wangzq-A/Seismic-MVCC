# -*-codeing = utf-8 -*-
# Time : 2026/6/3 20:40
# @Author: 沐
# @Description:
# @FileName：


import os
import sys
import logging
from datetime import datetime
import random

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
class SeismicDataset(Dataset):
    def __init__(self, data_root, patch_size=32, sampling_interval=4, time_depth=21, is_train=False):
        self.data_root = data_root
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.time_depth = time_depth
        self.n_inline = 1767
        self.n_crossline = 576
        self.global_mean = 6340984.5000
        self.global_std = 40601594.0000
        self.is_train = is_train
        self.samples = []
        for il in range(0, self.n_inline, sampling_interval):
            for xl in range(0, self.n_crossline, sampling_interval):
                self.samples.append({'center_il': il, 'center_xl': xl})
        print(f"Dataset initialized: {len(self.samples)} patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pos = self.samples[idx]
        cil, cxl = pos['center_il'], pos['center_xl']
        if self.is_train:
            cil = int(np.clip(cil + random.randint(-2, 2), 0, self.n_inline - 1))
            cxl = int(np.clip(cxl + random.randint(-2, 2), 0, self.n_crossline - 1))
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
            s = self._load_padded(axis, max(0, min(max_idx, center_idx + offset)), secondary_pos, shape_arg)
            slices.append(s)
        stack = torch.stack(slices, dim=0).permute(1, 0, 2, 3)
        return stack.reshape(stack.shape[0] * stack.shape[1], stack.shape[2], stack.shape[3])

    def _load_padded(self, view_type, fixed_idx, center_pos, shape):
        file_path = os.path.join(self.data_root, view_type, f'slice_{fixed_idx:04d}.dat')
        full_shape = (1767, 21, 10) if view_type == 'axis_0' else (
            (576, 21, 10) if view_type == 'axis_1' else (576, 1767, 10))
        full_max = 1767 if view_type == 'axis_0' else 576
        data = np.memmap(file_path, dtype='float32', mode='r', shape=full_shape)

        if view_type in ['axis_0', 'axis_1']:
            start, end = center_pos - self.half_size, center_pos + self.half_size
            v_start, v_end = max(0, start), min(full_max, end)
            patch = data[v_start:v_end, :, :]
            if v_start - start > 0 or end - v_end > 0:
                patch = np.pad(patch, ((v_start - start, end - v_end), (0, 0), (0, 0)), mode='edge')
            return torch.from_numpy(patch).permute(2, 1, 0).float()
        else:
            cil, cxl = center_pos
            il_start, il_end = cil - self.half_size, cil + self.half_size
            xl_start, xl_end = cxl - self.half_size, cxl + self.half_size
            v_il_start, v_il_end = max(0, il_start), min(1767, il_end)
            v_xl_start, v_xl_end = max(0, xl_start), min(576, xl_end)
            patch = data[v_xl_start:v_xl_end, v_il_start:v_il_end, :]
            pad_il_l, pad_il_r = v_il_start - il_start, il_end - v_il_end
            pad_xl_l, pad_xl_r = v_xl_start - xl_start, xl_end - v_xl_end
            if (pad_il_l + pad_il_r + pad_xl_l + pad_xl_r) > 0:
                patch = np.pad(patch, ((pad_xl_l, pad_xl_r), (pad_il_l, pad_il_r), (0, 0)), mode='edge')
            return torch.from_numpy(patch).permute(2, 0, 1).float()


# --------------------------- 6. Loss & Training ---------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j, fn_threshold):
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
        similarity_matrix.masked_fill_(self_mask | mask_to_ignore, -1e9)

        return self.criterion(similarity_matrix, labels)


def train_contrastive(model, dataloader, epochs=50, device='cuda'):
    setup_logger("/root/autodl-tmp/logs_production")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = ContrastiveLoss(temperature=0.1).to(device)
    scaler = GradScaler()

    attention_history = []
    print(">>> Starting Pre-Training...")

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0

        current_fn_threshold = 1.0 if epoch <= 4 else (0.8 if epoch <= 9 else 0.5)

        last_batch = None
        for x_i, x_t, x_c, _ in pbar:
            x_i, x_t, x_c = x_i.to(device), x_t.to(device), x_c.to(device)
            last_batch = (x_i, x_c, x_t)

            optimizer.zero_grad()
            with autocast():
                p1, p2 = model(x_i, x_t, x_c)
            loss = criterion(p1.float(), p2.float(), fn_threshold=current_fn_threshold)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        if last_batch:
            model.eval()
            with torch.no_grad():
                _, importance, _ = model.forward_features(last_batch[0], last_batch[1], last_batch[2], apply_aug=False,
                                                          return_weights=True)
                avg_importance = importance.mean(dim=0).cpu().numpy()
                attention_history.append(avg_importance / np.sum(avg_importance))
            model.train()

        log(f"Epoch {epoch} Completed. Avg Loss: {epoch_loss / len(dataloader):.5f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"/root/autodl-tmp/model_epoch_SM{epoch}.pth")

    return model


# --------------------------- Main---------------------------
def main():
    config = {
        "data_root": "../model_Predata/combined_3d_by_axis",
        "batch_size": 256,
        "epochs": 30, "seed": 42,
        "train_interval": 4
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Phase 1: Contrastive Pre-Training")
    train_ds = SeismicDataset(config["data_root"], sampling_interval=config["train_interval"], is_train=True)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=10, pin_memory=True)

    model = SeismicMVCC_Pretrain(embed_dim=128).to(device)
    model = train_contrastive(model, train_dl, epochs=config["epochs"], device=device)
    torch.save(model.state_dict(), "/root/autodl-tmp/model_SM_2.pth")
    print(">>> Phase 1 Completed. Model saved to /root/autodl-tmp/model_SM_2.pth")


if __name__ == "__main__":
    main()