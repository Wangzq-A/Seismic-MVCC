# -*-codeing = utf-8 -*-
# @Author:wzq
# @Description:Data Preprocessing
# @FileName：
import numpy as np
import os

input_dir = '../../data/'
output_root = './combined_3d_by_axis'
os.makedirs(output_root, exist_ok=True)

# Create an output directory for each axis
for axis in [0, 1, 2]:
    os.makedirs(os.path.join(output_root, f'axis_{axis}'), exist_ok=True)

orig_shape = (576, 1767, 21)
dtype = np.float32

# Read all 10 files into memory (if there is enough memory) or read them one by one
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.dat')])
print(file_list)
assert len(file_list) == 10

all_data = []
for fname in file_list:
    data = np.fromfile(os.path.join(input_dir, fname), dtype=dtype).reshape(orig_shape)
    all_data.append(data)

# all_data[i].shape = (576, 1767, 21)

all_data = np.stack(all_data, axis=0)  # shape: (10, 576, 1767, 21)

# all_data[:, i, :, :] → (10, 1767, 21) → 转为 (1767, 21, 10)
print(" axis=0 ...")
for i in range(orig_shape[0]):  # 0 ~ 575
    combined = all_data[:, i, :, :]          # (10, 1767, 21)
    combined = np.transpose(combined, (1, 2, 0))  # (1767, 21, 10)
    out_path = os.path.join(output_root, 'axis_0', f'slice_{i:04d}.dat')
    combined.astype(dtype).tofile(out_path)

#  all_data[:, :, j, :] → (10, 576, 21) → (576, 21, 10)
print(" axis=1 ...")
for j in range(orig_shape[1]):  # 0 ~ 1766
    combined = all_data[:, :, j, :]          # (10, 576, 21)
    combined = np.transpose(combined, (1, 2, 0))  # (576, 21, 10)
    out_path = os.path.join(output_root, 'axis_1', f'slice_{j:04d}.dat')
    combined.astype(dtype).tofile(out_path)

#  all_data[:, :, :, k] → (10, 576, 1767) → (576, 1767, 10)
print(" axis=2 ...")
for k in range(orig_shape[2]):  # 0 ~ 20
    combined = all_data[:, :, :, k]          # (10, 576, 1767)
    combined = np.transpose(combined, (1, 2, 0))  # (576, 1767, 10)
    out_path = os.path.join(output_root, 'axis_2', f'slice_{k:04d}.dat')
    combined.astype(dtype).tofile(out_path)

print("Done! All 3D data sets have been saved by axis.")