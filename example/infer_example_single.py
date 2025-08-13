#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PAZNet single-case inference example.

Usage:
  python example/infer_example_single.py \
      --mri-file testdata_single/pre_norm.nii \
      --tem-file testdata_single/tem_norm.nii \
      --weights module/net200.pth \
      --save-dir ./test_output
"""

import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
from paznet import generate_model


def main():
    parser = argparse.ArgumentParser("PAZNet single-case inference")
    parser.add_argument('--mri-file', required=True, help='Path to pre_norm.nii')
    parser.add_argument('--tem-file', required=True, help='Path to tem_norm.nii')
    parser.add_argument('--weights',  required=True, help='Path to model weights (.pth)')
    parser.add_argument('--save-dir', required=True, help='Directory to save prediction')
    parser.add_argument('--device',   default='cuda:0', help='cuda:0 or cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')

    # ---- 1. 加载模型 ----
    model, _ = generate_model(training=False, no_cuda=(device.type == 'cpu'),
                              gpu_id=[0], phase='test')
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # ---- 2. 读取 MRI 和 TEM ----
    mri_img = sitk.ReadImage(args.mri_file, sitk.sitkFloat32)
    mri_array = sitk.GetArrayFromImage(mri_img)

    tem_img = sitk.ReadImage(args.tem_file, sitk.sitkFloat32)
    tem_array = sitk.GetArrayFromImage(tem_img)

    # ---- 3. 转 tensor ----
    mri_tensor = torch.from_numpy(mri_array).unsqueeze(0).unsqueeze(0).to(device)
    tem_tensor = torch.from_numpy(tem_array).unsqueeze(0).unsqueeze(0).to(device)

    # ---- 4. 推理 ----
    with torch.no_grad():
        output = model(mri_tensor, tem_tensor)
        pred_prob = output.squeeze().cpu().numpy()

    pred_seg = (pred_prob >= 0.5).astype(np.uint8)

    # ---- 5. 保存结果 ----
    pred_img = sitk.GetImageFromArray(pred_seg)
    pred_img.CopyInformation(mri_img)
    save_path = os.path.join(args.save_dir, 'predicted_seg.nii.gz')
    sitk.WriteImage(pred_img, save_path)

    print(f"Saved prediction to: {save_path}")


if __name__ == '__main__':
    main()
