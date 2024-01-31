import os

import cv2
import imageio
import numpy as np
import torch
from scipy.ndimage import median_filter
from DPT.dpt.models import DPTDepthModel, DPT
import torchvision.transforms as T
depth_model = DPTDepthModel(
        path="dpt_weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
depth_transform = T.Compose(
[
    T.Resize((384, 384)),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
imgs = torch.randn((512, 512, 3)).numpy()
imgs = cv2.cvtColor(imgs, cv2.COLOR_BGRA2RGBA)
imgs = cv2.resize(imgs, (512, 512), interpolation=cv2.INTER_AREA)
imgs = (torch.from_numpy(imgs)/255.).unsqueeze(0).permute(0, 3, 1, 2)
imgs = imgs[:, :3, :, :] * imgs[:, 3:, :, :] + (1 - imgs[:, 3:, :, :])
with torch.no_grad():
    depth_prediction = depth_model.forward(depth_transform(imgs))
    depth_prediction = torch.nn.functional.interpolate(
        depth_prediction.unsqueeze(1),
        size=512,
        mode="bicubic",
        align_corners=True,
    )
    disparity = median_filter(depth_prediction, size=5)
    depth = 1. / np.maximum(disparity, 1e-2)

print(depth)