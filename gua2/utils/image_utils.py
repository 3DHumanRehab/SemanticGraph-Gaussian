#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import cv2
import numpy as np
import paddlehub as hub
import torch
import torchvision.transforms as T
from scipy.ndimage import median_filter

from submodules.DPT.dpt.models import DPTDepthModel


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def segment_human(image):
    module = hub.Module(name="ace2p")
    return module.segmentation(images=[image])


def get_depth(image):
    depth_model = DPTDepthModel(
        path="submodules/DPT/dpt_weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_transform = T.Compose(
        [
            T.Resize((384, 384)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = (torch.from_numpy(image) / 255.).unsqueeze(0).permute(0, 3, 1, 2)
    image = image[:, :3, :, :] * image[:, 3:, :, :] + (1 - image[:, 3:, :, :])
    with torch.no_grad():
        depth_prediction = depth_model.forward(depth_transform(image))
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=512,
            mode="bicubic",
            align_corners=True,
        )
        disparity = median_filter(depth_prediction, size=5)
        depth = 1. / np.maximum(disparity, 1e-2)
    return torch.tensor(depth).cuda()

