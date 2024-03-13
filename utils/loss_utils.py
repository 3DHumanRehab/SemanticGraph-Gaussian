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
import random
import clip
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics import PearsonCorrCoef


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cluster(x, y, gt_labels, gaussian_label, weight, neighbor=1):
    direction = {(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1), (-1, -1), (1, 1)}
    loss = torch.tensor(0.0)
    if x > gt_labels.shape[0] or y > gt_labels.shape[0] or y < 0 or y < 0: return loss
    loss += torch.tensor(0.0) if gt_labels[x][y] == gaussian_label else torch.tensor(weight)
    for i in range(1, neighbor):
        for (tx, ty) in direction:
            if i * tx + x > gt_labels.shape[0] or i * ty + y > gt_labels.shape[
                0] or i * tx + x < 0 or i * ty + y < 0: continue
            loss += -torch.tensor(weight) if gt_labels[i * ty + x][i * ty + y] == gaussian_label else torch.tensor(
                weight)
    return loss


def loss_self_supervision(gaussian_labels, colors, opacity, positions, k=30):
    types = dict()
    loss = torch.tensor(0.0).cuda()
    pearson = PearsonCorrCoef().cuda()
    # print("gaussian_labels={}".format(gaussian_labels))
    for i in range(gaussian_labels.shape[0]):
        if types.get(gaussian_labels[i].item()) is None:
            types[gaussian_labels[i].item()] = list()
        feature_tensor = torch.cat([colors[i], opacity[i], positions[i]], dim=0)
        types[gaussian_labels[i].item()].append(feature_tensor)
    for key in types.keys():
        pearson_loss = torch.tensor(0.0).cuda()
        for i in types[key]:
            for j in types[key]:
                # for i in random.sample(types[key], min(k,len(types[key]))):
                #     for j in random.sample(types[key], min(k,len(types[key]))):
                pearson_loss += 1 - pearson(i, j)
        loss += pearson_loss / (len(types[key]) * len(types[key]))
    return loss


def dice_loss(prediction, ground_truth, num_classes=20):
    """
    Parameters:
    - prediction: Model's prediction (probability values or binary segmented region).
    - ground_truth: Ground truth segmentation (binary labels).

    Returns:
    Dice Loss value.
    """
    dice_loss = 0.0

    for class_idx in range(num_classes):
        class_prediction = (prediction == class_idx).float()
        class_target = (ground_truth == class_idx).float()

        intersection = torch.sum(class_prediction * class_target)
        union = torch.sum(class_prediction) + torch.sum(class_target)

        class_dice = 1.0 - (2.0 * intersection) / (union + 1e-8)
        dice_loss += class_dice
    dice_loss /= num_classes

    return dice_loss


def loss_depth(pred_depth, depth_gt):
    pearson = PearsonCorrCoef().cuda()
    pred_depth = pred_depth.squeeze()
    pred_depth = torch.nan_to_num(pred_depth)
    depth_gt = depth_gt.squeeze().reshape(-1)
    pred_depth = pred_depth.reshape(-1)
    return 1 - pearson(pred_depth, depth_gt)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.

    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # print("sample_features={}".format(sample_features))
    # print("features={}".format(features))
    # print("sample_preds={}".format(sample_preds))
    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # print("dists={}".format(dists))
    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]
    # print("neighbor_preds={}".format(neighbor_preds))

    # Compute KL divergence
    # print("sample_preds={}".format(torch.log(sample_preds.unsqueeze(1) + 1e-6)))
    # print("neighbor_preds={}".format(neighbor_preds))
    sample_preds = torch.clamp(sample_preds, min=1e-7, max=1 - 1e-7)
    neighbor_preds = torch.clamp(neighbor_preds, min=1e-7, max=1 - 1e-7)
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1)) - torch.log(neighbor_preds))
    # print("kl={}".format(kl))
    loss = kl.sum(dim=-1).mean()

    # print("loss={}".format(loss))
    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss


def pointcloud_loss(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer Distance between two point clouds.

    Parameters:
    point_cloud1: The first point cloud, PyTorch tensor of shape (N, 3), where N is the number of points.
    point_cloud2: Second point cloud, PyTorch tensor of shape (M, 3), where M is the number of points.

    Back:
    chamfer_distance: chamfer distance
    """
    distances1 = torch.norm(point_cloud1[:, None, :] - point_cloud2, dim=2)
    distances2 = torch.norm(point_cloud2[:, None, :] - point_cloud1, dim=2)
    min_distances1, _ = torch.min(distances1, dim=1)
    min_distances2, _ = torch.min(distances2, dim=1)
    chamfer_distance = torch.sum(min_distances1) / point_cloud1.shape[0] + torch.sum(min_distances2) / \
                       point_cloud2.shape[0]
    return chamfer_distance


def img2text_clip_loss(text, image):
    clip_model, clip_preprocess = clip.load("assets/ViT-B-16.pt", device='cuda', jit=False)
    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    text_tokens = clip.tokenize(text).to('cuda')
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)
    similarity_score = (torch.nn.functional.cosine_similarity(image_features, text_features)).item()
    return 1.0 - similarity_score


def img2img_clip_loss(image, image2):
    clip_model, clip_preprocess = clip.load("assets/ViT-B-16.pt", device='cuda', jit=False)
    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    image2 = F.interpolate(image2, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image2_features = clip_model.encode_image(image2)
    similarity_score = (torch.nn.functional.cosine_similarity(image_features, image2_features))
    return 1.0 - similarity_score
