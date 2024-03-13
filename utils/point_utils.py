import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d
from torch import optim

from utils.mapper import COLOR_MAP_INSTANCES, MERGED_BODY_PART_COLORS
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, num_classes=16, num_point=1024):
        super(PointNet, self).__init__()
        self.num_point = num_point
        self.input_transform_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1024, kernel_size=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((num_point, 1))
        )
        self.input_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        self.input_fc[-1].weight.data = torch.zeros((256, 9))
        self.input_fc[-1].bias.data = torch.eye(3).view(-1)

        self.mlp_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.feature_transform_net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1024, kernel_size=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((num_point, 1))
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64)
        )

        self.mlp_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1024, kernel_size=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.seg_net = nn.Sequential(
            nn.Conv2d(1088, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=(1, 1)),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        batchsize = inputs.shape[0]

        t_net = self.input_transform_net(inputs)
        t_net = t_net.view(batchsize, -1)
        t_net = self.input_fc(t_net)
        t_net = t_net.view(batchsize, 3, 3)

        x = inputs.view(batchsize, 1024, 3)
        x = torch.matmul(x, t_net)
        x = x.unsqueeze(1)
        x = self.mlp_1(x)

        t_net = self.feature_transform_net(x)
        t_net = t_net.view(batchsize, -1)
        t_net = self.feature_fc(t_net)
        t_net = t_net.view(batchsize, 64, 64)

        x = x.view(batchsize, 64, 1024)
        x = x.transpose(1, 2)
        x = torch.matmul(x, t_net)
        x = x.transpose(1, 2)
        x = x.unsqueeze(-1)
        point_feat = x
        x = self.mlp_2(x)
        x = torch.max(x, dim=2)[0]

        global_feat_expand = x.unsqueeze(1).repeat(1, self.num_point, 1, 1)
        x = torch.cat([point_feat, global_feat_expand], dim=1)
        x = self.seg_net(x)
        x = x.squeeze(-1)
        x = x.transpose(1, 2)

        return x


class HumanSegmentationDataset():
    def __init__(self, file_list):
        self.file_list = file_list

        self.COLOR_MAP_INSTANCES = COLOR_MAP_INSTANCES
        self.MERGED_BODY_PART_COLORS = MERGED_BODY_PART_COLORS

        self.ORIG_BODY_PART_IDS = set(range(100, 126))

        self.LABEL_LIST = ["background", "rightHand", "rightUpLeg", "leftArm", "head", "leftEye", "rightEye", "leftLeg",
                           "leftToeBase", "leftFoot", "spine1", "spine2", "leftShoulder", "rightShoulder",
                           "rightFoot", "rightArm", "leftHandIndex1", "rightLeg", "rightHandIndex1",
                           "leftForeArm", "rightForeArm", "neck", "rightToeBase", "spine", "leftUpLeg",
                           "leftHand", "hips"]

        self.MERGED_LABEL_LIST = {
            0: "background",
            1: "rightHand",
            2: "rightUpLeg",
            3: "leftArm",
            4: "head",
            5: "leftLeg",
            6: "leftFoot",
            7: "torso",
            8: "rightFoot",
            9: "rightArm",
            10: "leftHand",
            11: "rightLeg",
            12: "leftForeArm",
            13: "rightForeArm",
            14: "leftUpLeg",
            15: "hips"
        }

        self.LABEL_MAPPER_FOR_BODY_PART_SEGM = {
            -1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
            11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0,
            22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0,
            33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0,  # background
            100: 1,  # rightHand
            101: 2,  # rightUpLeg
            102: 3,  # leftArm
            103: 4,  # head
            104: 4,  # head
            105: 4,  # head
            106: 5,  # leftLeg
            107: 6,  # leftFoot
            108: 6,  # leftFoot
            109: 7,  # torso
            110: 7,  # torso
            111: 7,  # torso
            112: 7,  # torso
            113: 8,  # rightFoot
            114: 9,  # rightArm
            115: 10,  # leftHand
            116: 11,  # rightLeg
            117: 1,  # rightHand
            118: 12,  # leftForeArm
            119: 13,  # rightForeArm
            120: 4,  # head
            121: 8,  # rightFoot
            122: 7,  # torso
            123: 14,  # leftUpLeg
            124: 10,  # leftHand
            125: 15,  # hips
        }

    def __len__(self):
        return len(self.file_list)

    def read_plyfile(self, file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
        if plydata.elements:
            return pd.DataFrame(plydata.elements[0].data).values

    def load_pc(self, file_path):
        pc = self.read_plyfile(file_path)  # (num_points, 8)
        mask = pc[:, 6] != 0
        pc = pc[mask]

        pc_coords = pc[:, 0:3]  # (num_points, 3)
        pc_rgb = pc[:, 3:6].astype(np.uint8)  # (num_points, 3) - 0-255
        pc_inst_labels = pc[:, 6].astype(np.uint8)  # (num_points,)
        pc_orig_segm_labels = pc[:, 7].astype(np.uint8)  # (num_points,)
        pc_part_segm_labels = np.asarray([self.LABEL_MAPPER_FOR_BODY_PART_SEGM[el] for el in pc_orig_segm_labels])

        vertex = np.core.records.fromarrays(
            [pc_coords[:, 0], pc_coords[:, 1], pc_coords[:, 2], pc_rgb[:, 0], pc_rgb[:, 1], pc_rgb[:, 2],
             pc_inst_labels, pc_orig_segm_labels], names='x, y, z, red, green, blue, inst_labels, orig_segm_labels',
            formats='f4, f4, f4, u1, u1, u1, u1, u1')
        el = PlyElement.describe(vertex, 'vertex')
        ply_data = PlyData([el])
        ply_data.write("filter.ply")
        return pc_coords, pc_rgb, pc_inst_labels, pc_orig_segm_labels, pc_part_segm_labels

    def export_colored_pcd_inst_segm(self, coords, pc_inst_labels, write_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        inst_colors = np.asarray([self.COLOR_MAP_INSTANCES[int(label_idx)] for label_idx in pc_inst_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(write_path, pcd)

    def export_colored_pcd_part_segm(self, coords, pc_part_segm_labels, write_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        part_colors = np.asarray(
            [self.MERGED_BODY_PART_COLORS[int(label_idx)] for label_idx in pc_part_segm_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(part_colors)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(write_path, pcd)

    def __getitem__(self, index):
        return self.load_pc(self.file_list[index])
    
    
class HumanSegmentationDataset():
    def __init__(self, file_list):
        self.file_list = file_list
        self.ORIG_BODY_PART_IDS = set(range(100, 126))
        self.MERGED_LABEL_LIST = {
            0: "background",
            1: "rightHand",
            2: "rightUpLeg",
            3: "leftArm",
            4: "head",
            5: "leftLeg",
            6: "leftFoot",
            7: "torso",
            8: "rightFoot",
            9: "rightArm",
            10: "leftHand",
            11: "rightLeg",
            12: "leftForeArm",
            13: "rightForeArm",
            14: "leftUpLeg",
            15: "hips"
        }

    def __len__(self):
        return len(self.file_list)

    def read_plyfile(self, file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
        if plydata.elements:
            return pd.DataFrame(plydata.elements[0].data).values

    def load_pc(self, file_path):
        pc = self.read_plyfile(file_path)  # (num_points, 8)
        pc_coords = pc[:, 0:3]  # (num_points, 3)
        pc_rgb = pc[:, 3:6].astype(np.uint8)  # (num_points, 3) - 0-255
        pc_orig_segm_labels = pc[:, 6].astype(np.uint8)  # (num_points,)
        return pc_coords, pc_rgb, torch.tensor(pc_orig_segm_labels).cuda()

    def export_colored_pcd_inst_segm(self, coords, pc_inst_labels, write_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        inst_colors = np.asarray([self.COLOR_MAP_INSTANCES[int(label_idx)] for label_idx in pc_inst_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(write_path, pcd)

    def export_colored_pcd_part_segm(self, coords, pc_part_segm_labels, write_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        part_colors = np.asarray(
            [self.MERGED_BODY_PART_COLORS[int(label_idx)] for label_idx in pc_part_segm_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(part_colors)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(write_path, pcd)

    def __getitem__(self, index):
        return self.load_pc(self.file_list[index])