import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

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


if __name__ == "__main__":
    pcd_path = 'smpl_predict.ply'  # sample input scene
    file_list = [pcd_path]  # for now just the demo scene
    input_tensor = torch.tensor([1, 2, 0, 1, 3])
    dataset = HumanSegmentationDataset(file_list=file_list)
    coords, colors, labels = dataset.load_pc(pcd_path)
    print(F.one_hot(labels.to(torch.int64),num_classes=20).unsqueeze(1).shape)
    # (num_points, 3), (num_points, 3), (num_points,), (num_points,), (num_points,)
    # print(len(coords))
    # pointnet = PointNet()
    # optimizer = optim.Adam(pointnet.parameters(), lr=0.001)

    # # Training loop
    # num_epochs = 10
    # checkpoint_path = 'pointnet_checkpoint.pth'
    # criterion = nn.CrossEntropyLoss()
    # for epoch in range(num_epochs):
    #     pointnet.train()
    #     running_loss = 0.0
    #     features = np.concatenate((coords, colors), axis=1)
    #     optimizer.zero_grad()
    #     outputs = pointnet(features)
    #     print(outputs)
    #     loss = criterion(outputs, full_body_part_labels)
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()
    #
    #     # Print average loss for the epoch
    #     average_loss = running_loss
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
    #
    # # Save model checkpoint
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': pointnet.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': average_loss
    # }, checkpoint_path)

    # pcd = o3d.io.read_point_cloud("smpl_predict.ply")
    # o3d.visualization.draw_geometries([pcd])

    # dataset.export_colored_pcd_inst_segm(coords, inst_labels, write_path='test2.ply')
    # dataset.export_colored_pcd_part_segm(coords, merged_body_part_labels, write_path='test3.ply')
