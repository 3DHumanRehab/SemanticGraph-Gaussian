from plyfile import PlyData
from smpl_np import SMPLModel
import numpy as np
from SegDataUtils import SegHumanPlyData
import ply_data_handel

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

PART_SEG_LABELS=[
    "background",
    "rightHand",
    "rightUpLeg",
    "leftArm",
    "head",
    "leftLeg",
    "leftFoot",
    "torso",
    "rightFoot",
    "rightArm",
    "leftHand",
    "rightLeg",
    "leftForeArm",
    "rightForeArm",
    "leftUpLeg",
    "hips",
]

if __name__=='__main__':
    smpl = SMPLModel('/root/HumanGaussian_zwy/Human3D/SMPL_NEUTRAL.pkl')
    points = smpl.verts[:, [0, 2, 1]]
    points[:, 1] = -points[:, 1]
    print(points.shape)
    ply = ply_data_handel.generate_plydata(points)
    ply_data_handel.save_plyfile(ply,'/root/HumanGaussian_zwy/Human3D/data/raw/test_set/smpl.ply')