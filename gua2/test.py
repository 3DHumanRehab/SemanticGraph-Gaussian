# import numpy as np
import numpy as np
# from torch import nn
#
# # from scene.dataset_readers import fetchPly
# # from utils.loss_utils import pointcloud_loss
#
# from sklearn.cluster import AffinityPropagation
# import matplotlib.pyplot as plt
# from PIL import Image
# import clip
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter

# from submodules.DPT.dpt.models import DPTDepthModel
import torchvision.transforms as T

# from utils.graph_utils import node_clustering

def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array(
            [data["red"], data["green"], data["blue"]], dtype=np.uint8
        ).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels

if __name__ == '__main__':
    # print(load_ply("test.ply"))
    index = torch.randperm(200)[1:30]
    print(index)
    pass
    # x = torch.rand((6890, 1, 3)).squeeze(1)
    # y = torch.rand((6890, 1))
    # print(x.shape)
    # print(y.shape)
    # print(torch.cat((x,y),dim = 1).shape)
    # labels = torch.rand((4,2,2)).permute(1,2,0)
    # print(labels)
    # max_indices = torch.argmax(labels, dim=2)
    # converted_labels = max_indices
    # print(max_indices)
    # print("原始 labels 形状:", labels.shape)
    # print("转换后 labels 形状:", converted_labels.shape)
    # point_cloud = np.random.rand(200, 3)
    # point_cloud_attr = np.random.rand(200, 15)
    # print(node_clustering(point_cloud,point_cloud_attr))
    # depth_model = DPTDepthModel(path="submodules/DPT/dpt_weights/dpt_hybrid-midas-501f0c75.pt",backbone="vitb_rn50_384",non_negative=True,enable_attention_hooks=False,)
    # depth_transform = T.Compose([T.Resize((384, 384)),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    # ref_imgs = torch.rand((512, 512, 4))
    # ref_imgs = (ref_imgs / 255.).unsqueeze(0).permute(0, 3, 1, 2)
    # ori_imgs = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + (1 - ref_imgs[:, 3:, :, :])
    # depth_prediction = depth_model.forward(depth_transform(ori_imgs))
    # depth_prediction = torch.nn.functional.interpolate(depth_prediction.unsqueeze(1),size=512,mode="bicubic",align_corners=True,).detach().numpy()
    # depth_prediction = torch.tensor(1. / np.maximum(median_filter(depth_prediction / 65535., size=5), 1e-2)).cuda()
    # print(depth_prediction.shape)
    # clip_model, clip_preprocess = clip.load("assets/ViT-B-16.pt", device='cuda', jit=False)
    # image_tensor = torch.rand(1, 3, 224, 224).to('cuda')
    # image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    # text = ["a photo of a cat"]
    # text_tokens = clip.tokenize(text).to('cuda')
    # with torch.no_grad():
    #     image_features = clip_model.encode_image(image_tensor)
    #     text_features = clip_model.encode_text(text_tokens)
    # similarity_score = (100 * torch.nn.functional.cosine_similarity(image_features, text_features)).item()
    # print(similarity_score)
    # pass
    # path = "/home/featurize/work/gua2/output/zju_mocap_refine/my_377_test/input.ply"
    # ply = torch.tensor(fetchPly(path).points)
    # tensor = torch.randn(6890,3)
    # print(ply.points.type)
    # print(ply.shape)
    # print(pointcloud_loss(tensor, ply))
    # 创建 AffinityPropagation 模型
    # data = np.random.rand(6890, 10)
    # affinity_propagation = AffinityPropagation(damping=0.9, preference=-50, max_iter=200, convergence_iter=15)
    # # 拟合数据并进行聚类
    # affinity_propagation.fit(data)
    # # 获取聚类标签
    # cluster_labels = affinity_propagation.labels_
    # # 获取聚类中心的索引
    # cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    # # 获取聚类中心的坐标
    # cluster_centers = data[cluster_centers_indices]
    # # 绘制聚类结果（假设只绘制前两个特征）
    # print("data={}".format(data))
    # print("cluster_labels={}".format(torch.tensor(cluster_labels).shape))
    # plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis')
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, linewidths=3, color='red',label='Cluster Centers')
    # plt.title('Affinity Propagation Clustering')
    # plt.legend()
    # plt.show()
    # tensor = torch.rand(10, 10)
    # cosine_similarity_matrix = torch.nn.functional.cosine_similarity(tensor, tensor, dim=1)
    # print(cosine_similarity_matrix)
    # one_hot_tensor = torch.tensor([[0,0,1],[1,0,0]])
    # print(one_hot_tensor.shape)
    # label_tensor = torch.argmax(one_hot_tensor, dim=1, keepdim=True)
    # result_tensor = torch.where(label_tensor == 2, label_tensor, torch.tensor(-1))
    # print(one_hot_tensor)
    # print(label_tensor)
    # print(result_tensor)
    # group = torch.randn(size=(3,3))
    # extension_tensor = torch.randn(size=(3,3))
    # print(nn.Parameter(torch.cat((group["params"][0], extension_tensor.unsqueeze(1)), dim=0).requires_grad_(True)))
    # tensor = torch.randn(3,3)
    # print(tensor)
    # result_tensor = torch.where(tensor == 3, tensor, torch.tensor(-1))
    # print(result_tensor)
    # a = torch.randn(3)
    # print(a.norm())
    # print(torch.__version__)
    # print(torch.version.cuda)#cuda版本
    # print(torch.backends.cudnn.version())
    # print(torch.cuda.is_available()) #cuda是否可用，返回为True表示可用
    # print(torch.cuda.device_count())#返回GPU的数量
    # print(torch.cuda.get_device_name(0))#返回gpu名字，设备索引默认从0开始
    # print(torch.rand(size=(3,5,5)).cuda)

    # input_tensor = torch.randn(6890, 3)
    # output_tensor = torch.randn(6890,20)
    # filter_tensor = torch.randn(800,3)
    # dists = torch.cdist(filter_tensor,input_tensor)
    # loss_cls_3d(input_tensor,output_tensor)
    # target_size = (20, 3, 3)
    # output_tensor = torch.zeros(target_size)
    # output_tensor[:input_tensor.size(0), :, :] = input_tensor
    # print(input_tensor)
    # print(output_tensor)
