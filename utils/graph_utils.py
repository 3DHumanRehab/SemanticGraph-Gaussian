import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from node2vec import Node2Vec
import networkx as nx
import math
def point_to_graph(point_cloud_distance, point_cloud_attr, k):
    kdtree = KDTree(point_cloud_distance, leaf_size=30, metric='euclidean')
    distances, indices = kdtree.query(point_cloud_distance, k)
    edge_index = []
    edge_attr = []
    for i in range(len(point_cloud_distance)):
        for index, j in enumerate(indices[i]):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append(distances[i, index])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    data = Data(x=torch.tensor(point_cloud_attr, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    return data

def compute_modularity(graph,community,node_attr):
    score = 0
    for i in graph.nodes:
        for j in graph.nodes:
            # print(node_attr[i])
            if community[i] == community[j]:
                r1, g1, b1 = node_attr[i][0],node_attr[i][1],node_attr[i][2]
                r2, g2, b2 = node_attr[j][0],node_attr[j][1],node_attr[j][2]
                color_score= math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
                color_score = color_score / math.sqrt(3 * (255 ** 2))
                opacity_score = abs(node_attr[i][3] - node_attr[j][3])
                score += color_score + opacity_score
    print(score)
    return score
               
def frequency_clustering(features,pos,frozen_labels,label):
    result, max_score = None, None
    origin_id = torch.zeros(features.shape[0]).cuda()
    for i in range(features.shape[0]-1):
        origin_id[i] = i
    part_feature = features[frozen_labels == label]
    part_pos = pos[frozen_labels == label]
    origin_id = origin_id[frozen_labels == label]
    graph = to_networkx(point_to_graph(part_pos.detach().cpu().numpy(), part_feature.detach().cpu().numpy(), 3),to_undirected=True)
    for i in graph.nodes:
        for j in graph.neighbors(i):
            r1, g1, b1 = part_feature[i][0],part_feature[i][1],part_feature[i][2]
            r2, g2, b2 = part_feature[j][0],part_feature[j][1],part_feature[j][2]
            color_score= math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
            color_score = color_score / math.sqrt(3 * (255 ** 2))
            opacity_score = abs(part_feature[i][3] - part_feature[j][3])
            score = color_score + opacity_score
        if max_score is None or score > max_score:
            max_score = score
            result = origin_id[i]
    return result.item()

# def frequency_clustering(features,pos,frozen_labels,label):
#     result, max_score = None, None
#     origin_id = torch.zeros(features.shape[0]).cuda()
#     for i in range(features.shape[0]-1):
#         origin_id[i] = i
#     part_feature = features[frozen_labels == label]
#     part_pos = pos[frozen_labels == label]
#     origin_id = origin_id[frozen_labels == label]
#     graph = to_networkx(point_to_graph(part_pos.detach().cpu().numpy(), part_feature.detach().cpu().numpy(), 3),to_undirected=True)
#     for i in graph.nodes:
#         for j in graph.neighbors(i):
#             r1, g1, b1 = part_feature[i][0],part_feature[i][1],part_feature[i][2]
#             r2, g2, b2 = part_feature[j][0],part_feature[j][1],part_feature[j][2]
#             color_score= math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
#             color_score = color_score / math.sqrt(3 * (255 ** 2))
#             opacity_score = abs(part_feature[i][3] - part_feature[j][3])
#             score = color_score + opacity_score
#         if max_score is None or score > max_score:
#             max_score = score
#             result = origin_id[i]
#     return result.item()
            
#     optimized, community, modularity, best_community, best_modularity = False, dict(), 0, dict(), 0
#     for i in range(len(graph.nodes)):
#         community[i] = i
#     best_modularity, best_community = compute_modularity(graph, community,part_feature), community
#     while not optimized:
#         optimized = True
#         for node in graph.nodes:
#             temp_community = best_community
#             for neighbor in graph.neighbors(node):
#                 temp_community[node] = temp_community[neighbor]
#                 modularity = compute_modularity(graph,community,part_feature)
#                 if modularity > best_modularity:
#                     best_community = community
#                     best_modularity = modularity
#                     optimized = False
#         print(optimized)
#     return best_community
    
def nodeEmbedding_node2vec(graph):
    G = to_networkx(graph, to_undirected=True)
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    results = {str(node): model.wv[str(node)] for node in G.nodes()}
    embeddings = None
    for x in results.values():
        embeddings = torch.tensor(x).unsqueeze(0) if embeddings is None else torch.cat((embeddings, torch.tensor(x).unsqueeze(0)))
    return embeddings


def node_clustering(point_cloud, point_cloud_attr, k=3):
    graph = point_to_graph(point_cloud, point_cloud_attr, 3).cuda()
    data = nodeEmbedding_node2vec(graph)
    affinity_propagation = AffinityPropagation(damping=0.9, preference=-50, max_iter=200, convergence_iter=15)
    affinity_propagation.fit(data)
    return affinity_propagation.labels_

def equal_node(node1,node2):
    pass

def equal_edges(edge1,edge2):
    pass

def graph_similar(graph1,graph2):
    graph1 = to_networkx(graph1, to_undirected=True)
    graph2 = to_networkx(graph2, to_undirected=True)
    distance = nx.optimize_graph_edit_distance(graph1, graph2)
    for dist in distance:
        print(dist)
    for dist in distance:
        return torch.tensor(dist,dtype=torch.float)