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


def point_to_graph(point_cloud_distance, point_cloud_attr, k):
    kdtree = KDTree(point_cloud_distance, leaf_size=30, metric='euclidean')
    distances, indices = kdtree.query(point_cloud_distance, k)
    edge_index = []
    edge_attr = []
    # print(distances)
    # print(indices)
    for i in range(len(point_cloud_distance)):
        for index, j in enumerate(indices[i]):
            # print("j={},index={}".format(j,index))
            if i != j:
                edge_index.append([i, j])
                edge_attr.append(distances[i, index])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    data = Data(x=torch.tensor(point_cloud_attr, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    return data


def nodeEmbedding_node2vec(graph):
    G = to_networkx(graph, to_undirected=True)
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    results = {str(node): model.wv[str(node)] for node in G.nodes()}
    # print(type(results['0']))
    embeddings = None
    for x in results.values():
        embeddings = torch.tensor(x).unsqueeze(0) if embeddings is None else torch.cat((embeddings, torch.tensor(x).unsqueeze(0)))
    # print("Embedding for node:={}".format(embeddings))
    return embeddings


def node_clustering(point_cloud, point_cloud_attr, k=3):
    graph = point_to_graph(point_cloud, point_cloud_attr, 3).cuda()
    data = nodeEmbedding_node2vec(graph)
    affinity_propagation = AffinityPropagation(damping=0.9, preference=-50, max_iter=200, convergence_iter=15)
    affinity_propagation.fit(data)
    return affinity_propagation.labels_

