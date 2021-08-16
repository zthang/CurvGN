from dgl.data import citation_graph as citegrh
import networkx as nx
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import pickle
import numpy as np

def load_cora_data():

    with open("data/Cora/Cora/raw/ind.cora.graph", 'rb') as f:
        out = pickle.load(f, encoding='latin1')

    edge_index=[]
    for key in range(2708):
        for item in out[key]:
            if (key,item) not in edge_index:
                edge_index.append((key,item))

    Gd = nx.DiGraph()
    Gd.add_edges_from(edge_index)
    edge_index_v1 = list(Gd.edges)
    edge_index_v1 = sorted(edge_index_v1)
    data = citegrh.load_cora()#加载数据集
    edge_index_v2 = list(data.graph.edges)
    edge_index_v2 = sorted(edge_index_v2)
    features=th.FloatTensor(data.features)#特征向量  张量的形式
    # for i in range(10556):
    #     if edge_index_v1[i] != edge_index_v2[i]:
    #         print(1)

    with open("data/Cora/Cora/raw/ind.cora.allx", 'rb') as f:
        allx = pickle.load(f, encoding='latin1').A
    with open("data/Cora/Cora/raw/ind.cora.tx", 'rb') as f:
        tx = pickle.load(f, encoding='latin1').A
    with open("data/Cora/Cora/raw/ind.cora.ally", 'rb') as f:
        ally = pickle.load(f, encoding='latin1')
    with open("data/Cora/Cora/raw/ind.cora.ty", 'rb') as f:
        ty = pickle.load(f, encoding='latin1')

    ori_x = np.vstack((allx, tx))
    ori_y = np.vstack((ally, ty))
    ori_y = np.array([np.argmax(i)for i in ori_y])
    features = data.features.numpy()
    for i in range(2708):
        if (ori_x[i]!=features[i]).all():
            print(2)
    for i in range(2708):
        if (ori_y[i]!=data.labels[i]):
            print(3)
    labels=th.LongTensor(data.labels)#所属类别
    train_mask=th.BoolTensor(data.train_mask)#那些参与训练
    test_mask=th.BoolTensor(data.test_mask)#哪些是测试集
    g=data.graph
    g.remove_edges_from(nx.selfloop_edges(g))#删除自循环的边
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

g, features, labels, train_mask, test_mask=load_cora_data()
