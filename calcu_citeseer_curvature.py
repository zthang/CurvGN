import os.path as osp
from torch_geometric.io import read_txt_array
import pickle
import torch
import json
import numpy as np
import scipy.sparse
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

# 0:theory 1:reinforcement_learning 2:genetic_algorithms 3:neural_networks 4:probabilistic_methods 5:case_based 6:rule_learning
method = 0   # 0:OllivierRicci 1:FormanRicci
is_DiGraph = False

def get_labels():
    labels = []
    with open("label_vector", 'rb') as f:
        for i in range(7):
            line = json.loads(bytes.decode(f.readline()))
            labels.append(line)
    labels = np.array(labels)
    return labels

def get_y():
    y_groundtruth = []
    with open("y_groundtruth", 'rb') as f:
        for i in range(7):
            line = json.loads(bytes.decode(f.readline()))
            y_groundtruth.append(line)
    y_groundtruth = np.array(y_groundtruth)
    return y_groundtruth

def save_file_x(tx, label):
    tx = tx.A
    tx = np.vstack((tx, label))
    tx = scipy.sparse.csr_matrix(tx)
    f = open("ind.cora.tx", 'wb')
    pickle.dump(tx, f)
    f.close()

def save_file_y(ty, y):
    ty = np.vstack((ty, y))
    f = open("ind.cora.ty", 'wb')
    pickle.dump(ty, f)
    f.close()

def save_dict():
    y = np.vstack((ally, ty))
    y_max = np.argmax(y, axis=1)
    for i in range(7):
        graph[2708 + i] = []
    for i in range(2708):
        graph[i].append(2708 + y_max[i])
        graph[2708 + y_max[i]].append(i)

    f = open("data/Cora/Cora/raw_own/ind.cora.graph", 'wb')
    pickle.dump(graph, f)
    f.close()

dic = {}
def DiGraph_to_graph():
    path = osp.join("raw", 'ind.{}.{}'.format("citeseer", "graph"))
    with open(path, 'rb') as f:
        out = pickle.load(f, encoding='latin1')
    for key in range(3327):
        for item in out[key]:
            if key not in out[item]:
                out[item].append(key)
    with open('ind.{}.{}'.format("citeseer", "graph"), 'wb') as f:
        pickle.dump(out, f)

def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        out = pickle.load(f, encoding='latin1')

    if name == 'graph':
        edge_index = []
        for key in range(3327):
            for item in out[key]:
                if (key, item) not in edge_index and key != item:
                    edge_index.append((key, item))

        Gd = nx.Graph() if not is_DiGraph else nx.DiGraph()
        Gd.add_edges_from(edge_index)
        Gd_OT = OllivierRicci(Gd, alpha=0.5, method="OTD", verbose="INFO") if method == 0 else FormanRicci(Gd)
        Gd_OT.compute_ricci_curvature()
        with open("graph_Citeseer.edge_list_OllivierRicci" if method == 0 else "graph_Citeseer.edge_list_FormanRicci", "w") as f:
            for item in Gd.edges:
                f.writelines(f"{item[0]} {item[1]} {Gd_OT.G[item[0]][item[1]]['ricciCurvature']}\n") if method == 0 else f.writelines(f"{item[0]} {item[1]} {Gd_OT.G[item[0]][item[1]]['formanCurvature']}\n")
            f.close()
        return out

    # out = out.todense() if hasattr(out, 'todense') else out
    # out = torch.Tensor(out)
    return out

# DiGraph_to_graph()

names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
items = [read_file("data/Citeseer/Citeseer/raw", "Citeseer", name) for name in names]
x, tx, allx, y, ty, ally, graph, test_index = items

# save_file_x(tx,get_labels())
# save_file_y(ty,get_y())
# save_dict()
print(1)