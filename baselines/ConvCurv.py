import os.path as osp
import random

import torch
from torch_scatter import scatter
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import pickle
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax
from baselines.curvGN import curvGN
config = None
def minmaxscaler(x):
    for i in range(len(x)):
        min = np.amin(x[i])
        max = np.amax(x[i])
        x[i]=(x[i] - min)/(max-min)
    return x

class Net(torch.nn.Module):
    def __init__(self,data,num_features,num_classes,w_mul):
        super(Net, self).__init__()
        self.conv1 = curvGN(num_features, 64, 64, w_mul, data, config)
        self.conv2 = curvGN(64, num_classes, num_classes, w_mul, data, config)
    def forward(self,data):
        x = F.dropout(data.x,p=0.6,training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)   #Todo:在前还是在后？

        num_edge_noloop = data.num_edges-data.num_nodes
        num_classes = torch.max(data.y)+1
        edge_mask = torch.tensor([i < 20*num_classes for i in data.edge_index[0][:num_edge_noloop]])
        source_feature = x.index_select(0, data.edge_index[0][:num_edge_noloop])[edge_mask]
        target_feature = x.index_select(0, data.edge_index[1][:num_edge_noloop])[edge_mask]
        t = source_feature - target_feature
        t = torch.sum(t*t, dim=1)*data.w_mul_sigmoid[edge_mask]
        Reg1 = t.sum()

        # x_t = x[torch.tensor(data.train_mask)]
        # temp_matrix = x_t.T@data.D@x_t - torch.eye(64)
        # Reg2 = (temp_matrix*temp_matrix).sum()
        Reg2 = 1

        x = F.dropout(x,p=0.6,training=self.training)
        x = self.conv2(x, data.edge_index)
        # x_n=x.detach().numpy()
        # x_n=minmaxscaler(x_n)
        # with open("vectors_norm","wb") as f:
        #     pickle.dump(x_n,f)
        return F.log_softmax(x, dim=1), Reg1, Reg2

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def random_mask(ricci_cur):
    edge_num = len(ricci_cur)
    random_index = random.sample(range(edge_num), (int)(config.mask_rate*edge_num))
    random_index.sort(reverse=True)
    if config.mask_mode == 0:
        for i in random_index:
            del ricci_cur[i]
    elif config.mask_mode == 1:
        ricci_cur.sort(key=lambda x: x[2])
        for i in random_index:
            del ricci_cur[i]
    elif config.mask_mode == 2:
        ricci_cur.sort(key=lambda x: x[2], reverse=True)
        for i in random_index:
            del ricci_cur[i]
    return ricci_cur

def call(data,name,num_features,num_classes, my_config):
    global config
    config = my_config
    filename='data/curvature/graph_'+name+'.edge_list_OllivierRicci'
    f=open(filename)
    cur_list=list(f)
    if name=='CS' or config.is_DiGraph:
        ricci_cur=[[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
    else:
        ricci_cur=[[] for i in range(2*len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
            ricci_cur[i+len(cur_list)]=[ricci_cur[i][1],ricci_cur[i][0],ricci_cur[i][2]]
    ricci_cur = random_mask(ricci_cur)
    ricci_cur = sorted(ricci_cur)
    eg_index0 = [i[0] for i in ricci_cur]
    eg_index1 = [i[1] for i in ricci_cur]
    eg_index = torch.stack((torch.tensor(eg_index0), torch.tensor(eg_index1)), dim=0)
    data.edge_index = eg_index
    w_mul = [i[2] for i in ricci_cur]

    filename='data/curvature/graph_'+name+'.edge_list_FormanRicci'
    f=open(filename)
    cur_list=list(f)
    if name=='CS' or config.is_DiGraph:
        ricci_cur=[[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
    else:
        ricci_cur=[[] for i in range(2*len(cur_list))]
        for i in range(len(cur_list)):
            ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
            ricci_cur[i+len(cur_list)]=[ricci_cur[i][1],ricci_cur[i][0],ricci_cur[i][2]]
    ricci_cur = random_mask(ricci_cur)
    ricci_cur = sorted(ricci_cur)
    w_mul_Forman = [i[2] for i in ricci_cur]

    w_mul_sigmoid = torch.sigmoid(torch.tensor(w_mul))
    data.w_mul_sigmoid=w_mul_sigmoid
    # D = scatter(torch.tensor(w_mul), data.edge_index[0], dim=0, reduce='add')
    # D = torch.diag(torch.sigmoid(D[torch.tensor(data.train_mask)]))
    # data.D = D

    w_mul = w_mul+[0 for i in range(data.x.size(0))]
    w_mul_Forman = w_mul_Forman+[0 for i in range(data.x.size(0))]
    w_mul=torch.tensor(w_mul, dtype=torch.float)
    w_mul_Forman=torch.tensor(w_mul_Forman, dtype=torch.float)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index,num_nodes=data.x.size(0))

    source_feature = data.x.index_select(0, data.edge_index[0])
    target_feature = data.x.index_select(0, data.edge_index[1])
    data.cosine_similarity = torch.cosine_similarity(source_feature, target_feature, dim=1)

    data.w_mul=w_mul.view(-1,1)
    data.w_mul_Forman=w_mul_Forman.view(-1,1)
    device = torch.device('cpu')
    #change to call function
    data.w_mul = data.w_mul.to(device)
    data.w_mul_Forman = data.w_mul_Forman.to(device)
    model, data = Net(data,num_features,num_classes,data.w_mul).to(device), data.to(device)
    return model, data
