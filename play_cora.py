import os.path as osp
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse
import pickle

def find_x(vec):
    index=-1
    for idx, row in enumerate(x):
        if (row==vec).all():
            if index != -1:
                return -1
            else:
                index = idx
    return index

def find_cite(source, target):
    for i in cite:
        if (i[0]==source and i[1]==target) or (i[1]==source and i[0]==target):
            return True
    return False


    # 导入数据：分隔符为空格
raw_data = pd.read_csv('cora/cora.content',sep = '\t',header = None)
raw_data_cites = pd.read_csv('cora/cora.cites',sep = '\t',header = None)

cite = raw_data_cites.to_numpy()

x = raw_data.iloc[:,1:1434]
x = x.to_numpy().astype('float32')

raw_data = raw_data.to_numpy()

with open("data/Cora/Cora/raw_忘了/ind.cora.allx", 'rb') as f:
    allx = pickle.load(f, encoding='latin1').A
with open("data/Cora/Cora/raw_忘了/ind.cora.tx", 'rb') as f:
    tx = pickle.load(f, encoding='latin1').A
with open("data/Cora/Cora/raw_忘了/ind.cora.graph", 'rb') as f:
    ori_graph = pickle.load(f, encoding='latin1')
ori_x = np.vstack((allx, tx))
num=0
for source in ori_graph:
    index1 = find_x(ori_x[source])  # 查找source节点的feature对应原始Cora数据集的index，若查到了两个及以上，则返回-1，略过
    if index1 == -1:
        continue
    for target in ori_graph[source]:
        index2 = find_x(ori_x[target]) # 查找target节点的feature对应原始Cora数据集的index，若查到了两个及以上，则返回-1，略过
        if index2 == -1:
            continue
        source_id = raw_data[index1][0]  # 获取source节点的id
        target_id = raw_data[index2][0]  # 获取target节点的id
        if not find_cite(source_id, target_id): # 查找source-->target or target-->source是否存在在原始Cora数据集
            num+=1

print(num)