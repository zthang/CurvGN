import os.path as osp
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse
import pickle

is_DiGraph = True

def save_file(folder, prefix, name, obj):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

#allx x tx都是csr_matrix ally y ty都是nd array graph是defaultdict
#0:Case_Based 1:Genetic_Algorithms 2:Neural_Networks 3:Probabilistic_Methods 4:Reinforcement_Learning 5:Rule_Learning 6:Theory

# 导入数据：分隔符为空格
nameDic={"AI":0,"Agents":1,"DB":2,"HCI":3,"IR":4,"ML":5}
raw_data = pd.read_csv('citeseer/citeseer.content',sep = '\t',header = None)
num_vertex = raw_data.shape[0]# 样本点数2708
raw_data = raw_data.sample(frac=1).reset_index(drop=True)
a = list(raw_data.index)
b = list(raw_data[0])
b = [str(i) for i in b]
c = zip(b,a)
map = dict(c)
raw_data = raw_data.to_numpy()
nums = [0, 0, 0, 0, 0, 0]
train_list = None
left_list = None
for row in raw_data:
    if nums[nameDic[row[3704]]] < 20:
        if train_list is not None:
            train_list = np.vstack((train_list, row))
        else:
            train_list = row
        nums[nameDic[row[3704]]] += 1
    else:
        if left_list is not None:
            left_list = np.vstack((left_list, row))
        else:
            left_list = row
raw_data = np.vstack((train_list, left_list))
raw_data = pd.DataFrame(raw_data).reset_index(drop=True)

labels = pd.get_dummies(raw_data[3704])
labels = labels.to_numpy().astype('int32')
x = raw_data.iloc[:,1:3704]
x = x.to_numpy().astype('float32')

# num0 = labels.Neural_Networks.value_counts()
# num1 = labels.Probabilistic_Methods.value_counts()
# num2 = labels.Reinforcement_Learning.value_counts()
# num3 = labels.Rule_Learning.value_counts()
# num4 = labels.Theory.value_counts()
# num5 = labels.Genetic_Algorithms.value_counts()
# num6 = labels.Case_Based.value_counts()


raw_data_cites = pd.read_csv('citeseer/citeseer.cites',sep = '\t',header = None)
num_edge=raw_data_cites.shape[0]

# 创建一个规模和邻接矩阵一样大小的矩阵
dic=defaultdict(list)
num=0
# 创建邻接矩阵
for i in range(3327):
    dic[i]=[]
for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
    if str(i) in map and str(j) in map:
        source = map[i]
        target = map[j]
        if source == target:
            continue
        dic[source].append(target)
        num+=1
        if not is_DiGraph:
            dic[target].append(source)
print(num)
allx = scipy.sparse.csr_matrix(x[:2312])
tx = scipy.sparse.csr_matrix(x[2312:])
x = scipy.sparse.csr_matrix(x[:120])
ally = labels[:2312]
ty = labels[2312:]
y = labels[:120]

names = [('x', x), ('tx', tx), ('allx', allx), ('y', y), ('ty', ty), ('ally', ally), ('graph', dic)]
items = [save_file("raw", "Citeseer", name[0], name[1]) for name in names]