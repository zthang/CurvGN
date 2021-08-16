import os.path as osp
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse
import pickle

is_DiGraph = False

def save_file(folder, prefix, name, obj):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_candidate_graph(index):
    id = raw_data[index][0]
    s = set()
    cite = raw_data_cites.to_numpy()
    for i in cite:
        if i[0]==id:
            s.add(i[1])
        elif i[1]==id:
            s.add(i[0])
    res_list=[map[node] if node in map else -1 for node in s]
    return res_list

def get_fin_index(idx, candidate):
    if len(candidate) == 1:
        return [candidate[0]]
    cite = raw_data_cites.to_numpy()
    num = len(set(ori_graph[idx]))
    digree=[]
    s_list=[]
    for index in candidate:
        if index in fin_list:
            digree.append(999999)
            continue
        id = raw_data[index][0]
        if id == 1153280:
            print(1)
        s = set()
        for i in cite:
            if i[0]==id:
                s.add(i[1])
            elif i[1]==id:
                s.add(i[0])
        digree.append(len(s))
        s_list.append(s)
    digree = np.array(digree) - num
    value = np.min(np.abs(digree))
    candidate = filter(lambda x:abs(x[1])==value, zip(candidate, digree))
    return [i[0] for i in candidate]

#allx x tx都是csr_matrix ally y ty都是nd array graph是defaultdict
#0:Case_Based 1:Genetic_Algorithms 2:Neural_Networks 3:Probabilistic_Methods 4:Reinforcement_Learning 5:Rule_Learning 6:Theory

# 导入数据：分隔符为空格
indexDic={0:"Theory", 1:"Reinforcement_Learning", 2:"Genetic_Algorithms", 3:"Neural_Networks", 4:"Probabilistic_Methods", 5:"Case_Based", 6:"Rule_Learning"}
nameDic={"Case_Based":0,"Genetic_Algorithms":1,"Neural_Networks":2,"Probabilistic_Methods":3,"Reinforcement_Learning":4,"Rule_Learning":5,"Theory":6}
raw_data = pd.read_csv('cora/cora.content',sep = '\t',header = None)
num_vertex = raw_data.shape[0]# 样本点数2708
raw_data_cites = pd.read_csv('cora/cora.cites',sep = '\t',header = None)
num_edge=raw_data_cites.shape[0]

raw_data = raw_data.reset_index(drop=True)
x = raw_data.iloc[:,1:1434]
x = x.to_numpy().astype('float32')
raw_data = raw_data.to_numpy()

with open("data/Cora/Cora/raw_ml/ind.cora.allx", 'rb') as f:
    allx = pickle.load(f, encoding='latin1').A
with open("data/Cora/Cora/raw_ml/ind.cora.tx", 'rb') as f:
    tx = pickle.load(f, encoding='latin1').A
with open("data/Cora/Cora/raw_ml/ind.cora.ally", 'rb') as f:
    ally = pickle.load(f, encoding='latin1')
with open("data/Cora/Cora/raw_ml/ind.cora.ty", 'rb') as f:
    ty = pickle.load(f, encoding='latin1')
with open("data/Cora/Cora/raw_ml/ind.cora.graph", 'rb') as f:
    ori_graph = pickle.load(f, encoding='latin1')
ori_x = np.vstack((allx, tx))
ori_y = np.vstack((ally, ty))

num=0
fin_list=[]
can_list=[]
sorted_list=None
temp_array=np.empty(1435)
for idx, i in enumerate(ori_x):
    candidate = []
    for index, j in enumerate(x):
        if (i == j).all() and indexDic[np.argmax(ori_y[idx])] == raw_data[index][1434]:
            candidate.append(index)
    fin_index = get_fin_index(idx, candidate)
    if len(fin_index)==1:
        fin_index=fin_index[0]
        fin_list.append(fin_index)
        if sorted_list is not None:
            sorted_list = np.vstack((sorted_list, raw_data[fin_index]))
        else:
            sorted_list = raw_data[fin_index]
    else:
        can_list.append((idx, fin_index))
        temp_array.fill(num)
        if sorted_list is not None:
            sorted_list = np.vstack((sorted_list, temp_array))
        else:
            sorted_list = temp_array
        num-=1

temp_raw_data = pd.DataFrame(sorted_list).reset_index(drop=True)

a = list(temp_raw_data.index)
b = list(temp_raw_data[0])
c = zip(b,a)
map = dict(c)

for i, node in enumerate(can_list):
    node_temp_list=[]
    for node_index in node[1]:
        node_temp_list.append({node_index:get_candidate_graph(node_index)})
    can_list[i]={"ori_graph":ori_graph[node[0]],"candidate_graph":node_temp_list}
raw_data = pd.DataFrame(sorted_list).reset_index(drop=True)

labels = pd.get_dummies(raw_data[1434])
labels = labels.to_numpy().astype('int32')

# num0 = labels.Neural_Networks.value_counts()
# num1 = labels.Probabilistic_Methods.value_counts()
# num2 = labels.Reinforcement_Learning.value_counts()
# num3 = labels.Rule_Learning.value_counts()
# num4 = labels.Theory.value_counts()
# num5 = labels.Genetic_Algorithms.value_counts()
# num6 = labels.Case_Based.value_counts()


# 将论文的编号转[0,2707]
a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b,a)
map = dict(c)


# 创建一个规模和邻接矩阵一样大小的矩阵
dic=defaultdict(list)
# 创建邻接矩阵
for i in range(2708):
    dic[i]=[]
for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
    source = map[i]
    target = map[j]
    dic[source].append(target)
    if not is_DiGraph:
        dic[target].append(source)

allx = scipy.sparse.csr_matrix(x[:1708])
tx = scipy.sparse.csr_matrix(x[1708:])
x = scipy.sparse.csr_matrix(x[:140])
ally = labels[:1708]
ty = labels[1708:]
y = labels[:140]

names = [('x', x), ('tx', tx), ('allx', allx), ('y', y), ('ty', ty), ('ally', ally), ('graph', dic)]
items = [save_file("raw", "Cora", name[0], name[1]) for name in names]