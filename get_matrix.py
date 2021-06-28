import pickle
import numpy as np

def cos_distance(vec1, vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2)))
def euclid_distance(vec1,vec2):
    return np.sqrt(np.sum(np.square(vec1-vec2)))

def make_matrix():
    with open("vectors_norm", 'rb') as f:
        vectors = pickle.load(f)

    res = np.empty((0, 2715))
    for i in range(2715):
        temp = np.array([])
        for j in range(2715):
            temp = np.append(temp, 1-cos_distance(vectors[i], vectors[j]))
        res = np.append(res, np.array([temp]), axis=0)
    with open("matrix_norm_cos.pickle", "wb") as f:
        pickle.dump(res, f)
        print("ok")
    f.close()

    # res = np.empty((0, 2715))
    # for i in range(2715):
    #     temp = np.array([])
    #     for j in range(2715):
    #         temp = np.append(temp, euclid_distance(vectors[i], vectors[j]))
    #     res = np.append(res, np.array([temp]), axis=0)
    # with open("matrix_norm_euclid.pickle", "wb") as f:
    #     pickle.dump(res, f)
    #     print("ok")
    # f.close()

def get_mean(arr, start_row, end_row, start_col, end_col):
    num=0
    sum=0.
    for i in range(start_row, end_row):
        for j in range(start_col,end_col):
            sum+=arr[i][j]
            num+=1
    return sum/num

with open("groundtruth_y",'rb') as f1:
    y=pickle.load(f1)

with open("matrix_norm_euclid.pickle",'rb') as f:
    v=pickle.load(f)
    print(get_mean(v,0,2708,0,2708))
    print(get_mean(v,0,2708,2708,2715))
    print(get_mean(v,2708,2715,2708,2715))
    t=v[:2708,2708:]
    t=np.argmin(t,axis=1)
    acc=np.count_nonzero(t==y[:2708])/2708
    print(1)
