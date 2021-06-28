import numpy as np
from gensim.models.word2vec import Word2Vec

# 0:theory 1:reinforcement_learning 2:genetic_algorithms 3:neural_networks 4:probabilistic_methods 5:case_based 6:rule_learning

model=Word2Vec.load("wiki.en.text.model")
lis=[model.wv["theory"]]
lis=np.vstack((lis,model.wv["reinforcement_learning"]))
lis=np.vstack((lis,model.wv["genetic_algorithms"]))
lis=np.vstack((lis,model.wv["neural_networks"]))
lis=np.vstack((lis,model.wv["probabilistic_methods"]))
lis=np.vstack((lis,model.wv["case"]))
lis=np.vstack((lis,model.wv["rule"]))

lis=lis.tolist()

with open("label_vector") as f:
    for i in lis:
        f.writelines(i)
        f.writelines('\n')
f.close()