import os.path as osp
import torch
import loaddatas as lds
import torch.nn.functional as F
from torch_geometric.data import Data
import random
import numpy as np
from config import Config
from baselines import ConvCurv
#load the neural networks
def train(train_mask):
    model.train()
    optimizer.zero_grad()
    nll_loss, Reg1, Reg2 = model(data)
    cross_entropy_loss = F.nll_loss(nll_loss[train_mask], data.y[train_mask])
    loss = cross_entropy_loss + config.gamma1 * Reg1 + config.gamma2 * Reg2 if config.loss_mode == 1 else cross_entropy_loss
    loss.backward()
    optimizer.step()

def test(train_mask,val_mask,test_mask):
    model.eval()
    logits, Reg1, Reg2 = model(data)
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        #print(pred)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    accs.append(F.nll_loss(logits[val_mask], data.y[val_mask]))
    print(accs)
    return accs

config = Config()
#load dataset
times = range(config.times)  #Todo:实验次数
is_train = config.is_train
epoch_num = config.epoch_num
wait_total = config.wait_total
pipelines = ['ConvCurv']
pipeline_acc = {'ConvCurv':[i for i in times]}
pipeline_acc_sum = {'ConvCurv':0}
# d_names=['Cora','Citeseer','PubMed','Photo','Computers','CS','Physics']
d_names = config.d_names
#d_names=['Photo','Computers','CS','Physics']
for d_name in d_names:
    f2=open('scores/pipe_benchmark_' +d_name+ '_scores.txt', 'w+')
    f2.write('{0:7} {1:7}\n'.format(d_name,'ConvCurv'))
    f2.flush()
    if d_name=='Cora' or d_name=='Citeseer' or d_name=='PubMed':
        d_loader='Planetoid'
    elif d_name=='Computers' or d_name=='Photo':
        d_loader='Amazon'
    else:
        d_loader='Coauthor'
    dataset=lds.loaddatas(d_loader,d_name)
    for time in times:
        for Conv_method in pipelines:
            data=dataset[0]
            index=[i for i in range(len(data.y))]
            if d_loader != 'Planetoid':
                train_len=20*int(data.y.max()+1)
                train_mask=torch.tensor([i < train_len for i in index])
                val_mask=torch.tensor([i >= train_len and i < 500+train_len for i in index])
                test_mask=torch.tensor([i >= len(data.y)-1000 for i in index])
            else:
                train_mask=data.train_mask.bool()
                val_mask=data.val_mask.bool()
                test_mask=data.test_mask.bool()
            model,data = locals()[Conv_method].call(data,dataset.name,data.x.size(1),dataset.num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
            best_val_acc = test_acc = 0.0
            best_val_loss = np.inf
            if is_train:
                for epoch in range(0, epoch_num):
                    train(train_mask)
                    train_acc, val_acc, tmp_test_acc, val_loss = test(train_mask, val_mask, test_mask)
                    if val_acc >= best_val_acc:
                        test_acc = tmp_test_acc
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        wait_step = 0
                    else:
                        wait_step += 1
                        if wait_step == wait_total:
                            print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                            break
            else:
                model.load_state_dict(torch.load(f"saved_model/ConvCurv_{epoch_num}"))
                print(test(train_mask,val_mask,test_mask))
            # del model
            del data
            pipeline_acc[Conv_method][time]=test_acc
            pipeline_acc_sum[Conv_method]=pipeline_acc_sum[Conv_method]+test_acc/len(times)
            log =f'Epoch: {epoch_num}, dataset name: '+ d_name + ', Method: '+ Conv_method + ' Test: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][time])))
        f2.write('{0:4d} {1:4f}\n'.format(time,pipeline_acc['ConvCurv'][time]))
        f2.flush()
        if not is_train:
            break
    if is_train:
        torch.save(model.state_dict(), f"saved_model/ConvCurv_{epoch_num}")
    f2.write('{0:4} {1:4f}\n'.format('std',np.std(pipeline_acc['ConvCurv'])))
    f2.write('{0:4} {1:4f}\n'.format('mean',np.mean(pipeline_acc['ConvCurv'])))
    f2.close()
