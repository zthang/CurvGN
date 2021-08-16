import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as seq, Parameter, LeakyReLU, init, Linear, Sigmoid, ReLU, PReLU, ELU, Tanh
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, degree
config = None
def maxabsscaler(data):
    max = torch.max(torch.abs(data))
    return data/max
    # max = torch.max(data)
    # min = torch.min(data)
    # zero = torch.tensor(0.)
    # data.map_(data, lambda x, *y: x/max if x > zero else -x/min)
    # return data

class  curvGN(MessagePassing):
    def __init__(self, in_channels, out_channels, curv_n, w_mul, data, my_config, bias=True):
        global config
        config = my_config
        super(curvGN, self).__init__(aggr='add') # "Add" aggregation.
        self.curvature_weight = Parameter((torch.tensor([[1.], [1.]])))
        self.w_mul = w_mul
        # self.w_mul_Forman = F.leaky_relu(data.w_mul_Forman, negative_slope=0.01)
        self.w_mul_Forman = maxabsscaler(data.w_mul_Forman)
        self.cosine_similarity = data.cosine_similarity.reshape(-1, 1)
        self.w_mul_activation = w_mul@torch.ones([1, out_channels])
        self.non_learnable_weight = torch.abs(torch.randn(w_mul.shape[0], 1, requires_grad=False))
        self.learnable_weight = Parameter(torch.abs(torch.randn(w_mul.shape[0], 1)), requires_grad=True)
        self.lin = Linear(in_channels, out_channels)
        widths = [1, out_channels]
        self.w_mlp_out = create_wmlp(widths, out_channels, 1)
        self.w_activation_out = create_activation_function(out_channels)
    def forward(self, x, edge_index):
        x = self.lin(x)
        if config.curvature_activate_mode == 0:
            if config.curvature_method == 2:
                # out_weight = self.w_mlp_out(torch.hstack((self.w_mul, self.w_mul_Forman)))
                out_weight = self.w_mlp_out(torch.hstack((self.w_mul, self.w_mul_Forman))@self.curvature_weight)
            elif config.curvature_method == 0:
                out_weight = self.w_mlp_out(self.w_mul)
            elif config.curvature_method == 1:
                out_weight = self.w_mlp_out(self.w_mul_Forman)
        else:
            out_weight = self.w_activation_out(self.w_mul_activation)
        out_weight = softmax(out_weight,edge_index[0])
        return self.propagate(x=x,edge_index=edge_index,out_weight=out_weight)
    def message(self,x_j,edge_index,out_weight):
        if config.learnable_curvature == 0:
            return out_weight*x_j
        elif config.learnable_curvature == 1:
            return self.non_learnable_weight@torch.ones(1, x_j.shape[1])*x_j
        elif config.learnable_curvature == 2:
            return self.learnable_weight@torch.ones(1, x_j.shape[1])*x_j
        elif config.learnable_curvature == 3:
            return self.cosine_similarity@torch.ones(1, x_j.shape[1])*x_j
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out

def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k], widths[k+1], bias=False))
        mlp_modules.append(LeakyReLU(0.2, True))
    mlp_modules.append(Linear(widths[len(widths)-1], nfeato, bias=lbias))
    return seq(*mlp_modules)

def create_activation_function(nfeato):
    if config.curvature_activate_mode == 1:
        return ReLU()
    elif config.curvature_activate_mode == 2:
        return LeakyReLU(0.1)
    elif config.curvature_activate_mode == 3:
        return PReLU()
    elif config.curvature_activate_mode == 4:
        return PReLU(nfeato)
    elif config.curvature_activate_mode == 5:
        return ELU()
    elif config.curvature_activate_mode == 6:
        return Tanh()
    elif config.curvature_activate_mode == 7:
        return Sigmoid()