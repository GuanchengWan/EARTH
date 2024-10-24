# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
# from torch_sparse import spmm
from utils import *
from torch_geometric.utils import degree, sort_edge_index

from mamba_ssm import Mamba
from torch import Tensor
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing, GATConv, GCNConv

import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from vector_fields import *



class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))

    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                   h0=h0,
                                   z0=z0,
                                   func_f=self.func_f,
                                   func_g=self.func_g,
                                   t=times,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:,...].transpose(0,1)

        #CNN based predictor
        output = self.end_conv(z_T)                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output




def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out


class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.3, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):

        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class SIRGCN(MessagePassing):
    def __init__(self, embedding_size):
        super().__init__(aggr='add')
        self.embedding_size = embedding_size
        #self.toS = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toI = nn.Linear(self.embedding_size * 2, self.embedding_size,bias=True)
        self.toR = nn.Linear(self.embedding_size, self.embedding_size,bias=True)

    def forward(self, s, i, r, edge_index, edge_weight):
        # x has shape [N, 3, embedding_size], where the second dim represents S, I, R
        return self.propagate(edge_index, edge_weight=edge_weight, s=s, x=i, r=r)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.reshape(-1,1)

    def update(self, neighbor_i, s, x, r):
        s1 = s - self.toI(torch.cat([s, neighbor_i], dim=-1))   # i.shape = 边数x节点数xembedding
        i1 = x + self.toI(torch.cat([s, neighbor_i], dim=-1)) - self.toR(x)
        r1 = self.toR(x) + r
        return s1, i1, r1
    
# class SIRGCN(MessagePassing):
#     def __init__(self, embedding_size):
#         super().__init__(aggr='add')
#         self.embedding_size = embedding_size
#         self.toI = nn.Linear(self.embedding_size * 2, self.embedding_size)
#         self.toR = nn.Linear(self.embedding_size, self.embedding_size)

#     def forward(self, s, i, r, edge_index, edge_weight):
#         # Add the batch dimension to edge_weight if necessary
#         if edge_weight.dim() == 1:
#             edge_weight = edge_weight.unsqueeze(1)

#         # x has shape [batch_size, n_nodes, embedding_size]
#         return self.propagate(edge_index, edge_weight=edge_weight, s=s, i=i, r=r)

#     def message(self, x_j, edge_weight):
#         # Multiply message by the edge weight
#         return x_j * edge_weight.view(-1, 1, 1)

#     def update(self, aggr_neighbor_i, s, i, r):
#         # aggr_neighbor_i has shape [batch_size, n_nodes, embedding_size]
#         s1 = s - self.toI(torch.cat([s, aggr_neighbor_i], dim=-1))
#         i1 = i + self.toI(torch.cat([s, aggr_neighbor_i], dim=-1)) - self.toR(i)
#         r1 = self.toR(i) + r
#         return s1, i1, r1



def remove_self_loops(edge_index):
    # 找到所有非自连接的边的索引
    non_self_loop_mask = edge_index[0] != edge_index[1]

    # 使用这个掩码过滤掉自连接的边
    filtered_edge_index = edge_index[:, non_self_loop_mask]

    return filtered_edge_index



class SIR_mamba(torch.nn.Module):
    def __init__(self, args, data):
        super(SIR_mamba, self).__init__()

        #############################################
        # # Parameters
        data.edge_index = remove_self_loops(data.edge_index)
        num_edges = data.edge_index.size(1)
        self.edge_index = data.edge_index

# 创建一个与data.edge_index第二个维度大小相同的可训练参数
        # self.g = g
        self.w = nn.Parameter(torch.ones(num_edges, requires_grad=True))
        
        self.feature_size = args.d_model
        self.activation = nn.ReLU()

        # Define GCN Layers
        self.base_gcn = SIRGCN(self.feature_size)
        #############################################
        # Define Input and Output MLP Layers
        self.input_s = nn.Sequential(nn.Linear(self.feature_size, self.feature_size, bias=True),)
        self.input_i = nn.Sequential(nn.Linear(self.feature_size, self.feature_size, bias=True),)
        self.input_r = nn.Sequential(nn.Linear(self.feature_size, self.feature_size, bias=True),)

        self.output = nn.Sequential(
            nn.Linear(self.feature_size * 3, self.feature_size, bias=True),
        )
        
        #############################################
        # Define Layer Norm
        self.batch_norm = nn.BatchNorm1d(self.feature_size)

        self.self_attn = Mamba(d_model=self.feature_size, # Model dimension d_model
        d_state=8,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2)   # Block expansion factor)
        
        
        self.d_model = self.feature_size
        
        self.d_state = 16
        
        self.graph_mamba = EncoderLayer(
                Mamba(
                    d_model=self.d_model,  # Model dimension d_model
                    d_state=self.d_state,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor)
                ),
                Mamba(
                    d_model=self.d_model,  # Model dimension d_model
                    d_state=self.d_state,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor)
                ),
            self.d_model,
        )
        
        
        #############################################
        # Initialize
        for l in self.input_s:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        for l in self.input_i:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        for l in self.input_r:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        for l in self.output:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        dropout = 0.5
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_local = nn.Dropout(dropout)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.ff_linear1 = nn.Linear(self.feature_size, self.feature_size * 2)
        self.ff_linear2 = nn.Linear(self.feature_size * 2, self.feature_size)
        
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
    
    
    def forward(self, x):
        orig_x = x
        # s = self.activation(self.batch_norm(self.input_s(x) + x))
        # i = self.activation(self.batch_norm(self.input_i(x) + x))
        # r = self.activation(self.batch_norm(self.input_r(x) + x))
        sl = []
        il = []
        rl = []
        s = self.activation(self.input_s(x) + x)
        i = self.activation(self.input_i(x) + x)
        r = self.activation(self.input_r(x) + x)
        
        for batch in range(s.shape[0]):
            sm, im, rm = self.base_gcn(s[batch], i[batch], r[batch], self.edge_index, self.w)
            sl.append(sm)
            il.append(im)
            rl.append(rm)
        
        s = torch.stack(sl,dim=0)
        i = torch.stack(il,dim=0)
        r = torch.stack(rl,dim=0)
        
        x_SIR = self.output(torch.cat([s,i,r],dim=-1))
    
        x_SIR = x + self.dropout_local(x_SIR)
        
        deg = degree(self.edge_index[0], x.shape[1]).to(torch.float).to(x.device)
        
        deg_noise = torch.rand_like(deg).to(deg.device)
        h_ind_perm = lexsort([deg + deg_noise, torch.zeros(x.shape[1]).to(x.device)])
        h_ind_perm_reverse = torch.argsort(h_ind_perm)
        h_attn = self.self_attn(x[:,h_ind_perm,:])[:,h_ind_perm_reverse,:]
        
        
        h_attn = self.dropout_attn(h_attn)
        
        h_attn = x + h_attn  # Residual connection.

        h = x_SIR
        
        h = h + self._ff_block(h)
        
        return h





class GraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 

 