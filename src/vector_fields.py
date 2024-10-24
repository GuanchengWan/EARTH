import torch
import torch.nn as nn
import torch.nn.functional as F
    
from torch_geometric.nn import MessagePassing, GATConv, GCNConv

from torch.nn import Parameter, Linear


class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()
        
        # z = self.dropout(z)

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = z.tanh()
        # z = self.dropout2(z)
        return z

class FinalTanh_f_prime(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f_prime, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        # self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)    
        z = z.tanh()
        return z

class FinalTanh_f2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f2, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        # self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

        self.start_conv = torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=(1,1))


        self.linears = torch.nn.ModuleList(torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=(1,1))
                                           for _ in range(num_hidden_layers - 1))
        
        self.linear_out = torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=input_channels*hidden_channels,
                                    kernel_size=(1,1))

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        # z: torch.Size([64, 207, 32])
        z = self.start_conv(z.transpose(1,2).unsqueeze(-1))
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()

        z = self.linear_out(z).squeeze().transpose(1,2).view(*z.transpose(1,2).shape[:-2], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z
    
def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1) # (bs, num_nodes) #[1000, m+1]
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

from torch.nn import Parameter
import math
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

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
    
    #667
class SIRLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 我们需要三组权重来分别处理 S, I, R 的更新
        self.weight_s = Parameter(torch.Tensor(in_features *2, out_features))
        self.weight_i = Parameter(torch.Tensor(in_features * 2, out_features))
        self.weight_r = Parameter(torch.Tensor(out_features, out_features))
        nn.init.xavier_uniform_(self.weight_s)
        nn.init.xavier_uniform_(self.weight_i)
        nn.init.xavier_uniform_(self.weight_r)
        self.dropout = 0.4

        if bias:
            self.bias_s = Parameter(torch.Tensor(out_features))
            self.bias_i = Parameter(torch.Tensor(out_features))
            self.bias_r = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias_s.size(0))
            self.bias_s.data.uniform_(-stdv, stdv)
            self.bias_i.data.uniform_(-stdv, stdv)
            self.bias_r.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias_s', None)
            self.register_parameter('bias_i', None)
            self.register_parameter('bias_r', None)

# 667

    def forward(self, s, i, r, adj):
        # 加入噪声项
        noise_s = torch.randn_like(s) * 0.01
        noise_i = torch.randn_like(i) * 0.01
        
        if self.training:
            support_s = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) 
            support_i = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) - torch.matmul(i, self.weight_r) 
            support_r = torch.matmul(i, self.weight_r)
        else:
            support_s = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) 
            support_i = torch.matmul(torch.cat([s, adj @ i], dim=-1), self.weight_i) - torch.matmul(i, self.weight_r) 
            support_r = torch.matmul(i, self.weight_r)


        s =  - F.dropout(support_s, self.dropout, training=self.training)


        i = F.dropout(support_i, self.dropout, training=self.training)


        r = F.dropout(support_r, self.dropout, training=self.training)
        
        
        # s =  - support_s


        # i = support_i


        # r = support_r        

        

        return s, i, r
    


class SIRGCN(MessagePassing):
    def __init__(self, embedding_size):
        super().__init__(aggr='add')
        self.embedding_size = embedding_size
        #self.toS = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toI = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toR = nn.Linear(self.embedding_size, self.embedding_size)


    def forward(self, s, i, r, edge_index, edge_weight):
        # x has shape [N, 3, embedding_size], where the second dim represents S, I, R
        return self.propagate(edge_index, edge_weight=edge_weight, s=s, x=i, r=r)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.reshape(-1,1)

    def update(self, neighbor_i, s, x, r):
        s1 =  - self.toI(torch.cat([s, neighbor_i], dim=1))   # i.shape = 边数x节点数xembedding
        i1 = + self.toI(torch.cat([s, neighbor_i], dim=1)) - self.toR(x)
        r1 = self.toR(x) 
        return s1, i1, r1



class ChebNetII(nn.Module):
    def __init__(self, num_features, hidden , K  = 6):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.prop1 = ChebnetII_prop(K)

        self.dprate = 0.5
        self.dropout = 0.5
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, X, edge_index):
        x = X
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        
        return x
    
import numpy as np      
              
# 定义图的邻接矩阵和度矩阵
def get_random_walk_matrix(adj_matrix):
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    inv_degree_matrix = np.linalg.inv(degree_matrix)
    random_walk_matrix = np.dot(inv_degree_matrix, adj_matrix)
    return random_walk_matrix

# 初始化节点级嵌入
def initialize_node_level_embeddings(adj_matrix, K):
    n = adj_matrix.shape[0]
    I = np.eye(n)
    M = get_random_walk_matrix(adj_matrix)
    embeddings = np.zeros((n, K))
    M_power = M
    for k in range(K):
        embeddings[:, k] = np.diag(M_power)
        M_power = np.dot(M_power, M)
    return embeddings


class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj
           
class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type, adj):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers


        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out_z = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out_s = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out_i = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out_r = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
            
        self.m = num_nodes
        self.n_hidden = hidden_channels
        
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.Wb1 = Parameter(torch.Tensor(self.m,self.m))
        self.wb1 = Parameter(torch.Tensor(1))     
        
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

        self.conv1 = GraphConvLayer(self.n_hidden, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_hidden)
        
        self.dropout = 0.5
        
        self.SIR_GCN = SIRGCN(self.n_hidden)
        self.sir = SIRLayer(self.n_hidden, self.n_hidden)
        
        self.EdgeWeightMLP = EdgeWeight(self.n_hidden, self.n_hidden, 1)
        
        self.ChebnetII = ChebNetII(self.n_hidden, self.n_hidden)
        
        
        self.weight = Parameter(torch.Tensor(self.n_hidden, self.n_hidden))
        nn.init.xavier_uniform_(self.weight)


        self.bias = Parameter(torch.Tensor(self.n_hidden))
        stdv = 1. / math.sqrt(self.bias.size(0))
        self.bias.data.uniform_(-stdv, stdv)
        
        self.graph_gen = GraphLearner(self.n_hidden)
        
        self.position = 16
        
        self.WQ = nn.Linear(self.n_hidden, self.n_hidden // 2)
        self.WK = nn.Linear(self.n_hidden, self.n_hidden // 2)
        
        # self.node_position_ebeddings = torch.Tensor(initialize_node_level_embeddings(adj.detach().cpu().numpy(), self.position)).to(adj.device)


        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.dropout1 = nn.Dropout(0.2)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)
# 666
    def forward(self, z_list):
        

        t = z_list[0]
        
        z = z_list[1]
        
        b = z.size(0)
        
        # hid_rpt_m = z.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        # hid_rpt_w = z.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        
        
        # a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        # a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        # query = self.WQ(z) # batch, N, hidden
        # query = self.dropout1(query)
        # key = self.WK(z)
        # key = self.dropout1(key)
        
        # a_mx = self.act( torch.bmm(query, key.transpose(1, 2))) 
        

        
        a_mx = F.normalize(self.graph_gen(z), p=2, dim=1, eps=1e-12, out=None)

        # adjs  = z_list[-1]
        
        
        adjs = self.adj.repeat(b,1)
        
        adjs = adjs.view(b, self.m, self.m)
        
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        # c = torch.sigmoid(a_mx)
        
        a_mx = adjs * c + a_mx * (1-c)
        
        
        
        Adj_soft = F.softmax(a_mx, dim=2)
        
        
        # adjs1 = self.dtw.repeat(b,1)
        
        # adjs1 = adjs1.view(b, self.m, self.m)
        
        # c1 = torch.sigmoid(a_mx @ self.Wb1 + self.wb1)
        
        # a_mx_semantic = adjs1 * c1 + a_mx * (1-c1)
        
        # Adj_soft_semantic = F.softmax(a_mx_semantic, dim=2)    
        

        
        
        z = self.agc(z, Adj_soft, self.fused_adj)
        
        
        # z = self.ChebnetII(z, self.edge)
        
        
        s, i, r = z_list[2:]
        
        
        
        s, i , r = self.sir(s, i, r, Adj_soft)

            

        z = self.linear_out_z(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        # z = F.dropout(z)
        
        s = self.linear_out_s(s).view(*s.shape[:-1], self.hidden_channels, self.hidden_channels)
        s = s.tanh()
        # s = F.dropout(s)
        
        i = self.linear_out_i(i).view(*i.shape[:-1], self.hidden_channels, self.hidden_channels)
        i = i.tanh()
        # i = F.dropout(i)
        
        r = self.linear_out_r(r).view(*r.shape[:-1], self.hidden_channels, self.hidden_channels)
        r = r.tanh()
        # r = F.dropout(r)
        
        return [z, s, i , r] #torch.Size([64, 307, 64, 1])

#         t = z_list[0]
        
#         z = z_list[1]
#         # z = self.linear_in(z)
#         # z = z.relu()
        
#         # Get the source and target nodes from edge_index
#         source_nodes = self.edge[0]
#         target_nodes = self.edge[1]

#         # Get the features of the source and target nodes
# # Get the features of the source and target nodes
#         z_i = t[:, source_nodes, :]
#         z_j = t[:, target_nodes, :]

#         # Calculate weights for all edges at once
#         weights = self.EdgeWeightMLP(z_i, z_j)
        
#         s , i , r = z_list[2:]
        
#         sl = []
#         il = []
#         rl = []
        
#         for batch in range(t.shape[0]):
#             sm, im, rm = self.SIR_GCN(s[batch], i[batch], r[batch], self.edge, weights[batch])
#             sl.append(sm)
#             il.append(im)
#             rl.append(rm)
        
#         s = torch.stack(sl,dim=0)
#         i = torch.stack(il,dim=0)
#         r = torch.stack(rl,dim=0)

#         # z = self.ChebnetII(z, self.dtw_edge)
        
#         z = self.linear_out_z(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
#         z = z.tanh()
        
#         s = self.linear_out_s(s).view(*s.shape[:-1], self.hidden_channels, self.hidden_channels)
#         s = s.tanh()
        
#         i = self.linear_out_i(i).view(*i.shape[:-1], self.hidden_channels, self.hidden_channels)
#         i = i.tanh()
        
#         r = self.linear_out_r(r).view(*r.shape[:-1], self.hidden_channels, self.hidden_channels)
#         r = r.tanh()
             
#         return [z, s, i , r] #torch.Size([64, 307, 64, 1])


    def agc(self, z, adj, origin_adj, power=1):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        # z [b, node_num, features]
        # self.m = z.size(1)

        # self.node_position_ebeddings = self.node_position_ebeddings
        # ---- for deep component
        
        z = self.linear_in(z)
        
        # z = self.linear_in(torch.cat([z,self.node_position_ebeddings.repeat(z.shape[0], 1, 1)],dim=-1))

        z = z.relu()
        
        # z = F.dropout(z, self.dropout, training=self.training) 
        
        global_h = torch.matmul(z, self.weight)
        
        for i in range(power):
            global_h = origin_adj @ global_h
            
        x = z + global_h
        
        
        # x = F.relu(self.conv1(z, adj))

        x = F.dropout(x, self.dropout, training=self.training)

        # # ---- not softmax
        # new_z = F.relu(self.conv2(x, adj))   
        
        
        
        z = x
        
        
        
        # node_num = self.node_embeddings.shape[0]
        
        # supports_adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        
        
        # supports = (supports_adj + self.norm_adj) / 2 
        
        # supports = self.norm_adj
        
        # # print("ahha")
        # # laplacian=False
        # laplacian=False
        # if laplacian == True:
        #     # support_set = [torch.eye(node_num).to(supports.device), -supports]
        #     support_set = [supports, -torch.eye(node_num).to(supports.device)]
        #     # support_set = [torch.eye(node_num).to(supports.device), -supports]
        #     # support_set = [-supports]
        # else:
        #     support_set = [torch.eye(node_num).to(supports.device), supports]
        # #default cheb_k = 3
        # for k in range(2, self.cheb_k):
        #     support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
            
        # supports = torch.stack(support_set, dim=0)
        # weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        # x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z




class EdgeWeight(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeWeight, self).__init__()
        self.f_e = nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self.f_self = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self.n_hidden = hidden_dim
        
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        
        
        
        # self.Wb = Parameter(torch.Tensor(self.m,self.m))
        # self.wb = Parameter(torch.Tensor(1))
        
        # hid_rpt_m = z.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        # hid_rpt_w = z.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        
        # a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        # a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        
        # adjs = self.adj.repeat(b,1)
        # adjs = adjs.view(b,self.m, self.m)
        
        # c = torch.sigmoid(a_mx @ self.Wb + self.wb)
    def forward(self, z_i, z_j):
        # Concatenate z_i and z_j and pass through f_e
        z_ij = torch.cat([z_i @ self.W1.t(), z_j @ self.W2.t()], dim=-1)
        
        edge_weight = self.f_e(z_ij)
        
        # Calculate self weight for z_i
        # self_weight = self.f_self(z_i)
        
        return edge_weight.squeeze(-1)


class VectorField_only_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_only_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        #FIXME:
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

        laplacian=False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z
    
    


class VectorField_g_prime(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g_prime, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size, num_nodes = query.size(0), query.size(1)

        # Project inputs to the multi-head space
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, num_nodes, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(2, 3)
        key = key.view(batch_size, num_nodes, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(2, 3)
        value = value.view(batch_size, num_nodes, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(2, 3)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted average
        context = torch.matmul(attn_weights, value)
        
        # Reshape back to [batch_size, num_nodes, seq_len, embed_dim]
        context = context.transpose(2, 3).contiguous().view(batch_size, num_nodes, -1, self.embed_dim)
        
        # Project the output back to the original embedding dimension
        output = self.out_proj(context)
        
        return output, attn_weights


class AttentionFusion(nn.Module):
    def __init__(self, features):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, X_list):
        # 拼接输入向量
        X = torch.stack(X_list, dim=1)  # 维度变为 (batch_size, 4, num_nodes, features)

        # 计算 Query, Key, Value
        Q = self.query(X)  # (batch_size, 4, num_nodes, features)
        K = self.key(X)    # (batch_size, 4, num_nodes, features)
        V = self.value(X)  # (batch_size, 4, num_nodes, features)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (features ** 0.5)  # (batch_size, 4, num_nodes, num_nodes)
        attention_weights = self.softmax(attention_scores)  # (batch_size, 4, num_nodes, num_nodes)

        # 聚合
        fusion = torch.matmul(attention_weights, V)  # (batch_size, 4, num_nodes, features)

        # 输出融合后的向量
        output = torch.mean(fusion, dim=1)  # (batch_size, num_nodes, features)

        return output




import torch
import torch.nn as nn

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, features, num_heads, dropout= 0.3):
        super(MultiHeadAttentionFusion, self).__init__()
        assert features % num_heads == 0, "Features must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)
        self.fc_out = nn.Linear(features, features)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, X_list):
        # batch_size, num_nodes, features = X1.shape

        # 拼接输入向量
        X = torch.stack(X_list, dim=1)  # 维度变为 (batch_size, 4, num_nodes, features)   # (batch_size, 4, num_nodes, features)
        batch_size, num, num_nodes,features = X.shape
        # 计算 Query, Key, Value
        Q = self.query(X)  # (batch_size, 4, num_nodes, features)
        K = self.key(X)    # (batch_size, 4, num_nodes, features)
        V = self.value(X)  # (batch_size, 4, num_nodes, features)

        # 拆分成多头
        Q = Q.view(batch_size, 4, num_nodes, self.num_heads, self.head_dim).transpose(2, 3)  # (batch_size, 4, num_heads, num_nodes, head_dim)
        K = K.view(batch_size, 4, num_nodes, self.num_heads, self.head_dim).transpose(2, 3)  # (batch_size, 4, num_heads, num_nodes, head_dim)
        V = V.view(batch_size, 4, num_nodes, self.num_heads, self.head_dim).transpose(2, 3)  # (batch_size, 4, num_heads, num_nodes, head_dim)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, 4, num_heads, num_nodes, num_nodes)
        attention_weights = self.softmax(attention_scores)  # (batch_size, 4, num_heads, num_nodes, num_nodes)

        # 聚合
        fusion = torch.matmul(attention_weights, V)  # (batch_size, 4, num_heads, num_nodes, head_dim)

        # 拼接多头输出
        fusion = fusion.transpose(2, 3).contiguous().view(batch_size, 4, num_nodes, features)  # (batch_size, 4, num_nodes, features)


        # 应用 Dropout 到多头拼接的结果
        fusion = self.dropout(fusion)
        
        
        # 最终线性变换
        output = self.fc_out(fusion.mean(dim=1))  # (batch_size, num_nodes, features)

        return output



class MultiViewFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.5):
        super(MultiViewFusion, self).__init__()
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic_feature, multi_view_features):
        # Apply cross-attention
        fusion_output, attn_weights = self.cross_attention(semantic_feature, multi_view_features, multi_view_features)
        
        # Apply residual connection, normalization, and dropout
        fusion_output = self.fc(fusion_output) + semantic_feature
        fusion_output = self.dropout(fusion_output)
        
        return fusion_output, attn_weights

