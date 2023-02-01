import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
import  torch.nn.functional as F
from torch_geometric.utils import add_self_loops,remove_self_loops,softmax
from torch_geometric.nn import GATConv

class GRUUint(nn.Module): #GRU单元

    def __init__(self, hid_dim, act):
        super(GRUUint, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim) #输入输出相同维度
        self.lin_z1 = nn.Linear(hid_dim, hid_dim)
        self.lin_r0 = nn.Linear(hid_dim, hid_dim)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim)
        self.lin_h0 = nn.Linear(hid_dim, hid_dim)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1) #初始化参数 均匀分布 ~ U ( − a , a )
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        z = (self.lin_z0(a) + self.lin_z1(x)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r)))
        return h * z + x * (1 - z)




class GraphLayer(gnn.MessagePassing):

    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False, step=2): #300->96 step代表gru层数
        super(GraphLayer, self).__init__(aggr='add') #图神经网络初始化
        self.step = step
        self.act = act
        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.gru = GRUUint(out_dim, act=act) #调用GRU单元
        # self.lpl=LPL(out_dim,act=act)
        self.gat=GATConv(out_dim,out_dim,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters() #更新参数

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, g):
        x = self.encode(x)# 单词数X300-》单词数X96
        x = self.act(x)
        for _ in range(self.step):#2层gru
            a = self.gat(x,g.edge_index)
            # a = self.propagate(edge_index=g.edge_index, x=x, edge_attr=self.dropout(g.edge_attr))#x 不变
            # ht = self.propagate(edge_index=g.edge_index, x=x, edge_attr=self.dropout(g.edge_attr))
            #同时调用message和update
            x = self.gru(x, a)
        x = self.graph2batch(x, g.length)#将图转化为batchsizeX最大长度Xhid_dim
        return x

    def message(self, x_i,x_j, edge_attr,edge_index):#x_j =边数xhid——dim,index是输入的边
        return x_j * edge_attr.unsqueeze(-1)

    def update(self, inputs):
        return inputs

    def graph2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False):
        super(ReadoutLayer, self).__init__()
        self.act = act
        self.bias = bias
        self.att = nn.Linear(in_dim, 1, bias=False)
        self.emb = nn.Linear(in_dim, in_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        # att = self.att(x).sigmoid() #软注意权重
        x = self.act(self.emb(x))
        # x = att * emb
        x = self.__max(x, mask) + self.__mean(x, mask)#batchsize*hid_dim
        x = self.mlp(x)#batchsize*out_dim
        return x

    def __max(self, x, mask):
        return (x + (mask - 1) * 1e9).max(1)[0]

    def __mean(self, x, mask):
        return (x * mask).sum(1) / mask.sum(1)


class Model(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=300, hid_dim=96,
                 step=2, dropout=0.5, word2vec=None, freeze=True):
        super(Model, self).__init__()
        if word2vec is None: # 没有就自己生成 但自己有
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), freeze, num_words)
            #Embedding(18754, 300, padding_idx=18753)
        self.gcn = GraphLayer(in_dim, hid_dim, act=torch.tanh, dropout=dropout, step=step) #使用上面2个单元 输入300 输出隐藏层维度
        self.read = ReadoutLayer(hid_dim, num_classes, act=torch.tanh, dropout=dropout) #输入隐藏层维度 输出文本分类数量 使用tanh
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.zeros_(m.bias)

    def forward(self, g):
        mask = self.get_mask(g)
        x = self.embed(g.x) #把输入转化为单词数量X300的向量
        x = self.gcn(x, g)
        x = self.read(x, mask)
        return x

    def get_mask(self, g):
        mask = pad_sequence([torch.ones(l) for l in g.length], batch_first=True).unsqueeze(-1) #增加一个维度
        if g.x.is_cuda: mask = mask.cuda() #转化为cuda
        return mask