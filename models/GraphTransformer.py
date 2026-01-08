# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)


class MultiheadAttention(MessagePassing):
    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.rel_encoding = Sequential(
            Linear(dim_model, 1),
            ReLU()
        )

        self.spatial_encoding = spatial_encoder


        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads  # 64/4=16

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)

        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self):
        self.rel_embedding.reset_parameters()
        self.rel_encoding[0].reset_parameters()
        self.spatial_encoding.reset_parameters()

        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.dense.reset_parameters()

    def message(self, x_j, edge_weight):
        r"""Constructs messages from node :math:`j` to node :math:`i`"""
        # 默认return x_j
        return edge_weight.view(-1, 1) * x_j  # 变为 [num_edges, 1]，与 x_j[num_edges, d] 做逐元素相乘

    def forward(self, x, sp_edge_index, sp_edge_rel, sp_value=None):

        q = self.wq(x)  # [nodes_num, dim_model]
        k = self.wk(x)
        v = self.wv(x).view(x.shape[0],self.num_heads,self.depth) ##[nodes_num, num_heads, depth]

        row, col = sp_edge_index  # [2, edge_nums]  (v,u)/(s,t)
        query_end, key_start = q[col], k[row]  # [edge_nums, dim_model]  (u,v) 要求v对u的重要性(与论文公式不对应) 在“目标节点 j”视角上衡量来自源节点 i 的信息的重要性

        rel_embedding = self.rel_embedding(sp_edge_rel)  # 创新点
        query_end += rel_embedding
        key_start += rel_embedding

        query_end = query_end.view(sp_edge_index.shape[1],self.num_heads,self.depth)  ##[edge_nums, num_heads, depths]
        key_start = key_start.view(sp_edge_index.shape[1],self.num_heads,self.depth)  # [e,h,d]

        # 原来代码采用加权线性归一化，仅对分子进行缩放，分子分母都没有计算exp
        # 计算注意力的分子numerator
        edge_attn_num = torch.einsum("ehd,ehd->eh", query_end, key_start) ##[edge_nums, num_heads]  # 点积  ψ(xv, xu, rc  v⇔u)
        # data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(edge_attn_num.shape[-1], dtype=torch.float32)))  # 1/sqrt(sqrt(num_heads)) ? => # 1/sqrt(self.depth)
        data_normalizer = 1.0 / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        edge_attn_num *= data_normalizer  # ψ(·, ·, ·)/√dk
        if sp_value is not None:
            edge_attn_bias = self.spatial_encoding(sp_value)  # scalar (reference Graphormer) [edge_nums, 1]
            edge_attn_num += edge_attn_bias  # 广播到每个 head

        # 为数值稳定性做减最大值 trick（每个目标节点分别做）每个目标节点 col（即接收节点）内，求最大
        max_per_node, _ = scatter_max(edge_attn_num, col, dim=0)  # scatter_max 返回 (值, 索引)
        max_per_node = max_per_node[col]  # 映射回每条边（按边顺序）
        edge_attn_num_exp = torch.exp(edge_attn_num - max_per_node)  # exp
        edge_attn_dem = scatter_add(edge_attn_num_exp, col, dim=0)[col]  # 分母：目标节点上 sum exp
        attention_weight = edge_attn_num / (edge_attn_dem + 1e-16)  # 防止除 0 ##[edge_nums, num_heads] ##scaled

        outputs = []
        for i in range(self.num_heads):
            v_single_head = v[:, i, :]  # [num_nodes, depth]
            output_per_head = self.propagate(edge_index=sp_edge_index, x = v_single_head, edge_weight = attention_weight[:, i], size=None)
            outputs.append(output_per_head)

        out = torch.cat(outputs,dim=-1)

        return self.dense(out), attention_weight


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout):
        super(GraphTransformerEncode, self).__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim)
        )

        self.multiHeadAttention = MultiheadAttention(dim_model = self.in_dim, num_heads = self.num_heads, rel_encoder=rel_encoder, spatial_encoder = spatial_encoder)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()

        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, feature, sp_edge_index, sp_edge_rel, sp_value=None):
        # h′(l) = MHA(LN(h(l−1))) + h(l−1)
        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(x_norm, sp_edge_index, sp_edge_rel, sp_value)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        # h(l) = FFN(LN(h′(l))) + h′(l)
        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight


class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super(SpatialEncoding, self).__init__()

        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU()
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap_ = torch.unsqueeze(lap, dim=-1) ##[n_edges, 1]
        out = self.fnn(lap_)

        return out


class GraphTransformer(torch.nn.Module):
    def __init__(self, layer_num = 2, embed_dim = 64, num_heads = 4, num_rel = 10, dropout = 0.2, type = 'graph'): ##type指示的是graph还是node，也就是对应的是图级别的表示学习，还是节点级别的表示学习
        super(GraphTransformer, self).__init__()

        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embed_dim)  ##权重共享的
        self.spatial_encoder = SpatialEncoding(embed_dim)  ##这两个是权重共享的

        self.encoder = torch.nn.ModuleList()
        for i in range(layer_num):  # 原来为什么要 -1 ?
            self.encoder.append(GraphTransformerEncode(num_heads = num_heads, in_dim = embed_dim, dim_forward = embed_dim*2, rel_encoder = self.rel_encoder, spatial_encoder = self.spatial_encoder, dropout=dropout))

        # if self.type == 'graph':
        #     self.out = SAGPoolReadout(embed_dim=embed_dim, out_dim=embed_dim, layer_num=2)

    def reset_parameters(self):
        for e in self.encoder:
            e.reset_parameters()


    def forward(self, feature, data):

        ##首先就是按照edge index计算attn_weight, 然后按照权重聚合就可以了！！
        x = feature
        graph_embedding_layer = []
        attn_layer = []
        for graphEncoder in self.encoder:  # GraphTransformerEncode
            x, attn = graphEncoder(x, data.sp_edge_index, data.sp_edge_rel, data.sp_value)
            # x, attn = graphEncoder(x, data.edge_index, data.edge_rel)
            graph_embedding_layer.append(x)
            attn_layer.append(attn)

        #all_out = torch.stack([x for x in graph_embedding_layer])

        if self.type == 'graph':
            ##pooling
            sub_representation = []
            for index, drug_mol_graph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]  ##第index个图中的各个节点的表示，[atom_number, emd_dim]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(x, batch=data.batch)  ##每个drug分子的图的表示
            # representation = self.out(x, adj=data.sp_edge_index, edge_attr=None, batch=data.batch)
        else:
            ##只返回 data.id 对应的节点的表示
            sub_representation = []
            for index, drug_subgraph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                #print(sub_embedding.shape)
                sub_representation.append(sub_embedding) ##只取那个节点的embedding
            #print(x.shape)
            #print(data.id.shape)
            representation = x[data.id.nonzero().flatten()]  # data.id

        return representation, sub_representation, attn_layer

        ##对于节点级别的表示，需要每一层的级联，然后做最后的互信息最大化，这个层级的优化可能要考虑一下，但是最终落到的还是节点和图
