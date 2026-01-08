import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from .ContrastiveLoss import ContrastiveLoss
from .GraphTransformer import GraphTransformer


class HeteroNodeFeatureEncoder(torch.nn.Module):
    def __init__(self, input_dims, embed_dim):
        super().__init__()
        self.encoders = torch.nn.ModuleDict()
        for ntype, in_dim in enumerate(input_dims):
            self.encoders[str(ntype)] = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, feature_list):
        encoded_features = []
        for ntype, features in enumerate(feature_list):
            encoded_features.append(self.encoders[str(ntype)](features))
        concatenated_features = torch.cat(encoded_features, dim=0)
        return concatenated_features


class NodeFeatures(nn.Module):
    def __init__(self, degree, feature_num, embed_dim, type, rand_init):
        super(NodeFeatures, self).__init__()

        self.rand_init = rand_init
        self.type = type

        if type == 'graph':
            self.node_encoder = nn.Linear(feature_num, embed_dim)
        elif rand_init:
            self.node_encoder = torch.nn.Embedding(feature_num, embed_dim)
        if type == 'graph' or rand_init:
            nn.init.kaiming_uniform_(self.node_encoder.weight.data, nonlinearity='relu')

        self.degree_encoder = nn.Embedding(degree, embed_dim, padding_idx=0)  ##将度的值映射成embedding
        nn.init.kaiming_uniform_(self.degree_encoder.weight.data, nonlinearity='relu')

    def reset_parameters(self):
        self.degree_encoder.reset_parameters()
        if self.type == 'graph' or self.rand_init:
            self.node_encoder.reset_parameters()

    def forward(self, data, X):
        row, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        if self.type == 'graph' or self.rand_init:
            node_feature = self.node_encoder(data.x)
        else:  # rel
            node_feature = X[data.x]
        node_feature += self.degree_encoder((x_degree).long())
        return node_feature


class HGTSynergy(nn.Module):
    def __init__(self, args, feature_list, in_dims, num_node, num_s_rel, num_s_deg, num_m_rel, num_m_deg, device, dropout = 0.2, num_m_dist=None):
        super(HGTSynergy, self).__init__()

        layer = args.layer
        embed_dim = args.hidden_dim
        arch = args.arch
        rand_init = args.rand_init
        num_heads = args.num_heads

        self.rand_init = rand_init
        if not self.rand_init:
            self.feature_list = feature_list
            self.feature_encoder = HeteroNodeFeatureEncoder(input_dims=in_dims, embed_dim=embed_dim)

        self.arch = arch
        if self.arch == 'rel' or self.arch == 'both':
            self.node_feature = NodeFeatures(degree=num_s_deg, feature_num=num_node, embed_dim=embed_dim, type='node', rand_init=rand_init)
            self.node_representation_learning = GraphTransformer(layer_num=layer, embed_dim=embed_dim, num_heads=num_heads, num_rel=num_s_rel, dropout=dropout, type='node')
        if self.arch == 'mol' or self.arch == 'both':
            self.atom_feature = NodeFeatures(degree=num_m_deg, feature_num=67, embed_dim=embed_dim, type='graph', rand_init=False)  # (old) feature_num=67
            self.mol_representation_learning = GraphTransformer(layer_num=layer, embed_dim=embed_dim, num_heads=num_heads, num_rel=num_m_rel, dropout=dropout, type='graph')
        if self.arch == 'both':
            #MLP
            self.fuser = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )

            self.lc = args.lc
            self.criterion = ContrastiveLoss(device, temperature=0.1).to(device)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 8, 2)
        )


    def forward(self, drug1_subgraph, drug2_subgraph, cell_subgraph, drug1_mol, drug2_mol, pred_only=False):
        X = None
        if not self.rand_init:
            X = self.feature_encoder(self.feature_list)

        if self.arch == 'rel':
            drug1_node_feature = self.node_feature(drug1_subgraph, X)
            drug2_node_feature = self.node_feature(drug2_subgraph, X)
            drug1_emb, drug1_sub_emb, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
            drug2_emb, drug2_sub_emb, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)
            cell_node_feature = self.node_feature(cell_subgraph, X)
            cell_emb, cell_sub_emb, cell_attn = self.node_representation_learning(cell_node_feature, cell_subgraph)

        if self.arch == 'mol':
            mol1_atom_feature = self.atom_feature(drug1_mol, X)
            mol2_atom_feature = self.atom_feature(drug2_mol, X)
            mol1_emb, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
            mol2_emb, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        if self.arch == 'both':
            drug1_node_feature = self.node_feature(drug1_subgraph, X)
            drug2_node_feature = self.node_feature(drug2_subgraph, X)
            drug1_emb, drug1_sub_emb, drug1_attn = self.node_representation_learning(drug1_node_feature, drug1_subgraph)
            drug2_emb, drug2_sub_emb, drug2_attn = self.node_representation_learning(drug2_node_feature, drug2_subgraph)
            cell_node_feature = self.node_feature(cell_subgraph, X)
            cell_emb, cell_sub_emb, cell_attn = self.node_representation_learning(cell_node_feature, cell_subgraph)

            mol1_atom_feature = self.atom_feature(drug1_mol, X)
            mol2_atom_feature = self.atom_feature(drug2_mol, X)
            mol1_emb, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
            mol2_emb, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        if self.arch == 'both':
            loss_con1 = self.criterion(drug1_emb, mol1_emb)
            loss_con2 = self.criterion(drug2_emb, mol2_emb)

            # MLP
            drug1_emb = self.fuser(torch.cat([drug1_emb, mol1_emb], dim=-1))
            drug2_emb = self.fuser(torch.cat([drug2_emb, mol2_emb], dim=-1))

        if self.arch == 'rel' or self.arch == 'both':
            concat_embed = torch.cat([drug1_emb, drug2_emb, cell_emb], dim=-1)
        else:
            concat_embed = torch.cat([mol1_emb, mol2_emb, X[cell_subgraph.x][cell_subgraph.id.nonzero().flatten()]], dim=-1)
        score = self.fc(concat_embed)

        pred = F.log_softmax(score, dim=-1)
        if pred_only:
            return torch.exp(pred)[:, 1]

        loss_label = F.nll_loss(pred, drug1_subgraph.y.view(-1))
        loss = loss_label + self.lc * (loss_con1 + loss_con2) if self.arch == 'both' else loss_label

        return torch.exp(pred)[:, 1], loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path

    def reset_parameters(self):
        if not self.rand_init:
            for module in self.feature_encoder.encoders.values():
                module.reset_parameters()
        self.node_feature.reset_parameters()
        self.node_representation_learning.reset_parameters()
        if self.use_mol:
            self.atom_feature.reset_parameters()
            self.mol_representation_learning.reset_parameters()