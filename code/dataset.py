import torch
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric.data import Data
import numpy as np


class DDSDataset(InMemoryDataset):
    def __init__(self, synergy=None, sub_graph=None, mol_graph=None):
        super(DDSDataset, self).__init__()
        self.synergy = synergy
        self.sub_graph = sub_graph
        self.mol_graph = mol_graph

    def construct_subgraph_data(self, id, label):
        subsets, sub_edge_index, sub_rel_index, mapping_list, s_edge_index, s_value, s_rel, num_degree = self.sub_graph[str(id)]
        data = Data(x=torch.LongTensor(subsets),
                    y=torch.LongTensor([label]),
                    edge_index=torch.LongTensor(sub_edge_index).transpose(1, 0),  # [2, N]
                    edge_rel=torch.LongTensor(np.array(sub_rel_index, dtype=int)),
                    sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),  # [2, N]
                    sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int)),
                    sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                    id=torch.LongTensor(np.array(mapping_list, dtype=bool)))
        return data

    def construct_molgraph_data(self, drug_id, label):
        c_size, atom_feats, edge_index, rel_index, s_edge_index, s_rel, s_value, max_deg = self.mol_graph[str(drug_id)]  ##drug_id是str类型的，不是int型的，这点要注意
        data = Data(x=torch.Tensor(np.array(atom_feats)),
                    y=torch.LongTensor([label]),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    edge_rel=torch.LongTensor(np.array(rel_index, dtype=int)),
                    sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                    sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int)),
                    sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                    )
        return data

    def __len__(self):
        return len(self.synergy)

    def __getitem__(self, idx):
        drug1_id, drug2_id, cell_id, label = self.synergy[idx]

        drug1_subgraph = self.construct_subgraph_data(drug1_id, label)
        drug2_subgraph = self.construct_subgraph_data(drug2_id, label)
        cell_subgraph = self.construct_subgraph_data(cell_id, label)

        drug1_mol = self.construct_molgraph_data(drug1_id, label)
        drug2_mol = self.construct_molgraph_data(drug2_id, label)

        return drug1_subgraph, drug2_subgraph, cell_subgraph, drug1_mol, drug2_mol


def collate(data_list):

    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = Batch.from_data_list([data[4] for data in data_list])

    return batchA, batchB, batchC, batchD, batchE