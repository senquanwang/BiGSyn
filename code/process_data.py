import sys
sys.path.append('..')

import os
import dgl
import json
import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from randomWalk import Node2vec
from torch_geometric.utils import degree, subgraph
from sklearn.model_selection import StratifiedKFold
# from sentence_transformers import SentenceTransformer

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, Descriptors

from drug_utils import single_smile_to_graph
from utils import setup_seed, calculate_shortest_path


def convert(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32): return int(o)
    else: print(type(o), o)
    raise TypeError


def rwExtractor(ids, edge_pairs, edge_rels, path, length, seed):

    json_path = os.path.join(path, f"subgraph_size{length}_seed{seed}.json")

    if os.path.exists(json_path):
        print(f'load subgraph from {json_path}')
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel+1

    my_graph = nx.Graph()
    my_graph.add_edges_from(edge_pairs)

    num_rel_update = []
    num_degree_update = []
    subgraphs = {}

    num_edge = len(edge_pairs)
    num_rel = max(edge_rels) + 1
    edge_index = torch.from_numpy(np.array(edge_pairs).T)  ##[2, num_edges]
    rel_index = torch.from_numpy(np.array(edge_rels))

    setup_seed(seed)

    for d in ids:
        subsets = Node2vec(start_nodes=[int(d)], graph=my_graph, path_length=length, num_paths=1, workers=6, dw=True).get_walks() ##返回一个list
        mapping_id = subsets.index(int(d))
        mapping_list = [False for _ in range(len((subsets)))]
        mapping_list[mapping_id] = True

        sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, rel_index, relabel_nodes=True)
        # relabel_nodes=True 会将 subsets 中的节点编号为 [0, 1, 2]（按照 subsets 的顺序）
        row_sub, col_sub = sub_edge_index
        num_degree = int(torch.max(degree(col_sub.to(torch.int64))).item())

        sub_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()  # list [N x 2]
        sub_rel_index = sub_rel_index.numpy().tolist()  # list [N]
        num_sub_edge = len(sub_edge_index)

        # 计算子图中节点之间的最短路径: 构建最短路径关系图
        s_edge_index = sub_edge_index.copy()
        s_value = [1 for _ in range(num_sub_edge)]
        s_rel = sub_rel_index.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index)  # 会计算节点到自身的距离
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]  # 最短距离

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里  # 原始子图中的边
                continue
            elif sp_value[i] == 0:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i]-1 + num_rel)  # 将最短距离作为新的关系

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_s_rel = max(s_rel) + 1

        num_rel_update.append(num_s_rel)  # 添加了最短距离后的关系数量
        num_degree_update.append(num_degree)  # 原始子图的最大度

        subgraphs[str(d)] = subsets, sub_edge_index, sub_rel_index, mapping_list, s_edge_index, s_value, s_rel, num_degree

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(num_degree_update), max(num_rel_update)


def generate_subgraphs(DrugToID, CellToID, edge_pairs, edge_rels, args):
    drug_id, cell_id = set(DrugToID.values()), set(CellToID.values())

    method = args.extractor
    path = os.path.join("../data/processed", args.dataset, method)
    os.makedirs(path, exist_ok=True)

    if method == "randomWalk":
        ids = drug_id.union(cell_id)
        subgraphs, max_degree, max_rel_num = rwExtractor(ids, edge_pairs, edge_rels, path, length=args.size, seed=args.seed)

    return subgraphs, max_degree, max_rel_num


def generate_molgraphs(dataset, ligands):

    smile_graph = {}

    paths = os.path.join("../data/processed", dataset, "molgraphs.json")

    if os.path.exists(paths):
        print(f'load molgraph from {paths}')
        with open(paths, 'r') as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        max_dist = 0

        for s in smile_graph.keys():
            max_rel = max(smile_graph[s][5]) if max(smile_graph[s][5]) > max_rel else max_rel  # 调换了s_rel和s_value的位置所以要重新生成
            max_degree = smile_graph[s][7] if smile_graph[s][7] > max_degree else max_degree
            # max_dist = smile_graph[s][8] if smile_graph[s][8] > max_dist else max_dist

        return smile_graph, max_rel+1, max_degree, max_dist

    num_mol_rel_update = 0
    max_node_degree = []
    max_path_dist = 0

    for d in ligands.keys():
        try:
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))  ##还是smiles序列（标准化处理）
        except Exception as e:
            # 输出错误信息并终止程序
            print(f"程序执行出错: {e}")
            print(d)
            print(ligands[d])
            sys.exit(1)
        c_size, atom_feats, edge_index, rel_index, s_edge_index, s_rel, s_value, max_deg = single_smile_to_graph(lg)
        # c_size, atom_feats, bond_feats, edge_index, rel_index, node_paths, edge_paths, max_deg, max_dist = single_smile_to_graph(lg)

        if c_size == 0: ##证明这个药物只由一个atom组成，这种的不考虑
            print(f'{d} is a single atom molecule, skip it.')
            sys.exit(1)

        if max(s_rel)+1 > num_mol_rel_update:
            num_mol_rel_update = max(s_rel)+1

        max_node_degree.append(max_deg)

        # if max_dist > max_path_dist:
        #     max_path_dist = max_dist

        smile_graph[str(d)] = c_size, atom_feats, edge_index, rel_index, s_edge_index, s_rel, s_value, max_deg
        # smile_graph[str(d)] = c_size, atom_feats, bond_feats, edge_index, rel_index, node_paths, edge_paths, max_deg, max_dist

    with open(paths, 'w') as f:
        json.dump(smile_graph, f, default=convert)

    return smile_graph, num_mol_rel_update, max(max_node_degree), max_path_dist


def load_data(args, device):
    data_file = '../data'
    Dataset_Name = args.dataset
    path = os.path.join(data_file, Dataset_Name)

    save_dir = '../data/processed'
    save_path = os.path.join(save_dir, Dataset_Name)
    os.makedirs(save_path, exist_ok=True)

    comb_file = os.path.join(path, 'drug_combinations.csv')
    comb = pd.read_csv(comb_file)
    comb['drug_a_name'] = comb['drug_a_name'].str.upper()
    comb['drug_b_name'] = comb['drug_b_name'].str.upper()

    drug = set(comb['drug_a_name']).union(set(comb['drug_b_name']))
    drug = sorted(list(drug))
    drug_num = len(drug)
    DrugToID = dict(zip(drug, range(drug_num)))

    cell = set(comb['cell'])
    cell = sorted(list(cell))
    cell_num = len(cell)
    CellToID = dict(zip(cell, range(drug_num, drug_num + cell_num)))

    comb['drug_a_name'] = comb['drug_a_name'].map(DrugToID)
    comb['drug_b_name'] = comb['drug_b_name'].map(DrugToID)
    comb['cell'] = comb['cell'].map(CellToID)

    # threshold = 30  # DeepSynergy
    # comb['label'] = (comb['synergy'] > threshold).astype(int)
    score = comb['synergy'].to_numpy()
    upper_quartile = np.percentile(score, 75)
    lower_quartile = np.percentile(score, 25)
    if Dataset_Name == 'Oneil':
        lower_quartile = -10
    comb = comb[(comb['synergy'] < lower_quartile) | (comb['synergy'] > upper_quartile)]
    comb['label'] = comb['synergy'].apply(lambda x: 1 if x > upper_quartile else (0 if x < lower_quartile else None))
    label_counts = comb['label'].value_counts()
    print("Positive:", label_counts.get(1, 0))
    print("Negative:", label_counts.get(0, 0))

    comb = comb[['drug_a_name', 'drug_b_name', 'cell', 'label']].to_numpy()

    # Construct molgraphs
    drug_info_file = os.path.join('../data/processed', Dataset_Name, 'drug_smiles.csv')
    Drug_Information = pd.read_csv(drug_info_file)
    smiles_dict = {}
    for d in drug:
        try:
            smiles_dict[DrugToID[d]] = Drug_Information[Drug_Information['Name'] == d]['SMILES'].values[0]
        except Exception as e:
            # 输出错误信息并终止程序
            print(f"程序执行出错: {e}")
            print(d)
            sys.exit(1)
    # molgraphs, max_mol_deg, num_mol_rel = generate_molgraphs(Dataset_Name, smiles_dict)
    molgraphs, num_mol_rel, max_mol_deg, max_mol_dist = generate_molgraphs(Dataset_Name, smiles_dict)
    # molgraphs, num_mol_rel, max_mol_deg, max_mol_dist = generate_mol_dglgraphs(Dataset_Name, smiles_dict)

    # Drug feature
    drug_feature_file = os.path.join(save_path, 'drug_features.pkl')
    if os.path.exists(drug_feature_file):
        print(f'load Drug_Features from {drug_feature_file}')
        with open(drug_feature_file, 'rb') as f:
            Drug_Features = pickle.load(f)
    # else:
    #     Drug_Features = []
    #     drug_model = SentenceTransformer('../pretrained/simcsesqrt-model', device=device)
    #     for d in tqdm(drug):
    #        smiles = smiles_dict[DrugToID[d]]
    #        f = drug_model.encode(smiles)
    #        Drug_Features.append(f.tolist())
    #     Drug_Features = np.array(Drug_Features)
    #     with open(drug_feature_file, 'wb') as f:
    #         pickle.dump(Drug_Features, f)
    print('Drug_num, Features_dim: ', Drug_Features.shape)

    # Cell_Line feature
    cell_embedding_dict = json.loads(open('../data/context_set_m.json', 'r').read())
    Cell_Line_Feature = []
    for c in cell:
        Cell_Line_Feature.append(cell_embedding_dict[c])
    Cell_Line_Feature = np.array(Cell_Line_Feature)
    print('Cell_Line_num, Features_dim: ', Cell_Line_Feature.shape)

    # Protein feature
    gene_fasta = pd.read_csv(f'../data/processed/gene_fasta.csv')
    protein = set(gene_fasta['ID'])
    protein = sorted(list(protein))
    protein_num = len(protein)
    ProteinToID = dict(zip(protein, range(drug_num + cell_num, drug_num + cell_num + protein_num)))
    protein_feature_file = os.path.join(save_path, 'protein_features.pkl')
    if os.path.exists(protein_feature_file):
        print(f'load Protein_Features from {protein_feature_file}')
        with open(protein_feature_file, 'rb') as f:
            Protein_Features = pickle.load(f)
    # else:
    #     Protein_Features = []
    #     protein_model = SentenceTransformer('../pretrained/simcsesqrt-model', device=device)
    #     for p in tqdm(protein):
    #         seq = gene_fasta[gene_fasta['ID'] == p]['Seq'].values[0]
    #         f = protein_model.encode(seq)
    #         Protein_Features.append(f.tolist())
    #     Protein_Features = np.array(Protein_Features)
    #     with open(protein_feature_file, 'wb') as f:
    #         pickle.dump(Protein_Features, f)
    print('Protein_num, Features_dim: ', Protein_Features.shape)

    # Tissue node
    CPI_file = os.path.join(path, 'cell_protein.csv')
    CPI = pd.read_csv(CPI_file)
    tissue = set(CPI['Tissue'])
    tissue = sorted(list(tissue))
    tissue_num = len(tissue)
    TissueToID = dict(
        zip(tissue, range(drug_num + cell_num + protein_num, drug_num + cell_num + protein_num + tissue_num)))
    Tissue_Feature = np.eye(tissue_num)  # one-hot encoding for tissue nodes
    print('Tissue_num, Features_dim: ', Tissue_Feature.shape)

    # Combine all features
    features_list = [Drug_Features, Cell_Line_Feature, Protein_Features, Tissue_Feature]
    in_dims = [features.shape[1] for features in features_list]
    features_list = [torch.tensor(features).float().to(device) for features in features_list]

    # load networks
    with open(os.path.join('../data/processed', Dataset_Name, 'edge_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        edge_pairs = data['edge_pairs']  # list [N x 2]
        edge_rels = data['edge_rels']  # list [N]
    num_edge = len(edge_pairs)
    num_rel = max(edge_rels) + 1
    dst = torch.from_numpy(np.array(edge_pairs).T)[1, :]
    num_degree = int(torch.max(degree(dst.to(torch.int64))).item())
    num_node = drug_num + cell_num + protein_num + tissue_num

    # Construct subgraphs
    subgraphs, max_sub_deg, num_sub_rel = generate_subgraphs(DrugToID, CellToID, edge_pairs, edge_rels, args)

    data_sta = {
        'num_node': num_node,
        'num_drug': drug_num,
        'num_cell': cell_num,
        'num_protein': protein_num,
        'num_tissue': tissue_num,
        'num_edge': num_edge,
        'num_rel': num_rel,
        'num_s_rel': num_sub_rel,  # 已+1
        'num_s_deg': max_sub_deg+1,  # +1是因为degree从0开始
        'num_m_rel': num_mol_rel,  # 已+1
        'num_m_deg': max_mol_deg+1,  # +1是因为degree从0开始
        'num_m_dist': max_mol_dist+1,  # +1是因为距离从0开始
    }

    return comb, subgraphs, data_sta, features_list, in_dims, molgraphs


def data_split(synergy, device, rd_seed):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    # -----split synergy into 5CV,test set
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),  # 对正样本数据进行随机打乱，其中 frac=1 表示对所有数据进行采样
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    # --CV set
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    # --test set
    synergy_test = np.concatenate((np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    # test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    # test_ind = torch.from_numpy(synergy_test).to(device)

    return synergy_cv_data, synergy_test#, test_ind, test_label

def k_fold(data, kf, folds, y):

    test_indices = []
    train_indices = []

    if len(y):
        for _, idx in kf.split(torch.zeros(len(data)), y):
            test_indices.append(idx)
    else:
        for _, idx in kf.split(data):
            test_indices.append(idx)

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def split_fold(folds, dataset, labels, scenario_type='random'):

    test_indices, train_indices, val_indices = [], [], []

    if scenario_type == 'random':##这是根据interactions在划分的数据集，也就是根据interactions的label进行的数据集划分
        skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
        train_indices, test_indices, val_indices = k_fold(dataset, skf, folds, labels)

    return train_indices, test_indices, val_indices