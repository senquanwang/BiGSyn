import sys
import dgl
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import FindBRICSBonds

from utils import calculate_shortest_path


e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))  # 返回布尔值，np.array会将布尔值转换为数值类型


def atom_features(atom):
    # 44 （+11） +11 +11 +1 = 67（78）
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    # one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 最后有单独返回 atom.GetDegree()
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def single_smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    atom_feats = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        atom_feats.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        print(f"No bonds found for {smile}")
        sys.exit(1)
        # return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:,:2].tolist()
    mol_rel_index = mol_index[:,2].tolist()
    num_mol_edge = len(mol_edge_index)

    # 计算节点之间的最短路径: 构建最短路径关系图
    s_edge_index = mol_edge_index.copy()
    s_value = [1 for _ in range(num_mol_edge)]
    s_rel = mol_rel_index.copy()

    ##在这个位置应该计算的是最短路径
    edge_index_value = calculate_shortest_path(mol_edge_index)
    sp_edge_index = edge_index_value[:, :2]
    sp_value = edge_index_value[:, 2]

    num_rel = len(e_map['bond_type'])  # bond_type的个数
    for i in range(len(sp_edge_index)):
        if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里  # 原始子图中的边
            continue
        elif sp_value[i] == 0:  # 会计算节点到自身的距离
            s_edge_index.append(sp_edge_index[i].tolist())
            s_value.append(sp_value[i])
            s_rel.append(sp_value[i] + num_rel)
        else:
            s_edge_index.append(sp_edge_index[i].tolist())
            s_value.append(sp_value[i])
            s_rel.append(sp_value[i] - 1 + num_rel)  # 将最短距离作为新的关系

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    return (c_size, atom_feats,
            # bond_feats,
            mol_edge_index, mol_rel_index,
            s_edge_index, s_rel, s_value,
            # node_paths, edge_paths,
            max(degrees),
            # max_dist
            )


