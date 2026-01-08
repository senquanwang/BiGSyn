import random
import torch
import os
import numpy as np
import json
import networkx as nx
import pandas as pd


# 固定随机数种子
def setup_seed(seed):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False


def save(save_dir, args, train_log, test_log):
    args.device = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=False, separators=(',', ':'))
    with open(save_dir + '/test_results.json', 'w') as f:
        json.dump(test_log, f, indent=4, sort_keys=False, separators=(',', ':'))
    with open(save_dir + '/train_log.json', 'w') as f:
        json.dump(train_log, f, indent=4, sort_keys=False, separators=(',', ':'))


def save_results(save_dir, args, results_list):
    """
    results_list: List[Dict]
        每个元素是一折的结果
        e.g. {'acc':..., 'prec':..., 'rec':..., 'auc':..., 'aupr':..., 'f1':...}
    """

    # ===== 1. 收集每一折 =====
    metrics = ['acc', 'prec', 'rec', 'auc', 'aupr', 'f1']
    fold_results = []

    for r in results_list:
        fold_results.append({m: float(r[m]) for m in metrics})

    # ===== 2. 计算 mean / std =====
    mean_std = {}
    for m in metrics:
        values = np.array([r[m] for r in fold_results])
        mean_std[m] = [float(values.mean()), float(values.std())]

    # ===== 3. 组织最终保存内容 =====
    save_dict = {
        'args': vars(args),
        'fold_results': fold_results,
        'mean_std': mean_std
    }

    # ===== 4. 保存 =====
    save_path = f"{save_dir}_{args.extractor}_all_results.json"
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=4, sort_keys=False)

    print(f"[INFO] Results saved to {save_path}")

    args = vars(args)
    results = mean_std  # 兼容旧逻辑

    if args['summary'] is not None:
        # 从 results 字典提取
        summary_row = {
            "flag": args['flag'],
            "acc_mean": results['acc'][0],
            "acc_std": results['acc'][1],
            "prec_mean": results['prec'][0],
            "prec_std": results['prec'][1],
            "rec_mean": results['rec'][0],
            "rec_std": results['rec'][1],
            "auc_mean": results['auc'][0],
            "auc_std": results['auc'][1],
            "aupr_mean": results['aupr'][0],
            "aupr_std": results['aupr'][1],
            "f1_mean": results['f1'][0],
            "f1_std": results['f1'][1],
        }

        summary_dir = '../hyper_search/'
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, args['summary'])

        # 是否存在，决定是否先写 header
        if not os.path.exists(summary_file):
            df = pd.DataFrame([summary_row])
            df.to_csv(summary_file, index=False)
        else:
            df = pd.read_csv(summary_file)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            df.to_csv(summary_file, index=False)

        print(f"Appended results to {summary_file}")


def calculate_shortest_path(edge_index):

    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index)

    paths = nx.all_pairs_shortest_path_length(g)
    # (source, dictionary) iterator with dictionary keyed by target and shortest path length as the key value.
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)