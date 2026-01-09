import os
import sys
sys.path.append('..')

import json
import copy
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from dataset import DDSDataset, collate
from models.BiGTSynergy import BiGTSynergy
from process_data import load_data, data_split
from utils import setup_seed, save, save_results
# from train_eval import train, eval,
from train_eval import test, add_log


def init_args():
    parser = argparse.ArgumentParser(description='DDS')
    '''data'''
    parser.add_argument('--dataset', type=str, default="DrugCombDB")  # Oneil or DrugCombDB
    '''model'''
    parser.add_argument('--model', type=str, default="BiGTSynergy")
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    '''train'''
    parser.add_argument('--n_epoch', type=int, default=200)  # 200 or 400
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)  # 1e-3 or 1e-4
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    '''subgraph'''
    parser.add_argument('--extractor', type=str, default="randomWalk")
    parser.add_argument('--size', type=int, default=32)

    parser.add_argument('--flag', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--arch', type=str, choices=["rel", "mol", "both"], default="both",
                        help="Architecture type: 'rel', 'mol', or 'both'")
    parser.add_argument('--rand_init', action='store_true')
    parser.add_argument('--cv_mode', type=int, default=1)

    parser.add_argument('--summary', type=str, default=None)
    parser.add_argument('--test', type=str, default='../best_save/DrugCombDB_both')  # Oneil_both or DrugCombDB_both
    parser.add_argument('--lc', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = init_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    if args.test is not None:
        with open(f"{args.test}/topk_metrics.txt", "w") as f:
            f.write(f"Test on {args.test}\n")

    seed = args.seed
    setup_seed(seed)

    synergy_data, subgraphs, data_sta, features_list, in_dims, molgraphs = load_data(args, device)
    print(data_sta)

    results_of_each_fold = []

    # # -----split synergy into 5CV,test set (ref: HyperGraphSynergy)
    synergy_cv, synergy_test = data_split(synergy_data, device, seed)  # np.array
    cv_mode = args.cv_mode
    if cv_mode == 1:
        cv_data = synergy_cv
    elif cv_mode == 2:  # cline_level
        cv_data = np.unique(synergy_cv[:, 2])  # 提取所有唯一值并排序
    else:  # drug pairs_level
        cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T  # 提取药物对的唯一组合(行)并排序

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_index, validation_index) in enumerate(kf.split(cv_data)):
        if cv_mode == 1:
            synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
        elif cv_mode == 2:  # cell line_level
            train_name, test_name = cv_data[train_index], cv_data[validation_index]
            synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
            synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
        else:  # drug pairs_level
            pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
            synergy_train = np.array([j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            synergy_validation = np.array([j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])

        train_data = DDSDataset(synergy=synergy_train, sub_graph=subgraphs, mol_graph=molgraphs)
        test_data = DDSDataset(synergy=synergy_test, sub_graph=subgraphs, mol_graph=molgraphs)
        eval_data = DDSDataset(synergy=synergy_validation, sub_graph=subgraphs, mol_graph=molgraphs)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        if args.model == "BiGTSynergy":
            model = BiGTSynergy(args=args, feature_list=features_list, in_dims=in_dims,
                                num_node=data_sta['num_node'], num_s_rel=data_sta['num_s_rel'], num_s_deg=data_sta['num_s_deg'],
                                num_m_rel=data_sta['num_m_rel'], num_m_deg=data_sta['num_m_deg'], device=device, num_m_dist=data_sta['num_m_dist'], dropout=0.2)
        model.to(device)

        if args.test is None:
            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)  # 对模型所有参数添加正则

            best_auc = 0.0
            early_stop_num = 0

            train_log = {'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_auc': [], 'train_aupr': [],
                         'train_f1': [], 'train_loss': [], 'eval_acc': [], 'eval_prec': [], 'eval_rec': [],
                         'eval_auc': [], 'eval_aupr': [], 'eval_f1': [], 'eval_loss': []}

            for i_epoch in range(args.n_epoch):
                loop = tqdm(train_loader, ncols=80)
                loop.set_description(f'{fold + 1}:Epoch[{i_epoch}/{args.n_epoch}]')
                '''Train'''
                train_acc, train_prec, train_rec, train_auc, train_aupr, train_f1, train_loss = train(loop, model,
                                                                                                      optimizer, device)
                '''Evaluate'''
                eval_acc, eval_prec, eval_rec, eval_auc, eval_aupr, eval_f1, eval_loss = eval(eval_loader, model,
                                                                                              device)

                print()
                print(args.dataset + '_' + args.flag)
                print(
                    f"train_loss:{train_loss:.3f} train_acc:{train_acc:.3f} train_prec:{train_prec:.3f} train_rec:{train_rec:.3f} train_auc:{train_auc:.3f} train_aupr:{train_aupr:.3f} train_f1:{train_f1:.3f}")
                print(
                    f"eval_loss :{eval_loss:.3f} eval_acc :{eval_acc:.3f} eval_prec :{eval_prec:.3f} eval_rec :{eval_rec:.3f} eval_auc :{eval_auc:.3f} eval_aupr :{eval_aupr:.3f} eval_f1 :{eval_f1:.3f}")
                print()

                add_log(train_log, train_acc, train_prec, train_rec, train_auc, train_aupr, train_f1, train_loss,
                        eval_acc,
                        eval_prec, eval_rec, eval_auc, eval_aupr, eval_f1, eval_loss)

                if eval_auc > best_auc:
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_auc = eval_auc
                    early_stop_num = 0
                else:
                    early_stop_num += 1
                    if early_stop_num > args.patience:
                        print("early stop!")
                        break

            model.load_state_dict(best_model_state)
        else:
            best_model_dir = '{}/{}/fold_{}/'.format(args.test, args.extractor, fold)
            subdirs = [d for d in os.listdir(best_model_dir) if os.path.isdir(os.path.join(best_model_dir, d))]
            best_model_dir = os.path.join(best_model_dir, subdirs[0])
            best_model = os.path.join(best_model_dir, '{}.pt'.format(args.model))
            model.load_state_dict(torch.load(best_model, map_location=device))

        model.to(device)
        '''Test'''
        test_log = test(test_loader, model, device, save_dir=args.test, fold=fold)  ##test_log是一个字典，里面存储着metrics

        if args.test is None:
            save_dir = os.path.join('../best_save/', args.model, args.dataset + '_' + args.flag, args.extractor,
                                    "fold_{}".format(fold), "{:.5f}".format(test_log['auc']))
            model.save(save_dir)  ##保存当前最好的模型
            save(save_dir, args, train_log, test_log)
            print(f"model and log save to {save_dir}")
        results_of_each_fold.append(test_log)

    save_results(os.path.join('../best_save/', args.dataset + '_' + args.flag), args, results_of_each_fold)







