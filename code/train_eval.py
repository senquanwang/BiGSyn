import sys
import time
import torch
from tqdm import *
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc, recall_score, precision_score, average_precision_score


def test(loader, model, device, save_dir=None, fold=-1):
    correct, total_loss, total_samples   = 0, 0, 0
    model.eval()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            data_drug1, data_drug2, data_cell, data_mol1, data_mol2 = [d.to(device) for d in data]
            labels = data_cell.y

            predicts, loss = model(data_drug1, data_drug2, data_cell, data_mol1, data_mol2)

            ##获取指标
            prob_all.append(predicts)
            label_all.append(labels)
            total_loss += loss.item() * labels.size(0)  # 将每个batch的平均loss转成总loss
            total_samples += labels.size(0)

    test_loss = total_loss / total_samples
    label_all = torch.cat(label_all).cpu().detach().numpy()
    prob_all = torch.cat(prob_all).cpu().detach().numpy()
    acc, prec, rec, auc, aupr, f1 = get_score(label_all, prob_all)

    print(f"test_acc:{acc:.3f} test_prec:{prec:.3f} test_rec:{rec:.3f} test_auc:{auc:.3f} test_aupr:{aupr:.3f} test_f1:{f1:.3f}")

    topk_metrics = evaluate_topk(label_all, prob_all, topk_list=[10, 50, 100])
    print(topk_metrics)

    if save_dir is not None:
        np.savez_compressed(f"{save_dir}/test_results_{fold}.npz", y_true=label_all, y_score=prob_all)
        with open(f"{save_dir}/topk_metrics.txt", "a") as f:
            f.write(f"fold{fold}: {topk_metrics}\n")

    return {"acc": acc, "prec": prec, "rec": rec, "auc": auc, "aupr": aupr, "f1": f1, "loss": test_loss}

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.c_size

def get_score(label_all, prob_all):

    predicts_label = [1 if prob >= 0.5 else 0 for prob in prob_all]

    acc = accuracy_score(label_all, predicts_label)
    f1 = f1_score(label_all, predicts_label)
    auc = roc_auc_score(label_all, prob_all)  # roc_auc
    prec = precision_score(label_all, predicts_label, average='binary')
    rec = recall_score(label_all, predicts_label, average='binary')
    aupr = average_precision_score(label_all, prob_all)

    return acc, prec, rec, auc, aupr, f1

def add_log(train_log, train_acc, train_prec, train_rec, train_auc, train_aupr, train_f1, train_loss,
         eval_acc, eval_prec, eval_rec, eval_auc, eval_aupr, eval_f1, eval_loss):
    train_log['train_acc'].append(train_acc)
    train_log['train_prec'].append(train_prec)
    train_log['train_rec'].append(train_rec)
    train_log['train_auc'].append(train_auc)
    train_log['train_aupr'].append(train_aupr)
    train_log['train_f1'].append(train_f1)
    train_log['train_loss'].append(train_loss)
    train_log['eval_acc'].append(eval_acc)
    train_log['eval_prec'].append(eval_prec)
    train_log['eval_rec'].append(eval_rec)
    train_log['eval_auc'].append(eval_auc)
    train_log['eval_aupr'].append(eval_aupr)
    train_log['eval_f1'].append(eval_f1)
    train_log['eval_loss'].append(eval_loss)

def evaluate_topk(y_true, y_score, topk_list=[10, 20, 50]):
    results = {}
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    sorted_indices = np.argsort(y_score)[::-1]  # 从大到小排序
    sorted_y_true = y_true[sorted_indices]

    total_positives = np.sum(y_true)

    for k in topk_list:
        topk_true = sorted_y_true[:k]
        tp = np.sum(topk_true)

        precision_k = tp / k
        recall_k = tp / total_positives if total_positives > 0 else 0

        results[f'Top-{k} Precision'] = precision_k
        results[f'Top-{k} Recall'] = recall_k

    return results