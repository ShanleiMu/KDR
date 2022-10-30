# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 21:10
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : base_evaluator.py

import torch
import numpy as np


def get_precision(pos_index):
    precision_list = np.cumsum(pos_index, axis=1) / np.arange(1, pos_index.shape[1] + 1)
    precision_list = np.sum(precision_list, axis=0) / precision_list.shape[0]
    return precision_list


def get_recall(pos_index, pos_len):
    recall_list = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    recall_list = np.sum(recall_list, axis=0) / recall_list.shape[0]
    return recall_list


def get_ndcg(pos_index, pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    result = np.sum(result, axis=0) / result.shape[0]
    return result


class BaseEvaluator(object):
    def __init__(self):
        super(BaseEvaluator, self).__init__()
        self.top_k_list = [1, 5, 10, 20]

    def evaluate(self, scores):
        _, topk_index = torch.topk(scores.unsqueeze(0), max(self.top_k_list), dim=-1)
        return topk_index

    def collect(self, index_list, eval_data):
        pos_len_list = np.array([ins[2] for ins in eval_data])
        topk_index = torch.cat(index_list, dim=0).cpu().numpy()
        assert len(pos_len_list) == len(topk_index)

        precision, recall, ndcg = self.calculate_metrics(topk_index, pos_len_list)
        return precision, recall, ndcg

    def calculate_metrics(self, topk_index, pos_len_list):
        pos_idx_matrix = (topk_index < pos_len_list.reshape(-1, 1))
        precision_list = get_precision(pos_idx_matrix)
        recall_list = get_recall(pos_idx_matrix, pos_len_list)
        ndcg_list = get_ndcg(pos_idx_matrix, pos_len_list)
        precision, recall, ndcg = [], [], []
        for k in self.top_k_list:
            precision.append(precision_list[k - 1])
            recall.append(recall_list[k - 1])
            ndcg.append(ndcg_list[k - 1])
        return precision, recall, ndcg
