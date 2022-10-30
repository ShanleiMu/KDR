# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 15:17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : utils.py

import os
import torch
import random
import numpy as np


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def get_saved_file(model_name, info_str, dataset):
    saved_path = model_name + '/saved/' + dataset
    ensure_dir(saved_path)
    return os.path.join(saved_path, info_str + '.pth')


def get_saved_path(model_name, info_str, dataset):
    saved_path = model_name + '/saved/' + dataset + '/' + info_str
    ensure_dir(saved_path)
    return saved_path


def get_result_file(model_name, dataset):
    result_path = model_name + '/result/' + dataset
    ensure_dir(result_path)
    return os.path.join(result_path, 'result.txt')


def get_already_paras(result_file):
    already_paras = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as fp:
            lines = fp.readlines()
            for i in range(len(lines)):
                if i % 5 == 0:
                    already_paras.add(lines[i].strip())
    return already_paras
