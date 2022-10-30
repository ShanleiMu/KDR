# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 12:08
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : init_parser.py


import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--data_path', type=str, default='../data/')

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--stopping_step', type=int, default=5)
    parser.add_argument('--valid_full', type=int, default=1)

    parser.add_argument('--seed', type=int, default=555)
    parser.add_argument('--gpu_id', type=int, default=0)

    return parser
