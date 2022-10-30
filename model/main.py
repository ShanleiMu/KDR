# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 11:11
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : main.py

import os
import argparse
import importlib

from base.init_parser import init_parser
from base.base_trainer import BaseTrainer
from base.utils import set_rng_seed, \
    set_device, get_saved_file, get_saved_path, get_result_file, get_already_paras


def import_parser(my_parser, model_name, params=None):
    parser_module = importlib.import_module(model_name + '.parser')
    _parse_args = getattr(parser_module, 'parse_args')
    args = _parse_args(my_parser, params)
    return args


def import_info_str(model_name, args):
    parser_module = importlib.import_module(model_name + '.parser')
    _build_info_str = getattr(parser_module, 'build_info_str')
    info_str = _build_info_str(args)
    return info_str


def import_dataloader(model_name, data_path, args):
    dataloader_module = importlib.import_module(model_name + '.dataloader')
    _DataLoader = getattr(dataloader_module, model_name + 'DataLoader')
    dataset = _DataLoader(data_path, args)
    return dataset


def import_model(model_name):
    model_module = importlib.import_module(model_name + '.model')
    _Model = getattr(model_module, model_name)
    return _Model


def import_trainer(model_name):
    try:
        trainer_module = importlib.import_module(model_name + '.trainer')
        _Trainer = getattr(trainer_module, 'Trainer')
    except ModuleNotFoundError:
        _Trainer = BaseTrainer
    return _Trainer


def whole_process(model_name, params=None):
    args = import_parser(init_parser(), model_name, params)
    set_rng_seed(args.seed)
    device = set_device(args.gpu_id)

    dataset = import_dataloader(model_name, os.path.join(args.data_path, args.dataset), args)
    model = import_model(model_name)(args, dataset, device).to(device)
    info_str = import_info_str(model_name, args)
    saved_file = get_saved_file(model_name, info_str, args.dataset)
    result_file = get_result_file(model_name, args.dataset)
    already_paras = get_already_paras(result_file)
    if info_str.strip() in already_paras:
        print('Already run these parameters')
        return {
            'best_valid_score': 0,
            'best_valid_result': None,
            'test_result': None
        }
    trainer = import_trainer(model_name)(args, dataset, model, device, saved_file)
    print(args.dataset)
    print(dataset)
    print(model)

    valid_data, test_data = dataset.build_valid_and_test_data()

    best_valid_score, best_valid_result = trainer.fit(valid_data)

    test_result = trainer.evaluate(test_data, load_best_model=True)
    precision = [str('%.4f' % p) for p in test_result[0]]
    recall = [str('%.4f' % r) for r in test_result[1]]
    ndcg = [str('%.4f' % n) for n in test_result[2]]
    result_output = 'valid precision:\t' + '\t'.join(precision) + '\n' \
                    '   valid recall:\t' + '\t'.join(recall) + '\n' \
                    '     valid ndcg:\t' + '\t'.join(ndcg)
    print('test result: ')
    print(result_output)
    with open(result_file, 'a') as fp:
        fp.write(info_str + '\n')
        fp.write(result_output + '\n\n')
    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KDR')
    args, _ = parser.parse_known_args()

    whole_process(args.model)
