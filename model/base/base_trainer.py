# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 15:45
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : base_trainer.py


import torch
import torch.optim as optim
import numpy as np


from time import time
from base.utils import early_stopping
from base.base_evaluator import BaseEvaluator


class BaseTrainer(object):
    def __init__(self, args, dataset, model, device, saved_file):
        super(BaseTrainer).__init__()
        self.dataset = dataset
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.evaluator = BaseEvaluator()
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.eval_step = args.eval_step
        self.stopping_step = args.stopping_step
        self.device = device

        self.saved_file = saved_file

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None

    def _train_epoch(self, train_data):
        self.model.train()
        start = 0
        total_loss = 0.
        np.random.shuffle(train_data)
        n_train = self.dataset.n_train
        while start + self.batch_size <= n_train:
            batch_data = train_data[start:start + self.batch_size, ]
            users = torch.from_numpy(batch_data[:, 0]).to(self.device)
            pos_items = torch.from_numpy(batch_data[:, 1]).to(self.device)
            neg_items = torch.from_numpy(batch_data[:, 2]).to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(users, pos_items, neg_items)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            start += self.batch_size
        return total_loss

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = valid_result[2][2]
        return valid_score, valid_result

    def _save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_file)

    def fit(self, valid_data=None):
        for epoch_idx in range(self.start_epoch, self.epochs + 1):
            # train
            training_start_time = time()
            train_data = self.dataset.build_train_data_with_sample()
            train_loss = self._train_epoch(train_data)
            training_end_time = time()
            train_loss_output = "epoch %d training [time: %.2fs, train loss: %.4f]" % \
                                (epoch_idx, training_end_time - training_start_time, train_loss)
            print(train_loss_output)

            # valid
            if (epoch_idx + 1) % self.eval_step == 0 and valid_data:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                precision = [str('%.4f' % p) for p in valid_result[0]]
                recall = [str('%.4f' % r) for r in valid_result[1]]
                ndcg = [str('%.4f' % n) for n in valid_result[2]]
                valid_result_output = 'valid precision:\t' + '\t'.join(precision) + '\n' \
                                      '   valid recall:\t' + '\t'.join(recall) + '\n' \
                                      '     valid ndcg:\t' + '\t'.join(ndcg)
                print(valid_score_output)
                print(valid_result_output)
                if update_flag:
                    self._save_checkpoint()
                    self.best_valid_result = valid_result
                if stop_flag:
                    break
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            print(message_output)

        self.model.eval()
        index_list = []
        for ins in eval_data:
            users = torch.from_numpy(np.array([ins[0]])).to(self.device)
            items = torch.from_numpy(np.array(ins[1])).to(self.device)
            scores = self.model.fast_predict(users, items)
            topk_index = self.evaluator.evaluate(scores)
            index_list.append(topk_index)
        precision, recall, ndcg = self.evaluator.collect(index_list, eval_data)

        return precision, recall, ndcg
