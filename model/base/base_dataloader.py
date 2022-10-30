# -*- coding: utf-8 -*-
# @Time   : 2020/8/14 9:47
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : base_dataloader.py

import os
import random
import collections
import numpy as np


class BaseDataLoader(object):
    def __init__(self, data_path, args=None):
        super(BaseDataLoader, self).__init__()
        self.valid_full = args.valid_full
        self.train_file = os.path.join(data_path, 'train.inter')
        self.valid_file = os.path.join(data_path, 'valid.inter')
        self.test_file = os.path.join(data_path, 'test.inter')

        [self.train_interactions, self.valid_interactions, self.test_interactions], dataset_info = \
            self.load_interactions([self.train_file, self.valid_file, self.test_file])
        self.n_users, self.n_items = dataset_info['n_users'], dataset_info['n_items']
        [self.n_train, self.n_valid, self.n_test] = dataset_info['n_interactions']
        self.user2candidates = self.build_user2candidates()
        # self.user2negs = self.build_user2negs()
        self.train_user_list, self.train_item_list, self.pos_len = self.build_train_data()

    @staticmethod
    def load_interactions(file_list):
        r = []
        dataset_info = {}
        n_items, n_users = 0, 0
        n_interactions = []
        for file in file_list:
            idx = 0
            user2poss = collections.defaultdict(list)
            with open(file, 'r') as fp:
                for line in fp:
                    idx += 1
                    user, pos_item = line.strip().split('\t')
                    user, pos_item = int(user), int(pos_item)
                    user2poss[user].append(pos_item)
                    n_users = max(n_users, user)
                    n_items = max(n_items, pos_item)
            r.append(user2poss)
            n_interactions.append(idx)
        dataset_info['n_users'] = n_users + 1
        dataset_info['n_items'] = n_items + 1
        dataset_info['n_interactions'] = n_interactions
        return r, dataset_info

    def build_user2candidates(self):
        user2candidates = dict()
        all_items = set(range(self.n_items))
        all_users = range(self.n_users)
        for user in all_users:
            user2candidates[user] = list(all_items - set(self.train_interactions[user]))
        return user2candidates

    def build_user2negs(self):
        user2negs = dict()
        all_users = range(self.n_users)
        for user in all_users:
            num = len(self.train_interactions[user]) * 5
            n_candidates = len(self.user2candidates[user])
            if num > n_candidates:
                user2negs[user] = list(np.random.choice(self.user2candidates[user], num))
            else:
                user2negs[user] = random.sample(self.user2candidates[user], num)
        return user2negs

    def sample_neg_items(self, pos_len):
        def sample_items(u, num):
            if num > len(self.user2candidates[u]):
                neg_items = list(np.random.choice(self.user2candidates[u], num))
            else:
                neg_items = random.sample(self.user2candidates[u], num)
            return neg_items

        neg_item_list = []
        for user, n in pos_len:
            neg_item_list += sample_items(user, n)
        return neg_item_list

    def build_train_data(self):
        user_list, item_list = [], []
        pos_len = []
        for user in self.train_interactions:
            interactions = self.train_interactions[user]
            n_interactions = len(interactions)
            pos_len.append((user, n_interactions))
            user_list += [user] * n_interactions
            item_list += interactions
        return user_list, item_list, pos_len

    def build_train_data_with_sample(self):
        neg_item_list = self.sample_neg_items(self.pos_len)
        train_data = np.transpose(np.array([self.train_user_list, self.train_item_list, neg_item_list]))
        return train_data

    def build_valid_and_test_data(self):
        valid_data, test_data = [], []
        valid_idx = 0
        for user in self.valid_interactions:
            true_items = self.valid_interactions[user]
            candidate_items = self.user2candidates[user]
            valid_data.append([user, true_items + list(set(candidate_items) - set(true_items)), len(true_items)])
            valid_idx += 1
            if valid_idx >= 2000 and not self.valid_full:
                break
        for user in self.test_interactions:
            true_items = self.test_interactions[user]
            candidate_items = self.user2candidates[user]
            test_data.append([user, true_items + list(set(candidate_items) - set(true_items)), len(true_items)])
        return valid_data, test_data

    def build_test_data_different_sparity_level(self, limits):
        test_data = []
        for user in self.test_interactions:
            n = len(self.train_interactions[user])
            if limits[0] < n <= limits[1]:
                true_items = self.test_interactions[user]
                candidate_items = self.user2candidates[user]
                test_data.append([user, true_items + list(set(candidate_items) - set(true_items)), len(true_items)])
        return test_data

    def __str__(self):
        output_string = '-------- dataset info --------\n'
        output_string += '[n_users, n_items]=[%d, %d]\n' % (self.n_users, self.n_items)
        output_string += '[n_train, n_eval, n_test]=[%d, %d, %d]\n' % (self.n_train, self.n_valid, self.n_test)
        return output_string
