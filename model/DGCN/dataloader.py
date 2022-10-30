# -*- coding: utf-8 -*-
# @Time   : 2020/8/16 20:19
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : dataloader.py

import torch
import collections
import numpy as np
import networkx as nx
from base.base_dataloader import BaseDataLoader


class DGCNDataLoader(BaseDataLoader):
    def __init__(self, data_path, args=None):
        super(DGCNDataLoader, self).__init__(data_path, args)
        self.graph = self.build_graph()
        self.neib_sampler = NeibSampler(self.graph, args.neighbor_size)
        self.user_nodes = torch.from_numpy(np.arange(self.n_users))
        self.item_nodes = torch.from_numpy(np.arange(self.n_items))

    def build_graph(self):
        graph = collections.defaultdict(list)
        for i in range(len(self.train_user_list)):
            node1 = self.train_user_list[i]
            node2 = self.train_item_list[i] + self.n_users
            graph[node1].append(node2)
            graph[node2].append(node1)
        graph = nx.from_dict_of_lists(graph)
        assert min(graph.nodes()) == 0
        for u in range(self.n_users + self.n_items):
            graph.add_node(u)
        assert graph.number_of_nodes() == self.n_users + self.n_items
        assert not graph.is_directed()
        return graph

    def get_neighbors(self):
        return torch.from_numpy(self.neib_sampler.sample())


class NeibSampler:
    def __init__(self, graph, nb_size):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        nb = {}
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = nb_v
            else:
                popkids.append(v)
        self.n, self.nb_size = n, nb_size
        self.graph, self.nb, self.popkids = graph, nb, popkids

    def sample(self):
        sample_nb = {}
        for i, v in enumerate(self.popkids):
            sample_nb[v] = np.random.choice(sorted(self.graph.neighbors(v)), self.nb_size)
        all_nb = {**self.nb, **sample_nb}
        nb_list = []
        for v in range(self.n):
            nb_list.append(list(all_nb[v]))
        return np.array(nb_list)
