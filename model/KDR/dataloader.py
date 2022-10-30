# -*- coding: utf-8 -*-
# @Time   : 2020/8/26 12:08
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : dataloader.py

import os
import torch
import random
import numpy as np
from DGCN.dataloader import DGCNDataLoader


class KDRDataLoader(DGCNDataLoader):
    def __init__(self, data_path, args=None):
        super(KDRDataLoader, self).__init__(data_path, args)
        self.ncaps = args.ncaps
        self.kg_file = os.path.join(data_path, 'data.kg')
        self.kg = self.load_kg()
        self.kg_graph = self.build_kg_graph()
        self.exist_heads = set(self.kg_graph.keys())
        self.n_entities, self.n_relations, self.n_triples = self.get_n_entities_n_relations_n_triples()
        self.n_attributes = self.n_entities - self.n_items
        self.attribute_nodes = torch.from_numpy(np.arange(self.n_attributes))

        self.kg_neib_sampler = KGNeibSampler(self.kg_graph, self.n_entities, args.kg_neighbor_size)

    def load_kg(self):
        kg_np = []
        with open(self.kg_file, 'r') as fp:
            for line in fp:
                h, r, t = line.strip().split('\t')
                if int(r) < self.ncaps:
                    kg_np.append([int(h), int(r), int(t)])
        return np.array(kg_np)

    def build_kg_graph(self):
        kg_graph = dict()
        for triple in self.kg:
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if head not in kg_graph:
                kg_graph[head] = []
            kg_graph[head].append((relation, tail))
            if tail not in kg_graph:
                kg_graph[tail] = []
            kg_graph[tail].append((relation, head))
        return kg_graph

    def get_n_entities_n_relations_n_triples(self):
        n_entities = max(list(set(self.kg[:, 0]) | set(self.kg[:, 2]))) + 1
        n_relations = len(set(self.kg[:, 1]))
        n_triples = len(self.kg)
        return n_entities, n_relations, n_triples

    def get_kg_neighbors(self):
        kg_neighbors_node, kg_neighbors_relation = self.kg_neib_sampler.sample()
        return torch.from_numpy(kg_neighbors_node), torch.from_numpy(kg_neighbors_relation)

    def generate_kg_batch(self, items, batch_size):

        def sample_triples_for_h(h, num):
            triples = self.kg_graph[h]
            n_triples = len(triples)
            rs, ts = [], []
            while True:
                if len(rs) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_triples, size=1)[0]
                r = triples[pos_id][0]
                t = triples[pos_id][1]
                if r not in rs and t not in ts:
                    rs.append(r)
                    ts.append(t)
            return rs, ts

        e_heads = list(self.exist_heads & set(items))
        if batch_size <= len(e_heads):
            heads = np.random.choice(e_heads, size=batch_size, replace=False)
        else:
            heads = np.random.choice(e_heads, size=batch_size, replace=True)
        relations_batch, tails_batch = [], []
        for h in heads:
            relations, tails = sample_triples_for_h(h, 1)
            relations_batch += relations
            tails_batch += tails
        return heads, np.array(relations_batch), np.array(tails_batch)


class KGNeibSampler:
    def __init__(self, graph, n_nodes, nb_size):
        self.graph = graph
        self.n_nodes = n_nodes
        self.nb_size = nb_size
        nb_v = {}
        nb_r = {}
        popkids = []

        for node in range(n_nodes):
            if node not in graph:
                rs = [0] * nb_size
                vs = [-1] * nb_size
                nb_v[node] = vs
                nb_r[node] = rs
            else:
                vs = [rv[1] for rv in graph[node]]
                rs = [rv[0] for rv in graph[node]]
                if len(vs) <= nb_size:
                    vs.extend([-1] * (nb_size - len(vs)))
                    rs.extend([0] * (nb_size - len(rs)))
                    nb_v[node] = vs
                    nb_r[node] = rs
                else:
                    popkids.append(node)
        self.nb_v, self.nb_r, self.popkids = nb_v, nb_r, popkids

    def sample(self):
        sample_nb_v = {}
        sample_nb_r = {}
        for i, v in enumerate(self.popkids):
            sample_rv = random.sample(self.graph[v], self.nb_size)
            sample_nb_v[v] = [rv[1] for rv in sample_rv]
            sample_nb_r[v] = [rv[0] for rv in sample_rv]
        all_nb_v = {**self.nb_v, **sample_nb_v}
        all_nb_r = {**self.nb_r, **sample_nb_r}
        nb_v_list = []
        nb_r_list = []
        for v in range(self.n_nodes):
            nb_v_list.append(all_nb_v[v])
            nb_r_list.append(all_nb_r[v])

        return np.array(nb_v_list), np.array(nb_r_list)
