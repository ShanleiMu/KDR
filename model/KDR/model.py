# -*- coding: utf-8 -*-
# @Time   : 2020/8/26 12:10
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : model_old.py


import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import xavier_normal_


class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)


class AggerateLayer(nn.Module):
    def __init__(self, ncaps, hidden_size):
        super(AggerateLayer, self).__init__()
        self.ncaps, self.hidden_size = ncaps, hidden_size
        self._cache_zero_d = torch.zeros(1, ncaps, hidden_size)
        self.linear = nn.Linear(ncaps * hidden_size, ncaps * hidden_size)
        xavier_normal_(self.linear.weight)

    def forward(self, x, neighbors_node, neighbors_relation):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
        n, m = x.size(0), neighbors_node.size(0) // x.size(0)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors_node]
        r = torch.unsqueeze(neighbors_relation, dim=1)
        r = torch.zeros(n * m, self.ncaps).to(dev).scatter_(1, r, 1)
        r = torch.unsqueeze(r, dim=2).expand(-1, -1, self.hidden_size)
        z = (z * r).view(n, m, self.ncaps, self.hidden_size).sum(dim=1)
        x = z + x
        x = x.view(n, self.ncaps * self.hidden_size)
        x = fn.relu(self.linear(x))
        x = x.view(n, self.ncaps, self.hidden_size)

        return x


class MyEmbeddingLayer(nn.Module):
    def __init__(self, n_users, n_items, n_attributes, ncaps, hidden_size):
        super(MyEmbeddingLayer, self).__init__()
        self.user_emb_matrix = nn.Embedding(n_users, ncaps * hidden_size)
        self.item_emb_matrix = nn.Embedding(n_items, ncaps * hidden_size)
        self.attribute_emb_matrix = nn.Embedding(n_attributes, hidden_size)
        xavier_normal_(self.user_emb_matrix.weight)
        xavier_normal_(self.item_emb_matrix.weight)
        xavier_normal_(self.attribute_emb_matrix.weight)

    def forward(self, user_nodes, item_nodes, attribute_nodes):
        user_emb = self.user_emb_matrix(user_nodes)
        item_emb = self.item_emb_matrix(item_nodes)
        attribute_emb = self.attribute_emb_matrix(attribute_nodes)
        return user_emb, item_emb, attribute_emb


class KDR(nn.Module):
    def __init__(self, args, dataset, device):
        super(KDR, self).__init__()

        self.n_users, self.n_items, self.n_attributes = dataset.n_users, dataset.n_items, dataset.n_attributes
        self.n_entities = self.n_items + self.n_attributes
        self.rs_neighbors = dataset.get_neighbors().to(device)
        kg_neighbors_node, kg_neighbors_relation = dataset.get_kg_neighbors()
        self.kg_neighbors_node = kg_neighbors_node.to(device)
        self.kg_neighbors_relation = kg_neighbors_relation.to(device)

        self.ncaps = args.ncaps
        self.hidden_size = args.embedding_size // args.ncaps
        self.rep_dim = args.embedding_size
        self.dropout = args.dropout
        self.kg_weight, self.mim_weight, self.reg_weight = args.kg_weight, args.mim_weight, args.reg_weight

        self.embedding_layer = MyEmbeddingLayer(self.n_users, self.n_items, self.n_attributes, self.ncaps, self.hidden_size)

        # DisenGCN U-I Graph
        self.rout_iteration = args.rs_rout_iteration
        self.rs_nlayer = args.rs_nlayer
        rs_layers = []
        for i in range(self.rs_nlayer):
            conv = RoutingLayer(self.rep_dim, self.ncaps)
            self.add_module('rs_layer_%d' % i, conv)
            rs_layers.append(conv)
        self.rs_layers = rs_layers

        # KG Graph
        kg_layers = []
        self.kg_nlayer = args.kg_nlayer
        for i in range(self.kg_nlayer):
            conv = AggerateLayer(self.ncaps, self.hidden_size)
            self.add_module('kg_layer_%d' % i, conv)
            kg_layers.append(conv)
        self.kg_layers = kg_layers

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def get_reg_loss(self):
        reg_loss = None
        for W in self.parameters():
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        reg_loss = self.reg_weight * reg_loss
        return reg_loss

    def get_infonce_loss(self, target_item, rs_item_emb, kg_item_emb):
        mim_loss = None
        rs_item_emb = rs_item_emb.view(-1, self.ncaps, self.hidden_size)
        kg_item_emb = kg_item_emb.view(-1, self.ncaps, self.hidden_size)

        rs_item_emb = rs_item_emb[target_item]
        kg_item_emb = kg_item_emb[target_item]  # (batch, ncaps, hidden_size)
        for i in range(self.ncaps):
            x = rs_item_emb[:, i, :].unsqueeze(2)     # (batch, hidden_size, 1)
            f_value = torch.bmm(kg_item_emb, x).squeeze()     # (batch, ncaps)
            exp_f_value = torch.exp(f_value)
            loss_i = (f_value[:, i] - torch.log(exp_f_value.sum(dim=1) - exp_f_value[:, i])).mean()
            if i == 0:
                mim_loss = loss_i
            else:
                mim_loss += loss_i
        mim_loss = - mim_loss / self.ncaps
        return mim_loss

    def get_js_loss(self, target_item, rs_item_emb, kg_item_emb):
        mim_loss = None
        rs_item_emb = rs_item_emb.view(-1, self.ncaps, self.hidden_size)
        kg_item_emb = kg_item_emb.view(-1, self.ncaps, self.hidden_size)

        rs_item_emb = rs_item_emb[target_item]
        kg_item_emb = kg_item_emb[target_item]  # (batch, ncaps, hidden_size)
        for i in range(self.ncaps):
            x = rs_item_emb[:, i, :].unsqueeze(2)     # (batch, hidden_size, 1)
            f_value = torch.bmm(kg_item_emb, x).squeeze()  # (batch, ncaps)
            softplus_f_value = fn.softplus(f_value)            # (batch, ncaps)
            e_n = (softplus_f_value.sum(dim=1) - softplus_f_value[:, i]) / (self.ncaps - 1)     # (batch)
            e_p = - fn.softplus(- f_value[:, i])    # (batch)
            loss_i = (e_n - e_p).mean()
            if i == 0:
                mim_loss = loss_i
            else:
                mim_loss += loss_i
        mim_loss = mim_loss / self.ncaps
        mim_loss = self.mim_weight * mim_loss
        return mim_loss

    def get_kg_loss(self, head_e, tail_e, relation):
        relation_predicted = torch.sum(head_e * tail_e, dim=2)
        kg_predict_loss = fn.cross_entropy(relation_predicted, relation)
        kg_predict_loss = self.kg_weight * kg_predict_loss
        return kg_predict_loss

    def get_bpr_loss(self, user_e, pos_item_e, neg_item_e):
        u_pos_item = torch.mul(user_e, pos_item_e).sum(dim=1)
        u_neg_item = torch.mul(user_e, neg_item_e).sum(dim=1)
        bpr_loss = torch.log(1e-10 + torch.sigmoid(u_pos_item - u_neg_item)).mean()
        return - bpr_loss

    def calculate_loss(self, u, pos_item, neg_item, head, tail, relation):
        rs_neighbors = self.rs_neighbors.view(-1)
        kg_neighbors_node = self.kg_neighbors_node.view(-1)
        kg_neighbors_relation = self.kg_neighbors_relation.view(-1)

        user_emb = self.embedding_layer.user_emb_matrix.weight
        item_emb = self.embedding_layer.item_emb_matrix.weight
        attribute_emb = self.embedding_layer.attribute_emb_matrix.weight

        ui_emb = torch.cat([user_emb, item_emb], dim=0)   # (n_users + n_items, rep_dim)
        for conv in self.rs_layers:
            ui_emb = self._dropout(conv(ui_emb, rs_neighbors, self.rout_iteration))
        user_emb, rs_item_emb = ui_emb[:self.n_users], ui_emb[self.n_users:]

        item_emb = item_emb.view(self.n_items, self.ncaps, self.hidden_size)
        attribute_emb = torch.unsqueeze(attribute_emb, 1)
        attribute_emb = attribute_emb.expand(-1, self.ncaps, -1)
        ie_emb = torch.cat([item_emb, attribute_emb], dim=0)
        for conv in self.kg_layers:
            ie_emb = self._dropout(conv(ie_emb, kg_neighbors_node, kg_neighbors_relation))
        ie_emb = ie_emb.view(self.n_entities, self.rep_dim)
        kg_item_emb, attribute_emb = ie_emb[:self.n_items], ie_emb[self.n_items:]

        head_e = ie_emb[head].view(-1, self.ncaps, self.hidden_size)
        tail_e = ie_emb[tail].view(-1, self.ncaps, self.hidden_size)
        kg_predict_loss = self.get_kg_loss(head_e, tail_e, relation)

        mim_loss = self.get_js_loss(pos_item, rs_item_emb, kg_item_emb)

        item_emb = rs_item_emb + kg_item_emb
        user_e = ui_emb[u]
        pos_item_e = item_emb[pos_item]
        neg_item_e = item_emb[neg_item]
        bpr_loss = self.get_bpr_loss(user_e, pos_item_e, neg_item_e)

        reg_loss = self.get_reg_loss()

        return bpr_loss, kg_predict_loss, mim_loss, reg_loss

    def get_ui_embeddings(self):
        rs_neighbors = self.rs_neighbors.view(-1)
        kg_neighbors_node = self.kg_neighbors_node.view(-1)
        kg_neighbors_relation = self.kg_neighbors_relation.view(-1)

        user_emb = self.embedding_layer.user_emb_matrix.weight
        item_emb = self.embedding_layer.item_emb_matrix.weight
        attribute_emb = self.embedding_layer.attribute_emb_matrix.weight

        ui_emb = torch.cat([user_emb, item_emb], dim=0)   # (n_users + n_items, rep_dim)
        for conv in self.rs_layers:
            ui_emb = self._dropout(conv(ui_emb, rs_neighbors, self.rout_iteration))
        user_emb, rs_item_emb = ui_emb[:self.n_users], ui_emb[self.n_users:]

        item_emb = item_emb.view(self.n_items, self.ncaps, self.hidden_size)
        attribute_emb = torch.unsqueeze(attribute_emb, 1)
        attribute_emb = attribute_emb.expand(-1, self.ncaps, -1)
        ie_emb = torch.cat([item_emb, attribute_emb], dim=0)
        for conv in self.kg_layers:
            ie_emb = self._dropout(conv(ie_emb, kg_neighbors_node, kg_neighbors_relation))
        ie_emb = ie_emb.view(self.n_entities, self.rep_dim)
        kg_item_emb, attribute_emb = ie_emb[:self.n_items], ie_emb[self.n_items:]

        item_emb = rs_item_emb + kg_item_emb
        return user_emb, item_emb

    def fast_predict(self, user, candidate_items, user_emb, item_emb):
        u_e = user_emb[user]
        item_e = item_emb[candidate_items]
        score = torch.matmul(item_e, u_e.transpose(0, 1))
        return score.view(-1)
