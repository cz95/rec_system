#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import torch
import pickle
import pandas as pd
import torch.nn.functional as F
from torch import nn


class FastText(nn.Module):
    def __init__(self, word_size, emb_dim, n_classes):
        super(FastText, self).__init__()
        self.bon_embed = nn.Embedding(word_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, n_classes)
        self._init_linears()
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x, x_len):
        embed = self.bon_embed(x)  # 增加了一个维度emb_dim
        embed = torch.sum(embed, 1).squeeze(1)  # 去除第二维度
        batch_size = embed.size(0)
        x_len = x_len.float().unsqueeze(1)
        x_len = x_len.expand(batch_size, self.config.embedding_dim)
        embed /= x_len
        embed = F.dropout(embed, p=0.5, training=self.training)
        embed = self.bn(embed)
        out = self.fc(embed)
        return F.log_softmax(out, dim=1)

    def _init_linears(self):
        nn.init.xavier_normal_(self.fc.weight, gain=1)  # 一种初始化方法，服从高斯分布
        nn.init.uniform_(self.fc.bias)


class AGData(object):
    def __init__(self, data_path, n_classes):
        self.n_classes = n_classes
        self.max_len = 0
        self.int_to_gram = dict()
        self.gram_to_int = dict()
        # self.int_to_gram[0] = 'PAD'
        # self.int_to_gram[1] = 'UNK'
        # self.gram_to_int['PAD'] = 0
        # self.gram_to_int['UNK'] = 1
        # self.train_data, self.valid_data, self.test_data = self._load_csv(
        #     data_path)
        self._load_csv(data_path)

    def _load_csv(self, data_path):
        columns = ['class', 'title', 'dec']
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')
        train_data = pd.read_csv(train_path, ',', header=None, names=columns)
        test_data = pd.read_csv(test_path, ',', header=None, names=columns)
        train_data['title'] = train_data['title'].str.cat(train_data['dec'])
        test_data['title'] = test_data['title'].str.cat(test_data['dec'])


class Classify(object):
    def __init__(self, data):
        pass


if __name__ == '__main__':
    data_path = './data/ag_news/'
    n_classes = 4
    n_gram = 2
    data = AGData(data_path, n_classes)
