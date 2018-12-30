#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import re
import spacy
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from sklearn.model_selection import train_test_split

"""
Dataset: AG's News Topic Classification Dataset（Version 3, Updated 09/09/2015）
Uncompleted: Hierarchical softmax
"""


class FastText(nn.Module):
    def __init__(self, word_size, emb_dim=300, n_classes=4):
        super(FastText, self).__init__()
        self.emb_dim = emb_dim
        self.bon_embed = nn.Embedding(word_size, emb_dim, padding_idx=0)
        linear_hidden_size = 128
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, n_classes)
        )

    def forward(self, x, x_len):
        """
        Calculate the loss of forward propagation
        :param x:
        :param x_len:
        :return:
        """
        embed = self.bon_embed(x)  # Add a dimension: emb_dim
        embed = torch.sum(embed, 1).squeeze(1)  # Remove the second dimension
        batch_size = embed.size(0)
        x_len = x_len.float().unsqueeze(1)
        x_len = x_len.expand(batch_size, self.emb_dim)
        embed /= x_len
        embed = F.dropout(embed, p=0.5, training=self.training)
        out = self.fc(embed)
        return F.log_softmax(out, dim=1)

    def save_embed(self, int_to_gram):
        """
        save the word embedding
        :param int_to_gram: a dictionary
        :return:
        """
        embed = self.bon_embed.weight.data.numpy()
        with open(self.embedding_dir, 'w', encoding='utf-8') as f:
            for id, w in int_to_gram.items():
                e = embed[id]
                e = ' '.join(map(lambda x: str(x), e))
                f.write('%s %s\n' % (str(w), e))


class AGData(object):
    def __init__(self, data_path, n_classes):
        self.data_path = data_path
        self.n_classes = n_classes
        self.int_to_gram = dict()
        self.gram_to_int = dict()
        self.train_pickle = os.path.join(self.data_path, 'train_pickle.arr')
        self.test_pickle = os.path.join(self.data_path, 'test_pickle.arr')
        if not os.path.exists(self.train_pickle):
            self.nlp = spacy.load('en_core_web_lg',
                                  disable=['parser', 'tagger', 'ner'])
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
            self.int_to_gram[0] = 'PAD'
            self.gram_to_int['PAD'] = 0
            self.int_to_gram[1] = 'UNK'
            self.gram_to_int['UNK'] = 1
            self.train_data_arr, self.test_data_arr = self._load_csv(data_path)
            self._save()

    def _add_ngram(self, bow, n=3):
        """
        add word-n_gram in word bag
        :param bow:
        :param n:
        :return:
        """
        bow_ngram = bow
        for word in bow:
            words = '<' + word + '>'
            n_grams = [''.join(words[i: i + n]) for i in
                       range(len(words) - (n - 1))]
            bow_ngram = bow_ngram + n_grams
        return bow_ngram

    def _get_process(self, content):
        """
        preprocessing include handling symbols、vectoring text
        :param content:
        :return:
        """
        content = content.replace('\\', ' ')
        content = re.sub("[+.!/_,$%^*()+\"']", '', content)
        content = self.nlp(content)
        bow = [token.text for token in content]
        bow_ngram = self._add_ngram(bow)
        if len(bow_ngram) > self.max_len:
            bow_ngram = bow_ngram[:self.max_len]
        for ng in bow_ngram:
            idx = self.gram_to_int.get(ng)
            if idx is None:
                idx = len(self.gram_to_int)
                self.gram_to_int[ng] = idx
                self.int_to_gram[idx] = ng
        int_bow_ngram = [self.gram_to_int[ng] if ng in self.gram_to_int else
                         self.gram_to_int['UNK'] for ng in bow_ngram]
        return int_bow_ngram

    def _load_csv(self, data_path):
        """
        load csv
        :param data_path:
        :return:
        """
        columns = ['class', 'title', 'dec']
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')
        train_data = pd.read_csv(train_path, ',', header=None, names=columns)
        test_data = pd.read_csv(test_path, ',', header=None, names=columns)

        train_data['title'] = train_data['title'].str.cat(
            ' ' + train_data['dec'])
        test_data['title'] = test_data['title'].str.cat(' ' + test_data['dec'])

        train_data['title'] = train_data['title'].apply(self._get_process)
        test_data['title'] = test_data['title'].apply(self._get_process)

        train_data = train_data.drop(['dec'], axis=1)
        test_data = test_data.drop(['dec'], axis=1)

        train_data_arr = train_data.values
        test_data_arr = test_data.values

        return train_data_arr, test_data_arr

    def load(self):
        """
        load model
        :return:
        """
        with open(self.train_pickle, 'rb') as f:
            train_data_arr = pickle.load(f)
        with open(self.test_pickle, 'rb') as f:
            test_data_arr = pickle.load(f)
        with open(os.path.join(self.data_path, 'int_to_gram.dict'), 'rb') as f:
            int_to_gram = pickle.load(f)
        with open(os.path.join(self.data_path, 'gram_to_int.dict'), 'rb') as f:
            gram_to_int = pickle.load(f)
        return train_data_arr, test_data_arr, int_to_gram, gram_to_int

    def _save(self):
        """
        save model
        :return:
        """
        with open(self.train_pickle, 'wb') as f:
            pickle.dump(self.train_data_arr, f)
        with open(self.test_pickle, 'wb') as f:
            pickle.dump(self.test_data_arr, f)
        with open(os.path.join(self.data_path, 'int_to_gram.dict'), 'wb') as f:
            pickle.dump(self.int_to_gram, f)
        with open(os.path.join(self.data_path, 'gram_to_int.dict'), 'wb') as f:
            pickle.dump(self.gram_to_int, f)


class Classify(object):
    def __init__(self, data_path, epochs=1, lr=0.1, n_classes=4, batch_size=32,
                 max_len=800):
        self.data = AGData(data_path, n_classes)
        self.embed_path = os.path.join(data_path, 'embed.dict')
        self.model_save_path = os.path.join(data_path, 'fasttext.pkl')
        self.train_data, self.test_data, self.int_to_gram, self.gram_to_int = self.data.load()
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.model = FastText(len(self.int_to_gram))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                    amsgrad=True)

    def _get_batches(self, data):
        n_batches = len(data) // self.batch_size
        data_trim = data[: n_batches * self.batch_size]  # 防止出现不够用
        for idx in range(0, len(data_trim), self.batch_size):
            batch = data_trim[idx: idx + self.batch_size]
            x, y, x_len = [], [], []
            for cls, data in batch:
                z = len(data)
                y.append(cls - 1)
                if z < self.max_len:
                    data = data + [self.gram_to_int['PAD']] * (self.max_len - z)
                x.append(data)
                x_len.append(z)
            yield x, y, x_len

    def train(self):
        self.model.train()
        for i in range(self.epochs):
            batches = self._get_batches(self.train_data)
            for x, y, x_len in batches:
                y = Variable(torch.LongTensor(y))
                x = Variable(torch.LongTensor(x))
                x_len = Variable(torch.LongTensor(x_len))
                self.optimizer.zero_grad()
                output = self.model.forward(x, x_len)
                loss = F.nll_loss(output, y)
                loss.backward()
                self.optimizer.step()
                pre = output.max(1, keepdim=True)[1]
                cor = pre.eq(y.view_as(pre)).sum().item()
                print(
                    'Train Epoch {}, acc={},loss={}'.format(i, cor / len(x_len),
                                                            loss))
                print(loss)
        self._save_model(self.model)

    def _save_model(self, model):
        # save a model
        model_dict = dict()
        model_dict['state_dict'] = model.state_dict()
        torch.save(model_dict, self.model_save_path)
        self.model.save_embed(self.embed_path, self.int_to_gram)


if __name__ == '__main__':
    data_path = './data/ag_news/'  # folder need include train.csv and test.csv
    cla = Classify(data_path)
    cla.train()
