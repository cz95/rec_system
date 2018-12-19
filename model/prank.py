#!/usr/bin/Python
# -*- coding: utf-8 -*-
import pickle
import os
import pandas as pd
from tqdm import tqdm


class Graph(object):
    graph_path = './data/prank/prank.graph'

    @classmethod
    def pre_process(cls, data):
        if not os.path.exists('./data/prank/'):
            os.mkdir('./data/prank/')
        if os.path.exists(cls.graph_path):
            return
        cls.data = data
        cls._gen_graph()
        cls.save()

    @classmethod
    def _gen_user_graph(cls, user_id):
        user_data = cls.data[cls.data['userId'] == user_id][
            ['movieId', 'rating']]
        item_data = user_data.set_index("movieId").to_dict()['rating']
        item_ids = user_data['movieId'].unique()
        graph_dict = {'item_{}'.format(item_id): (item_data[item_id] / 5.0) for
                      item_id in item_ids}
        return graph_dict

    @classmethod
    def _gen_item_graph(cls, item_id):
        user_ids = cls.data[cls.data['movieId'] == item_id]['userId'].unique()
        graph_dict = {'user_{}'.format(user_id): 1 for user_id in user_ids}
        return graph_dict

    @classmethod
    def _gen_graph(cls):
        """
        产生二元图，节点为用户和电影，边为用户对电影的打分
        :return:
        """
        user_ids = cls.data['userId'].unique()
        item_ids = cls.data['movieId'].unique()
        cls.graph = {'user_{}'.format(user_id): cls._gen_user_graph(user_id) for
                     user_id in user_ids}
        for item_id in item_ids:
            cls.graph['item_{}'.format(item_id)] = cls._gen_item_graph(item_id)

    @classmethod
    def save(cls):
        f = open(cls.graph_path, 'wb')
        pickle.dump(cls.graph, f)
        f.close()

    @classmethod
    def load(cls):
        f = open(cls.graph_path, 'rb')
        graph = pickle.load(f)
        f.close()
        return graph


class PersonalRank(object):
    def __init__(self, file_path):
        self.path = file_path
        self.alpha = 0.6
        self.iter_n = 30
        self._init_model()

    def _init_model(self):
        self.data = pd.read_csv(self.path)
        Graph.pre_process(self.data)
        self.graph = Graph.load()
        self.params = {k: 0 for k in self.graph.keys()}

    def train(self, user_id):
        self.params['user_{}'.format(user_id)] = 1
        # for _ in tqdm(range(self.iter_n)):
        for _ in range(self.iter_n):
            temp = {k: 0 for k in self.graph.keys()}
            for node, edge in self.graph.items():
                for next_node, rating in edge.items():
                    temp[next_node] += self.alpha * rating * self.params[
                        node] / len(edge)
            temp['user_{}'.format(user_id)] += 1 - self.alpha
            self.params = temp
        self.params = sorted(self.params.items(), key=lambda x: x[1],
                             reverse=True)
        self.save(user_id)

    def predict(self, user_id, top_n=10):
        if not os.path.exists('./data/prank/prank_{}.model'.format(user_id)):
            self.train(user_id)
        self.load(user_id)
        item_ids = ['item_{}'.format(item_id) for item_id in
                    self.data[self.data['userId'] == user_id][
                        'movieId'].unique()]
        candi_items = [(key, value) for key, value in self.params if
                       key not in item_ids and 'user' not in key]
        return candi_items[:top_n]

    def save(self, user_id):
        f = open('./data/prank/prank_{}.model'.format(user_id), 'wb')
        pickle.dump(self.params, f)
        f.close()

    def load(self, user_id):
        f = open('./data/prank/prank_{}.model'.format(user_id), 'rb')
        self.params = pickle.load(f)
        f.close()


if __name__ == '__main__':
    file_path = './data/ml-latest-small/ratings.csv'
    prank = PersonalRank(file_path)
    print(prank.predict(1))
