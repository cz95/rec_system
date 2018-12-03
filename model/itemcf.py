#!/usr/bin/Python
# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm


class ItemCF:
    def __init__(self, data):
        self.data = data

    def _cosine_sim(self, target_user, user):
        """
        方法1中：用于计算余弦相似性
        :param target_user:
        :param user:
        :return:
        """
        union_len = len(set(target_user) & set(user))
        if union_len == 0:
            return 0.0
        product = len(target_user) * len(user)
        return union_len / math.sqrt(product)

    def _get_n_items(self, item_id, item_n):
        """
        方法1中：用于计算与item_id最相似的top n项
        :param item_id:
        :param item_n:
        :return:
        """
        cur_user = self.data[self.data['movieId'] == item_id]['userId']
        other_item = set(self.data['movieId'].unique()) - set([item_id])
        # 二维矩阵，每一维包含当前看过此电影id的用户
        other_users = [self.data[self.data['movieId'] == i]['userId'] for i in
                       other_item]
        sim_list = [self._cosine_sim(cur_user, user) for user in other_users]
        sim_list = sorted(zip(other_item, sim_list), key=lambda x: x[1],
                          reverse=True)
        return sim_list[:item_n]

    def _get_top_n_items(self, user_id, item_n, top_n):
        """
        方法1中：给用户推荐TopN项目
        :param user_id:
        :return:
        """
        candi_items = dict()
        user_data = self.data[self.data['userId'] == user_id][['movieId', 'rating']]
        watched_item = user_data.set_index("movieId").to_dict()['rating']
        watched_item_list = user_data['movieId'].unique()
        for item_id in tqdm(watched_item_list):
            top_n_items = self._get_n_items(item_id, item_n=item_n)
            for item in top_n_items:
                if item[0] not in candi_items.keys():
                    candi_items[item[0]] = 0
                candi_items[item[0]] += watched_item[item_id] * item[1]
        # 去掉看过的
        for item in watched_item_list:
            if item in candi_items.keys():
                candi_items.pop(item)
        recom_items = sorted(candi_items.items(), key=lambda x: x[1], reverse=True)
        return recom_items[:top_n]

    def item_similarity(self):
        """
        方法2中：用于计算物品间相似矩阵
        :return:
        """
        item = dict()
        count = dict()
        item_unique = self.data['movieId'].unique()
        item_len = len(item_unique)
        sim_matrix = np.zeros([item_len, item_len])
        user_list = self.data['userId'].unique()
        self.item_order = {item_unique[i]: i for i in range(item_len)}
        self.order_item = {i: item_unique[i] for i in range(item_len)}
        for userId in tqdm(user_list):
            item_list = self.data[self.data['userId'] == userId]['movieId'].unique()
            for i in item_list:
                if i not in count:
                    count[i] = 0
                count[i] += 1
                for j in item_list:
                    if i != j:
                        if (i, j) not in item:
                            item[i, j] = 0
                        item[i, j] += 1
        for i, j in tqdm(item.keys()):
            sim_matrix[self.item_order[i]][self.item_order[j]] = \
                item[i, j] / math.sqrt(count[i] * count[j])
        return sim_matrix

    def get_item_n(self, item_id, sim_matrix, item_n):
        """
        方法2中：基于相似矩阵计算与item_id最相似的top n项
        :param item_id:
        :param sim_matrix:
        :param item_n:
        :return:
        """
        candi_item = {}
        for j in range(sim_matrix.shape[0]):
            if sim_matrix[self.item_order[item_id]][j] != 0:
                item = self.order_item[j]
                if item not in candi_item.keys():
                    candi_item[item] = 0
                    candi_item[item] = sim_matrix[self.item_order[item_id]][j]
        sim_items = sorted(candi_item.items(), key=lambda x: x[1], reverse=True)
        return sim_items[:item_n]

    def get_top_n(self, item_similar, user_id, item_n, top_n):
        """
        方法2中：给用户推荐top_n项目
        :param item_similar:
        :param user_id:
        :param item_n:
        :param top_n:
        :return:
        """
        candi_items = dict()
        user_data = self.data[self.data['userId'] == user_id][['movieId', 'rating']]
        watched_item_data = user_data.set_index("movieId").to_dict()['rating']
        watched_item_list = user_data['movieId'].unique()
        for movie_id in tqdm(watched_item_list):
            sim_items = self.get_item_n(movie_id, item_similar, item_n)
            for data in sim_items:
                if data[0] not in candi_items.keys():
                    candi_items[data[0]] = 0
                candi_items[data[0]] += data[1] * watched_item_data[movie_id]
        # 去掉看过的
        for item in watched_item_list:
            if item in candi_items.keys():
                candi_items.pop(item)
        recom_items = sorted(candi_items.items(), key=lambda x: x[1],
                              reverse=True)
        return recom_items[:top_n]

    def calculate_a(self, user_id=1, item_n=20, top_n=10):
        """
        用方法1计算，用时很长= =
        :param user_id:
        :param item_n:
        :param top_n:
        :return:
        """
        # 最相似的top N个items
        top_n_item = self._get_top_n_items(user_id=user_id,
                                           item_n=item_n, top_n=top_n)
        return top_n_item

    def calculate_b(self, user_id=1, item_n=20, top_n=10):
        """
        用方法2计算，用时很短
        :param user_id:
        :param item_n:
        :param top_n:
        :return:
        """
        item_similar = self.item_similarity()
        top_n_item = self.get_top_n(item_similar, user_id=user_id,
                                    item_n=item_n, top_n=top_n)
        return top_n_item


if __name__ == "__main__":
    file_path = '../data/ml-latest-small/ratings.csv'
    data = pd.read_csv(file_path)
    user_cf = ItemCF(data=data)
    print(user_cf.calculate_a())
    print(user_cf.calculate_a())
