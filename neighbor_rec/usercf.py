#!/usr/bin/Python
# -*- coding: utf-8 -*-
import pandas as pd
import math
from time import time


class Similar(object):
    @classmethod
    def _cosine_sim(cls, target_movies, other_movies):
        """
        Calculate the cosine similarity
        watch = 1
        unwatch = 0
        :param target_movies:
        :param other_movies:
        :return:
        """
        target_set = set(target_movies['movieId'].unique())
        other_set = set(other_movies['movieId'].unique())
        union_len = len(target_set & other_set)
        if union_len == 0:
            return 0.0
        product = len(target_set) * len(other_set)
        return union_len / math.sqrt(product)

    @classmethod
    def _cosine_sim_score(cls, target_movies, other_movies):
        """
        Calculating cosine similarity, consider score
        :param target_movies:
        :param other_movies:
        :return:
        """
        target_dict = target_movies.set_index("movieId").to_dict()['rating']
        movies_dict = other_movies.set_index("movieId").to_dict()['rating']
        union_movies = set(target_dict.keys()) & set(movies_dict.keys())
        if len(union_movies) == 0:
            return 0.0
        score_1 = 0
        for movie in union_movies:
            score_1 += (target_dict[movie] * movies_dict[movie])
        rating_1 = sum(target_movies['rating'].values ** 2)
        rating_2 = sum(other_movies['rating'].values ** 2)
        score_2 = math.sqrt(rating_1 * rating_2)
        return score_1 / score_2

    @classmethod
    def _iif_sim_(cls):
        """
        John S. Breese等人在1998年的工作中给出，两个用于对冷门物品采取过同样的行为更能说明他们兴趣的相似度
        e.g: 两人都买了《新华字典》，不能说明兴趣相似，但如果两人都买了《数据挖掘导论》，则认为兴趣相似
        In 1998, John s. Breese et al. showed that the similarity of their
        interests was more evident when two people used the same behavior for
        unpopular items
        e.g.:if two people buy 《dictionary》, they may not have the same interest,
             but if two people buy 《introduction to data mining》,
             they are considered to have the same interest.
        :return:
        """

        pass

    @classmethod
    def get_sim(cls, target_movies, other_movies, sim_type=1):
        """
        Provide similarity function
        :param target_movies: target user list
        :param other_movies: Target item user group with rating
        :param sim_type: Other item user group with rating
        :return:
        """
        if sim_type == 1:
            return cls._cosine_sim(target_movies, other_movies)
        else:
            return cls._cosine_sim_score(target_movies, other_movies)


class UserCF(object):
    def __init__(self, data):
        self.data = data

    def _get_top_n_users(self, target_user_id, user_n):
        """
        Calculate similarity between target users and other users based on ratings
        :param target_user_id:
        :param user_n:
        :return:
        """
        target_movies = self.data[self.data['userId'] == target_user_id][
            ['movieId', 'rating']]
        other_users_id = set(self.data['userId'].unique()) - set(
            [target_user_id])
        # A 2-d matrix
        # Each dimension contains the movie_id that the current user has watched
        other_movies = [
            self.data[self.data['userId'] == i][['movieId', 'rating']] for i in
            other_users_id]
        sim_list = [Similar.get_sim(target_movies, movies, self.type) for movies
                    in other_movies]
        sim_list = sorted(zip(other_users_id, sim_list), key=lambda x: x[1],
                          reverse=True)
        return sim_list[:user_n]

    def _get_candidates_items(self, target_user_id):
        """
        Find all movies from the source data that the target user has not seen
        :param target_user_id:
        :return:
        """
        target_user_movies = set(
            self.data[self.data['userId'] == target_user_id]['movieId'])
        candidates_movies = set(
            self.data['movieId'].unique()) - target_user_movies
        return candidates_movies

    def _get_top_m_items(self, top_n_users, candidates_movies, item_n):
        """
        Calculate the movies that top n is interested in in candidate movies
        :param top_n_users:
        :param candidates_movies:
        :param item_n:
        :return:
        """
        top_n_user_data = [self.data[self.data['userId'] == k] for k, _ in
                           top_n_users]
        interest_list = []
        for movie_id in candidates_movies:
            temp = []
            for user_data in top_n_user_data:
                if movie_id in user_data['movieId'].values:
                    temp.append(user_data[user_data['movieId'] == movie_id][
                                    'rating'].values[0] / 5)
                else:
                    temp.append(0)
            interest = sum(
                [top_n_users[i][1] * temp[i] for i in range(len(top_n_users))])
            interest_list.append((movie_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:item_n]

    def calculate(self, target_user_id=1, user_n=20, item_n=10, type=2):
        """
       Use userCF for recommendations
        :param target_user_id:  target user_id
        :param user_n: the number of most similar users
        :param item_n: the number of recommend_items
        :param type:  type 1: only distinguish the movie watch or unwatched,
                           2: considering score
        :return:
        """
        self.type = type
        # Calculate the most similar top N users
        top_n_users = self._get_top_n_users(target_user_id, user_n)
        # Movies candidates for the recommendation system
        candidates_movies = self._get_candidates_items(target_user_id)
        # Top M movies of greatest interest
        top_m_items = self._get_top_m_items(top_n_users, candidates_movies,
                                            item_n)
        return top_m_items


if __name__ == "__main__":
    start = time()
    file_path = './data/ml-latest-small/ratings.csv'
    data = pd.read_csv(file_path)
    user_cf = UserCF(data=data)
    # type = 1 only distinguish the movie watch or unwatched, ignore score
    # type = 2 watched movie with rating
    # type = 3
    print(user_cf.calculate(type=1))
    print('total time is:', time() - start)
