#!/usr/bin/Python
# -*- coding: utf-8 -*-

import os
import jieba.posseg as pseg
import networkx as nx
import numpy as np
from operator import itemgetter


class TextRank(object):
    """
    基于TextRank提取关键词
    """

    # 分割句子
    sentence_delimiters = ['.', '?', '!', ';', '？', '！', '。', '；', '……', '…',
                           '\n']

    # 允许的词性
    allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt',
                         'nz', 't', 'v', 'vd', 'vn', 'eng']

    def __init__(self):
        self.stop_words = set()
        self._pro_stop_words()

    def _pro_stop_words(self):
        """
        加入停用词
        :return:
        """
        stop_dir = os.path.join('./data/stop_words')
        content = open(stop_dir, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def _get_sentence(self, text):
        """
        基于文章得到句子
        :param text:
        :return:
        """
        res = [text]
        for sep in self.sentence_delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res

    def _get_segment(self, sentence):
        """
        分词，过滤不要的词性和停用词
        :param sentence:
        :return:
        """
        res = []
        words = pseg.cut(sentence)
        words_filter = [w for w in words if w.flag in self.allow_speech_tags]
        for w, flag in words_filter:
            if len(w.strip()) < 2 or w.lower() in self.stop_words:
                continue
            res.append(w)
        return res

    def combine(self, word_list, window=2):
        """
        构造在window下的词语组合，用来构建词语之间的关系
        :param word_list: 单词列表
        :param window: 窗口大小
        :return: 返回两个词语
        """
        if window < 2: window = 2
        for x in range(1, window):
            if x >= len(word_list):
                break
            word_list2 = word_list[x:]
            res = zip(word_list, word_list2)
            for r in res:
                yield r

    def get_text_rank(self, text, window=2, topk=10,
                      pagerank_config={'alpha': 0.85}):
        """
        实现textrank算法，并返回关键词
        :param text: 文章
        :param window: 滑动窗口
        :param topk: topk个词
        :param pagerank_config: pagerank的阻尼系数 一般设置为0.85
        :return: topk个关键词
        """
        sentences = self._get_sentence(text)
        res = []  # 二维列表
        for sen in sentences:
            res.append(self._get_segment(sen))
        word_index = {}
        index_word = {}
        words_number = 0
        for word_list in res:
            for word in word_list:
                if not word in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1

        graph = np.zeros((words_number, words_number))  # 矩阵
        for word_list in res:
            for w1, w2 in self.combine(word_list, window):
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] = 1.0
                graph[index2][index2] = 1.0

        sorted_words = []
        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **pagerank_config)
        sorted_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)
        for index, score in sorted_scores:
            item = {}
            item[index_word[index]] = score
            sorted_words.append(item)
        return sorted_words[:topk]


if __name__ == "__main__":
    idf_test = './data/idf_test.txt'
    text = open(idf_test, 'rb').read().decode('utf-8')
    tr = TextRank()
    print(tr.get_text_rank(text, window=5))
