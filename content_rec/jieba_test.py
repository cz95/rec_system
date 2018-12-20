#!/usr/bin/Python
# -*- coding: utf-8 -*-
import jieba.analyse

file_name = './data/idf_test.txt'
topK = 10
content = open(file_name, 'rb').read()
jieba.analyse.set_stop_words('./data/stop_words')
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt',
                     'nz', 't', 'v', 'vd', 'vn', 'eng']
print('tf-idf : ')
for x, w in jieba.analyse.extract_tags(content, withWeight=True, topK=topK,
                                       allowPOS=allow_speech_tags):
    print('%s %s' % (x, w))

print('TextRank : ')
for x, w in jieba.analyse.textrank(content, withWeight=True, topK=topK,
                                   allowPOS=allow_speech_tags):
    print('%s %s' % (x, w))
