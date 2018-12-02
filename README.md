# movie_recommend
电影推荐系统，内含主流模型，不定期更新模型

## 数据集地址：
数据集（小）：http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
数据集（大）：http://files.grouplens.org/datasets/movielens/ml-latest.zip

## 评价指标(indicators)
- 精确率：推荐正确item集合 / 所有推荐item集合
- 召回率：推荐正确item集合 / 用户在测试集上看的item集合
- 覆盖率：最终推荐的item / 所有item
- 新颖性：推荐物品越热门，新颖性越低

## 模型介绍(model)
- usercf.py：基于用户的协同过滤来做推荐，利用余弦相似性

