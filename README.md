# Recommend System
Recommend System contain mainstream models and are updated rregularly.

## 1 Content-based recommendation（content_rec）
This chapter mainly implements the basic algorithms required by content-based recommendation system.

These algorithms can extract structured data based on user behavior and then select labels
### 1.1 Structured data
1. Keyword extraction —— TF-IDF（√）/TextRank（√）
2. Content classification —— FastText（√）
3. Entity recognition —— BiLSTM
4. Topic model —— LDA
5. Word Embedding —— Word2Vec(√)
### 1.2 Select labels
Filter labels extracted from structured data
1. Chi-square
2. Comentropy


## 2 Neighbor-based recommendation（neighbor_rec）
This chapter mainly implements the basic algorithms required by neighbor-based recommendation system.
### 2.1 User-based collaborative filtering

### 2.2 Item-based collaborative filtering

### 2.3 Model-based collaborative filtering