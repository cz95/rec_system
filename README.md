# Recommend System
Recommend System contain mainstream models and are updated rregularly.

## 1 Content-based recommendation (content_rec)
This chapter mainly implements the basic algorithms required by content-based recommendation system.

These algorithms can extract structured data based on user behavior and then select labels
### 1.1 Structured data
1. Keyword extraction —— TF-IDF(√)/TextRank(√)
2. Content classification —— FastText(√)
3. Entity recognition —— BiLSTM
4. Topic model —— LDA
5. Word Embedding —— Word2Vec(√)
### 1.2 Select labels
Filter labels extracted from structured data
1. Chi-square
2. Comentropy


## 2 Neighbor-based recommendation (neighbor_rec)
This chapter mainly implements the basic algorithms of recommendation system based on collaborative filtering. 

Collaborative-filtering includes Memory-Based CF and Model-Based CF.

The  Memory-Based CF is mainly introduced.

1. User-based collaborative filtering —— usercf(√)
2. Item-based collaborative filtering —— itemcf(√)

## 3 Matrix-decomposition recommendation (matrix_rec)
This chapter mainly implements the basic algorithm of recommendation system based on matrix decomposition.

The most commonly matrix decomposition is SVD.

1. Regularized singular value decomposition —— RSVD(√)