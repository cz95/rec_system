# Dataset
- Dataset(small): http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- Dataset(large): http://files.grouplens.org/datasets/movielens/ml-latest.zip

# Model introducation
## Completed
- rsvd.py: Regular SVD, is the most common form of SVD
- bpr.py: Bayesian personalized ranking. Compare with rsvd, I used numpy instead of pandas for optimization. SGD speed is significantly impoved.
## Uncompleted
- svd++.py: It can be used to calculate large user-item matrices.
- als.py: We can use ALS instead of SGD when updating parameters.
