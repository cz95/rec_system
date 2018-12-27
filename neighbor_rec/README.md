# Dataset
- Dataset(small): http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- Dataset(large): http://files.grouplens.org/datasets/movielens/ml-latest.zip

# Evaluation indicator
- Precision: (recommend correct items) / (all recommended items)
- Recall: (recommend correct items) / (all items that the user likes on the test set)
- Coverage: (all recommended items) / (all items)
- Popularity: The more popular the item, the less popularity it is

# Model introducation
## Completed
- usercf.py: Find the most similar m users and recommend top-n movies from them. Cosine similarity is utilized.
- itemcf.py: Find the matrix to directly find the similarity between all movies.Then recommend top-n movies based on user's movie records.
- lfm.py: Gradient descent is used to solve RSVD and save it, which is very slow. The time can be controlled by adjusting the number of feauture and the number of iterations.
- prank.py: Peasonal Rank algorithm was used for solving binary graphs, which was quite fast.
## Uncompleted
- Calculate the evaluation indicator of itemcf
- Improve the calculation of user similarity in usercf.py 
- Improve the calculation of item similarity in itemcf.py 
