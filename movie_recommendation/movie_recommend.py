import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# reading users file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zipcode']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

# reading raings file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

# reading items file
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

# checking the content of each file
# print(users.shape)
# print(users.head())

# print(ratings.shape)
# print(ratings.head())

# print(items.shape)
# print(items.head())

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
# print(ratings_train.shape, ratings_test.shape)


# Collaborative filtering model
# no of unique users
n_users = ratings.user_id.unique().shape[0]

# no of unique items
n_items = ratings.movie_id.unique().shape[0]

# User - Item matrix to calculate similarity between users and items
similarity_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():
	similarity_matrix[line[1]-1, line[2]-1] = line[3]

# print(similarity_matrix)

# pairwise distance using cosine similarity
from sklearn.metrics.pairwise import pairwise_distances 

user_similarity = pairwise_distances(similarity_matrix, metric='cosine')
item_similarity = pairwise_distances(similarity_matrix.T, metric='cosine')

# print(user_similarity)
# print(item_similarity)

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        # Normalize the ratings by subtracting it from mean user rating
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Make preictions based on user similarity and item similarity

user_prediction = predict(similarity_matrix, user_similarity, type= 'user')
item_prediction = predict(similarity_matrix, item_similarity, type= 'item')
user_prediction.sort()
print(user_prediction)
