import numpy as np
import csv

profiles_file = 'profiles.csv'
train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'
artists_file = 'artists.csv'

# Load the training data.
train_data = {}
num_users = 0.
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])

        if not user in train_data:
            train_data[user] = {'artists':{}}
            num_users += 1

        train_data[user]['artists'][artist] = plays

with open(profiles_file, 'r') as prof_fh:
    profiles_csv = csv.reader(prof_fh, delimiter=',', quotechar='"')
    next(profiles_csv, None)
    for row in profiles_csv:
        user  = row[0]
        if not user in train_data:
            train_data[user] = {'artists':{}}
            num_users += 1
        train_data[user]['sex'] = row[1]
        train_data[user]['age'] = row[2]
        train_data[user]['country'] = row[3]

pca_data_vectors = []
artists = []

with open(artists_file, 'r') as art_fh:
    artists_csv = csv.reader(art_fh, delimiter=',', quotechar='"')
    next(artists_csv, None)
    for row in artists_csv:
        artists.append(row[0])

train_data_items = train_data.items()
num_train = len(train_data_items)
train_data1 = dict(train_data_items[:num_train/4])
train_data2 = dict(train_data_items[num_train/4:num_train/2])
train_data3 = dict(train_data_items[num_train/2:num_train*3/4])
train_data4 = dict(train_data_items[num_train*3/4:])

"""
for user in train_data:
    for artist in artists:
        if not (train_data[user]['artists']).get(artist):
            train_data[user]['artists'][artist] = 0
    data_vector = np.array([train_data[user]['artists'][artist] for artist in train_data[user]['artists']])
    pca_data_vectors.append(data_vector)
"""
for user in train_data1:
    for artist in artists:
        if not (train_data1[user]['artists']).get(artist):
            train_data1[user]['artists'][artist] = 0
    data_vector = np.array([train_data1[user]['artists'][artist] for artist in train_data1[user]['artists']])
    pca_data_vectors.append(data_vector)

for user in train_data2:
    for artist in artists:
        if not (train_data2[user]['artists']).get(artist):
          train_data2[user]['artists'][artist] = 0
    data_vector = np.array([train_data2[user]['artists'][artist] for artist in train_data2[user]['artists']])
    pca_data_vectors.append(data_vector)

for user in train_data3:
    for artist in artists:
        if not (train_data3[user]['artists']).get(artist):
          train_data3[user]['artists'][artist] = 0
    data_vector = np.array([train_data3[user]['artists'][artist] for artist in train_data3[user]['artists']])
    pca_data_vectors.append(data_vector)

for user in train_data4:
    for artist in artists:
        if not (train_data4[user]['artists']).get(artist):
            train_data4[user]['artists'][artist] = 0
    data_vector = np.array([train_data4[user]['artists'][artist] for artist in train_data4[user]['artists']])
    pca_data_vectors.append(data_vector)

print pca_data_vectors[0]
pca_data_vectors = np.array(pca_data_vectors)
mean_vector = np.sum(pca_data_vectors, axis=0)/num_users

def get_dist_to_mean_matrix(data_vector, mean_vector):
    vec = np.matrix(data_vector-mean_vector)
    mat = np.multiply(vec.transpose(), vec)
    return mat

dist_to_mean_matrices = [get_dist_to_mean_matrix(user, mean_vector) for user in pca_data_vectors]

sample_covariance_matrix = np.sum(dist_to_mean_matrices, axis=0)/len(pca_data_vectors)

evalues, evectors = np.linalg.eig(sample_covariance_matrix)
alpha = [[np.dot((data_vector-mean_vector),evector) for evector in evectors]


